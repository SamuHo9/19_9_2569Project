"""Leakage-free, common trainer for the six non-PointNet coefficient models.

The runner uses the untouched ``All_coef`` cohorts.  Every fold fits its own
scaler and model on the fold's training subjects only.  Test data is read once
for final scoring and is never used for model selection, threshold selection,
early stopping, or preprocessing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_contract import (RAW_COEF_PROTOCOL, load_raw_cohort,
                           patient_group_id)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError('Torch is required. Use run_leakage_free_coef.ps1.') from exc


COHORTS = ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602')
SIDES = ('left', 'right')
MODELS = ('SVM', 'MLP', 'ResNet', 'ResNetAE', 'MobileNet', 'SqueezeNet')
META = {'Subject', 'Group', 'Class', 'BinaryClass', 'DataType'}
AUGMENTATION_PROTOCOL = 'within-class-mixup-jitter-v1'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def feature_columns(frame):
    cols = [c for c in frame.columns if c.startswith('Coef_')]
    cols.sort(key=lambda value: int(value.split('_')[1]))
    if cols != [f'Coef_{i}' for i in range(1, 508)]:
        raise ValueError(f'Expected Coef_1..Coef_507, got {cols[:3]}..{cols[-3:]} ({len(cols)})')
    return cols


def matrix(frame):
    cols = feature_columns(frame)
    x = frame[cols].to_numpy(dtype=np.float32)
    y = frame['BinaryClass'].to_numpy(dtype=np.int64)
    groups = np.asarray([patient_group_id(v) for v in frame['Subject']], dtype=str)
    if len(set(groups)) != len(groups):
        # Duplicate patient rows are allowed only when they share a group; the
        # grouped splitter below keeps them together.
        pass
    return x, y, groups, cols


def grouped_splits(y, groups, requested=5, seed=42):
    """Create deterministic stratified splits while keeping each group intact."""
    unique, first = np.unique(groups, return_index=True)
    group_y = y[first]
    if any((y[groups == group] != label).any() for group, label in zip(unique, group_y)):
        raise ValueError('A patient group contains conflicting binary labels')
    n_splits = min(requested, int(np.bincount(group_y, minlength=2).min()), len(unique))
    if n_splits < 2:
        raise ValueError('At least two groups per class are required for CV')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for group_train, group_val in skf.split(unique, group_y):
        train_groups = set(unique[group_train])
        val_groups = set(unique[group_val])
        train_idx = np.asarray([i for i, group in enumerate(groups) if group in train_groups], dtype=int)
        val_idx = np.asarray([i for i, group in enumerate(groups) if group in val_groups], dtype=int)
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise AssertionError('Grouped split overlap')
        splits.append((train_idx, val_idx))
    return splits


def metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    result = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'sensitivity': float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        'specificity': float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        'f1_macro': float(f1_score(y_true, pred, average='macro', zero_division=0)),
        'threshold': float(threshold),
    }
    try:
        result['roc_auc'] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        result['roc_auc'] = None
    try:
        result['pr_auc'] = float(average_precision_score(y_true, probabilities))
    except ValueError:
        result['pr_auc'] = None
    return result


def augment_training_fold(x, y, seed, method='balanced_jitter', noise_scale=0.02,
                          children_per_pair=8):
    """Balance a training fold without PLS-DA or cross-fold information.

    Synthetic rows are generated only from real rows in the same class and
    fold.  A convex within-class mix followed by small feature-wise Gaussian
    jitter keeps the class label fixed.  Validation and test rows never enter
    this function.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    counts = {label: int((y == label).sum()) for label in (0, 1)}
    info = {
        'augmentation_protocol': method if method != 'none' else 'none',
        'children_per_pair': int(children_per_pair),
        'original_train_n': int(len(y)),
        'synthetic_train_n': 0,
        'fit_train_n': int(len(y)),
        'original_class_0': counts[0], 'original_class_1': counts[1],
        'fit_class_0': counts[0], 'fit_class_1': counts[1],
    }
    if method == 'none' or counts[0] == counts[1]:
        return x, y, info
    if method != 'balanced_jitter':
        raise ValueError(f'Unknown augmentation method: {method}')
    if children_per_pair < 1:
        raise ValueError('children_per_pair must be positive')
    target = max(counts.values())
    rng = np.random.default_rng(seed)
    feature_std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    generated_x, generated_y = [], []
    for label in (0, 1):
        needed = target - counts[label]
        if needed <= 0:
            continue
        base = x[y == label]
        n_pairs = int(np.ceil(needed / children_per_pair))
        pair_first = rng.integers(0, len(base), size=n_pairs)
        pair_second = rng.integers(0, len(base), size=n_pairs)
        first = np.repeat(pair_first, children_per_pair)[:needed]
        second = np.repeat(pair_second, children_per_pair)[:needed]
        alpha = rng.uniform(0.25, 0.75, size=(needed, 1)).astype(np.float32)
        synthetic = alpha * base[first] + (1.0 - alpha) * base[second]
        synthetic += rng.normal(0.0, noise_scale * feature_std,
                                size=synthetic.shape).astype(np.float32)
        generated_x.append(synthetic)
        generated_y.append(np.full(needed, label, dtype=np.int64))
    if generated_x:
        x = np.vstack([x] + generated_x).astype(np.float32)
        y = np.concatenate([y] + generated_y).astype(np.int64)
        order = rng.permutation(len(y))
        x, y = x[order], y[order]
    info['synthetic_train_n'] = int(len(y) - info['original_train_n'])
    info['fit_train_n'] = int(len(y))
    info['fit_class_0'] = int((y == 0).sum())
    info['fit_class_1'] = int((y == 1).sum())
    return x, y, info


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)


class ResNet1D(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.GroupNorm(1, 32), nn.ReLU())
        self.body = nn.Sequential(ResidualBlock(32), nn.Conv1d(32, 64, 3, stride=2, padding=1),
                                   nn.GroupNorm(1, 64), nn.ReLU(), ResidualBlock(64),
                                   nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(1, 128),
                                   nn.ReLU(), ResidualBlock(128))
        # Keep a small ordered grid of coefficient positions instead of
        # collapsing all 507 positions into one global average.  The latter
        # made the convolutional models nearly constant on coefficient data.
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 8, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


class ResNetAE1D(ResNet1D):
    def __init__(self, dropout=None):
        super().__init__(dropout=dropout)
        self.decode = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 507))

    def forward(self, x):
        latent = self.body(self.stem(x))
        pooled = F.adaptive_avg_pool1d(latent, 1).flatten(1)
        return self.head(latent), self.decode(pooled)


class MobileNet1D(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.GroupNorm(1, 16), nn.ReLU6(),
            nn.Conv1d(16, 16, 3, stride=2, padding=1, groups=16), nn.GroupNorm(1, 16), nn.ReLU6(),
            nn.Conv1d(16, 32, 1), nn.GroupNorm(1, 32), nn.ReLU6(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, groups=32), nn.GroupNorm(1, 32), nn.ReLU6(),
            nn.Conv1d(32, 64, 1), nn.GroupNorm(1, 64), nn.ReLU6(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, groups=64), nn.GroupNorm(1, 64), nn.ReLU6(),
            nn.Conv1d(64, 128, 1), nn.GroupNorm(1, 128), nn.ReLU6(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 8, 1))

    def forward(self, x):
        return self.head(self.features(x))


class FireBlock(nn.Module):
    def __init__(self, inp, squeeze, out):
        super().__init__()
        self.squeeze = nn.Conv1d(inp, squeeze, 1)
        self.e1 = nn.Conv1d(squeeze, out // 2, 1)
        self.e3 = nn.Conv1d(squeeze, out // 2, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.e1(x)), F.relu(self.e3(x))], dim=1)


class SqueezeNet1D(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        self.features = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                                      FireBlock(16, 8, 32), FireBlock(32, 16, 64),
                                      FireBlock(64, 32, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 8, 1))

    def forward(self, x):
        return self.head(self.features(x))


class MLP1D(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        if dropout is None:
            dropout1, dropout2 = 0.25, 0.2
        else:
            dropout1 = dropout2 = float(dropout)
        self.net = nn.Sequential(nn.Linear(507, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout1),
                                 nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout2),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.flatten(1))


def make_torch_model(name, dropout=None):
    return {'MLP': MLP1D, 'ResNet': ResNet1D, 'ResNetAE': ResNetAE1D,
            'MobileNet': MobileNet1D, 'SqueezeNet': SqueezeNet1D}[name](dropout=dropout)


def tensor_for_model(name, values):
    """Use a flat tensor for MLP and a one-channel sequence for Conv1D models."""
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor if name == 'MLP' else tensor.unsqueeze(1)


def train_torch_fold(name, x_train, y_train, x_val, y_val, device, seed, epochs, patience,
                     lr=1e-3, weight_decay=1e-3, batch_size=32,
                     recon_weight=0.05, dropout=None):
    seed_everything(seed)
    model = make_torch_model(name, dropout=dropout).to(device)
    pos = max(1, int((y_train == 0).sum()))
    neg = max(1, int((y_train == 1).sum()))
    pos_weight = torch.tensor([pos / neg], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    tx = tensor_for_model(name, x_train)
    ty = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    vx = tensor_for_model(name, x_val)
    vy = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(tx, ty)
    batch = min(int(batch_size), max(1, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == 'ResNetAE':
                logits, recon = model(bx)
                loss = criterion(logits, by) + float(recon_weight) * F.mse_loss(recon, bx.flatten(1))
            else:
                logits = model(bx)
                loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(vx.to(device))
            if name == 'ResNetAE':
                val_logits = val_logits[0]
            val_loss = criterion(val_logits, vy.to(device)).item()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(vx.to(device))
        if name == 'ResNetAE':
            logits = logits[0]
        val_prob = torch.sigmoid(logits).flatten().cpu().numpy()
    return model, val_prob, best_epoch


def run_svm(x, y, splits, seed, augmentation='balanced_jitter', noise_scale=0.02,
            children_per_pair=8, C=1.0, gamma='scale'):
    oof = np.zeros(len(y), dtype=float)
    test_models = []
    fold_info = []
    # Defaults preserve the original fixed baseline; tuning callers can pass C/gamma.
    base = Pipeline([('scaler', StandardScaler()),
                     ('model', SVC(C=float(C), kernel='rbf', gamma=gamma, class_weight='balanced',
                                   probability=True, random_state=seed))])
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, aug_info = augment_training_fold(
            x[tr], y[tr], seed + fold, augmentation, noise_scale, children_per_pair)
        model = clone(base)
        model.fit(fit_x, fit_y)
        oof[va] = model.predict_proba(x[va])[:, 1]
        test_models.append(model)
        aug_info.update({'fold': fold + 1, 'raw_train_n': int(len(tr)), 'val_n': int(len(va))})
        fold_info.append(aug_info)
    return oof, test_models, fold_info


def run_torch(name, x, y, splits, test_x, device, seed, epochs, patience,
              augmentation='balanced_jitter', noise_scale=0.02, children_per_pair=8,
              config=None):
    config = dict(config or {})
    oof = np.zeros(len(y), dtype=float)
    test_probs = []
    fold_info = []
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, aug_info = augment_training_fold(
            x[tr], y[tr], seed + fold, augmentation, noise_scale, children_per_pair)
        scaler = StandardScaler()
        xtr = scaler.fit_transform(fit_x).astype(np.float32)
        xva = scaler.transform(x[va]).astype(np.float32)
        xte = scaler.transform(test_x).astype(np.float32)
        model, val_prob, best_epoch = train_torch_fold(
            name, xtr, fit_y, xva, y[va], device, seed + fold, epochs, patience,
            lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 1e-3),
            batch_size=config.get('batch_size', 32),
            recon_weight=config.get('recon_weight', 0.05),
            dropout=config.get('dropout'))
        oof[va] = val_prob
        with torch.no_grad():
            logits = model(tensor_for_model(name, xte).to(device))
            if name == 'ResNetAE':
                logits = logits[0]
            test_probs.append(torch.sigmoid(logits).flatten().cpu().numpy())
        aug_info.update({'fold': fold + 1, 'best_epoch': int(best_epoch),
                         'raw_train_n': int(len(tr)), 'val_n': int(len(va))})
        fold_info.append(aug_info)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, np.mean(test_probs, axis=0), fold_info


def run_one(cohort, side, model_name, args):
    seed_everything(args.seed)
    train_df, test_df = load_raw_cohort(cohort, side, kind='coef')
    x, y, groups, columns = matrix(train_df)
    test_x, test_y, test_groups, _ = matrix(test_df)
    if set(groups) & set(test_groups):
        raise ValueError(f'Patient group overlap before training: {cohort}/{side}')
    splits = grouped_splits(y, groups, args.folds, args.seed)
    output_root = Path(args.output_root).resolve()
    run_token = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / cohort / side / model_name / run_token
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f'[{cohort}/{side}/{model_name}] train={len(train_df)} test={len(test_df)} folds={len(splits)}')
    if model_name == 'SVM':
        oof, models, fold_info = run_svm(x, y, splits, args.seed,
                                         args.augmentation, args.noise_scale,
                                         args.children_per_pair)
        # Each fold model owns its scaler; test prediction is the fold ensemble.
        test_prob = np.mean([m.predict_proba(test_x)[:, 1] for m in models], axis=0)
        with (run_dir / 'models.pkl').open('wb') as stream:
            pickle.dump(models, stream)
    else:
        device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        oof, test_prob, fold_info = run_torch(model_name, x, y, splits, test_x, device,
                                              args.seed, args.epochs, args.patience,
                                              args.augmentation, args.noise_scale,
                                              args.children_per_pair)
    cv_metrics = metrics(y, oof)
    test_metrics = metrics(test_y, test_prob)
    pred = pd.DataFrame({'Subject': test_df['Subject'].astype(str), 'PatientID': test_groups,
                         'BinaryClass': test_y, 'Probability': test_prob,
                         'Prediction': (test_prob >= 0.5).astype(int)})
    pred.to_csv(run_dir / 'test_predictions.csv', index=False)
    (run_dir / 'metrics.json').write_text(json.dumps({'cv_oof': cv_metrics, 'test': test_metrics}, indent=2), encoding='utf-8')
    manifest = {
        'protocol': RAW_COEF_PROTOCOL,
        'cohort': cohort, 'side': side, 'model': model_name,
        'feature_kind': 'coef', 'pls_da_applied_to_model_input': False,
        'train_rows': len(train_df), 'test_rows': len(test_df), 'features': columns,
        'train_classes': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'test_classes': {str(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'train_feature_sha256': digest_frame(train_df), 'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False, 'folds': fold_info, 'seed': args.seed,
        'augmentation': args.augmentation,
        'augmentation_protocol': (AUGMENTATION_PROTOCOL if args.augmentation != 'none' else 'none'),
        'augmentation_noise_scale': args.noise_scale,
        'augmentation_children_per_pair': args.children_per_pair,
        'augmentation_applied_to_validation': False,
        'augmentation_applied_to_test': False,
        'synthetic_training_rows_total': int(sum(item['synthetic_train_n'] for item in fold_info)),
        'device': str(args.device), 'elapsed_seconds': round(time.time() - started, 2),
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'  CV balanced_accuracy={cv_metrics["balanced_accuracy"]:.4f} | test accuracy={test_metrics["accuracy"]:.4f} | test balanced_accuracy={test_metrics["balanced_accuracy"]:.4f} | test F1={test_metrics["f1_macro"]:.4f}')
    return {'Cohort': cohort, 'Side': side, 'Model': model_name,
            'CV_BalancedAccuracy': cv_metrics['balanced_accuracy'],
            'CV_F1_Macro': cv_metrics['f1_macro'],
            'Test_Accuracy': test_metrics['accuracy'],
            'Test_BalancedAccuracy': test_metrics['balanced_accuracy'],
            'Test_Sensitivity': test_metrics['sensitivity'],
            'Test_Specificity': test_metrics['specificity'],
            'Test_F1_Macro': test_metrics['f1_macro'],
            'Test_ROC_AUC': test_metrics['roc_auc'],
            'RunDir': str(run_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=COHORTS + ('all',), default='all')
    parser.add_argument('--side', choices=SIDES + ('all',), default='all')
    parser.add_argument('--model', choices=MODELS + ('all',), default='all')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--augmentation', choices=('none', 'balanced_jitter'), default='balanced_jitter')
    parser.add_argument('--noise_scale', type=float, default=0.02)
    parser.add_argument('--children_per_pair', type=int, default=8)
    parser.add_argument('--output_root', default=str(Path(__file__).resolve().parent / 'leakage_free_runs'))
    args = parser.parse_args()
    cohorts = COHORTS if args.cohort == 'all' else (args.cohort,)
    sides = SIDES if args.side == 'all' else (args.side,)
    models = MODELS if args.model == 'all' else (args.model,)
    rows = []
    for cohort in cohorts:
        for side in sides:
            for model_name in models:
                try:
                    rows.append(run_one(cohort, side, model_name, args))
                except Exception as exc:
                    print(f'[FAIL] {cohort}/{side}/{model_name}: {exc}')
                    rows.append({'Cohort': cohort, 'Side': side, 'Model': model_name, 'Status': 'FAIL', 'Error': repr(exc)})
    summary = Path(args.output_root).resolve() / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)
    failed = [row for row in rows if row.get('Status') == 'FAIL']
    print(f'Wrote summary: {summary}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
