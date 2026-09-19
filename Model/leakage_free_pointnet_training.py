"""Leakage-free PointNet baseline on original SPHARM XYZ point clouds."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_contract import RAW_COEF_PROTOCOL, load_raw_xyz_cohort, patient_group_id
from leakage_free_coef_training import COHORTS, grouped_splits, metrics, seed_everything
from leakage_free_coef_training import augment_training_fold


POINTNET_PROTOCOL = 'original-xyz-pointnet-no-pls-v1'


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(batch, 512)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k)
        return (x + identity).view(-1, self.k, self.k)


class PointNetDual(nn.Module):
    def __init__(self, num_classes=2, dropout=None):
        super().__init__()
        dropout = 0.4 if dropout is None else float(dropout)
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        transform = self.input_transform(x)
        x = torch.bmm(transform, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        global_feat = torch.cat([torch.max(x, 2)[0], torch.mean(x, 2)], dim=1)
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


def xyz_matrix(frame):
    columns = [f'{axis}_{i}' for i in range(1002) for axis in ('x', 'y', 'z')]
    if [c for c in frame.columns if c.startswith(('x_', 'y_', 'z_'))] != columns:
        raise ValueError('Expected x_0,y_0,z_0 through x_1001,y_1001,z_1001')
    points = frame[columns].to_numpy(dtype=np.float32).reshape(-1, 1002, 3)
    # Match the existing PointNet preprocessing while fitting no statistics
    # across subjects: each point cloud is normalized independently.
    points = (points - points.mean(axis=1, keepdims=True)) / (points.std(axis=1, keepdims=True) + 1e-8)
    return np.transpose(points, (0, 2, 1)), frame.BinaryClass.to_numpy(dtype=np.int64), \
        np.asarray([patient_group_id(v) for v in frame.Subject], dtype=str)


def augment_xyz_fold(x, y, seed, method='balanced_jitter', noise_scale=0.02, children_per_pair=8):
    flat, labels, info = augment_training_fold(
        x.reshape(len(x), -1), y, seed, method, noise_scale, children_per_pair)
    return flat.reshape(-1, 3, 1002), labels, info


def train_fold(x_train, y_train, x_val, y_val, device, seed, epochs, patience,
               lr=1e-3, weight_decay=1e-3, batch_size=None, dropout=None):
    seed_everything(seed)
    model = PointNetDual(num_classes=2, dropout=dropout).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = torch.tensor(len(y_train) / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    tx = torch.tensor(x_train, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.long)
    vx = torch.tensor(x_val, dtype=torch.float32, device=device)
    vy = torch.tensor(y_val, dtype=torch.long, device=device)
    if batch_size is None:
        batch_size = min(32, max(8, len(tx) // 4))
    batch_size = min(int(batch_size), max(1, len(tx)))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size,
                        shuffle=True, drop_last=(len(tx) >= batch_size))
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, stale = float('inf'), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(vx), vy).item()
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
        val_prob = F.softmax(model(vx), dim=1)[:, 1].cpu().numpy()
    return model, val_prob, best_epoch


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def run_one(cohort, side, args):
    seed_everything(args.seed)
    train_df, test_df = load_raw_xyz_cohort(cohort, side)
    x, y, groups = xyz_matrix(train_df)
    test_x, test_y, test_groups = xyz_matrix(test_df)
    if set(groups) & set(test_groups):
        raise ValueError(f'Patient overlap: {cohort}/{side}')
    splits = grouped_splits(y, groups, args.folds, args.seed)
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / cohort / side / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    oof = np.zeros(len(y), dtype=float)
    test_probs, fold_info, state_dicts = [], [], []
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[{cohort}/{side}/PointNet] train={len(train_df)} test={len(test_df)} folds={len(splits)}')
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, info = augment_xyz_fold(
            x[tr], y[tr], args.seed + fold, args.augmentation,
            args.noise_scale, args.children_per_pair)
        model, val_prob, best_epoch = train_fold(
            fit_x, fit_y, x[va], y[va], device, args.seed + fold,
            args.epochs, args.patience)
        oof[va] = val_prob
        with torch.no_grad():
            test_prob = F.softmax(model(torch.tensor(test_x, dtype=torch.float32, device=device)), dim=1)[:, 1]
            test_probs.append(test_prob.cpu().numpy())
        info.update({'fold': fold + 1, 'best_epoch': int(best_epoch),
                     'raw_train_n': int(len(tr)), 'val_n': int(len(va))})
        fold_info.append(info)
        state_dicts.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_prob = np.mean(test_probs, axis=0)
    cv_metrics, test_metrics = metrics(y, oof), metrics(test_y, test_prob)
    pd.DataFrame({
        'Subject': test_df.Subject.astype(str), 'PatientID': test_groups,
        'BinaryClass': test_y, 'Probability': test_prob,
        'Prediction': (test_prob >= 0.5).astype(int),
    }).to_csv(run_dir / 'test_predictions.csv', index=False)
    torch.save(state_dicts, run_dir / 'fold_models.pt')
    (run_dir / 'metrics.json').write_text(json.dumps({'cv_oof': cv_metrics, 'test': test_metrics}, indent=2), encoding='utf-8')
    manifest = {
        'protocol': POINTNET_PROTOCOL, 'source_protocol': RAW_COEF_PROTOCOL,
        'cohort': cohort, 'side': side, 'model': 'PointNet',
        'feature_kind': 'xyz', 'points': 1002, 'point_order': 'x,y,z',
        'pls_da_applied_to_model_input': False,
        'augmentation': args.augmentation, 'augmentation_protocol': 'within-class-mixup-jitter-v1',
        'augmentation_applied_to_validation': False, 'augmentation_applied_to_test': False,
        'train_rows': len(train_df), 'test_rows': len(test_df),
        'train_classes': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'test_classes': {str(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'train_feature_sha256': digest_frame(train_df), 'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False, 'folds': fold_info, 'seed': args.seed,
        'children_per_pair': args.children_per_pair, 'device': str(device),
        'elapsed_seconds': round(time.time() - started, 2),
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'  CV balanced_accuracy={cv_metrics["balanced_accuracy"]:.4f} | '
          f'test accuracy={test_metrics["accuracy"]:.4f} | '
          f'test balanced_accuracy={test_metrics["balanced_accuracy"]:.4f} | '
          f'test F1={test_metrics["f1_macro"]:.4f}')
    return {
        'Cohort': cohort, 'Side': side, 'Model': 'PointNet',
        'CV_BalancedAccuracy': cv_metrics['balanced_accuracy'],
        'Test_Accuracy': test_metrics['accuracy'],
        'Test_BalancedAccuracy': test_metrics['balanced_accuracy'],
        'Test_Sensitivity': test_metrics['sensitivity'],
        'Test_Specificity': test_metrics['specificity'],
        'Test_F1_Macro': test_metrics['f1_macro'],
        'Test_ROC_AUC': test_metrics['roc_auc'], 'RunDir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=COHORTS + ('all',), default='all')
    parser.add_argument('--side', choices=('left', 'right', 'all'), default='all')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--augmentation', choices=('none', 'balanced_jitter'), default='balanced_jitter')
    parser.add_argument('--noise_scale', type=float, default=0.02)
    parser.add_argument('--children_per_pair', type=int, default=8)
    parser.add_argument('--output_root', default=str(Path(__file__).resolve().parent / 'pointnet_runs'))
    args = parser.parse_args()
    cohorts = COHORTS if args.cohort == 'all' else (args.cohort,)
    sides = ('left', 'right') if args.side == 'all' else (args.side,)
    rows = []
    for cohort in cohorts:
        for side in sides:
            try:
                rows.append(run_one(cohort, side, args))
            except Exception as exc:
                print(f'[FAIL] {cohort}/{side}/PointNet: {exc}')
                rows.append({'Cohort': cohort, 'Side': side, 'Model': 'PointNet',
                             'Status': 'FAIL', 'Error': repr(exc)})
    summary = Path(args.output_root).resolve() / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f'Wrote summary: {summary}')
    if any(row.get('Status') == 'FAIL' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
