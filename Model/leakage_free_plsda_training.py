"""Leakage-free PLS-DA augmentation runner.

PLS-DA is used only to choose same-class parent pairs and reconstruct synthetic
coefficient rows.  The PLS model, parent pairing, scaler, early stopping and
classifier are fitted separately inside every training fold.  Validation and
test rows are always original and untouched.
"""
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
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_contract import RAW_COEF_PROTOCOL, load_raw_cohort, patient_group_id
from leakage_free_coef_training import (
    COHORTS, MODELS, SIDES, grouped_splits, matrix, metrics, run_torch,
    seed_everything, tensor_for_model, train_torch_fold,
)


PLSDA_PROTOCOL = 'fold-local-plsda-augmentation-v1'


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def pair_indices(indices, scores, rng):
    pool = list(np.asarray(indices, dtype=int))
    rng.shuffle(pool)
    pairs = []
    while len(pool) >= 2:
        first = pool.pop(0)
        distances = np.linalg.norm(scores[pool] - scores[first], axis=1)
        second = pool.pop(int(np.argmin(distances)))
        pairs.append((first, second))
    return pairs


def plsda_augment_training_fold(x, y, seed, children_per_pair=8, n_components=8):
    """Fit PLS-DA and synthesize only the under-represented fold class."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    counts = {label: int((y == label).sum()) for label in (0, 1)}
    info = {
        'augmentation_protocol': PLSDA_PROTOCOL,
        'pls_fit_scope': 'training_fold_only',
        'pls_components': 0,
        'children_per_pair': int(children_per_pair),
        'original_train_n': int(len(y)),
        'synthetic_train_n': 0,
        'fit_train_n': int(len(y)),
        'original_class_0': counts[0], 'original_class_1': counts[1],
        'fit_class_0': counts[0], 'fit_class_1': counts[1],
    }
    if counts[0] == counts[1]:
        return x.astype(np.float32), y, info
    if children_per_pair < 1:
        raise ValueError('children_per_pair must be positive')
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    y_onehot = np.column_stack([(y == 0).astype(float), (y == 1).astype(float)])
    n_comp = min(int(n_components), x.shape[1], len(y) - 1)
    if n_comp < 1:
        raise ValueError('Not enough fold training rows for PLS-DA')
    pls = PLSRegression(n_components=n_comp, scale=False, max_iter=1000)
    scores = pls.fit_transform(x_scaled, y_onehot)[0]
    info['pls_components'] = int(n_comp)
    rng = np.random.default_rng(seed)
    target = max(counts.values())
    generated_x, generated_y = [], []
    for label in (0, 1):
        needed = target - counts[label]
        if needed <= 0:
            continue
        class_indices = np.flatnonzero(y == label)
        pairs = pair_indices(class_indices, scores, rng)
        if not pairs:
            # This is only a fallback for a very small fold class.  It still
            # uses a training-fold row and never reaches validation/test.
            pairs = [(int(class_indices[0]), int(class_indices[0]))]
        rows = []
        for child_idx in range(needed):
            pair = pairs[(child_idx // children_per_pair) % len(pairs)]
            alpha = (0.1 + 0.8 * ((child_idx % children_per_pair) + 0.5) /
                     children_per_pair)
            score_new = ((1.0 - alpha) * scores[pair[0]] + alpha * scores[pair[1]])[None, :]
            reconstructed_scaled = pls.inverse_transform(score_new)
            reconstructed = scaler.inverse_transform(reconstructed_scaled)[0]
            rows.append(reconstructed)
        generated_x.append(np.asarray(rows, dtype=np.float64))
        generated_y.append(np.full(needed, label, dtype=np.int64))
    if generated_x:
        x = np.vstack([x] + generated_x)
        y = np.concatenate([y] + generated_y)
        order = rng.permutation(len(y))
        x, y = x[order], y[order]
    if not np.isfinite(x).all():
        raise ValueError('PLS-DA reconstruction produced nonfinite features')
    info['synthetic_train_n'] = int(len(y) - info['original_train_n'])
    info['fit_train_n'] = int(len(y))
    info['fit_class_0'] = int((y == 0).sum())
    info['fit_class_1'] = int((y == 1).sum())
    return x.astype(np.float32), y.astype(np.int64), info


def run_svm_plsda(x, y, splits, seed, children_per_pair, n_components):
    oof = np.zeros(len(y), dtype=float)
    models, fold_info = [], []
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
                      probability=True, random_state=seed)),
    ])
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, info = plsda_augment_training_fold(
            x[tr], y[tr], seed + fold, children_per_pair, n_components)
        model = clone(base).fit(fit_x, fit_y)
        oof[va] = model.predict_proba(x[va])[:, 1]
        models.append(model)
        info.update({'fold': fold + 1, 'raw_train_n': int(len(tr)), 'val_n': int(len(va))})
        fold_info.append(info)
    return oof, models, fold_info


def run_torch_plsda(name, x, y, splits, test_x, device, seed, epochs, patience,
                    children_per_pair, n_components):
    oof = np.zeros(len(y), dtype=float)
    test_probs, fold_info = [], []
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, info = plsda_augment_training_fold(
            x[tr], y[tr], seed + fold, children_per_pair, n_components)
        scaler = StandardScaler().fit(fit_x)
        xtr = scaler.transform(fit_x).astype(np.float32)
        xva = scaler.transform(x[va]).astype(np.float32)
        xte = scaler.transform(test_x).astype(np.float32)
        model, val_prob, best_epoch = train_torch_fold(
            name, xtr, fit_y, xva, y[va], device, seed + fold, epochs, patience)
        oof[va] = val_prob
        import torch
        with torch.no_grad():
            logits = model(tensor_for_model(name, xte).to(device))
            if name == 'ResNetAE':
                logits = logits[0]
            test_probs.append(torch.sigmoid(logits).flatten().cpu().numpy())
        info.update({'fold': fold + 1, 'best_epoch': int(best_epoch),
                     'raw_train_n': int(len(tr)), 'val_n': int(len(va))})
        fold_info.append(info)
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
        raise ValueError(f'Patient group overlap: {cohort}/{side}')
    splits = grouped_splits(y, groups, args.folds, args.seed)
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / cohort / side / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f'[{cohort}/{side}/{model_name}] train={len(train_df)} test={len(test_df)} folds={len(splits)}')
    if model_name == 'SVM':
        oof, models, fold_info = run_svm_plsda(
            x, y, splits, args.seed, args.children_per_pair, args.n_components)
        test_prob = np.mean([m.predict_proba(test_x)[:, 1] for m in models], axis=0)
        with (run_dir / 'models.pkl').open('wb') as stream:
            pickle.dump(models, stream)
    else:
        import torch
        device = torch.device(args.device if args.device != 'auto' else
                              ('cuda' if torch.cuda.is_available() else 'cpu'))
        oof, test_prob, fold_info = run_torch_plsda(
            model_name, x, y, splits, test_x, device, args.seed, args.epochs,
            args.patience, args.children_per_pair, args.n_components)
    cv_metrics = metrics(y, oof)
    test_metrics = metrics(test_y, test_prob)
    pd.DataFrame({
        'Subject': test_df['Subject'].astype(str), 'PatientID': test_groups,
        'BinaryClass': test_y, 'Probability': test_prob,
        'Prediction': (test_prob >= 0.5).astype(int),
    }).to_csv(run_dir / 'test_predictions.csv', index=False)
    (run_dir / 'metrics.json').write_text(
        json.dumps({'cv_oof': cv_metrics, 'test': test_metrics}, indent=2), encoding='utf-8')
    manifest = {
        'protocol': PLSDA_PROTOCOL, 'source_protocol': RAW_COEF_PROTOCOL,
        'cohort': cohort, 'side': side, 'model': model_name,
        'feature_kind': 'coef', 'model_input_features': 'Coef_1..Coef_507',
        'pls_da_used_for_augmentation': True,
        'pls_da_fit_scope': 'training_fold_only',
        'pls_da_applied_to_model_input': False,
        'pls_components_requested': args.n_components,
        'train_rows': len(train_df), 'test_rows': len(test_df), 'features': columns,
        'train_classes': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'test_classes': {str(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'train_feature_sha256': digest_frame(train_df), 'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False, 'folds': fold_info, 'seed': args.seed,
        'children_per_pair': args.children_per_pair,
        'augmentation_applied_to_validation': False,
        'augmentation_applied_to_test': False,
        'synthetic_training_rows_total': int(sum(item['synthetic_train_n'] for item in fold_info)),
        'device': str(args.device), 'elapsed_seconds': round(time.time() - started, 2),
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'  CV balanced_accuracy={cv_metrics["balanced_accuracy"]:.4f} | '
          f'test accuracy={test_metrics["accuracy"]:.4f} | '
          f'test balanced_accuracy={test_metrics["balanced_accuracy"]:.4f} | '
          f'test F1={test_metrics["f1_macro"]:.4f}')
    return {
        'Cohort': cohort, 'Side': side, 'Model': model_name,
        'CV_BalancedAccuracy': cv_metrics['balanced_accuracy'],
        'CV_F1_Macro': cv_metrics['f1_macro'],
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
    parser.add_argument('--side', choices=SIDES + ('all',), default='all')
    parser.add_argument('--model', choices=MODELS + ('all',), default='all')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--children_per_pair', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=8)
    parser.add_argument('--output_root', default=str(Path(__file__).resolve().parent / 'leakage_free_plsda_runs'))
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
                    rows.append({'Cohort': cohort, 'Side': side, 'Model': model_name,
                                 'Status': 'FAIL', 'Error': repr(exc)})
    summary = Path(args.output_root).resolve() / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)
    failed = [row for row in rows if row.get('Status') == 'FAIL']
    print(f'Wrote summary: {summary}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
