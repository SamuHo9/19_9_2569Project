"""Leakage-free PointNet with fold-local PLS-DA augmentation.

This runner is deliberately separate from the raw-XYZ PointNet baseline.  It
fits PLS-DA only on each training fold, uses the PLS score space to pair
same-class point clouds, reconstructs synthetic *normalized XYZ* clouds, and
then trains PointNet on the original plus synthetic training clouds.  The
validation and held-out test clouds remain original and untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from data_contract import RAW_COEF_PROTOCOL, load_raw_xyz_cohort, patient_group_id
from leakage_free_coef_training import COHORTS, grouped_splits, metrics, seed_everything
from leakage_free_pointnet_training import PointNetDual, train_fold, xyz_matrix


POINTNET_PLSDA_PROTOCOL = 'fold-local-plsda-pointnet-augmentation-v1'
POINTNET_PLSDA_MODEL = 'PointNet_PLSDA'
N_POINTS = 1002
N_AXES = 3


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def pair_indices(indices, scores, rng):
    """Pair nearby same-class rows in PLS score space."""
    pool = list(np.asarray(indices, dtype=int))
    rng.shuffle(pool)
    pairs = []
    while len(pool) >= 2:
        first = pool.pop(0)
        distances = np.linalg.norm(scores[pool] - scores[first], axis=1)
        second = pool.pop(int(np.argmin(distances)))
        pairs.append((first, second))
    return pairs


def normalize_clouds(x):
    """Apply the same per-cloud normalization used by PointNet inference."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (N_AXES, N_POINTS):
        raise ValueError(f'Expected clouds shaped (n, 3, {N_POINTS}), got {x.shape}')
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)
    return (x - mean) / (std + 1e-8)


def plsda_augment_xyz_fold(x, y, seed, children_per_pair=8, n_components=8):
    """Balance one training fold using PLS-DA reconstruction in XYZ space."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim != 3 or x.shape[1:] != (N_AXES, N_POINTS):
        raise ValueError(f'Expected x shaped (n, 3, {N_POINTS}), got {x.shape}')
    counts = {label: int((y == label).sum()) for label in (0, 1)}
    info = {
        'augmentation_protocol': POINTNET_PLSDA_PROTOCOL,
        'pls_fit_scope': 'training_fold_only',
        'pls_feature_space': 'per-cloud-normalized-xyz-flattened',
        'pls_feature_count': int(N_AXES * N_POINTS),
        'pls_components': 0,
        'children_per_pair': int(children_per_pair),
        'original_train_n': int(len(y)),
        'synthetic_train_n': 0,
        'fit_train_n': int(len(y)),
        'original_class_0': counts[0], 'original_class_1': counts[1],
        'fit_class_0': counts[0], 'fit_class_1': counts[1],
        'postprocess': 'per-cloud-zscore-after-pls-reconstruction',
    }
    if counts[0] == counts[1]:
        return x.astype(np.float32), y, info
    if children_per_pair < 1:
        raise ValueError('children_per_pair must be positive')
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError('PLS-DA augmentation requires both classes in each training fold')

    flat = x.reshape(len(x), -1).astype(np.float64)
    scaler = StandardScaler().fit(flat)
    flat_scaled = scaler.transform(flat)
    y_onehot = np.column_stack([(y == 0).astype(float), (y == 1).astype(float)])
    n_comp = min(int(n_components), flat.shape[1], len(y) - 1)
    if n_comp < 1:
        raise ValueError('Not enough fold training rows for PLS-DA')
    pls = PLSRegression(n_components=n_comp, scale=False, max_iter=1000)
    scores = pls.fit_transform(flat_scaled, y_onehot)[0]
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
            # A one-row class is still kept inside the training fold.  This
            # fallback never reads validation or test rows.
            pairs = [(int(class_indices[0]), int(class_indices[0]))]
        rows = []
        for child_idx in range(needed):
            first, second = pairs[(child_idx // children_per_pair) % len(pairs)]
            alpha = (0.1 + 0.8 * ((child_idx % children_per_pair) + 0.5) /
                     children_per_pair)
            score_new = ((1.0 - alpha) * scores[first] + alpha * scores[second])[None, :]
            reconstructed_scaled = pls.inverse_transform(score_new)
            reconstructed = scaler.inverse_transform(reconstructed_scaled)[0]
            cloud = reconstructed.reshape(N_AXES, N_POINTS)
            rows.append(normalize_clouds(cloud[None, ...])[0])
        generated_x.append(np.asarray(rows, dtype=np.float32))
        generated_y.append(np.full(needed, label, dtype=np.int64))

    if generated_x:
        x = np.concatenate([x] + generated_x, axis=0)
        y = np.concatenate([y] + generated_y, axis=0)
        order = rng.permutation(len(y))
        x, y = x[order], y[order]
    if not np.isfinite(x).all():
        raise ValueError('PLS-DA XYZ reconstruction produced nonfinite values')
    info['synthetic_train_n'] = int(len(y) - info['original_train_n'])
    info['fit_train_n'] = int(len(y))
    info['fit_class_0'] = int((y == 0).sum())
    info['fit_class_1'] = int((y == 1).sum())
    return x.astype(np.float32), y.astype(np.int64), info


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
    print(f'[{cohort}/{side}/{POINTNET_PLSDA_MODEL}] train={len(train_df)} '
          f'test={len(test_df)} folds={len(splits)}')
    for fold, (tr, va) in enumerate(splits):
        fit_x, fit_y, info = plsda_augment_xyz_fold(
            x[tr], y[tr], args.seed + fold, args.children_per_pair, args.n_components)
        model, val_prob, best_epoch = train_fold(
            fit_x, fit_y, x[va], y[va], device, args.seed + fold,
            args.epochs, args.patience)
        oof[va] = val_prob
        with torch.no_grad():
            test_logits = model(torch.tensor(test_x, dtype=torch.float32, device=device))
            test_prob = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            test_probs.append(test_prob)
        info.update({'fold': fold + 1, 'best_epoch': int(best_epoch),
                     'raw_train_n': int(len(tr)), 'val_n': int(len(va)),
                     'augmentation_applied_to_validation': False,
                     'augmentation_applied_to_test': False})
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
    (run_dir / 'metrics.json').write_text(
        json.dumps({'cv_oof': cv_metrics, 'test': test_metrics}, indent=2),
        encoding='utf-8')
    manifest = {
        'protocol': POINTNET_PLSDA_PROTOCOL,
        'source_protocol': RAW_COEF_PROTOCOL,
        'cohort': cohort, 'side': side, 'model': POINTNET_PLSDA_MODEL,
        'feature_kind': 'xyz', 'points': N_POINTS, 'point_order': 'x,y,z',
        'model_input_features': 'normalized XYZ point clouds (3 x 1002)',
        'pls_da_used_for_augmentation': True,
        'pls_da_fit_scope': 'training_fold_only',
        'pls_da_feature_space': 'per-cloud-normalized-xyz-flattened',
        'pls_da_components_requested': int(args.n_components),
        'pls_da_applied_to_model_input': False,
        'augmentation_protocol': POINTNET_PLSDA_PROTOCOL,
        'augmentation_applied_to_validation': False,
        'augmentation_applied_to_test': False,
        'train_rows': len(train_df), 'test_rows': len(test_df),
        'train_classes': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'test_classes': {str(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'train_feature_sha256': digest_frame(train_df),
        'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False, 'folds': fold_info, 'seed': args.seed,
        'children_per_pair': int(args.children_per_pair),
        'device': str(device), 'elapsed_seconds': round(time.time() - started, 2),
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'  CV balanced_accuracy={cv_metrics["balanced_accuracy"]:.4f} | '
          f'test accuracy={test_metrics["accuracy"]:.4f} | '
          f'test balanced_accuracy={test_metrics["balanced_accuracy"]:.4f} | '
          f'test F1={test_metrics["f1_macro"]:.4f}')
    return {
        'Cohort': cohort, 'Side': side, 'Model': POINTNET_PLSDA_MODEL,
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
    parser.add_argument('--side', choices=('left', 'right', 'all'), default='all')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--children_per_pair', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=8)
    parser.add_argument('--output_root', default=str(Path(__file__).resolve().parent / 'pointnet_plsda_runs'))
    args = parser.parse_args()
    cohorts = COHORTS if args.cohort == 'all' else (args.cohort,)
    sides = ('left', 'right') if args.side == 'all' else (args.side,)
    rows = []
    for cohort in cohorts:
        for side in sides:
            try:
                rows.append(run_one(cohort, side, args))
            except Exception as exc:
                print(f'[FAIL] {cohort}/{side}/{POINTNET_PLSDA_MODEL}: {exc}')
                rows.append({'Cohort': cohort, 'Side': side,
                             'Model': POINTNET_PLSDA_MODEL,
                             'Status': 'FAIL', 'Error': repr(exc)})
    summary = Path(args.output_root).resolve() / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f'Wrote summary: {summary}')
    if any(row.get('Status') == 'FAIL' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
