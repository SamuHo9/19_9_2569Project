"""Leakage-free direct PLS-DA classifier baseline.

This runner is deliberately separate from ``leakage_free_plsda_training.py``.
There, PLS-DA is used only to generate fold-local synthetic training rows for
other classifiers.  Here PLS-DA itself is the classifier: it is fit to a
two-column one-hot target and the class with the larger predicted score is the
diagnosis.

The PLS-DA model and its scaler are fit separately in every grouped CV fold
for out-of-fold validation.  After CV, one final model is fit once on all
original training subjects and evaluated once on the untouched test set.  The
test result therefore does not average fold models and does not use an
ensemble.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_contract import RAW_COEF_PROTOCOL, load_raw_cohort, patient_group_id


COHORTS = ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602')
SIDES = ('left', 'right')
PLSDA_CLASSIFIER_PROTOCOL = 'fold-local-plsda-classifier-v1'
META = {'Subject', 'Group', 'Class', 'BinaryClass', 'DataType'}


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def feature_columns(frame):
    cols = [column for column in frame.columns if column.startswith('Coef_')]
    cols.sort(key=lambda value: int(value.split('_')[1]))
    expected = [f'Coef_{index}' for index in range(1, 508)]
    if cols != expected:
        raise ValueError(f'Expected Coef_1..Coef_507, got {len(cols)} coefficient columns')
    return cols


def matrix(frame):
    cols = feature_columns(frame)
    x = frame[cols].to_numpy(dtype=np.float64)
    y = frame['BinaryClass'].to_numpy(dtype=np.int64)
    groups = np.asarray([patient_group_id(value) for value in frame['Subject']], dtype=str)
    return x, y, groups, cols


def grouped_splits(y, groups, requested=10, seed=42):
    """Stratified folds at patient level, never splitting a patient."""
    unique, first = np.unique(groups, return_index=True)
    group_y = y[first]
    if any((y[groups == group] != label).any()
           for group, label in zip(unique, group_y)):
        raise ValueError('A patient group contains conflicting binary labels')
    class_group_counts = np.bincount(group_y, minlength=2)
    n_splits = min(int(requested), int(class_group_counts.min()), len(unique))
    if n_splits < 2:
        raise ValueError('At least two groups per class are required for CV')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for group_train, group_val in splitter.split(unique, group_y):
        train_groups = set(unique[group_train])
        val_groups = set(unique[group_val])
        train_idx = np.asarray(
            [index for index, group in enumerate(groups) if group in train_groups],
            dtype=int)
        val_idx = np.asarray(
            [index for index, group in enumerate(groups) if group in val_groups],
            dtype=int)
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise AssertionError('Grouped split overlap')
        splits.append((train_idx, val_idx))
    return splits


def metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= threshold).astype(int)
    result = {
        'accuracy': float(accuracy_score(y_true, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, prediction)),
        'sensitivity': float(recall_score(y_true, prediction, pos_label=1,
                                          zero_division=0)),
        'specificity': float(recall_score(y_true, prediction, pos_label=0,
                                          zero_division=0)),
        'f1_macro': float(f1_score(y_true, prediction, average='macro',
                                   zero_division=0)),
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


def fit_plsda(x, y, requested_components):
    """Fit a scaler plus one-hot PLS-DA and return its effective dimension."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    target = np.column_stack([(y == 0).astype(float), (y == 1).astype(float)])
    effective = min(int(requested_components), x.shape[1], len(y) - 1)
    if effective < 1:
        raise ValueError('PLS-DA needs at least two training rows')
    model = PLSRegression(n_components=effective, scale=False, max_iter=1000)
    model.fit(x_scaled, target)
    return {'scaler': scaler, 'model': model,
            'requested_components': int(requested_components),
            'effective_components': int(effective)}


def predict_probability(bundle, x):
    """Convert the two one-hot PLS predictions to class-1 probability.

    PLS-DA scores are not calibrated probabilities.  A two-class softmax is
    used only to obtain a stable [0, 1] score; its 0.5 decision is exactly the
    larger-score (argmax) PLS-DA decision.
    """
    scores = np.asarray(bundle['model'].predict(
        bundle['scaler'].transform(np.asarray(x, dtype=np.float64))), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(f'Expected two PLS-DA class scores, got {scores.shape}')
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probabilities = exp_scores[:, 1] / exp_scores.sum(axis=1)
    return np.clip(probabilities, 0.0, 1.0)


def run_one(cohort, side, args):
    train_df, test_df = load_raw_cohort(cohort, side, kind='coef')
    x, y, groups, columns = matrix(train_df)
    test_x, test_y, test_groups, _ = matrix(test_df)
    if set(groups) & set(test_groups):
        raise ValueError(f'Patient group overlap: {cohort}/{side}')
    splits = grouped_splits(y, groups, args.folds, args.seed)

    output_root = Path(args.output_root).resolve()
    run_dir = output_root / cohort / side / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f'[{cohort}/{side}/PLSDA_classifier] train={len(train_df)} '
          f'test={len(test_df)} folds={len(splits)}')

    # Out-of-fold score: every fold learns PLS-DA from its training subjects.
    oof = np.zeros(len(y), dtype=float)
    fold_info = []
    for fold, (tr, va) in enumerate(splits, start=1):
        bundle = fit_plsda(x[tr], y[tr], args.n_components)
        oof[va] = predict_probability(bundle, x[va])
        fold_info.append({
            'fold': fold,
            'raw_train_n': int(len(tr)),
            'val_n': int(len(va)),
            'train_groups': int(len(set(groups[tr]))),
            'val_groups': int(len(set(groups[va]))),
            'pls_components_requested': int(args.n_components),
            'pls_components_effective': bundle['effective_components'],
            'pls_fit_scope': 'training_fold_only',
        })

    # Single final PLS-DA model, fitted once on all original train subjects.
    final_bundle = fit_plsda(x, y, args.n_components)
    test_probability = predict_probability(final_bundle, test_x)
    cv_metrics = metrics(y, oof, args.threshold)
    test_metrics = metrics(test_y, test_probability, args.threshold)

    with (run_dir / 'model.pkl').open('wb') as stream:
        pickle.dump(final_bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame({
        'Subject': test_df['Subject'].astype(str),
        'PatientID': test_groups,
        'BinaryClass': test_y,
        'Probability': test_probability,
        'Prediction': (test_probability >= args.threshold).astype(int),
    }).to_csv(run_dir / 'test_predictions.csv', index=False)
    pd.DataFrame({
        'Subject': train_df['Subject'].astype(str),
        'PatientID': groups,
        'BinaryClass': y,
        'Probability': oof,
        'Prediction': (oof >= args.threshold).astype(int),
    }).to_csv(run_dir / 'oof_predictions.csv', index=False)
    (run_dir / 'metrics.json').write_text(json.dumps({
        'cv_oof': cv_metrics, 'test': test_metrics,
    }, indent=2), encoding='utf-8')
    manifest = {
        'protocol': PLSDA_CLASSIFIER_PROTOCOL,
        'source_protocol': RAW_COEF_PROTOCOL,
        'cohort': cohort, 'side': side, 'model': 'PLSDA_classifier',
        'feature_kind': 'coef',
        'model_input_features': '507 original coefficient features',
        'classifier': 'PLSRegression(one-hot target)+two-class-softmax',
        'pls_da_used_for_augmentation': False,
        'pls_da_applied_to_model_input': True,
        'pls_da_fit_scope': 'training_fold_only_for_cv_and_all_train_only_for_final',
        'pls_components_requested': int(args.n_components),
        'final_pls_components_effective': final_bundle['effective_components'],
        'test_evaluation': 'single_final_model_no_fold_ensemble',
        'train_rows': len(train_df), 'test_rows': len(test_df),
        'features': columns,
        'train_classes': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        'test_classes': {str(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        'train_feature_sha256': digest_frame(train_df),
        'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False,
        'folds': fold_info,
        'fold_count': len(splits),
        'seed': int(args.seed),
        'threshold': float(args.threshold),
        'augmentation_applied_to_validation': False,
        'augmentation_applied_to_test': False,
        'elapsed_seconds': round(time.time() - started, 2),
    }
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2),
                                                encoding='utf-8')
    print(f'  CV balanced_accuracy={cv_metrics["balanced_accuracy"]:.4f} | '
          f'test accuracy={test_metrics["accuracy"]:.4f} | '
          f'test balanced_accuracy={test_metrics["balanced_accuracy"]:.4f} | '
          f'test F1={test_metrics["f1_macro"]:.4f}')
    return {
        'Protocol': 'Fold-local PLS-DA classifier',
        'Cohort': cohort, 'Side': side, 'Model': 'PLSDA_classifier',
        'CV_BalancedAccuracy': cv_metrics['balanced_accuracy'],
        'CV_F1_Macro': cv_metrics['f1_macro'],
        'Test_Accuracy': test_metrics['accuracy'],
        'Test_BalancedAccuracy': test_metrics['balanced_accuracy'],
        'Test_Sensitivity': test_metrics['sensitivity'],
        'Test_Specificity': test_metrics['specificity'],
        'Test_F1_Macro': test_metrics['f1_macro'],
        'Test_ROC_AUC': test_metrics['roc_auc'],
        'RunDir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description='Leakage-free direct PLS-DA classifier')
    parser.add_argument('--cohort', choices=COHORTS + ('all',), default='all')
    parser.add_argument('--side', choices=SIDES + ('all',), default='all')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_components', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_root', default=str(
        Path(__file__).resolve().parent / 'plsda_classifier_runs'))
    args = parser.parse_args()
    cohorts = COHORTS if args.cohort == 'all' else (args.cohort,)
    sides = SIDES if args.side == 'all' else (args.side,)
    rows = []
    for cohort in cohorts:
        for side in sides:
            try:
                rows.append(run_one(cohort, side, args))
            except Exception as exc:
                print(f'[FAIL] {cohort}/{side}/PLSDA_classifier: {exc}')
                rows.append({'Protocol': 'Fold-local PLS-DA classifier',
                             'Cohort': cohort, 'Side': side,
                             'Model': 'PLSDA_classifier', 'Status': 'FAIL',
                             'Error': repr(exc)})
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    pd.DataFrame(rows).to_csv(summary, index=False)
    failed = [row for row in rows if row.get('Status') == 'FAIL']
    print(f'Wrote summary: {summary}')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
