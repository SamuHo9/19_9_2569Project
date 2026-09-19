"""Preflight checks for the no-PLS coefficient training protocol.

This is intentionally executable before a long training batch.  It checks the
dataset manifests, row/feature integrity, patient-group split isolation, the
shared model interfaces, and (when supplied) one completed run's output files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract import (RAW_COEF_PROTOCOL, load_raw_cohort,
                           patient_group_id)
from leakage_free_coef_training import (MODELS, COHORTS, SIDES,
                                        grouped_splits, make_torch_model)


def check_dataset(cohort, side):
    train, test = load_raw_cohort(cohort, side, kind='coef')
    feature_cols = [c for c in train if c.startswith('Coef_')]
    assert feature_cols == [f'Coef_{i}' for i in range(1, 508)]
    assert train.columns.tolist() == test.columns.tolist()
    assert train['DataType'].str.lower().isin(['original', 'real']).all()
    assert test['DataType'].str.lower().isin(['original', 'real']).all()
    assert not train['Subject'].duplicated().any()
    assert not test['Subject'].duplicated().any()
    train_groups = {patient_group_id(v) for v in train['Subject']}
    test_groups = {patient_group_id(v) for v in test['Subject']}
    assert not train_groups & test_groups
    train_hash = set(pd.util.hash_pandas_object(train[feature_cols], index=False).tolist())
    test_hash = set(pd.util.hash_pandas_object(test[feature_cols], index=False).tolist())
    assert not train_hash & test_hash
    splits = grouped_splits(train['BinaryClass'].to_numpy(),
                            np.asarray([patient_group_id(v) for v in train['Subject']]),
                            requested=5, seed=42)
    seen = []
    for tr, va in splits:
        tr_groups = set(train_groups_for_indices(train, tr))
        va_groups = set(train_groups_for_indices(train, va))
        assert not tr_groups & va_groups
        seen.extend(va_groups)
    assert len(seen) == len(set(seen)) == len(train_groups)
    return {
        'cohort': cohort, 'side': side, 'train_rows': len(train),
        'test_rows': len(test), 'train_class_0': int((train.BinaryClass == 0).sum()),
        'train_class_1': int((train.BinaryClass == 1).sum()),
        'test_class_0': int((test.BinaryClass == 0).sum()),
        'test_class_1': int((test.BinaryClass == 1).sum()),
        'groups_train': len(train_groups), 'groups_test': len(test_groups),
        'folds': len(splits),
    }


def train_groups_for_indices(frame, indices):
    return [patient_group_id(frame.iloc[i]['Subject']) for i in indices]


def check_model_interfaces():
    import torch

    for name in MODELS:
        if name == 'SVM':
            continue
        model = make_torch_model(name).cpu().eval()
        x = torch.zeros((2, 507), dtype=torch.float32) if name == 'MLP' else torch.zeros((2, 1, 507), dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
        if name == 'ResNetAE':
            logits, reconstruction = output
            assert tuple(logits.shape) == (2, 1)
            assert tuple(reconstruction.shape) == (2, 507)
        else:
            assert tuple(output.shape) == (2, 1)


def check_runner_source():
    source = (Path(__file__).with_name('leakage_free_coef_training.py')).read_text(encoding='utf-8')
    for forbidden in ('PLSRegression', 'augment_plsda_balanced', 'load_training_data'):
        assert forbidden not in source, f'legacy/leaky dependency found: {forbidden}'


def check_run(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / 'metrics.json'
    pred_path = run_dir / 'test_predictions.csv'
    manifest_path = run_dir / 'run_manifest.json'
    for path in (metrics_path, pred_path, manifest_path):
        assert path.is_file(), f'missing run output: {path}'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['protocol'] == RAW_COEF_PROTOCOL
    assert manifest['pls_da_applied_to_model_input'] is False
    assert manifest['patient_overlap'] is False
    assert manifest.get('augmentation') == 'balanced_jitter'
    assert manifest.get('augmentation_protocol') == 'within-class-mixup-jitter-v1'
    assert manifest.get('augmentation_children_per_pair') == 8
    assert manifest.get('augmentation_applied_to_validation') is False
    assert manifest.get('augmentation_applied_to_test') is False
    assert all(fold['fit_class_0'] == fold['fit_class_1'] for fold in manifest['folds'])
    assert all(fold['synthetic_train_n'] >= 0 for fold in manifest['folds'])
    assert manifest['test_rows'] == len(pd.read_csv(pred_path))
    for section in ('cv_oof', 'test'):
        for key in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_macro'):
            assert 0.0 <= float(metrics[section][key]) <= 1.0
    pred = pd.read_csv(pred_path)
    assert pred['Probability'].between(0, 1).all()
    assert pred['Prediction'].isin([0, 1]).all()
    return {'run_dir': str(run_dir), 'protocol': manifest['protocol'],
            'test_rows': len(pred), 'test_balanced_accuracy': metrics['test']['balanced_accuracy']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=Path)
    args = parser.parse_args()
    rows = [check_dataset(cohort, side) for cohort in COHORTS for side in SIDES]
    check_model_interfaces()
    check_runner_source()
    if args.run_dir:
        run = check_run(args.run_dir)
    else:
        run = None
    print(json.dumps({'datasets': rows, 'model_interfaces': 'ok',
                      'runner_source': 'ok', 'run': run}, indent=2))


if __name__ == '__main__':
    main()
