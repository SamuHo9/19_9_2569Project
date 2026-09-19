"""Audit a completed fold-local PLS-DA batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_contract import load_raw_cohort
from leakage_free_plsda_training import MODELS, PLSDA_PROTOCOL


EXPECTED = {
    (cohort, side, model)
    for cohort in ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602')
    for side in ('left', 'right')
    for model in MODELS
}


def check_run(row):
    run_dir = Path(row.RunDir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    pred = pd.read_csv(run_dir / 'test_predictions.csv')
    assert manifest['protocol'] == PLSDA_PROTOCOL
    assert manifest['pls_da_fit_scope'] == 'training_fold_only'
    assert manifest['pls_da_used_for_augmentation'] is True
    assert manifest['pls_da_applied_to_model_input'] is False
    assert manifest['augmentation_applied_to_validation'] is False
    assert manifest['augmentation_applied_to_test'] is False
    assert manifest['children_per_pair'] == 8
    assert all(fold['pls_fit_scope'] == 'training_fold_only' for fold in manifest['folds'])
    assert all(fold['pls_components'] >= 1 for fold in manifest['folds'])
    assert all(fold['fit_class_0'] == fold['fit_class_1'] for fold in manifest['folds'])
    assert manifest['test_rows'] == len(pred)
    assert pred['Probability'].between(0, 1).all()
    assert pred['Prediction'].isin([0, 1]).all()
    train, test = load_raw_cohort(row.Cohort, row.Side, kind='coef')
    assert set(pred.Subject) == set(test.Subject)
    assert not set(train.Subject) & set(pred.Subject)
    for section in ('cv_oof', 'test'):
        for key in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_macro'):
            assert 0.0 <= float(metrics[section][key]) <= 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    keys = {(row.Cohort, row.Side, row.Model) for row in summary.itertuples()}
    assert len(summary) == 36
    assert keys == EXPECTED
    assert 'Status' not in summary or not summary['Status'].eq('FAIL').any()
    for row in summary.itertuples():
        check_run(row)
    result = {
        'summary': str(args.summary.resolve()), 'runs': len(summary),
        'protocol': PLSDA_PROTOCOL,
        'test_balanced_accuracy_mean': float(summary.Test_BalancedAccuracy.mean()),
        'test_balanced_accuracy_by_model': summary.groupby('Model').Test_BalancedAccuracy.mean().round(6).to_dict(),
        'test_accuracy_by_model': summary.groupby('Model').Test_Accuracy.mean().round(6).to_dict(),
    }
    output = args.summary.with_name(args.summary.stem + '_audit.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Wrote audit: {output}')


if __name__ == '__main__':
    main()
