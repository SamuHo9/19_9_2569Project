"""Audit leakage-free PointNet PLS-DA runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_contract import load_raw_xyz_cohort
from leakage_free_pointnet_plsda_training import (
    POINTNET_PLSDA_MODEL,
    POINTNET_PLSDA_PROTOCOL,
)


EXPECTED = {
    (cohort, side, POINTNET_PLSDA_MODEL)
    for cohort in ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602')
    for side in ('left', 'right')
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    keys = {(row.Cohort, row.Side, row.Model) for row in summary.itertuples()}
    assert len(summary) == 6 and keys == EXPECTED
    assert 'Status' not in summary or not summary['Status'].eq('FAIL').any()
    for row in summary.itertuples():
        run = Path(row.RunDir)
        manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
        metric = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        pred = pd.read_csv(run / 'test_predictions.csv')
        assert manifest['protocol'] == POINTNET_PLSDA_PROTOCOL
        assert manifest['model'] == POINTNET_PLSDA_MODEL
        assert manifest['feature_kind'] == 'xyz' and manifest['points'] == 1002
        assert manifest['pls_da_used_for_augmentation'] is True
        assert manifest['pls_da_fit_scope'] == 'training_fold_only'
        assert manifest['pls_da_applied_to_model_input'] is False
        assert manifest['augmentation_applied_to_validation'] is False
        assert manifest['augmentation_applied_to_test'] is False
        assert manifest['children_per_pair'] == 8
        assert all(fold['fit_class_0'] == fold['fit_class_1']
                   for fold in manifest['folds'])
        assert all(fold['pls_fit_scope'] == 'training_fold_only'
                   for fold in manifest['folds'])
        assert (run / 'fold_models.pt').is_file()
        _, test = load_raw_xyz_cohort(row.Cohort, row.Side)
        assert set(pred.Subject) == set(test.Subject)
        assert not pred.Probability.isna().any()
        assert pred.Probability.between(0, 1).all()
        for section in ('cv_oof', 'test'):
            for key in ('accuracy', 'balanced_accuracy', 'sensitivity',
                        'specificity', 'f1_macro'):
                assert 0.0 <= float(metric[section][key]) <= 1.0
    result = {
        'summary': str(args.summary.resolve()),
        'runs': len(summary),
        'protocol': POINTNET_PLSDA_PROTOCOL,
        'test_balanced_accuracy_mean': float(summary.Test_BalancedAccuracy.mean()),
        'test_accuracy_mean': float(summary.Test_Accuracy.mean()),
    }
    out = args.summary.with_name(args.summary.stem + '_audit.json')
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Wrote audit: {out}')


if __name__ == '__main__':
    main()
