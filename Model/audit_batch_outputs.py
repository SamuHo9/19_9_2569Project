"""Audit one completed leakage-free batch summary and every referenced run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_contract import RAW_COEF_PROTOCOL
from verify_leakage_free_pipeline import check_run


EXPECTED = {
    (cohort, side, model)
    for cohort in ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602')
    for side in ('left', 'right')
    for model in ('SVM', 'MLP', 'ResNet', 'ResNetAE', 'MobileNet', 'SqueezeNet')
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    keys = {(row.Cohort, row.Side, row.Model) for row in summary.itertuples()}
    assert len(summary) == len(EXPECTED) == 36, f'expected 36 rows, got {len(summary)}'
    assert keys == EXPECTED, f'summary key mismatch: missing={EXPECTED - keys}, extra={keys - EXPECTED}'
    assert 'Status' not in summary or not summary['Status'].eq('FAIL').any()
    audits = []
    for row in summary.itertuples():
        audit = check_run(Path(row.RunDir))
        manifest = json.loads((Path(row.RunDir) / 'run_manifest.json').read_text(encoding='utf-8'))
        assert manifest['protocol'] == RAW_COEF_PROTOCOL
        assert manifest['cohort'] == row.Cohort and manifest['side'] == row.Side
        assert manifest['model'] == row.Model
        assert manifest['augmentation_children_per_pair'] == 8
        audits.append(audit)
    output = {
        'summary': str(args.summary.resolve()),
        'runs': len(audits),
        'protocol': RAW_COEF_PROTOCOL,
        'augmentation': 'within-class-mixup-jitter-v1',
        'test_balanced_accuracy_mean': float(summary.Test_BalancedAccuracy.mean()),
        'test_balanced_accuracy_by_model': summary.groupby('Model').Test_BalancedAccuracy.mean().round(6).to_dict(),
        'test_accuracy_by_model': summary.groupby('Model').Test_Accuracy.mean().round(6).to_dict(),
    }
    out_path = args.summary.with_name(args.summary.stem + '_audit.json')
    out_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))
    print(f'Wrote audit: {out_path}')


if __name__ == '__main__':
    main()
