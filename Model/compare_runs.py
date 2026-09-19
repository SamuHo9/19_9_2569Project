"""Compare only predictions from exactly the same identified test subjects/labels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from data_contract import PROTOCOL


def compare(files):
    reference = None
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {'subject_ids', 'protocol', 'y_test', 'y_pred', 'y_prob'}
            if not required.issubset(data.files) or str(data['protocol']) != PROTOCOL:
                raise ValueError(f'Missing subject identity or incompatible protocol: {path}')
            ids = data['subject_ids'].astype(str).reshape(-1)
            truth = data['y_test'].reshape(-1)
            pred = data['y_pred'].reshape(-1)
            prob = data['y_prob'].reshape(-1)
            if not len(ids) or len(set(ids)) != len(ids) or not (len(ids) == len(truth) == len(pred) == len(prob)):
                raise ValueError(f'Duplicate subjects or inconsistent lengths: {path}')
            if not np.isin(truth, [0,1]).all() or not np.isin(pred, [0,1]).all() or len(np.unique(truth)) != 2:
                raise ValueError(f'Invalid binary labels: {path}')
            if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
                raise ValueError(f'Invalid probability values: {path}')
            ordered = sorted(zip(ids.tolist(), truth.tolist()))
            if reference is not None and ordered != reference:
                raise ValueError('Cannot compare different test subjects or labels')
            reference = ordered
            rows.append({'File': str(path), 'Test_N': len(truth),
                         'Accuracy': accuracy_score(truth, pred),
                         'Sensitivity': recall_score(truth, pred, pos_label=1),
                         'Specificity': recall_score(truth, pred, pos_label=0),
                         'F1_Macro': f1_score(truth, pred, average='macro'),
                         'AUC': roc_auc_score(truth, prob)})
    if len(rows) < 2:
        raise ValueError('At least two prediction artifacts are required')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    frame = compare(args.predictions)
    with output.open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)
    print(frame.to_string(index=False))
