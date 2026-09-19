"""Validated, reproducible bootstrap metrics using the classifier's own decisions."""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_bootstrap_stats(y_true, y_prob, n_boot=1000, y_pred=None, random_state=42):
    truth = np.asarray(y_true).reshape(-1)
    prob = np.asarray(y_prob, dtype=float)
    if prob.ndim == 2 and prob.shape[1] == 2:
        prob = prob[:, 1]
    prob = prob.reshape(-1)
    pred = (prob > 0.5).astype(int) if y_pred is None else np.asarray(y_pred).reshape(-1)
    if not len(truth) or not (len(truth) == len(prob) == len(pred)):
        raise ValueError('Prediction arrays must be nonempty and have equal lengths')
    if not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Binary labels must be 0 or 1; unknown labels are not allowed')
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if len(np.unique(truth)) != 2:
        raise ValueError('Both classes are required for sensitivity/specificity/AUC')
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError('n_boot must be a positive integer')
    rng = np.random.RandomState(random_state)
    values = {key: [] for key in ('Acc', 'Sens', 'Spec', 'F1', 'AUC')}
    skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, len(truth), size=len(truth))
        yt, yp, score = truth[idx], pred[idx], prob[idx]
        if len(np.unique(yt)) != 2:
            skipped += 1
            continue
        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        values['Acc'].append(np.mean(yt == yp))
        values['Sens'].append(tp / (tp + fn))
        values['Spec'].append(tn / (tn + fp))
        values['F1'].append(0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)))
        values['AUC'].append(roc_auc_score(yt, score))
    if not values['AUC']:
        raise ValueError('No bootstrap resample contained both classes')
    result = {'Bootstrap_Valid': len(values['AUC']), 'Bootstrap_Skipped': skipped,
              'Decision_Source': 'saved_y_pred' if y_pred is not None else 'threshold_0.5'}
    for key, arr in values.items():
        factor = 1 if key == 'AUC' else 100
        arr = np.asarray(arr) * factor
        lo, hi = np.percentile(arr, [2.5, 97.5])
        result['Mean_' + key] = f'{arr.mean():.4f}' if key == 'AUC' else f'{arr.mean():.2f}%'
        result[key + '_95CI'] = f'[{lo:.4f}, {hi:.4f}]' if key == 'AUC' else f'[{lo:.2f}%, {hi:.2f}%]'
        if key == 'Acc':
            result['Acc_SD'] = f'+/-{arr.std():.2f}%'
    return result
