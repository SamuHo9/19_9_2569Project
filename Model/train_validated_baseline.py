"""Nested subject CV + untouched test, using fold-local augmentation.

This is a reproducible research baseline on historical features, not a clinically
validated model. It never replaces the Desktop ResNet weights.
"""
import argparse
import json
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from data_contract import load_cohort, META, MODEL_ROOT, PROTOCOL, cohort_identity
from evaluation import compute_bootstrap_stats
from validated_model import FoldAugmentedSVC


def run(cohort, side, outer_folds=5, children=0, components=(2, 5, 10)):
    train, test = load_cohort(cohort, side)
    columns = [c for c in train if c not in META]
    X, y = train[columns].to_numpy(), train.BinaryClass.to_numpy()
    Xt, yt = test[columns].to_numpy(), test.BinaryClass.to_numpy()
    if outer_folds < 2 or outer_folds > min(np.bincount(y)):
        raise ValueError('Invalid outer fold count')
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=42)
    oof_pred, oof_prob = np.zeros(len(y), dtype=int), np.zeros(len(y))
    fold_info = []

    def select(fit_X, fit_y):
        inner = StratifiedKFold(3, shuffle=True, random_state=42)
        if min(np.bincount(fit_y)) < 3:
            raise ValueError('At least three subjects per class required for inner CV')
        max_comp = min(fit_X.shape[1], min(len(a) - 1 for a, _ in inner.split(fit_X, fit_y)))
        candidates = [int(c) for c in components if 1 <= c <= max_comp]
        if not candidates:
            raise ValueError('No valid component candidate fits the inner folds')
        search = GridSearchCV(FoldAugmentedSVC(children_per_subject=children),
                              {'n_components': candidates, 'C': [0.1, 1.0]},
                              cv=inner, scoring='balanced_accuracy', n_jobs=1, error_score='raise')
        return search.fit(fit_X, fit_y)

    for fold, (fit_idx, val_idx) in enumerate(outer.split(X, y), 1):
        search = select(X[fit_idx], y[fit_idx])
        oof_pred[val_idx] = search.predict(X[val_idx])
        oof_prob[val_idx] = search.predict_proba(X[val_idx])[:, 1]
        fold_info.append({'fold': fold, 'train_n': len(fit_idx), 'validation_n': len(val_idx),
                          'train_subjects': train.Subject.iloc[fit_idx].tolist(),
                          'validation_subjects': train.Subject.iloc[val_idx].tolist(),
                          'best_params': search.best_params_,
                          'synthetic_train_n': search.best_estimator_.n_synthetic_fit_})
        print(f'Outer fold {fold}/{outer_folds} complete', flush=True)
    final = select(X, y)
    pred, prob = final.predict(Xt), final.predict_proba(Xt)[:, 1]
    def metrics(truth, decision, score):
        return {'accuracy': float(accuracy_score(truth, decision)),
                'sensitivity': float(recall_score(truth, decision, pos_label=1)),
                'specificity': float(recall_score(truth, decision, pos_label=0)),
                'f1_macro': float(f1_score(truth, decision, average='macro')),
                'auc': float(roc_auc_score(truth, score)),
                'confusion_matrix': confusion_matrix(truth, decision, labels=[0, 1]).tolist()}
    parent = MODEL_ROOT / 'validated_runs' / cohort / side / 'nested_svm'
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
    report = {'protocol': PROTOCOL, 'model': 'FoldAugmentedSVC', 'cohort': cohort, 'side': side,
              'train_n': len(train), 'test_n': len(test), 'feature_columns': columns,
              'train_identity': cohort_identity(train), 'test_identity': cohort_identity(test),
              'train_sha256': hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
              'test_sha256': hashlib.sha256(test.to_csv(index=False).encode()).hexdigest(),
              'image_preprocessing_provenance': 'historical features; not certified independent',
              'clinical_validation': False, 'children_per_subject': children,
              'folds': fold_info, 'best_params': final.best_params_,
              'nested_cv': metrics(y, oof_pred, oof_prob), 'test': metrics(yt, pred, prob),
              'bootstrap': compute_bootstrap_stats(yt, prob, y_pred=pred)}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez(out / 'test_predictions.npz', y_test=yt, y_pred=pred, y_prob=prob,
             subject_ids=test.Subject.to_numpy(dtype=str), protocol=np.asarray(PROTOCOL))
    np.savez(out / 'out_of_fold_predictions.npz', y_test=y, y_pred=oof_pred, y_prob=oof_prob,
             subject_ids=train.Subject.to_numpy(dtype=str), protocol=np.asarray(PROTOCOL))
    joblib.dump(final.best_estimator_, out / 'model.joblib')
    print(json.dumps({'output': str(out), 'nested_cv': report['nested_cv'], 'test': report['test']}, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', default='Ds004469')
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--children', type=int, default=0)
    parser.add_argument('--components', type=int, nargs='+', default=[2, 5, 10])
    args = parser.parse_args()
    run(args.cohort, args.side, args.outer_folds, args.children, args.components)
