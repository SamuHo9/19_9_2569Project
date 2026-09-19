"""PLS and optional same-class augmentation are fitted inside each CV training fold."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


class FoldAugmentedSVC(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components=5, C=1.0, children_per_subject=0, random_state=42):
        self.n_components = n_components
        self.C = C
        self.children_per_subject = children_per_subject
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError('Invalid feature matrix')
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Both binary classes are required')
        if self.children_per_subject < 0 or int(self.children_per_subject) != self.children_per_subject:
            raise ValueError('children_per_subject must be a nonnegative integer')
        if not 1 <= self.n_components <= min(X.shape[1], len(X) - 1):
            raise ValueError('PLS components exceed the training fold dimensions')
        self.n_features_in_ = X.shape[1]
        self.n_original_fit_ = len(X)
        self.scaler_ = StandardScaler().fit(X)
        self.pls_ = PLSRegression(n_components=self.n_components).fit(self.scaler_.transform(X), y)
        latent = self.pls_.transform(self.scaler_.transform(X))
        extra_X, extra_y = [], []
        rng = np.random.RandomState(self.random_state)
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            if len(indices) < 2:
                continue
            for i in indices:
                candidates = indices[indices != i]
                neighbor = candidates[np.argmin(np.linalg.norm(latent[candidates] - latent[i], axis=1))]
                for _ in range(self.children_per_subject):
                    alpha = rng.uniform(0.2, 0.8)
                    extra_X.append((1 - alpha) * latent[i] + alpha * latent[neighbor])
                    extra_y.append(label)
        self.n_synthetic_fit_ = len(extra_y)
        fit_X = np.vstack([latent, extra_X]) if extra_y else latent
        fit_y = np.concatenate([y, extra_y]) if extra_y else y
        self.classifier_ = SVC(C=self.C, class_weight='balanced', probability=True, random_state=self.random_state)
        self.classifier_.fit(fit_X, fit_y)
        self.classes_ = self.classifier_.classes_
        return self

    def _transform(self, X):
        check_is_fitted(self, 'classifier_')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('Feature schema/values do not match training')
        return self.pls_.transform(self.scaler_.transform(X))

    def predict(self, X):
        return self.classifier_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.classifier_.predict_proba(self._transform(X))
