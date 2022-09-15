import time

from sklearn.base import BaseEstimator


class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=None):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """
        self.estimator = estimator
        self.features_score = {}

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X, y=None):
        predict = self.estimator.predict(X)
        return predict

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    @property
    def classes_(self):
        if hasattr(self.estimator, 'classes_'):
            return self.estimator.classes_
        raise ValueError('selected model does not provide classes_')