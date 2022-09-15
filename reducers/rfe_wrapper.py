from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from dimension_reduction import DimReduce


class Rfe(DimReduce):
    def __init__(self, n_features_to_select=None) -> None:
        super().__init__()
        self.n_features_to_select = n_features_to_select

    def run(self, x, y) -> tuple:
        return RFE(SVR(kernel="linear"), n_features_to_select=self.n_features_to_select, step=1).fit(x, y).ranking_

    def set_n_features(self, n):
        self.n_features_to_select = n



