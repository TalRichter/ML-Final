from sklearn.feature_selection import f_classif, SelectFdr

from dimension_reduction import DimReduce


class FClassif(DimReduce):
    def __init__(self, alpha=0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def run(self, x, y):
        return SelectFdr(f_classif, alpha=self.alpha).fit(x, y).scores_

