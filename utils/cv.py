import time

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeavePOut, StratifiedKFold, KFold


# python 3.10 but matlab engine not yet supported
# def get_cv(X):
#     match len(X):
#         case num if num < 50:
#             return LeavePOut(2)
#         case num if 50 <= num < 100:
#             return KFold(num)
#         case num if 100 <= num < 1000:
#             return StratifiedKFold(10)
#         case num if num > 1000:
#             return StratifiedKFold(5)
#         case _:
#             raise ValueError("unhandled case")


def get_cv(X):
    size = len(X)
    if size < 50:
        return LeavePOut(2)
    if 50 <= size < 100:
        return KFold(size)
    if 100 <= size < 1000:
        return StratifiedKFold(10)
    if size >= 1000:
        return StratifiedKFold(5)
    else:
        raise ValueError("unhandled case")


class SelectKBestWrapper(SelectKBest):

    def __init__(self, score_func=f_classif, *, k=10):
        super().__init__(score_func, k=k)
        if hasattr(self.score_func, 'set_n_features'):
            self.score_func.set_n_features(self.k)

    def fit(self, X, y):
        start_time = time.time()
        fit = super().fit(X, y)
        self.total_runtime = time.time() - start_time
        return fit
