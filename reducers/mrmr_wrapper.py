# https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/MRMR.py
import os

import numpy as np
import pandas as pd
from mrmr import mrmr_classif

from dimension_reduction import DimReduce


class Mrmr(DimReduce):
    def __init__(self, n_selected_features=100) -> None:
        super().__init__()
        self.n_selected_features = n_selected_features

    def run(self,x,y):
        features, scores, _ = mrmr_classif(pd.DataFrame(x),
                                           pd.Series(np.squeeze(y)),
                                           K=self.n_selected_features,
                                           return_scores=True,
                                           n_jobs=int(os.environ['N_JOBS']))
        return scores

    def set_n_features(self, n):
        self.n_selected_features = n

