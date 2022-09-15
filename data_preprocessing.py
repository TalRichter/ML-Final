import os

import joblib
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer


# missing values in the explanatory variables SimpleImputer

# Coding all categorical variables into numbers (if any),

# removing variables with zero variance (VarianceThreshold)

# normalization (PowerTransformer)

class FClassifWrapper(TransformerMixin, BaseEstimator):

    def __init__(self, k=1000) -> None:
        self.select_k_best = SelectKBest(f_classif, k=k)
        self.k = k
        self.dask = k

    def fit_transform(self, X, y=None, **fit_params):
        if X.shape[1] < self.k:
            self.select_k_best.k = X.shape[1]
        return self.select_k_best.fit_transform(X, y, **fit_params)

    def fit(self, X, y=None):
        return self.select_k_best.fit(X, y)

    def transform(self, X):
        return self.select_k_best.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.select_k_best.get_feature_names_out(input_features)


def get_pipeline() -> Pipeline:
    return make_pipeline(

        make_column_transformer((SimpleImputer(), make_column_selector(dtype_include=np.number)),
                                n_jobs=int(os.environ['N_JOBS'])),
        # OneHotEncoder(),  # Encode categorical features - no need as dna microarray has no categorical features
        FClassifWrapper(),
        VarianceThreshold(),
        PowerTransformer(),
        memory=joblib.Memory(cachedir=r'./cache/preprocess/', verbose=0),
        verbose=True)
