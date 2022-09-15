import os
from math import log
from typing import List

import joblib
import numpy as np
# Discrete estimators
from joblib import Parallel

from dimension_reduction import DimReduce


def entropy_d(sx: List[float], base=2):
    """
    Discrete entropy estimator from list of samples.
    """
    return -sum(map(elog, hist(sx))) / log(base)


def hist(sx: List[float]):
    """
    Normalized histogram from list of samples.
    """

    d_ = dict()
    for s in sx:
        d_[s] = d_.get(s, 0) + 1
    return map(lambda z: float(z) / len(sx), d_.values())


def su_calculation(f1: np.array, f2: np.array) -> float:
    """
    Compute the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    """
    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = entropy_d(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = entropy_d(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0 * t1 / (t2 + t3)

    return su


def information_gain(f1: np.array, f2: np.array):
    """
    Compute the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)
    """
    ig = entropy_d(f1) - conditional_entropy(f1, f2)
    return ig


def midd(x, y):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """

    return -entropy_d(list(zip(x, y))) + entropy_d(x) + entropy_d(y)


def conditional_entropy(f1: np.array, f2: np.array) -> float:
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
    """

    ce = entropy_d(f1) - midd(f1, f2)
    return ce


def elog(x: float):
    if x <= 0 or x >= 1:
        return 0
    else:
        return x * log(x)


from typing import List
from tqdm import tqdm

import numpy as np


def merit_calculation(X: np.array, y: np.array) -> float:
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    """

    n_samples, n_features = X.shape
    rff, rcf = 0, 0
    for i in range(n_features):
        f_i = X[:, i]
        rcf += su_calculation(f_i, y)
        for j in range(n_features):
            if j > i:
                f_j = X[:, j]
                rff += su_calculation(f_i, f_j)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def isUpping(A: List[float]):
    """
    This function check if the serie is increasing.
    """
    return all(A[i] <= A[i + 1] for i in range(len(A) - 1))


def cfs(x, y, min_features: int = 5, parallel: bool = False) -> np.array:
    """
    This function uses a correlation based greedy to evaluate the worth of features.
    The algorithm works as following:
    - at each iteration we will add the best feature to the candidate set regarding the heuristic function defined in
    Chapter 4 Correlation-based Feature Selection of given refenrence.
    - we stop of the algorithm is based on convergence
    - one can specify the minimum number of features
    Mark A. Hall "Correlation-based Feature Selection for Machine Learning" 1999.
    """

    # X, y = X_.to_numpy(), y_.to_numpy().squeeze()
    print('parallel', parallel)
    n_samples, n_features = x.shape
    # F store the index of features
    features = []
    # M stores the merit values
    merits = []
    availables_features = [k for k in range(n_features)]
    # progress bar
    pbar = tqdm(total=min_features, unit="features")
    while availables_features:
        if parallel:
            p = Parallel(n_jobs=-1)
            merit_candidates = p(
                joblib.delayed(merit_calculation)(
                    x[:, features + [next_]], y)
                for next_ in availables_features)
        # if parallel:
        #     pool = Pool()
        #     merit_candidates = [
        #         pool.apply(merit_calculation, args=(x[:, features + [next_]], y))
        #         for next_ in availables_features
        #     ]

        else:
            merit_candidates = [
                merit_calculation(x[:, features + [next_]], y)
                for next_ in availables_features
            ]
        next_merit = max(merit_candidates)
        next_feature = availables_features[merit_candidates.index(next_merit)]

        features.append(next_feature)
        merits.append(next_merit)

        availables_features.remove(next_feature)

        pbar.update(1)
        pbar.set_description("Added {}".format(next_feature))
        # converge criterion with greedy
        if len(features) >= min_features and not (isUpping(merits[min_features - 1:])):
            best = merits.index(max(merits[min_features:])) + 1
            features = features[:best]
            break

    pbar.close()

    return features, merits


class CFS(DimReduce):
    def __init__(self, n_selected_features=100) -> None:
        super().__init__()
        self.n_selected_features = n_selected_features

    def run(self, x, y):
        index, scores = cfs(x, y, x.shape[1], int(os.environ['N_JOBS']) == -1)
        return np.array(scores)

    def set_n_features(self, n):
        self.n_selected_features = n
