import multiprocessing
import os
import sys
import time
import traceback

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.metrics._scorer import _BaseScorer, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearnex import patch_sklearn

import data_loader
from reducers.cfs import CFS
from reducers.dufs import DUFS
from reducers.f_classif_wrapper import FClassif
from reducers.mrmr_wrapper import Mrmr
from reducers.rfe_wrapper import Rfe
from utils.clf_switch import ClfSwitcher
from utils.cv import get_cv, SelectKBestWrapper


patch_sklearn()


def is_debug():
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True


if is_debug():
    os.environ['N_JOBS'] = '1'
else:
    os.environ['N_JOBS'] = '-1'

### Load Data
import numpy as np
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def loadData(dir: str, file: str = None) -> pandas.DataFrame:
    import data_loader
    load_function = getattr(data_loader, 'load_' + dir)
    if (file):
        return load_function(file)


Ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

##Preprocess
import data_preprocessing


def preprocess() -> Pipeline:
    return data_preprocessing.get_pipeline()


### Create Pipeline
def create_pipeline_estimator(preprocess_pipeline):
    preprocess_pipeline.steps.extend([
        ('reduce_dim', SelectKBestWrapper()),
        ('clf', ClfSwitcher())
    ])


### Run pipeline
def main(dataset_dir, dataset_file, features_score):
    df = loadData(dataset_dir, dataset_file)
    x, y = df.iloc[:, :-1], np.squeeze(df.iloc[:, -1:])
    pipline = preprocess()
    create_pipeline_estimator(pipline)

    REDUCE_DIM_ALGS = [Mrmr(), FClassif(), Rfe()]  # DUFS(), CFS()
    parameters = [
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": Ks,
            'clf__estimator': [GaussianNB()],
        },
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": Ks,
            'clf__estimator': [KNeighborsClassifier()],
        },
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": Ks,
            'clf__estimator': [RandomForestClassifier()],
        },
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": Ks,
            'clf__estimator': [LogisticRegression()],
        },
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": Ks,
            'clf__estimator': [SVC()],
            'clf__estimator__probability': [True],
        }
    ]
    cv = get_cv(x)
    gscv = GridSearchCV(pipline, parameters, cv=cv, error_score='raise',
                        n_jobs=int(os.environ['N_JOBS']), return_train_score=False,
                        verbose=10, scoring=get_scoring(y), refit='ACC')
    gscv.fit(x, y)
    # not pickleable
    gscv.scorer_ = {}
    gscv.scoring = {}
    metadata = {
        'file': dataset_file,
        'samples': len(x)
    }
    np.savez('grid-search-best-{}.npz'.format(dataset_file), gscv=gscv, features_score=dict(features_score),
             data=metadata)


class ScoringWrapper(_BaseScorer):

    def __init__(self, score_func, needs_proba=True, save_metrics=False):
        super().__init__(score_func, 1, {})
        self.scorer = make_scorer(score_func, needs_proba=needs_proba)
        if save_metrics:
            self._score = self.metrics_scorer

    def _select_proba_binary(self, y_pred, classes):
        return self.scorer._select_proba_binary(y_pred, classes)

    def metrics_scorer(self, method_caller, estimator, X, y_true, sample_weight=None):
        start_time = time.time()
        return_val = self.scorer._score(method_caller, estimator, X, y_true, sample_weight)
        runtime = time.time() - start_time
        self.save_features(estimator, runtime)
        return return_val

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        return self.scorer._score(method_caller, estimator, X, y_true, sample_weight)

    def save_features(self, estimator, runtime):
        reduce_dim_step = estimator['reduce_dim']
        clf = type(estimator['clf'].estimator).__name__
        score_func = reduce_dim_step.score_func
        reduce_dim_name = type(score_func).__name__
        key = (reduce_dim_name, reduce_dim_step.k, clf)
        global features_score
        if key not in features_score:  # train call after test, so we don't want to override
            features_in = estimator.named_steps['fclassifwrapper'].get_feature_names_out(
                input_features=estimator.feature_names_in_)
            clf_features = reduce_dim_step.get_feature_names_out(features_in)
            features_score[key] = (
                {k: v for k, v in zip(clf_features, reduce_dim_step.scores_)},
                score_func.total_runtime, [runtime])
        elif runtime is not None:
            fs, score_time, inference_time = features_score[key]
            inference_time.append(runtime)
            features_score[key] = (fs, score_time, inference_time)


def get_scoring(y):
    scoring = {'ACC': ScoringWrapper(accuracy_score, False, save_metrics=True),
               'MCC': ScoringWrapper(matthews_corrcoef, False),
               }

    def pr_auc(y_yest, y_score):

        precision, recall, thresholds = precision_recall_curve(y_yest, y_score)
        # Use AUC function to calculate the area under the curve of precision recall curve
        return auc(recall, precision)

    classes = np.unique(y)
    if len(classes) > 2:
        def score(y_test, y_score):
            return roc_auc_score(y_test, y_score, multi_class='ovo', average='weighted', labels=classes)

        scoring['AUC'] = ScoringWrapper(score)

        # Based on https://stackoverflow.com/a/56092736
        def pr_auc_multiclass(y_test, y_score):
            y_bin = label_binarize(y_test, classes=classes)
            auc_score = []
            for i in range(len(set(y_test))):
                auc_score.append(pr_auc(y_bin[:, i], y_score[:, i]))
            return np.mean(auc_score)

        scoring['PR-AUC'] = ScoringWrapper(pr_auc_multiclass)
    else:
        scoring['PR-AUC'] = ScoringWrapper(pr_auc)

        def failsafe_roc_auc(y_test, y_score):
            try:
                return roc_auc_score(y_test, y_score)
            except:
                return 0.5

        scoring['AUC'] = ScoringWrapper(failsafe_roc_auc)
    return scoring


def run_toy_example(dataset_dir, dataset_file, features_score):
    df = loadData(dataset_dir, dataset_file)
    x, y = df.iloc[:, 1:], np.squeeze(df.iloc[:, 0])
    x = x.add_prefix('f_')
    pipline = preprocess()
    create_pipeline_estimator(pipline)

    REDUCE_DIM_ALGS = [DUFS(), CFS()]
    parameters = [
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": [5, 10],
            'clf__estimator': [KNeighborsClassifier()],
        },
        {
            "reduce_dim__score_func": REDUCE_DIM_ALGS,
            "reduce_dim__k": [5],
            'clf__estimator': [SVC()],
            'clf__estimator__probability': [True],
        }
    ]
    cv = get_cv(x)
    gscv = GridSearchCV(pipline, parameters, cv=cv, error_score='raise',
                        n_jobs=1, return_train_score=False,
                        verbose=10, scoring=get_scoring(y), refit='ACC')

    gscv.fit(x, y)
    # not pickleable
    gscv.scorer_ = {}
    gscv.scoring = {}
    metadata = {
        'file': dataset_file,
        'samples': len(x)
    }
    np.savez('grid-search-{}.npz'.format(dataset_file), gscv=gscv, features_score=dict(features_score), data=metadata)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    features_score = manager.dict()
    main('bioconductor','CLL.csv', features_score)
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == 'toy':
        manager = multiprocessing.Manager()
        features_score = manager.dict()
        run_toy_example('toy_example', 'SPECT.train', features_score)
    else:
        for db, datasets in data_loader.data.items():
            for dataset in datasets:
                try:
                    print(db, dataset)
                    start_time = time.time()
                    manager = multiprocessing.Manager()
                    features_score = manager.dict()
                    main(db, dataset, features_score)
                    print('finished, took:', time.time() - start_time, db, dataset)
                except:
                    print('failed to run:', db, dataset)
                    traceback.print_exc()

