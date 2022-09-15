from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics._scorer import _BaseScorer, make_scorer


# creating dummy class to support unpickle

class ScoringWrapper(_BaseScorer):

    def __init__(self, score_func, needs_proba=True):
        super().__init__(score_func, 1, {})
        self.scorer = make_scorer(score_func, needs_proba=needs_proba)
        self.feature_selection_time = -1

    def _select_proba_binary(self, y_pred, classes):
        return self.scorer._select_proba_binary(y_pred, classes)

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        return_val = self.scorer._score(method_caller, estimator, X, y_true, sample_weight)
        return return_val


### Aggregate results
def analyze_grid_search(file):
    results = np.load(file, allow_pickle=True)
    gscv = results['gscv'].item(0)
    feature_score = results['features_score'].item()
    metadata = results['data'].item()
    report = []
    if hasattr(gscv, 'best_params_'):
        best = {'estimator': gscv.best_params_['clf__estimator'],
                'k': gscv.best_params_['reduce_dim__k'],
                'Best Score': gscv.best_score_,
                'Algorithm': str(gscv.best_params_['reduce_dim__score_func'])}
        pd.DataFrame([best]).to_csv('reports/best-config-' + metadata['file'])
    for j, experiment in enumerate(gscv.cv_results_['params']):
        for i in range(gscv.n_splits_):
            split = {k[len(f'split{i}_'):]: v[j] for k, v in gscv.cv_results_.items() if
                     k.startswith(f'split{i}_')}
            clf = type(experiment['clf__estimator']).__name__
            k = experiment['reduce_dim__k']
            score_func = type(experiment['reduce_dim__score_func']).__name__
            key = (score_func, k, clf)
            score, reduce_dim_time, inference_time = feature_score[key]
            mean_fit_time = np.mean(gscv.cv_results_['mean_fit_time'])
            inference_time = np.mean(gscv.cv_results_['mean_score_time'])
            for measure_type, measure_value in split.items():
                if 'test' in measure_type:
                    _, mt = measure_type.split('_')
                    report.append({
                        'Database Name': metadata['file'],
                        'Number of samples': metadata['samples'],
                        'Filtering Algorithm': score_func,
                        'Learning Algorithm': clf,
                        'Number of features selected (K)': k,
                        'CV Method': type(gscv.cv).__name__,
                        'Fold': j,
                        'Measure Type': mt,
                        'Measure Value': measure_value,
                        'List of Selected Features Names (Long STRING)': ','.join(list(score.keys())),
                        'Selected Features scores': ','.join(['{:.3f}'.format(x) for x in score.values()]),
                        'FSMethod Time': reduce_dim_time,
                        'Fit Time': mean_fit_time,
                        'Inference Time': inference_time,
                    })
    df = pd.DataFrame(report)
    df.to_csv('reports/report-' + metadata['file'] + '.csv')
    return df


def find_best_param(file):
    df = pd.read_csv(file)
    df = df.loc[df['Measure Type'] == 'AUC']
    df = df.loc[df['Measure Value'].idxmax()]
    df.to_csv('reports/best-config-' + file.split('\\')[1] + '.csv')


if __name__ == '__main__':
    for file in glob('results/*best*.npz'):
        analyze_grid_search(file)
