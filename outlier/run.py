#!/usr/bin/env python3

import time
import os
import sys
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from dSalmon.outlier import SWKNN, SWLOF, SWRRCT, RSHash, LODA
from tpSDOs.outlier import tpSDOs

dataset_name = sys.argv[1]

dataset = np.load('%s.npz' % dataset_name)
data = np.concatenate((dataset['X'], dataset['times'][:,None]), axis=1)
labels = dataset['y']

print ('Dataset shape: %s' % str(dataset['X'].shape))

# How much data to skip during scoring to overcome transient starting phase
TRAIN_LEN = int(data.shape[0]*0.1)

val_indices = np.sort(np.random.permutation(data.shape[0])[:data.shape[0]//2])
test_indices = np.delete(np.arange(data.shape[0]), val_indices)

# data for parameter search
val_data = data[val_indices,:]
val_labels = labels[val_indices]

# data for evaluation
test_data = data[test_indices,:]
test_labels = labels[test_indices]

while len(sys.argv) > 3:
    # by forking now we can parallelize
    # with very high memory efficiency
    if os.fork() == 0: del sys.argv[3:]
    else: del sys.argv[2]

algorithm = sys.argv[2]

time_parameters = {
    'swan': [0.5*24*3600*1000, 1*24*3600*1000, 10*24*3600*1000, 100*24*3600*1000, 1000*24*3600*1000],
    'kddcup': [100, 500, 1000, 5000, 10000, 50000]
}[dataset_name]

param_distribution = {
    # we use randomized search for the following algorithms
    'tpsdose': {
        'k': [500, 800, 1100, 1400],
        'freq_bins': [100],
        'x': np.arange(3, 50, 4).tolist(),
        'qv': [0.1, 0.3, 0.5, 0.7],
        'T': time_parameters,
        'n_estimators': [9]
    },
    'swrrct': {
        'n_estimators': [ int(x) for x in np.linspace(10, 300, num=20) ],
        'window': time_parameters,
        'n_jobs': [2]
    },
    'rshash': {
        'n_estimators': [ int(x) for x in np.linspace(10, 300, num=20) ],
        'cms_w': [4, 6, 8, 10, 12],
        'cms_d': [500, 1000, 5000, 10000, 50000],
        'window': time_parameters
    },
    'loda': {
        'window': time_parameters,
        'n_projections': [10, 20, 50, 100],
        'n_bins': [5, 10, 50, 100]
    },
    # we use gridsearch for the following algorithms
    'swknn': {
        'k_max': 50,
        'window': time_parameters
    },
    'swlof': {
        'k_max': 35,
        'window': time_parameters
    }
}[algorithm]

estimator = {
    'swknn': SWKNN,
    'swlof': SWLOF,
    'swrrct': SWRRCT,
    'rshash': RSHash,
    'loda': LODA,
    'tpsdose': tpSDOs
}[algorithm]

def set_rshash_s(params, X):
    params['s_param'] = int(params['window'] * X.shape[0] / (X[-1,-1] - X[0,-1]))

def get_indices(y_true, scores):
    m = y_true.size
    num_outliers = np.sum(y_true)
    sort_perm = np.argsort(scores)
    y_true_invs = y_true[sort_perm[::-1]]
    res = {}
    # P@n
    res['Patn'] = np.sum(y_true_invs[:num_outliers]) / num_outliers
    res['adj_Patn'] = (res['Patn'] - num_outliers/m) / (1 - num_outliers/m)
    y_true_cs = np.cumsum(y_true_invs[:])
    # average precision
    res['ap'] = np.sum( y_true_cs[:num_outliers] / np.arange(1, num_outliers + 1) ) / num_outliers
    res['adj_ap'] = (res['ap'] - num_outliers/m) / (1 - num_outliers/m)
    # Max. F1 score
    res['maxf1'] = 2 * np.max(y_true_cs[:m] / np.arange(1 + num_outliers, m + 1 + num_outliers))
    res['adj_maxf1'] = (res['maxf1'] - num_outliers/m) / (1 - num_outliers/m)
    # ROC-AUC
    res['auc'] = roc_auc_score(y_true, np.searchsorted(scores[sort_perm], scores))
    return res

def compute_training_score(y_true, scores):
    # roc_auc_score() cannot handle inf values. This transforms the scores,
    # so that order is retained, equal scores remain equal, but all scores
    # are finite.
    scores_sorted = np.sort(scores)
    return roc_auc_score(y_true, np.searchsorted(scores_sorted, scores))

def sw_gridsearch(windows, k_max):
    # The SWLOF and SWKNN algorithms allow returning outlier scores
    # for a range of possible k values. We use this to perform
    # grid search more efficiently.
    best_overall = None
    best_overall_score = -1
    for window in windows:
        print('Running for window=%d' % window)
        det = estimator(window, k_max, k_is_max=True)
        scores = det.fit_predict(val_data[:,:-1], val_data[:,-1])
        if np.any(np.isnan(scores)):
            print ('Warning: marking NaN as non-outlier')
            scores[np.isnan(scores)] = 0
        results = np.array([ compute_training_score(val_labels[TRAIN_LEN:], scores[TRAIN_LEN:,i]) for i in range(k_max) ])
        best_score = np.max(results)
        best_k = int(np.argmax(results) + 1)
        print ('Best k: %d with score=%.3f' % (best_k, best_score))
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = {'k': best_k, 'window': window}
    return best_overall, best_overall_score

class estimator_wrapper(BaseEstimator):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    def get_params(self, deep=True):
        return self.params
    def set_params(self, **kwargs):
        self.params = kwargs
        return self
    def fit(self, X, y):
        pass
    def score(self, X, y):
        params = self.params.copy()
        if algorithm == 'rshash':
            set_rshash_s(params, val_data)
        scores = np.zeros(val_data.shape[0])
        det = estimator(**params)
        for i in range(0,scores.size,10000):
            scores[i:i+10000] = det.fit_predict(val_data[i:i+10000,:-1], val_data[i:i+10000,-1])
        sc = compute_training_score(val_labels[TRAIN_LEN:], scores[TRAIN_LEN:])
        return sc

if algorithm in ['swknn', 'swlof']:
    best_params, best_score = \
        sw_gridsearch(param_distribution['window'], param_distribution['k_max'])
else:
    parameter_search = RandomizedSearchCV(
        estimator_wrapper(),
        param_distribution,
        cv=((np.arange(5),np.arange(5)),),
        n_iter=100,
        n_jobs=2,
        verbose=1).fit(np.random.rand(5,5), np.zeros(5))
    best_params = parameter_search.best_params_
    best_score = parameter_search.best_score_
    del parameter_search

print ('Best: %s with score=%.3f' % (str(best_params), best_score))
if algorithm == 'rshash':
    set_rshash_s(best_params, test_data)

scores = np.zeros(test_data.shape[0])
chunk_size = test_data.shape[0]//100
detector = estimator(**best_params)
start_time = time.time()
for start in range(0,scores.size,chunk_size):
    end = start + chunk_size
    scores[start:end] += \
        detector.fit_predict(test_data[start:end,:-1], test_data[start:end,-1])
    print('.', end='', flush=True)
end_time = time.time()
print('')

indices = ['Patn', 'adj_Patn', 'ap', 'adj_ap', 'maxf1', 'adj_maxf1', 'auc']
results = get_indices(test_labels[TRAIN_LEN:], scores[TRAIN_LEN:])
with open('results.csv', 'a') as results_file:
    if results_file.tell() == 0:
        results_file.write('algorithm,dataset,start_time,end_time,%s\n' % (','.join(indices)))
    results_file.write('%s,%s,%.2f,%.2f,%s\n' % (algorithm, dataset_name, start_time, end_time, ','.join('%.6f' % results[i] for i in indices)))
