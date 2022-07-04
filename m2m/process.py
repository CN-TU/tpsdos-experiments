#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tpSDOs.outlier import tpSDOs
import pickle
from tqdm import tqdm

SUBSAMPLE = 1

d = pd.read_csv('capture.csv').fillna(0)

remove_columns = ['flowStartMilliseconds', 'sourceIPAddress', 'destinationIPAddress', 'apply(_tcpCwrTotalCount,backward)']
times = d['flowStartMilliseconds']
keep_cols = [ i for i in range(d.shape[1]) if d.columns[i] not in remove_columns ]

values = d.iloc[:,keep_cols].values
values = (values - values.mean(axis=0)) / values.std(axis=0)
values_with_index = np.concatenate((values, np.arange(values.shape[0])[:,None]), axis=1)

sampled_indices = np.sort(np.random.choice(values_with_index.shape[0], int(values_with_index.shape[0]*SUBSAMPLE), replace=False))
values_sampled = values_with_index[sampled_indices,:]
times_sampled = times[sampled_indices]
det = tpSDOs(k=400, T=3600*24*7*1000, freq_bins=2000, min_period=60*1000, metric='trunc_euclidean', metric_params={'len': values.shape[1]}, return_sampling=True)
scores_total = []
sampled_total = []
for i in tqdm(range(0,values_sampled.shape[0], 10000)):
    scores, sampled = det.fit_predict(values_sampled[i:i+10000,:], times_sampled[i:i+10000])
    scores_total.append(scores)
    sampled_total.append(sampled)

with open('results.pickle', 'wb') as f:
    pickle.dump({
        'sampled_indices': sampled_indices,
        'params': det.get_params(),
        'scores': np.concatenate(scores_total),
        'sampled': np.concatenate(sampled_total),
        'obs': det.get_observers()}, f)
