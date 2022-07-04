#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tpSDOs.outlier import tpSDOs
import pickle
from tqdm import tqdm

APPROX_TOTAL = 2986256723
CHUNKSIZE = 1000000

numerical_fields = [
    'packetTotalCount',
    'distinct(dst_ip)',
    'modeCount(dst_ip)',
    'distinct(src_port)',
    'mode(src_port)',
    'modeCount(src_port)',
    'distinct(dst_port)',
    'mode(dst_port)',
    'modeCount(dst_port)',
    'distinct(proto)',
    'modeCount(proto)',
    'distinct(ttl)',
    'mode(ttl)',
    'modeCount(ttl)',
    'distinct(tcp_flags)',
    'modeCount(tcp_flags)',
    'distinct(ip_len)',
    'mode(ip_len)',
    'modeCount(ip_len)']

total_processed = 0
sums = pd.Series(np.zeros(len(numerical_fields)), index=numerical_fields)
sq_sums = pd.Series(np.zeros(len(numerical_fields)), index=numerical_fields)

for chunk in tqdm(pd.read_csv('agm.csv', chunksize=CHUNKSIZE),total=APPROX_TOTAL//CHUNKSIZE):
    total_processed += chunk.shape[0]
    sums += chunk[numerical_fields].sum(axis=0)
    sq_sums += (chunk[numerical_fields]**2).sum(axis=0)

means = sums/total_processed
stds = (sq_sums/total_processed-means**2)**(1/2)
    
det = tpSDOs(k=100, T=3600*24*7*10, freq_bins=100, min_period=3600*24*7/100, metric='trunc_euclidean', metric_params={'len': len(numerical_fields)})

for chunk in tqdm(pd.read_csv('agm.csv', chunksize=CHUNKSIZE),total=APPROX_TOTAL//CHUNKSIZE):
    values = ((chunk[numerical_fields]-means)/stds).values
    det.fit_predict(np.concatenate((values, chunk.index.values[:,None]), axis=1), chunk['flowStart'])

with open('results.pickle', 'wb') as f:
    pickle.dump({
        'norm_data': (means.values,stds.values),
        'params': det.get_params(),
        'obs': det.get_observers()}, f)
