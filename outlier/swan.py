#!/usr/bin/env python3

#     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM

# Downloaded files:
# - partition1_instances.tar.gz
# - partition2_instances.tar.gz
# - partition3_instances.tar.gz
# - partition4_instances.tar.gz
# - partition5_instances.tar.gz

# repo: https://bitbucket.org/gsudmlab/swan_features/src/master/

import os
import sys
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import scale, minmax_scale

sys.path.append('gsudmlab-swan_features')
import features.feature_collection as fc
from features.feature_extractor import FeatureExtractor

# Choose statistical features and physical parameters
features_list = [fc.get_min, fc.get_max, fc.get_median]
params_name_list = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ']

path_to_dest =  'swan_output'

configurations = []
for partition in range(1,6):
    for flare in ('FL', 'NF'):
        configurations.append((partition,flare))
def process_file(args):
    partition, flare = args
    root = 'partition%d/%s' % (partition,flare)
    filename = 'raw_features_p%d_%s.csv' % (partition,flare)
    pc = FeatureExtractor(root, path_to_dest, filename)
    pc.calculate_all(features_list, params_name_list=params_name_list)

with multiprocessing.Pool(40) as pool:
    pool.map(process_file, configurations)

raw_files = ['swan_output/raw_features_p%d_%s.csv' % (partition,flare) for partition in range(1,6) for flare in ('FL','NF')]
df = pd.concat([ pd.read_csv(fn, sep='\t') for fn in raw_files ])

df['START_TIME'] = pd.to_datetime(df['START_TIME']).astype(np.int64)/10**9
df.sort_values(by='START_TIME', inplace=True)
y = (df['LABEL'] != 'F').values
times = df['START_TIME'].values
del df['LABEL']
del df['START_TIME']
del df['NOAA_AR_NO']
del df['END_TIME']
X = scale(df.values)

np.savez('swan.npz', X=X, y=y, times=times)
