#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

BINS = 14

results = pickle.load(open('results.pickle', 'rb'))
sampled = results['sampled']
df = pd.read_csv('capture.csv')

times = df['flowStartMilliseconds'].values
mask = times < times[0]+1000*3600*24*7*2
times = times[mask]

sampled = sampled[mask]


assert len(times) == len(sampled)
boundaries = np.linspace(times[0], times[-1], BINS+1)
boundaries_i = [0] + np.searchsorted(times, boundaries[1:-1]).tolist() + [len(times)]

os.makedirs('figs', exist_ok=True)
plt.figure(figsize=(6,2.5))
plt.bar(range(1,BINS+1), [ np.sum(sampled[bin_from:bin_to]) for bin_from, bin_to in zip(boundaries_i[:-1], boundaries_i[1:]) ], 1)
plt.xlabel('Day after capture start')
plt.ylabel('Sampled observers')
plt.xticks(range(1,BINS+1))
plt.tight_layout()
plt.savefig('figs/sampling.pdf')
plt.show()
