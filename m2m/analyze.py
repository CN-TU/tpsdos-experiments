#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess
from multiprocessing import Pool
from datetime import datetime

results = pickle.load(open('results.pickle', 'rb'))
obs = results['obs']
params = results['params']

# plot the magnitude spectrum of observers into the fts directory
os.makedirs('fts', exist_ok=True)
for i in range(obs[0].shape[0]):
    plt.plot(np.linspace(0, 60, len(obs[1][i,:])), np.abs(obs[1][i,:]))
    plt.xlabel('Frequency (1/h)')
    plt.savefig('fts/%d.png' % i)
    plt.clf()

# plot 24h and 1h temporal plots into the temporal_1h and temporal_24h directories
def ift(times):
    hist = obs[1]
    freq_bins = params['freq_bins']
    max_freq = 2*np.pi/params['min_period']
    return [ np.real(np.matmul(hist[index,None,:], np.exp(1j* np.arange(freq_bins)[:,None]/freq_bins*max_freq * times[None,:]))[0,:]) for index in range(hist.shape[0])]

os.makedirs('temporal_1h', exist_ok=True)
times = np.linspace(0, 3600*1000, 1000)
for i,temporal_shape in enumerate(ift(times)):
    plt.plot(times/(60*1000), temporal_shape)
    plt.xlabel ('Time (minutes)')
    plt.savefig('temporal_1h/%d.png' % i)
    plt.clf()

os.makedirs('temporal_24h', exist_ok=True)
times = np.linspace(0, 24*3600*1000, 1000)
for i,temporal_shape in enumerate(ift(times)):
    plt.plot(times/(3600*1000), temporal_shape)
    plt.xlabel ('Time (hours)')
    plt.savefig('temporal_24h/%d.png' % i)
    plt.clf()
