#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

results = pickle.load(open('results.pickle', 'rb'))
obs = results['obs']
params = results['params']


def ift(times):
    hist = obs[1]
    freq_bins = params['freq_bins']
    max_freq = 2*np.pi/params['min_period']
    return [ np.real(np.matmul(hist[index,None,:], np.exp(1j* np.arange(freq_bins)[:,None]/freq_bins*max_freq * times[None,:]))[0,:]) for index in range(hist.shape[0])]

times = np.linspace(0, 3600*24*7, 1000)

to_plot = [1, 13, 28]
ifts = ift(times)

fig = plt.figure(figsize=(6,2.3))
for i,j in enumerate(to_plot):
    plt.semilogy(np.abs(obs[1][j,:]), label='Observer %d' % (i+1))
plt.xlabel('Frequency (week$^{-1}$)')
plt.ylabel('Observations')
plt.legend(bbox_to_anchor=(0,1,1,0), loc="lower right", ncol=3)
plt.tight_layout()
plt.savefig('fts_darkspace.pdf', bbox_inches='tight', pad_inches=0)
plt.close(fig)

fig = plt.figure(figsize=(6,2.3))
for i,j in enumerate(to_plot):
    plt.plot(times/3600, ifts[j], label='Observer %d' % (i+1))
plt.xlabel('Time (h)')
plt.ylabel('Observations')
plt.legend(bbox_to_anchor=(0,1,1,0), loc="lower right", ncol=3)
plt.tight_layout()
plt.savefig('temporal_1w_darkspace.pdf', bbox_inches='tight', pad_inches=0)
plt.close(fig)

for i in range(3):
    plt.semilogy(np.abs(obs[1][i,:]))
plt.show()

os.makedirs('fts', exist_ok=True)
os.makedirs('temporal_1w', exist_ok=True)
times = np.linspace(0, 3600*24*7, 1000)
for i,temporal_1w in enumerate(ift(times)):
    fig = plt.figure()
    plt.semilogy(np.abs(obs[1][i,:]))
    plt.xlabel('Frequency ($w^{-1}$)')
    plt.ylabel('Observations')
    plt.savefig('fts/%d.pdf' % i)
    plt.close(fig)
    fig = plt.figure()
    plt.plot(times/3600, temporal_1w)
    plt.xlabel('Time (h)')
    plt.ylabel('Observations')
    plt.savefig('temporal_1w/%d.pdf' % i)
    plt.close(fig)