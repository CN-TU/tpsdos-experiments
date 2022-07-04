#!/usr/bin/env python3

from streamgen import StreamGenerator
from tpSDOs.outlier import tpSDOs
from dSalmon.outlier import SWKNN, SWRRCT, SWLOF, LODA
import numpy as np
import pickle
import sys
import subprocess
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

if len(sys.argv) <= 1:
    fractions = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    for fraction in fractions:
        subprocess.check_call([sys.executable, sys.argv[0], '%.3f' % fraction])
    sys.exit(0)

seed = 1
np.random.seed(seed)

clusters = 3
period = 50000

outliers = 0.05 # 5% outliers compared to one active cluster
temp_outliers = float(sys.argv[1])
batch = 500

k_param = 400
max_period = 3*period
x_param = 3
qv_param = 0.9
freq_bins_param = 200
min_period_param = max_period / freq_bins_param

print ('Running for temporal outlier fraction %.2f' % temp_outliers)
gen = StreamGenerator(k=clusters, seed=seed)

detectors = [
    ('tpsdos', tpSDOs(k=k_param, T=10*max_period, qv=qv_param, x=x_param, freq_bins=freq_bins_param, min_period=min_period_param)),
    ('swknn', SWKNN(k=9, window=period)),
    ('swrrct', SWRRCT(n_estimators=20, window=period))
]

cluster_offsets = [ np.random.randint(period) for i in range(clusters) ]
cluster_widths = [ np.random.randint(period) for i in range(clusters) ]

def make_probabilities(start_time, i):
    times = np.arange(start_time, start_time + batch)
    return (1-outliers)*(temp_outliers + (1-temp_outliers)*( ((times - cluster_offsets[i]) % period) < cluster_widths[i]).astype(np.float64))

probabilities = np.zeros((batch,clusters+1))
probabilities[:,0] = outliers
time = 0

scores_total = [ np.zeros([0]) for _ in detectors ]
labels_total = np.zeros([0], dtype=int)
roc_aucs = [ [] for _ in detectors ]

for _ in tqdm(range(100000)):
    for i in range(clusters):
        probabilities[:,i+1] = make_probabilities(time, i)
    mask = np.sum(probabilities, axis=1)/clusters > np.random.rand(probabilities.shape[0])
    time += len(mask)
    if np.sum(mask) == 0:
        continue
    times = time + np.arange(probabilities.shape[0])[mask]

    data, labels = gen.generate(probabilities[mask,:])

    if temp_outliers > 0:
        temp_outlier_labels = np.zeros(labels.shape, dtype=bool)
        for i in range(clusters):
            temp_outlier_labels += (probabilities[mask,i+1] <= temp_outliers) & (labels == i)
        labels[temp_outlier_labels] = -2

    for i, (_,det) in enumerate(detectors):
        s = det.fit_predict(data, times)
        scores_total[i] = np.concatenate((scores_total[i], s))
    labels_total = np.concatenate([labels_total, labels])
    assert scores_total[0].shape == labels_total.shape
    if len(scores_total[0]) > period:
        l = (labels_total[:period] == -1) + (labels_total[:period] == -2)
        for i,ra in enumerate(roc_aucs):
            ra.append(roc_auc_score(l, -1/(1+scores_total[i][:period])))
            scores_total[i] = scores_total[i][period:]
        print (time, detectors[-2][1].window_size(), detectors[-1][1].window_size())
        print (roc_aucs)
        labels_total = labels_total[period:]

with open('results.csv', 'a') as f:
    for ra,(name,_) in zip(roc_aucs,detectors):
        f.write('%s,%s,%s\n' % (str(temp_outliers),name,','.join([str(x) for x in ra])))
        