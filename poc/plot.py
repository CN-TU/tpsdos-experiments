#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lines = [ l.strip().split(',') for l in open('results.csv') ]

x = np.unique([ float(x[0]) for x in lines])

plt.figure(figsize=(7,1.7))
for algo in ['tpsdos', 'swknn', 'swrrct']:
    plt.plot(x, [ next(float(l[-1]) for l in lines if float(l[0]) == v and l[1]==algo) for v in x])
plt.xlabel('Fraction of out-of-phase outliers')
plt.ylabel('ROC-AUC')
plt.legend(['Our method', 'SW-KNN', 'RRCT'], loc=(1.04,0.35))
plt.tight_layout()
plt.savefig('poc.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
