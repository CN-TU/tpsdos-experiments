#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

results = pickle.load(open('results.pickle', 'rb'))
scores = results['scores']

os.makedirs('figs', exist_ok=True)
plt.figure(figsize=(6,2))
plt.plot(scores)
plt.xlabel('Processed data point')
plt.ylabel('Outlier score')
plt.tight_layout()
plt.savefig('figs/scores.pdf')
plt.show()
