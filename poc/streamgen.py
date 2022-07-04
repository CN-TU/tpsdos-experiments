
import numpy as np
from mdcgenpy.clusters import ClusterGenerator

n_samples = 1000000000
BATCH=1000
CACHE_SIZE = 1000

rel_outliers = 0.05
n_outliers = n_samples*rel_outliers

class StreamGenerator:
    def __init__(self, **kwargs):
        self.k = kwargs.get('k', 5)
        self.n_feats = kwargs.get('n_feats', 2)
        self.mdcgen = ClusterGenerator(n_samples=n_samples, outliers=n_outliers, **kwargs)
        self.generator = self.mdcgen.generate_data(BATCH)
        self.cache = [ np.zeros((0,self.n_feats)) for _ in range(self.k+1) ]

    def fill_caches(self, to_len):
        cache_filled = [ self.cache[i].shape[0] >= to_len[i] for i in range(self.k+1) ]
        to_fill = self.k+1 - sum(cache_filled)

        while to_fill > 0:
            data, labels = next(self.generator)
            indices = np.zeros((BATCH, self.k+1),dtype=bool)
            indices[range(BATCH),labels[:,0]+1] = True
            available = np.sum(indices, axis=0)
            for i in range(self.k+1):
                to_copy = CACHE_SIZE+to_len[i]-self.cache[i].shape[0]
                if to_copy > 0:
                    to_copy = min(to_copy, available[i])
                    data_to_copy = data[indices[:,i],:][:to_copy,:]
                    self.cache[i] = np.vstack((self.cache[i],data_to_copy))
                if not cache_filled[i] and self.cache[i].shape[0] >= to_len[i]:
                    cache_filled[i] = True
                    to_fill -= 1

    def generate(self, probabilities):
        samples = probabilities.shape[0]
        probabilities /= np.sum(probabilities, axis=1)[:,None]
        cprobabilities = np.cumsum(probabilities, axis=1)
        labels = np.argmax(np.random.rand(samples)[:,None] <= cprobabilities, axis=1)
        masks = np.zeros((samples,self.k+1), dtype=bool)
        masks[range(samples),labels] = True
        no_sampleds = np.sum(masks, axis=0)
        self.fill_caches(no_sampleds)
        results = np.zeros((samples,self.n_feats))
        for i,no_sampled in enumerate(no_sampleds):
            results[masks[:,i],:] = self.cache[i][:no_sampled,:]
            self.cache[i] = self.cache[i][no_sampled:]
        return results, labels-1