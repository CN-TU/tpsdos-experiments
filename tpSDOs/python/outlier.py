# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import numpy as np
import random
import multiprocessing as mp

from . import swig as dSalmon_cpp
from .util import sanitizeData, lookupDistance

class OutlierDetector(object):

	def _init_model(self, p):
		pass

	def get_params(self, deep=True):
		return self.params

	def set_params(self, **params):
		p = self.params.copy()
		for key in params:
			assert key in p, 'Unknown parameter: %s' % key
			p[key] = params[key]
		self._init_model(p)

	def _processData(self, data):
		data = sanitizeData(data, self.params['float_type'])
		assert self.dimension == -1 or data.shape[1] == self.dimension
		self.dimension = data.shape[1]
		return data

	def _processTimes(self, data, times):
		if times is None:
			times = np.arange(self.last_time + 1, self.last_time + 1 + data.shape[0])
		else:
			times = np.array(times, dtype=self.params['float_type'])
			assert len(times.shape) <= 1
			if len(times.shape) == 0:
				times = np.repeat(times[None], data.shape[0])
			else:
				assert times.shape[0] == data.shape[0]
		self.last_time = times[-1]
		return times


class tpSDOs(OutlierDetector):
	"""Streaming outlier detection based on Sparse Data Observers."""

	def __init__(self, k, T, qv=0.3, x=6, freq_bins=10, min_period=None, metric='euclidean', metric_params={}, float_type=np.float64, seed=0, return_sampling=False, n_estimators=1):
		"""
		Parameters
		----------
		k: int
			Number of observers to use.
			
		T: int
			Characteristic time for the model.
			Increasing T makes the model adjust slower, decreasing T
			makes it adjust quicker.
			
		qv: float, optional (default=0.3)
			Ratio of unused observers due to model cleaning.
			
		x: int (default=6)
			Number of nearest observers to consider for outlier scoring
			and model cleaning.

		metric: string
			Which distance metric to use. Currently supported metrics
			include 'chebyshev', 'cityblock', 'euclidean' and
			'minkowsi'.

		metric_params: dict
			Additional keywords to pass to the metric.

		float_type: np.float32 or np.float64
			The floating point type to use for internal processing.

		seed: int (default=0)
			Random seed to use.
			
		return_sampling: bool (default=False)
			Also return whether a data point was adopted as observer.
		"""
		self.params = { k: v for k, v in locals().items() if k != 'self' }
		self._init_model(self.params)

	def _max_freq(self, p):
		if p['min_period'] is None:
			return 2*np.pi/(p['freq_bins']*p['T']/10)
		else:
			return 2*np.pi/p['min_period']
		
	def _init_model(self, p):
		assert p['float_type'] in [np.float32, np.float64]
		assert 0 <= p['qv'] < 1, 'qv must be in [0,1)'
		assert p['x'] > 0, 'x must be > 0'
		assert p['k'] > 0, 'k must be > 0'
		assert p['T'] > 0, 'T must be > 0'
		max_freq = self._max_freq(p)
		distance_function = lookupDistance(p['metric'], p['float_type'], **p['metric_params'])
		cpp_obj = {np.float32: dSalmon_cpp.tpSDOs32, np.float64: dSalmon_cpp.tpSDOs64}[p['float_type']]
		self.model = cpp_obj(p['k'], p['T'], p['qv'], p['x'], p['freq_bins'], max_freq, distance_function, p['seed'], p['n_estimators'])
		self.last_time = 0
		self.dimension = -1
		
	def fit_predict(self, data, times=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		times: ndarray, shape (n_samples,), optional
			Timestamps for input data. If None,
			timestamps are linearly increased for
			each sample. 
		
		Returns	
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for provided input data.
		"""
		data = self._processData(data)
		times = self._processTimes(data, times)
		scores = np.zeros(data.shape[0], dtype=self.params['float_type'])
		if self.params['return_sampling']:
			sampling = np.zeros(data.shape[0], dtype=np.int32)
			self.model.fit_predict_with_sampling(data, scores, np.array(times, dtype=self.params['float_type']), sampling)
			return scores, sampling
		else:
			self.model.fit_predict(data, scores, np.array(times, dtype=self.params['float_type']))
			return scores
		
	def observer_count(self):
		"""Return the current number of observers."""
		return self.model.observer_count()
		
		
	def get_observers(self, time=None):
		"""
		Return observer data.
		
		Returns	
		---------------
		data: ndarray, shape (n_observers, n_features)
			Sample used as observer.
			
		observations: ndarray, shape (n_observers,)
			Exponential moving average of observations.
			
		av_observations: ndarray, shape (n_observers,)
			Exponential moving average of observations
			normalized according to the theoretical maximum.
		"""
		if time is None:
			time = self.last_time
		observer_cnt = self.model.observer_count()
		freq_bins = self.model.frequency_bin_count()
		complex_type = np.complex64 if self.params['float_type'] == np.float32 else np.complex128
		if observer_cnt == 0:
			return np.zeros([0], dtype=self.params['float_type']), np.zeros([0,freq_bins], dtype=complex_type), np.zeros([0], dtype=self.params['float_type'])
		data = np.zeros([observer_cnt, self.dimension], dtype=self.params['float_type'])
		observations = np.zeros([observer_cnt, freq_bins], dtype=complex_type)
		av_observations = np.zeros(observer_cnt, dtype=self.params['float_type'])
		self.model.get_observers(data, observations, av_observations, self.params['float_type'](time))
		return data, observations, av_observations

	def ift(self, length, resolution=1000):
		_,hist,_ = self.get_observers()
		freq_bins = self.params['freq_bins']
		return [ np.real(np.matmul(hist[index,None,:], np.exp(1j* np.arange(freq_bins)[:,None]/freq_bins*self._max_freq(self.params) *  np.linspace(0,length,resolution)[None,:]))[0,:]) for index in range(hist.shape[0])]
