// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#include <algorithm>
#include <vector>
#include <assert.h>
#include <random>
#include <atomic>
#include <mutex>
#include <thread>

#include "outlier_wrapper.h"

#include <iostream>

template<typename FloatType>
tpSDOs_wrapper<FloatType>::tpSDOs_wrapper(int observer_cnt, FloatType T, FloatType idle_observers, int neighbour_cnt, int freq_bins, FloatType max_freq, Distance_wrapper<FloatType>* distance, int seed, int n_estimators) :
	dimension(-1),
	freq_bins(freq_bins), // use freq_bins and max_freq parameters when implementing periodic tpSDOs
	// sdo(observer_cnt, T, idle_observers, neighbour_cnt, distance->getFunction(), seed)
	//sdo(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance->getFunction(), seed),
	n_estimators(n_estimators)
{
	std::mt19937 rng(seed);
	estimators.reserve(n_estimators);
	for (int k = 0; k < n_estimators; k++)
		estimators.emplace_back(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance->getFunction(), rng());
}

template<typename FloatType>
void tpSDOs_wrapper<FloatType>::fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times) {
	auto thread_worker = [&](int k) {
		for (int i = 0; i < data.dim1; i++) {
			estimators[k].fit(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
		}
	};
	std::thread threads[n_estimators];
	for (int k = 0; k < n_estimators; k++)
		threads[k] = std::thread{thread_worker, k};
	for (int k = 0; k < n_estimators; k++)
		threads[k].join();
}

template<typename FloatType>
void tpSDOs_wrapper<FloatType>::fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled) {
	std::mutex lock;
	std::fill(scores.data, scores.data + data.dim1, 0);
	auto thread_worker = [&](int k) {
		for (int i = 0; i < data.dim1; i++) {
			FloatType score = estimators[k].fitPredict(Vector<FloatType>{&data.data[i * data.dim2], data.dim2}, times.data[i]);
			lock.lock();
			scores.data[i] += score;
			if (sampled.data && k==0)
				sampled.data[i] = estimators[0].lastWasSampled();
			lock.unlock();
		}
	};
	std::thread threads[n_estimators];
	for (int k = 0; k < n_estimators; k++)
		threads[k] = std::thread{thread_worker, k};
	for (int k = 0; k < n_estimators; k++)
		threads[k].join();
}

template<typename FloatType>
void tpSDOs_wrapper<FloatType>::fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times) {
	NumpyArray1<int> sampled_dummy;
	sampled_dummy.data = 0;
	fit_predict_with_sampling(data, scores, times, sampled_dummy);
}

template<typename FloatType>
int tpSDOs_wrapper<FloatType>::observer_count() {
	return estimators[0].observerCount();
}

template<typename FloatType>
int tpSDOs_wrapper<FloatType>::frequency_bin_count() {
	return freq_bins;
}

template<typename FloatType>
void tpSDOs_wrapper<FloatType>::get_observers(NumpyArray2<FloatType> data, NumpyArray2<std::complex<FloatType>> observations, NumpyArray1<FloatType> av_observations, FloatType time) {
	// TODO: check dimensions
	int i = 0;
	for (auto observer : estimators[0]) {
		Vector<FloatType> vec_data = observer.getData();
		std::copy(vec_data.begin(), vec_data.end(), &data.data[i * data.dim2]);
		// observations.data[i] = observer.getObservations(time);
		//use complex ft when implementing periodic tpSDOs
		std::vector<std::complex<FloatType>> observations_ft = observer.getObservations(time);
		std::copy(observations_ft.begin(), observations_ft.end(), &observations.data[i * observations.dim2]);
		av_observations.data[i] = observer.getAvObservations(time);
		i++;
	}
}

template<typename FloatType>
FloatType tpSDOs_wrapper<FloatType>::active_observations_threshold() {
	return estimators[0].getActiveObservationsThreshold();
}

template class tpSDOs_wrapper<double>;
template class tpSDOs_wrapper<float>;