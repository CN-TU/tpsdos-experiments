// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_SDOSTREAM_H
#define DSALMON_SDOSTREAM_H

#include <algorithm>
#include <boost/container/set.hpp>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <random>

#include "Vector.h"

template<typename FloatType=double>
class tpSDOs {
  public:
	typedef std::function<FloatType(const Vector<FloatType>&, const Vector<FloatType>&)> DistanceFunction;

  private:
	const std::complex<FloatType> imag_unit{0.0, 1.0};

	std::size_t observer_cnt;
	FloatType active_observers; // as fraction
	FloatType sampling_prefactor;
	FloatType fading;
	std::size_t neighbor_cnt;

	std::size_t freq_bins;
	FloatType max_freq;

	int last_index;
	FloatType last_time;
	int last_added_index;
	FloatType last_added_time;

	DistanceFunction distance_function;
	std::mt19937 rng;

	struct Observer {
		Vector<FloatType> data;
		std::vector<std::complex<FloatType>> observations;
		FloatType time_touched;
		FloatType age;
		// FloatType time_added;
		int index;
	};

	struct ObserverCompare{
		FloatType fading;
		ObserverCompare(FloatType fading) : fading(fading) {}
		bool operator()(const Observer& a, const Observer& b) const {
			FloatType common_touched = std::max(a.time_touched, b.time_touched);
			
			FloatType observations_a = real(a.observations[0])
				* std::pow(fading, common_touched - a.time_touched);
			
			FloatType observations_b = real(b.observations[0])
				* std::pow(fading, common_touched - b.time_touched);
			
			// tie breaker for reproducibility
			if (observations_a == observations_b)
				return a.index < b.index;
			return observations_a > observations_b;
		}
	} observer_compare;
	
	struct ObserverAvCompare{
		FloatType fading;
		ObserverAvCompare(FloatType fading) : fading(fading) {}
		bool operator()(FloatType now, const Observer& a, const Observer& b) {
			FloatType common_touched = std::max(a.time_touched, b.time_touched);
			
			FloatType observations_a = real(a.observations[0]) * std::pow(fading, common_touched - a.time_touched);
			// FloatType age_a = 1-std::pow(fading, now-a.time_added);
			
			FloatType observations_b = real(b.observations[0]) * std::pow(fading, common_touched - b.time_touched);
			// FloatType age_b = 1-std::pow(fading, now-b.time_added);
			
			return observations_a * b.age > observations_b * a.age;
			// do not necessarily need a tie breaker here
			// return observations_a * age_b > observations_b * age_a;
		}
	} observer_av_compare;

	typedef boost::container::multiset<Observer,ObserverCompare> MapType;
	typedef typename MapType::iterator MapIterator;
	MapType observers;

	void initNowVector(FloatType now, std::vector<std::complex<FloatType>>& now_vector) {
		now_vector.resize(freq_bins);
		for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
			FloatType frequency = max_freq * freq_ind / freq_bins;
			now_vector[freq_ind] = exp(imag_unit * (-frequency) * now);
		}
	}

	void updateNearestHeap(std::vector<std::pair<FloatType,MapIterator>>& nearest, FloatType distance, MapIterator it) {
		// TODO: make this static for performance ?
		const auto cmp = [] (const std::pair<FloatType,MapIterator>& a, const std::pair<FloatType,MapIterator>& b) {
			// in case of ties prefer older observers
			if (a.first == b.first)
				return a.second->index < b.second->index;
			return a.first < b.first;
		};
		if (nearest.size() < neighbor_cnt) {
			nearest.emplace_back(distance, it);
			std::push_heap(nearest.begin(), nearest.end(), cmp);
		}
		else if (nearest[0].first > distance) {
			std::pop_heap(nearest.begin(), nearest.end(), cmp);
			nearest.back().first = distance;
			nearest.back().second = it;
			std::push_heap(nearest.begin(), nearest.end(), cmp);
		}
	}

	FloatType fitPredict_impl(const Vector<FloatType>& data, FloatType now, bool fit_only) {
		FloatType score = 0;
		std::vector<std::pair<FloatType,MapIterator>> nearest;
		std::vector<std::complex<FloatType>> now_vector;
		initNowVector(now, now_vector);
		FloatType active_observations_thresh = getActiveObservationsThreshold();

		std::vector<FloatType> distances;
		std::vector<FloatType> proj_observations;
		distances.resize(observers.size());
		proj_observations.resize(observers.size());

		FloatType age_factor = std::pow<FloatType>(fading, now-last_time);
		FloatType observations_sum = 0;
		int i = 0;
		for (auto it = observers.begin(); it != observers.end(); it++, i++) {
			distances[i] = distance_function(data, it->data);
			proj_observations[i] = 0;
			FloatType factor = std::pow<FloatType>(fading, now-it->time_touched);
			observations_sum += real(it->observations[0]) * factor;
			FloatType norm_factor = 0;
			for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
				proj_observations[i] += real(it->observations[freq_ind] * conj(now_vector[freq_ind])) * factor;
			}
			if (proj_observations[i] >= active_observations_thresh) {
				updateNearestHeap(nearest, distances[i], it);
			}
			it->age = it->age * age_factor + 1;
		}
		{
			// we now have the closest active observers in nearest.
			// compute the score.
			std::size_t len = nearest.size();
			if (len < neighbor_cnt) {
				score = std::numeric_limits<FloatType>::infinity();
			}
			else {
				std::vector<FloatType> sorted_nearest(len);
				for (std::size_t j = 0; j < len; j++)
					sorted_nearest[j] = nearest[j].first;
				std::sort_heap(sorted_nearest.begin(), sorted_nearest.end());
				if (len % 2 == 0) {
					score = (sorted_nearest[len/2] + sorted_nearest[len/2 + 1]) / 2;
				}
				else {
					score = sorted_nearest[len/2];
				}
			}
		}
		i = 0;
		for (auto it = observers.begin(); it != observers.end(); it++, i++) {
			if (proj_observations[i] < active_observations_thresh)
				updateNearestHeap(nearest, distances[i], it);
		}
		FloatType observations_nearest_sum = 0;
		for (auto& observed : nearest) {
			MapIterator it = observed.second;
			auto node = observers.extract(it);
			Observer& observer = node.value();
			FloatType factor = std::pow<FloatType>(fading, now-observer.time_touched);
			for (std::size_t freq_ind = 0; freq_ind < freq_bins; freq_ind++) {
				observer.observations[freq_ind] *= factor;
				observer.observations[freq_ind] += now_vector[freq_ind];
			}
			observations_nearest_sum += real(observer.observations[0]);
			observations_sum += 1;
			observer.time_touched = now;
			observers.insert(std::move(node));
		}

		bool add_as_observer = 
			observers.empty() ||
			(rng() - rng.min()) * observations_sum * (last_index - last_added_index) < sampling_prefactor * (rng.max() - rng.min()) * observations_nearest_sum * (now - last_added_time) ;

		if (add_as_observer) {
			MapIterator worst_observer = observers.begin();
			for (auto it = observers.begin(); it != observers.end(); it++) {
				if (observer_av_compare(now, *worst_observer, *it))
					worst_observer = it;
			}
			if (observers.size() < observer_cnt) {
				observers.insert({data, now_vector, now, now, last_index});
			}
			else {
				auto node = observers.extract(worst_observer);
				Observer& observer = node.value();
				observer.data = data;
				observer.observations = now_vector;
				observer.time_touched = now;
				observer.age = 1;
				observer.index = last_index;
				observers.insert(std::move(node));
			}
			last_added_index = last_index;
			last_added_time = now;
		}
		last_index++;
		last_time = now;
		return score;
	}



  public:
	tpSDOs(std::size_t observer_cnt, FloatType T, FloatType idle_observers, std::size_t neighbor_cnt, std::size_t freq_bins, FloatType max_freq, DistanceFunction distance_function=Vector<FloatType>::euclidean, int seed=0) :
	  observer_cnt(observer_cnt), 
	  active_observers(1-idle_observers), 
	  sampling_prefactor(observer_cnt * observer_cnt / neighbor_cnt / T),
	  fading(std::exp(-1/T)),
	  neighbor_cnt(neighbor_cnt),
	  freq_bins(freq_bins),
	  max_freq(max_freq),
	  last_index(0),
	  last_added_index(0),
	  distance_function(distance_function),
	  rng(seed),
	  observer_compare(fading),
	  observer_av_compare(fading),
	  observers(observer_compare)
	{ }

	void fit(const Vector<FloatType>& data, FloatType now) {
		fitPredict_impl(data, now, true);
	}

	FloatType fitPredict(const Vector<FloatType>& data, FloatType now) {
		return fitPredict_impl(data, now, false);
	}

	int observerCount() { return observers.size(); }
	
	bool lastWasSampled() { return last_added_index == last_index - 1; }

	FloatType getActiveObservationsThreshold() {
		if (observers.size() > 1) {
			int active_threshold = ceil(active_observers * (observers.size()-1));
			return real(std::next(observers.begin(), active_threshold)->observations[0]);
		} 
		else {
			return 0;
		}
	}
	


	class ObserverView{
		tpSDOs *sdo;
		MapIterator it;
	public:
		ObserverView(tpSDOs *sdo, MapIterator it) :
			sdo(sdo),
			it(it)
		{ }
		Vector<FloatType> getData() { return it->data; }
		std::vector<std::complex<FloatType>> getObservations(FloatType now) {
			FloatType factor = std::pow<FloatType>(sdo->fading, now-it->time_touched);
			std::vector<std::complex<FloatType>> now_vector;
			sdo->initNowVector(now, now_vector);
			std::vector<std::complex<FloatType>> observations_ft;
			observations_ft.resize(sdo->freq_bins);
			for (std::size_t freq_ind = 0; freq_ind < sdo->freq_bins; freq_ind++) {
				observations_ft[freq_ind] = it->observations[freq_ind] * factor * conj(now_vector[freq_ind]);
			}
			return observations_ft;
		}
		FloatType getAvObservations(FloatType now) {
			return real(it->observations[0]) * std::pow(sdo->fading, now - it->time_touched) / it->age;
			// return (1-sdo->fading) * real(it->observations[0]) * std::pow(sdo->fading, now - it->time_touched) /
			// 	(1-std::pow(sdo->fading, now - it->time_added));
		}
	};
	


	class iterator : public MapIterator {
		tpSDOs *sdo;
	  public:
		ObserverView operator*() { return ObserverView(sdo, MapIterator(*this)); };
		iterator() {}
		iterator(tpSDOs *sdo, MapIterator it) : 
			MapIterator(it),
			sdo(sdo)
		{ }
	};
	
	iterator begin() { return iterator(this, observers.begin()); }
	iterator end() { return iterator(this, observers.end()); }

};

#endif

