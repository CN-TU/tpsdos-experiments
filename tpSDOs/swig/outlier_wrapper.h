// Released under the GNU Lesser General Public License version 3,
// see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

#ifndef DSALMON_OUTLIER_WRAPPER_H
#define DSALMON_OUTLIER_WRAPPER_H

#include <complex>

#include "tpSDOs.h"
#include "array_types.h"
#include "distance_wrappers.h"

template<typename FloatType>
class tpSDOs_wrapper {
	std::vector<tpSDOs<FloatType>> estimators;
	int dimension;
	int n_estimators;
	std::size_t freq_bins;
	//tpSDOs<FloatType> sdo;
	
  public:
	tpSDOs_wrapper(int observer_cnt, FloatType T, FloatType idle_observers, int neighbour_cnt, int freq_bins, FloatType max_freq, Distance_wrapper<FloatType>* distance, int seed, int n_estimators);
	void fit(const NumpyArray2<FloatType> data, const NumpyArray1<FloatType> times);
	void fit_predict(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times);
	void fit_predict_with_sampling(const NumpyArray2<FloatType> data, NumpyArray1<FloatType> scores, const NumpyArray1<FloatType> times, NumpyArray1<int> sampled);
	int observer_count();
	int frequency_bin_count();
	void get_observers(NumpyArray2<FloatType> data, NumpyArray2<std::complex<FloatType>> observations, NumpyArray1<FloatType> av_observations, FloatType time);
	FloatType active_observations_threshold();
};
DEFINE_FLOATINSTANTIATIONS(tpSDOs)

#endif
