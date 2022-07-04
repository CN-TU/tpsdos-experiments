# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _tpSDOs
else:
    import _tpSDOs

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class Distance32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

# Register Distance32 in _tpSDOs:
_tpSDOs.Distance32_swigregister(Distance32)

class Distance64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr

# Register Distance64 in _tpSDOs:
_tpSDOs.Distance64_swigregister(Distance64)

class EuclideanDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.EuclideanDist32_swiginit(self, _tpSDOs.new_EuclideanDist32())
    __swig_destroy__ = _tpSDOs.delete_EuclideanDist32

# Register EuclideanDist32 in _tpSDOs:
_tpSDOs.EuclideanDist32_swigregister(EuclideanDist32)

class EuclideanDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.EuclideanDist64_swiginit(self, _tpSDOs.new_EuclideanDist64())
    __swig_destroy__ = _tpSDOs.delete_EuclideanDist64

# Register EuclideanDist64 in _tpSDOs:
_tpSDOs.EuclideanDist64_swigregister(EuclideanDist64)

class ManhattanDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.ManhattanDist32_swiginit(self, _tpSDOs.new_ManhattanDist32())
    __swig_destroy__ = _tpSDOs.delete_ManhattanDist32

# Register ManhattanDist32 in _tpSDOs:
_tpSDOs.ManhattanDist32_swigregister(ManhattanDist32)

class ManhattanDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.ManhattanDist64_swiginit(self, _tpSDOs.new_ManhattanDist64())
    __swig_destroy__ = _tpSDOs.delete_ManhattanDist64

# Register ManhattanDist64 in _tpSDOs:
_tpSDOs.ManhattanDist64_swigregister(ManhattanDist64)

class ChebyshevDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.ChebyshevDist32_swiginit(self, _tpSDOs.new_ChebyshevDist32())
    __swig_destroy__ = _tpSDOs.delete_ChebyshevDist32

# Register ChebyshevDist32 in _tpSDOs:
_tpSDOs.ChebyshevDist32_swigregister(ChebyshevDist32)

class ChebyshevDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _tpSDOs.ChebyshevDist64_swiginit(self, _tpSDOs.new_ChebyshevDist64())
    __swig_destroy__ = _tpSDOs.delete_ChebyshevDist64

# Register ChebyshevDist64 in _tpSDOs:
_tpSDOs.ChebyshevDist64_swigregister(ChebyshevDist64)

class TruncEuclideanDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, len):
        _tpSDOs.TruncEuclideanDist32_swiginit(self, _tpSDOs.new_TruncEuclideanDist32(len))
    __swig_destroy__ = _tpSDOs.delete_TruncEuclideanDist32

# Register TruncEuclideanDist32 in _tpSDOs:
_tpSDOs.TruncEuclideanDist32_swigregister(TruncEuclideanDist32)

class TruncEuclideanDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, len):
        _tpSDOs.TruncEuclideanDist64_swiginit(self, _tpSDOs.new_TruncEuclideanDist64(len))
    __swig_destroy__ = _tpSDOs.delete_TruncEuclideanDist64

# Register TruncEuclideanDist64 in _tpSDOs:
_tpSDOs.TruncEuclideanDist64_swigregister(TruncEuclideanDist64)

class MinkowskiDist32(Distance32):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, p):
        _tpSDOs.MinkowskiDist32_swiginit(self, _tpSDOs.new_MinkowskiDist32(p))
    __swig_destroy__ = _tpSDOs.delete_MinkowskiDist32

# Register MinkowskiDist32 in _tpSDOs:
_tpSDOs.MinkowskiDist32_swigregister(MinkowskiDist32)

class MinkowskiDist64(Distance64):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, p):
        _tpSDOs.MinkowskiDist64_swiginit(self, _tpSDOs.new_MinkowskiDist64(p))
    __swig_destroy__ = _tpSDOs.delete_MinkowskiDist64

# Register MinkowskiDist64 in _tpSDOs:
_tpSDOs.MinkowskiDist64_swigregister(MinkowskiDist64)

class tpSDOs32(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed, n_estimators):
        _tpSDOs.tpSDOs32_swiginit(self, _tpSDOs.new_tpSDOs32(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed, n_estimators))

    def fit(self, data, times):
        return _tpSDOs.tpSDOs32_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _tpSDOs.tpSDOs32_fit_predict(self, data, scores, times)

    def fit_predict_with_sampling(self, data, scores, times, sampled):
        return _tpSDOs.tpSDOs32_fit_predict_with_sampling(self, data, scores, times, sampled)

    def observer_count(self):
        return _tpSDOs.tpSDOs32_observer_count(self)

    def frequency_bin_count(self):
        return _tpSDOs.tpSDOs32_frequency_bin_count(self)

    def get_observers(self, data, observations, av_observations, time):
        return _tpSDOs.tpSDOs32_get_observers(self, data, observations, av_observations, time)

    def active_observations_threshold(self):
        return _tpSDOs.tpSDOs32_active_observations_threshold(self)
    __swig_destroy__ = _tpSDOs.delete_tpSDOs32

# Register tpSDOs32 in _tpSDOs:
_tpSDOs.tpSDOs32_swigregister(tpSDOs32)

class tpSDOs64(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed, n_estimators):
        _tpSDOs.tpSDOs64_swiginit(self, _tpSDOs.new_tpSDOs64(observer_cnt, T, idle_observers, neighbour_cnt, freq_bins, max_freq, distance, seed, n_estimators))

    def fit(self, data, times):
        return _tpSDOs.tpSDOs64_fit(self, data, times)

    def fit_predict(self, data, scores, times):
        return _tpSDOs.tpSDOs64_fit_predict(self, data, scores, times)

    def fit_predict_with_sampling(self, data, scores, times, sampled):
        return _tpSDOs.tpSDOs64_fit_predict_with_sampling(self, data, scores, times, sampled)

    def observer_count(self):
        return _tpSDOs.tpSDOs64_observer_count(self)

    def frequency_bin_count(self):
        return _tpSDOs.tpSDOs64_frequency_bin_count(self)

    def get_observers(self, data, observations, av_observations, time):
        return _tpSDOs.tpSDOs64_get_observers(self, data, observations, av_observations, time)

    def active_observations_threshold(self):
        return _tpSDOs.tpSDOs64_active_observations_threshold(self)
    __swig_destroy__ = _tpSDOs.delete_tpSDOs64

# Register tpSDOs64 in _tpSDOs:
_tpSDOs.tpSDOs64_swigregister(tpSDOs64)



