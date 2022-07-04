#!/usr/bin/env python3

# Released under the GNU Lesser General Public License version 3,
# see accompanying file LICENSE or <https://www.gnu.org/licenses/>.

import glob
from distutils.core import setup, Extension

import os

try:
	import numpy
except:
	raise ImportError('Numpy is required for building this package.', name='numpy')

numpy_path = os.path.dirname(numpy.__file__)
numpy_include = numpy_path + '/core/include'

CPP_SOURCES = [
	'swig/outlier_wrapper.cpp',
	'swig/tpSDOs_wrap.cxx'
]

dSalmon_cpp = Extension(
	'tpSDOs.swig._tpSDOs',
	CPP_SOURCES,
	include_dirs=['cpp', numpy_include, 'contrib/boost'],
        extra_compile_args=['-g0'] # Strip .so file to an acceptable size
)

setup(
	name='tpSDOs',
	version='0.1',
	url='none',
	packages=['tpSDOs', 'tpSDOs.swig'],
	package_dir={'tpSDOs': 'python', 'tpSDOs.swig': 'swig'},
	ext_modules = [ dSalmon_cpp ],
	install_requires=['numpy']
)
