#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize( Extension("fkmixture",
                     sources=["fkmixture.pyx","covariance.cpp","component.cpp","mixture.cpp",\
                              "../gpu_decoder/gpu_kde.cpp"],
                     extra_compile_args=["-std=c++11"],
                     libraries=['gsl', 'gslcblas', 'm',\
                                'gpu_kde', 'cudart' ],
                     library_dirs=["../gpu_decoder", '/usr/local/cuda-8.0/lib64', '/usr/local/cuda-8.0'],
                     include_dirs=[numpy.get_include(), "../gpu_decoder","/usr/local/cuda-8.0/targets/x86_64-linux/include"],
                     language="c++"))

)
