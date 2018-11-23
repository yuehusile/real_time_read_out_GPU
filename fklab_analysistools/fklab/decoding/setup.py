#!/usr/bin/env python

#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

from Cython.Build import cythonize

import numpy

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("mixed_kde_cython",
#                             sources=["mixed_kde_cython.pyx","mixed_kde_c.c"],
#                             include_dirs=[numpy.get_include()])],
#)

def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('decoding', parent_package, top_path)
    
    config.add_extension( 'mixed_kde_cython',
                          sources = ['mixed_kde_cython.pyx','mixed_kde_c.c'],
                          include_dirs = [numpy.get_include()] )
    
    config.ext_modules[-1] = cythonize( config.ext_modules[-1] )[0]
    
    return config
    

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
