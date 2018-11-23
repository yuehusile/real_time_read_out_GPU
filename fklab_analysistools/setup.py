#!/usr/bin/env python

#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Build import cythonize

#import numpy

#extensions = [
  #Extension("fklab.gmmcompression.fkmixture", sources=["fklab/gmmcompression/fkmixture.pyx","fklab/gmmcompression/covariance_c.c","fklab/gmmcompression/component_c.c","fklab/gmmcompression/mixture_c.c"],
            #libraries=['gsl', 'gslcblas', 'm'], include_dirs=[numpy.get_include()] )
#]

#setup(name='fklab',
      #version='1.0',
      #description='Kloosterman Lab Data Analysis Tools',
      #author='Fabian Kloosterman',
      #author_email='fabian.kloosterman@nerf.be',
      #url='http://kloostermanlab.org',
      #packages=['fklab', 'fklab.utilities'],
      ##cmdclass = {'build_ext': build_ext},
      ##ext_package = 'fklab', 
      #ext_modules = cythonize(extensions),
     #)

import os

DISTNAME = 'fklab'
DESCRIPTION = 'Kloosterman Lab Data Analysis Tools'
LONG_DESCRIPTION = '' # read from readme??
MAINTAINER = 'Fabian Kloosterman'
MAINTAINER_EMAIL = 'fabian.kloosterman@nerf.be'
URL = 'http://kloostermanlab.org'
LICENSE = '?'

VERSION = '0.1.0' # get from fklab.__version__ ??

import setuptools

extra_setuptools_args = dict(
    zip_safe=False,
    include_package_data=True,
)

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
    
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    
    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ..."
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.add_subpackage('fklab')
    
    return config

def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    install_requires=['pybind11>=1.9', 'numpy', 'scipy', 'enum'],
                    **extra_setuptools_args)
    
    from numpy.distutils.core import setup
    
    metadata['configuration'] = configuration
    
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
