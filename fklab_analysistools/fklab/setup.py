
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('fklab', parent_package, top_path)
    
    # packages that do not have their own setup.py
    config.add_subpackage('utilities')
    config.add_subpackage('segments')
    config.add_subpackage('events')
    config.add_subpackage('signals')
    config.add_subpackage('statistics')
    config.add_subpackage('io')
    config.add_subpackage('plot')
    config.add_subpackage('geometry')
    config.add_subpackage('behavior')
    
    # packages that supply their own setup.py
    config.add_subpackage('radon')
    config.add_subpackage('decode')
    config.add_subpackage('decoding')
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
