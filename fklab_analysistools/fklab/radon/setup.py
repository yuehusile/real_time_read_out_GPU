class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user
    
    def __iter__(self):
        for k in str(self):
            yield k
    
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)
    
    


def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('radon', parent_package, top_path)
    
    config.add_extension( 'radonc',
                          sources = ['src/pybind_radon.cpp', 'src/radon.cpp'],
                          libraries = [],
                          include_dirs = [get_pybind_include(), get_pybind_include(user=True)],
                          language = "c++",
                          extra_compile_args = ['-std=c++11', '-Ofast'])
    
    return config
    

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
