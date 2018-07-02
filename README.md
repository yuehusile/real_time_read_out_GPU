# real_time_read_out_GPU

Step-by-step setup instructions:

Operation system: Linux, has been tested with Ubuntu 16.04
Python version: Python2.7

1. Setup FKLab toolbox:
'FKLab Data Analysis in Python' toolbox is required to run this code.
The toolbox code and setup instructions can be found here:
https://bitbucket.org/kloostermannerflab/dataanalysispython/src/master/
some packages maybe required to use this toolbox:
scipy, numba, pyyaml, h5py, natsort, libgsl
They can be installed by pip:

pip install scipy, numba, pyyaml, h5py, natsort

remember add the path to this toolbox into PYTHONPATH.
add the following line into ~/.bashrc or ~/.profile, replace path/to/dataanalysispython with your path:

PYTHONPATH=$PYTHONPATH:"path/to/dataanalysispython"

2. Intall CUDA toolkit if not installed yet.
The download link and instrctions can be found here: 
https://developer.nvidia.com/cuda-downloads
Our code was tested with cuda-8.0 version of the toolkit, this or a newer version is recommanded.

3. add the path to real_time_read_out_GPU/gpu_decoders folder into LD_LIBRARY_PATH.
add the following line into ~/.bashrc or ~/.profile, replace path/to/gpu_decoder with your path:

export LD_LIBRARY_PATH=path/to/gpu_decoder:$LD_LIBRARY_PATH

4. run demo.py, it will output a few results on the screen.

Python demo.py


Compile the binaries by yourself:
This repository is provided with the precompiled shared libraries for ubuntu 64 bit environment. 
If it doesn't work for you, you can re-compile the shared libraries by yourself.

1. install libgsl.
For ubuntu users, type this command in the terminal:
sudo apt-get install libgsl2
sudo apt-get install libgsl0-dev
For other linux disributions, please refer to the instrutions here:
https://www.gnu.org/software/gsl/

2. compile GPU code.
Go to gpu_decoders folder, edit makefile for cuda path if necessary, and then type make to compile in the terminal:

cd gpu_decoders
make

If succeeded, a libgpu_kde.so shared library will be found in this folder

3. compile cython interface.
Install Cython if not been installed yet:

pip install cython

compile Cython code:

python setup.py install --prefix=.
cp lib/python2.7/site-packages/fkmixture.so .





