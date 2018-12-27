# What is this?
This repository contains the source code for reading out the neural ensemble places code in real-time with GPU.
We got 20-50X speedup for decoding of unsorted ensemble place codes. Online real-time (20ms bin) significance assessment (with 1000 shuffle samples) for replay events during sleep is realized with our approuch.
More details and results can be found in our Cell Reports paper 
[Real-Time Readout of Large-Scale Unsorted Neural Ensemble Place Codes](https://www.sciencedirect.com/science/article/pii/S2211124718317960)

# files:

```
gpu_decode_demo.py         - The python script for the demostration of GPU decoding speed up

online_assess_demo.py      - The python script for the demostration of online decoding and assessment

kde_gpu.py                 - KDE decoding related python classes

util.py                    - tool functions for this demo code

config_file.py             - configuration options

files under gpu_decoder    - CUDA codes and C++ wrappers for decoding with GPU

files under gmmcompression - Cython and C codes for decoding

files under data           - data for testing

files under mixture        - preprocessed mixtures(trained model)

files under fklab_analysistools - fklab python toolbox, for data preprocessing
```

The codes under `gmmcompression,mixture,fklab_analysistools` are borrowed from
[Real Time Replay](https://bitbucket.org/kloostermannerflab/ciliberti_elife2018_realtimereplay)

`kde_gpu.py` is modified based on the orignal version in the repo above to adapt to GPU computation

See more about our rearch on our website:
[Computational Neuroscience, Neuroengineering & Neuropsychiatry Laboratory](http://www.cn3lab.org/home.html)

# Step-by-step setup instructions:

This code has been tested with the following setup:

Operating system: `Ubuntu 16.04`; 
Python version: `Python2.7.2`; 
CUDA version: `8.0`

1. some packages maybe required to use this toolbox:
scipy, numba, pyyaml, h5py, natsort, libgsl
They can be installed by pip:

```
pip install scipy, numba, pyyaml, h5py, natsort
```

remember add the path to this toolbox into PYTHONPATH.
add the following line into ~/.bashrc or ~/.profile, replace path/to/dataanalysispython with your path:
```
export PYTHONPATH=$PYTHONPATH:"path/to/dataanalysispython"
```
2. Intall CUDA toolkit if not installed yet.
The download link and instrctions can be found here: 
https://developer.nvidia.com/cuda-downloads
Our code was tested with cuda-8.0 version of the toolkit, this or a newer version is recommanded.

3. add the path to real_time_read_out_GPU/gpu_decoders folder into LD_LIBRARY_PATH.
add the following line into ~/.bashrc or ~/.profile, replace path/to/gpu_decoder with your path:

```
export LD_LIBRARY_PATH=path/to/gpu_decoder:$LD_LIBRARY_PATH
```

4. pre-trained models and testing data: https://drive.google.com/open?id=11tdFXoXyFvqeo_dxkpWttTMjnO-X8je4
download data to the repository folder, and extract the data files in data.zip:

```
unzip data.zip
```

5. run `gpu_decode_demo.py` and `online_assess_demo.py`, it takes from 30 seconds to a few minutes to run the decoding process, depending on the hardware. 

```
python gpu_decode_demo.py
python online_assess_demo.py
```

# Compile the binaries by yourself:

This repository is provided with the precompiled shared libraries for ubuntu 64 bit environment. 
If it doesn't work for you, you can re-compile the shared libraries by yourself.

1. install libgsl.
For ubuntu users, type this command in the terminal:

```
sudo apt-get install libgsl2
sudo apt-get install libgsl0-dev
```

For other linux disributions, please refer to the instrutions here:
https://www.gnu.org/software/gsl/

2. compile GPU code.
Go to gpu_decoders folder, edit makefile for cuda path if necessary, and then type make to compile in the terminal:

```
cd gpu_decoders
make
```

If succeeded, a libgpu_kde.so shared library will be found in this folder

3. compile Cython interface.

Install Cython if not been installed yet:

```
pip install cython
```

Go to gmmcompression folder and compile Cython code:

```
cd gmmcompression
python setup.py install --prefix=.
cp lib/python2.7/site-packages/fkmixture.so .
```

