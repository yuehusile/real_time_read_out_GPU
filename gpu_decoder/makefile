cc = g++ -O2
ncc = nvcc -O2 -Xptxas="-v" -default-stream per-thread -lineinfo

slib = libgpu_kde.so
deps = ../compressed_decoder/gmmcompression/mixture.h kde_kernel.h gpu_kde.hpp
obj = kde_kernel.o
libs = -lgsl -lgslcblas -lm -lcuda -lcudart -lstdc++ 
# change cuda lib/include path based on your system
flags = -L/usr/local/cuda-8.0/lib64 -I/usr/local/cuda-8.0/include 

$(slib): $(obj)
	 $(cc) -shared -fPIC -o $(slib) -Wl,-no-undefined $(obj) $(libs) $(flags) 

kde_kernel.o: kde_kernel.cu $(deps)
	$(ncc) $(flags) $(libs) -Xcompiler '-fPIC' -c kde_kernel.cu

clean:
	rm -rf $(obj) libgpu_kde.so

