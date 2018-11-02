#ifndef __GPU_KDE_H__
#define __GPU_KDE_H__

#include <vector>
#include "../gmmcompression/mixture.h"
#include <iostream>
#include "kde_kernel.h"
#include <time.h>
double getElapse(timespec ts_start,timespec ts_end);
class HostMemory;

class GpuMemory {
    private:
    // parameters related to memory size/dimension
    int spikeDim;
    int maxNcomponents;
    int nGrid;
    int maxNspikes;
    int batchSize;
    int nComponents;

    // pitch size of pitched memory
    size_t gPitch;//pitch of rows with size of nGrid
    size_t cPitch;//pitch of rows with size of nComponents
    
    //disable default constructor
    GpuMemory(){};
public:
    // device memory pointers
    float* pSpike;
    float* pMean;
    float* pCovDiag;
    float* pScale;
    float* pResult;
    float* pGridAcc;
    float* pComponentAcc;

    //constructor for multiple spikes
    GpuMemory(const int spikeDim_, const int maxNcomponents_, const int nGrid_, const int maxNspikes_)
        :spikeDim(spikeDim_), maxNcomponents(maxNcomponents_), nGrid(nGrid_), maxNspikes(maxNspikes_), batchSize(0), nComponents(0), gPitch(0), cPitch(0),
         pSpike(nullptr),pMean(nullptr),pCovDiag(nullptr),pScale(nullptr),pResult(nullptr),pGridAcc(nullptr),pComponentAcc(nullptr) 
    {}

    // constructor for batch mode 
    GpuMemory(const int spikeDim_, const int maxNcomponents_, const int nGrid_, const int maxNspikes_, const int batchSize_)
        :spikeDim(spikeDim_), maxNcomponents(maxNcomponents_), nGrid(nGrid_), maxNspikes(maxNspikes_), batchSize(batchSize_), nComponents(0), gPitch(0), cPitch(0),
         pSpike(nullptr),pMean(nullptr),pCovDiag(nullptr),pScale(nullptr),pResult(nullptr),pGridAcc(nullptr),pComponentAcc(nullptr) 
    {}

    // gpu memory deallocation
    int releaseGpuMem();
    // gpu memory allocation
    int allocGpuMem();

    // get memory dimension
    int getSpikeDim() const { return spikeDim;}
    int getMaxNcomponents() const { return maxNcomponents;}
    int getNcomponents() const { return nComponents;}
    int getNgrid() const {return nGrid;}
    int getMaxNspikes() const {return maxNspikes;}
    size_t getGpitch() const {return gPitch;}
    size_t getCpitch() const {return cPitch;}
    int getBatchSize() const {return batchSize;}

    //upload model
    int uploadModelToGpu(const HostMemory &hostMem);

    //destructor
    ~GpuMemory()
    {
       if (0==releaseGpuMem())
       {
           //printf("gpu memory release succeeded\n");
       }
       else
       {
           printf("gpu memory release failded\n");
       }
    }

};

class HostMemory{
private:
    // parameters related to memory size/dimension
    int spikeDim;
    int maxNcomponents;
    int nComponents;
    int nGrid;
    int maxNspikes;
    double scale; // some times we have to scale the scaling factor into some suitable range for single precision limits before computing(*scale).
                  // when the results come out(/scale), 
    //disable default constructor
    HostMemory(){};

public:
    // host memory pointers
    float* pSpike;
    float* pMean;
    float* pCovDiag;
    float* pScale;
    float* pResult;
    float* pGridAcc;
    double* pResult_d;

   //constructor for multiple spikes
    HostMemory(const int spikeDim_, const int maxNcomponents_, const int nGrid_, const int maxNspikes_)
        :spikeDim(spikeDim_), maxNcomponents(maxNcomponents_), nComponents(0), nGrid(nGrid_), maxNspikes(maxNspikes_), scale(1),
         pSpike(nullptr),pMean(nullptr),pCovDiag(nullptr),pScale(nullptr),pResult(nullptr),pGridAcc(nullptr),pResult_d(nullptr) 
    {}

    // constructor for single spike 
    HostMemory(const int spikeDim_, const int maxNcomponents_, const int nGrid_)
        :spikeDim(spikeDim_), maxNcomponents(maxNcomponents_), nComponents(0), nGrid(nGrid_), maxNspikes(1), scale(1),
         pSpike(nullptr),pMean(nullptr),pCovDiag(nullptr),pScale(nullptr),pResult(nullptr),pGridAcc(nullptr),pResult_d(nullptr) 
    {}

    // gpu memory deallocation
    int releaseHostMem();
    // gpu memory allocation
    int allocHostMem();

    // get memory dimension
    int getSpikeDim() const { return spikeDim;}
    int getMaxNcomponents() const { return maxNcomponents;}
    int getNcomponents() const { return nComponents;}
    int getNgrid() const {return nGrid;}
    int getMaxNspikes() const {return maxNspikes;}
    double getScale() const {return scale;}

    int fillHostModelMem(Mixture* m, double* grid_acc, uint16_t* pointdim);
    int fillHostSpikeMem(double* spikes);
    int setDoubleResult(int nSpikes);
 
    //destructor
    ~HostMemory()
    {
       if (0==releaseHostMem())
       {
           //printf("host memory release succeeded\n");
       }
       else
       {
           printf("host memory release failded\n");
       }
    }

};

int uploadModelToGpu(Mixture* m, double* grid_acc, uint16_t* pointdim, HostMemory *hostMem, GpuMemory *gpuMem);

const int default_bs = 64;

class GpuDecoder{
private:
    int n_tt_;// number of tetrodes(models)
    int maxBatchSize_;// maximum number of spikes been decoded together in batch mode
                      // the pre-allocated memory size also been decided based
		      // on this value
    int batchSize_;// current batchSize, can't be larger than maxBatchSize
    int gridDim_;// dimension of grid
    int n_spikes;// total number of collected spikes
    
    int bs_ca;// component_acc kernel block size = (bs_ca, n_spikedim)
    int bs_ev;// evaluation kernel block size = (bs_ev, 4)
    int maxSmem;// maximum size of shared memory can be dynamically allocated
                // full share memory= 48KB, available 29904bytes, so max components=29904/(2*size(half))=9976 
    std::vector<HostMemory*> hostMems;
    std::vector<GpuMemory*> gpuMems;

    std::vector<cudaStream_t> stream;
    GpuDecoder();//disable default constructor, must set a grid size

    // private methods for calling cuda apis & kernels
    int uploadKdeSpikes(const int tt_idx, double* spikes, const int n_spikes);
    int downloadKdeResults(const int tt_idx, const int n_spikes);
    int launchKde(const int tt_idx, const int n_spikes);
public:
    GpuDecoder(int gridDim, int maxBatchSize=8192):
	    n_tt_(0),maxBatchSize_(maxBatchSize),batchSize_(maxBatchSize),
	    gridDim_(gridDim),n_spikes(0),bs_ca(default_bs),bs_ev(default_bs),maxSmem(41952-default_bs*4*4),
	    hostMems(),gpuMems(),stream()
    {}

    int n_tt() const {return n_tt_;}
    int maxBatchSize() const {return maxBatchSize_;}
    int batchSize() const {return batchSize_;}
    int gridDim() const {return gridDim_;}
    int component_acc_block_size() const {return bs_ca;}
    int evaluation_block_size() const {return bs_ev;}

    void showMemRequirement(const int ncomponents, const int spikeDim) const;

    void setBatchSize(const int bs)
    {
	if (bs<1)
	{
            std::cout << "invalid batchSize="<<bs<<std::endl;
	    return;
	}
	if (bs>maxBatchSize_)
	{
	    batchSize_ = maxBatchSize_;
	}
	else
	{
	    batchSize_ = bs;
	}
	std::cout << "batchSize has been set to "<<batchSize_<<std::endl;
	return;
    }

    void set_component_acc_block_size(const int bs)
    {
	if (bs<1 || bs>1024)
	{
	    std::cout << "invalid block size = "<<bs <<std::endl;
	    return;
	}
	bs_ca = bs;
    }

    void set_evaluation_block_size(const int bs)
    {
	if (bs<1 || bs >1024)
	{
	    std::cout << "invalid block size = "<<bs<<std::endl;
	    return;
	}
	bs_ev = bs;
	// update available shared memory
	maxSmem = 41952-bs*4*4;
    }

    // add a new tetrode/model to the decoder
    int addTT(Mixture* m, double* grid_acc, int spikeDim, uint16_t* pointDim);

    // decode offline/online data
    // -------------------------------------------------------------------------
    // decode spikes from single tetrode(may be called by different host threads)
    // spikes will be grouped to process, the group size is min(n_spikes,batchSize)
    int decodeTT(const int tt_idx, double* spikes, const int n_spikes, double* results);
    // decode all spikes from all tetrodes(called by single host thread)
    // to be done:
    // tetrodes are paralelled on different STREAMS
    // void decodeAll(double* spikes, int* n_spikes)
    // get result
    double result(const int tt_idx, const int spike_idx, const int grid_idx)
    {
        return hostMems[tt_idx]->pResult_d[spike_idx*gridDim_ + grid_idx];
    }
    void clear()
    {
        while(gpuMems.size()>0)
	{
	    if (gpuMems.back()!=nullptr)
	        delete gpuMems.back();
   	    gpuMems.pop_back();
	}
        while(hostMems.size()>0)
	{
	    if (hostMems.back()!=nullptr)
	        delete hostMems.back();
   	    hostMems.pop_back();
	}
    }
    ~GpuDecoder()
    {
        clear();
    }
};

// the GPU memory buffer used for significance assessment
class SignificanceAnalyzer{
private:
    int n_shuffle_;     // number of shuffle samples
    int n_time_bin_;    // number of time bins need for assessment
    int n_spatial_bin_; // number of spatial bins
    int max_spikes_;    // max number of spikes allowed PER SPIKE GROUP
    int n_spike_group_; // spike group means tetrodes or shanks, or any other way to group the spike channels
    int n_gr_;          // the pax block size
    float bin_size_;    // size of time bin (s)
    float dmax_;
    float max_prob_;

    int n_pax_;         // number of paxes in current pax buffer
    int shf_idx_size_;  // size of shuffle indexes,eg. how many paxes to be shuffled.
    size_t g_pitch_;    // for aligned memory allocations (spatial bin)
    size_t s_pitch_;    // for aligned memory allocations (shuffle idx)

    int head;           // the time bins are circular buffered, this is the index of NEXT available pax address to renew
    int bin_head;       // the next bin buffer index to fill in
    bool full_;         // whether the circular buffer full or not
    // gpu memory pointers
    float* pax_;        // [g_pitch, shf_idx_size*2] p(a,x) of all the spikes
    float* offset_;     // [g_pitch, n_spike_group*max_spikes] p(a,x) of all the spikes
    int* cum_n_spikes_; // [n_time_bin+1] the cumulative number of spikes in each time bin 
    float* prob_;       // [g_pitch,n_time_bin*n_shuffle] probablities, the output
    // shuffle indexes are generated from the beginning
    int* shf_idx_;      // [shf_idx_size, n_shuffle] the shuffled pax index
    int* shift_idx_;    // [max_spikes*n_spike_group*n_time_bin] the memory address shift of each spike_idx

    float* mo_;         // [g_pitch] offset to be minused by the sum of p(a,x)
    float* pix_;        // [n_spatial_bin] pi(x)
    float* lx_;         // [n_spatial_bin] lambda(x)
    float* pix_cpu_;
    float* lx_cpu_;
    float* pax_cpu_;
    float* offset_cpu_;
    int* n_spikes_cpu_;
    int* cum_n_spikes_cpu_;
    int* shift_idx_cpu_;
    int* shf_idx_cpu_;
    float* prob_cpu_;
    float* dsmat_;
    float* dsmat_cpu_;
    float* rwd_;
    float* rwd_cpu_;
    // memory allocator
    int allocateMem();
    // memory de-allocator
    int releaseMem();
    // launch kernel
    int launchKernel();
    //disable default constructor
    SignificanceAnalyzer(){}
public:

    // constructor
    SignificanceAnalyzer(const int n_pos, const int n_group, const float bin_size, const float dmax,
	const int n_shf = 1000, const int n_tbin = 10, const int max_sp = 100, const  int shf_id_sz = 5000):
	//const int n_shf = 1000, const int n_tbin = 10, const int max_sp = 100, const  int shf_id_sz = 10):
      n_shuffle_(n_shf),n_time_bin_(n_tbin),n_spatial_bin_(n_pos),max_spikes_(max_sp),n_spike_group_(n_group),n_gr_(n_group*n_tbin),
      bin_size_(bin_size), dmax_(dmax), max_prob_(0), shf_idx_size_(shf_id_sz), n_pax_(0),
      g_pitch_(0), s_pitch_(0), head(0), bin_head(0),full_(false),
      pax_(nullptr),offset_(nullptr),cum_n_spikes_(nullptr),prob_(nullptr),shf_idx_(nullptr),pix_(nullptr),lx_(nullptr),shift_idx_(nullptr),
      pix_cpu_(nullptr),lx_cpu_(nullptr),pax_cpu_(nullptr),offset_cpu_(nullptr),n_spikes_cpu_(nullptr),cum_n_spikes_cpu_(nullptr),shift_idx_cpu_(nullptr),shf_idx_cpu_(nullptr),prob_cpu_(nullptr),dsmat_(nullptr),dsmat_cpu_(nullptr), rwd_(nullptr), rwd_cpu_(nullptr)
    {
        allocateMem();
	//printf("g_pitch=%d",(int)g_pitch_);
	//uploadModelParam();
	//printf("upload done=%d",(int)g_pitch_);
	//for (int i =0;i<=n_time_bin_;i++)
	//{
	    //updatePax();
	//}
	//computeLL();
    }

    // destructor
    ~SignificanceAnalyzer()
    {
        releaseMem();
    }

    void clear()
    {
        releaseMem();
    }
    
    // copy pix/lx to GPU
    void uploadParam(double* pix, double* lx, double* dsmat);
    // add decoding results of new bin into buffer
    //void updateBin(double* pax, int* n_spikes_g, const int n_spikes, const int n_group);
    void updateBin(double* pax, int* n_spikes_g, double* mu, const int n_spikes, const int n_group);
    // generate shuffle idx
    //void updateShuffle(); 
    bool getProb(double* prob); 
    int updatePax()
    {
        // now it's a fake data generator
	// fill pax
	printf("head=%d\n",head);
	launchFakeDataKernelpax(pax_, 35, g_pitch_, n_spatial_bin_, head, 1);
        launchFakeDataKernelnspike(cum_n_spikes_, n_spike_group_*n_time_bin_, 10);
        // init shuffle
        launchFakeDataKernelshf(shf_idx_, s_pitch_, 35*n_spike_group_*n_time_bin_, n_shuffle_);
	head = (head+1)%n_time_bin_;
	if (full_==false && head == n_time_bin_-1)
	{
	    full_ = true;
	    printf("full now\n");
	}
	return 0;
    }

    int computeLL()
    {
        struct timespec t1;
        struct timespec t2;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        launch_ll_kernel(pax_,mo_,prob_,shf_idx_,cum_n_spikes_,g_pitch_,n_spatial_bin_,n_time_bin_,max_spikes_,n_shuffle_);
        clock_gettime(CLOCK_MONOTONIC, &t2);
	double total = getElapse(t1,t2);
	printf("total=%funit=ms\n",total);
    }

    //int normalize();

    void computeRwd(int n_assess_bin);
    bool getRwd(double* rwd);

    // buffer access methods
    float* pax(const int spike_idx);// return the buffer pitch pointer based on spike index(inside shuffle access)
    float* pax(const int bin, const int gr);// return the buffer block pointer based on time bin/spike group index(update access)
    float* prob(const int bin, const int shf); //return the buffer pitch pointer based on bin and shuffle index
    float* shf_idx(const int shf); // return the shuffle index buffer of all spikes based on the shuffle index

    // check private values
    int n_shuffle() const {return n_shuffle_;}
    int n_time_bin() const {return n_time_bin_;}
    int n_spatial_bin() const {return n_spatial_bin_;}
    int max_spikes() const {return max_spikes_;}
    int n_spike_group() const {return n_spike_group_;}
    //int n_spikes(const int bin, const int gr) const {return n_spikes_[bin*n_spike_group_ + gr];}
    // to do: gpu functions for getting/setting the correct pax/probabilities
};
#endif
