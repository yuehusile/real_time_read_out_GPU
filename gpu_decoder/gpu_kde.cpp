#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <numeric>
#include <iostream>


#if defined __cplusplus
extern "C" {
#endif
#include "../gmmcompression/mixture.h"
#if defined __cplusplus
}
#endif

//#include "kde_kernel.h"

#include "pthread.h"

#include "gpu_kde.hpp"

#include <cstdlib>

//typedef cudaStream_t cudaStream_t;
//****************************
//--- methods of GpuMemory---
//****************************
int GpuMemory::releaseGpuMem()
{
    if (pSpike != nullptr)
        cudaFree(pSpike);
    if (pMean != nullptr)
        cudaFree(pMean);
    if (pCovDiag != nullptr)
        cudaFree(pCovDiag);
    if (pScale != nullptr)
        cudaFree(pScale);
    if (pResult != nullptr)
        cudaFree(pResult);
    if (pGridAcc != nullptr)
        cudaFree(pGridAcc);
    if (pComponentAcc != nullptr)
        cudaFree(pComponentAcc);
    return 0;
}

// gpu memory allocation
int GpuMemory::allocGpuMem()
{
    //printf("alloc gpu mem\n");
    bool fail = false;
    cudaError_t cuda_result_code;
    // memory for spike(s)
    if (spikeDim>0)
    {
        cudaMalloc(&pSpike, spikeDim*sizeof(float)*maxNspikes);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message Malloc spikes_d: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid spikeDim=%d\n", spikeDim);
        fail = true;
    }
    //printf("alloc gpu mem spike done\n");
    // memory for mean of components 
    if (!fail && maxNcomponents>0)
    {
        cudaMallocPitch(&pMean, &cPitch, maxNcomponents*sizeof(float), spikeDim);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message mean_d mallocPitch: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    //printf("alloc gpu mem mean done\n");
    // memory for diagnal covariance of components 
    if (!fail && maxNcomponents>0)
    {
        cudaMallocPitch(&pCovDiag, &cPitch, maxNcomponents*sizeof(float), spikeDim);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message cov_diag_d mallocPitch: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    //printf("alloc gpu mem cov done\n");
    // memory for scale of components 
    if (!fail && maxNcomponents>0)
    {
        cudaMalloc(&pScale, maxNcomponents*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message scael_d malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    //printf("alloc gpu mem scale done\n");
    // memory for pre computed grid_acc  
    if (!fail && maxNcomponents>0 && nGrid>0)
    {
        cudaMallocPitch(&pGridAcc, &gPitch, nGrid*sizeof(float), maxNcomponents);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message grid_acc_d mallocPitch: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d or nGrid=%d\n", maxNcomponents,nGrid);
        fail = true;
    }
    //printf("alloc gpu mem grid_acc done\n");
    // memory for accumulator of components 
    if (!fail && maxNcomponents>0 && (maxNspikes>0 || batchSize>0))
    {
        if (0==batchSize)
        {
            cudaMallocPitch(&pComponentAcc, &cPitch, maxNcomponents*sizeof(float), maxNspikes);
        }
        else
        {
            cudaMallocPitch(&pComponentAcc, &cPitch, maxNcomponents*sizeof(float), batchSize);
        }
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message component_acc_d malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d or nSpikes=%d\n", maxNcomponents,maxNspikes);
        fail = true;
    }
    //printf("alloc gpu mem comp_acc done\n");
    // memory for results 
    if (!fail && nGrid>0 && maxNspikes>0)
    {
        cudaMallocPitch(&pResult, &gPitch, nGrid*sizeof(float), maxNspikes);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message result_d malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid nGrid=%d or nSpikes=%d\n", nGrid,maxNspikes);
        fail = true;
    }
    //printf("alloc gpu mem results done\n");
 
    if (fail)
    {
        printf("allocation failed\n");
        releaseGpuMem();
        return -1; 
    }
    //printf("alloc gpu mem done\n");
    return 0;
}

// memory upload (synchronized memory copy)
int GpuMemory::uploadModelToGpu(const HostMemory &hostMem)
{
    cudaError_t cuda_result_code;

    bool fail = false;

    nComponents = hostMem.getNcomponents();
    //to be done: check if the dimension of gpu and cpu memory are match
   
    //printf("upload start cPitch=%d,nComponents=%d\n",cPitch,nComponents);
    //upload model data
    if (!fail)
    {
        cudaMemcpy2D(pMean, cPitch, hostMem.pMean, nComponents*sizeof(float), nComponents*sizeof(float), spikeDim, cudaMemcpyHostToDevice); 
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message mean_d memory cpy 2D: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    //printf("upload mean done");
    if (!fail)
    {
        cudaMemcpy2D(pCovDiag, cPitch, hostMem.pCovDiag, nComponents*sizeof(float), nComponents*sizeof(float), spikeDim, cudaMemcpyHostToDevice); 
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message cov_diag_d memory cpy 2D: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    //printf("upload cov done");
    if (!fail)
    {
        cudaMemcpy(pScale, hostMem.pScale, nComponents*sizeof(float), cudaMemcpyHostToDevice);   
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message cudaMemCpy scale %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    //printf("upload scale done");
    if (!fail)
    {
        cudaMemcpy2D(pGridAcc, gPitch, hostMem.pGridAcc, nGrid*sizeof(float), nGrid*sizeof(float), nComponents, cudaMemcpyHostToDevice);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message cudaMemCpy2D grid_acc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    //printf("upload grid_acc done");
 
    if (fail)
    { 
        printf("upload model failed\n");
        return -1;
    }
    return 0;
}
//****************************
//--- methods of HostMemory---
//****************************
int HostMemory::releaseHostMem()
{
    if (pSpike != nullptr)
        cudaFreeHost(pSpike);
    if (pMean != nullptr)
        cudaFreeHost(pMean);
    if (pCovDiag != nullptr)
        cudaFreeHost(pCovDiag);
    if (pScale != nullptr)
        cudaFreeHost(pScale);
    if (pResult != nullptr)
        cudaFreeHost(pResult);
    if (pGridAcc != nullptr)
        cudaFreeHost(pGridAcc);
    if (pResult_d != nullptr)
    {
        free(pResult_d);
        pResult_d = nullptr;
    }
    
    return 0;
}
// gpu memory allocation
int HostMemory::allocHostMem()
{
    //printf("alloc host mem: maxNspikes=%d,maxNcomponents=%d\n",maxNspikes,maxNcomponents);
    bool fail = false;
    cudaError_t cuda_result_code;
    // memory for spike(s)
    if (spikeDim>0)
    {
        cudaHostAlloc(&pSpike, spikeDim*sizeof(float)*maxNspikes, cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message Malloc spikes_h: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid spikeDim=%d\n", spikeDim);
        fail = true;
    }
    // memory for mean of components 
    if (!fail && maxNcomponents>0)
    {
        cudaHostAlloc(&pMean, maxNcomponents*sizeof(float)*spikeDim, cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message mean_h malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    // memory for diagnal covariance of components 
    if (!fail && maxNcomponents>0)
    {
        cudaHostAlloc(&pCovDiag, maxNcomponents*sizeof(float)*spikeDim, cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message cov_diag_h malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    // memory for scale of components 
    if (!fail && maxNcomponents>0)
    {
        cudaHostAlloc(&pScale, maxNcomponents*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message scale_h malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d\n", maxNcomponents);
        fail = true;
    }
    // memory for pre computed grid_acc  
    if (!fail && maxNcomponents>0 && nGrid>0)
    {
        cudaHostAlloc(&pGridAcc, nGrid*sizeof(float)*maxNcomponents, cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message grid_acc_h malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid maxNcomponents=%d or nGrid=%d\n", maxNcomponents,nGrid);
        fail = true;
    }
    // memory for results 
    if (!fail && nGrid>0 && maxNspikes>0)
    {
        cudaHostAlloc(&pResult, nGrid*sizeof(float)*maxNspikes, cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message result_h malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid nGrid=%d or nSpikes=%d\n", nGrid,maxNspikes);
        fail = true;
    }
    // double memory for results 
/*    if (!fail && nGrid>0 && maxNspikes>0)
    {
        pResult_d = (double*)malloc(nGrid*sizeof(double)*maxNspikes);
        if (nullptr==pResult_d) {
            printf("result_h_d malloc failed\n");
            fail = true;
        }
    }
    else if (!fail)
    {
        printf("invalid nGrid=%d or nSpikes=%d\n", nGrid,maxNspikes);
        fail = true;
    }
*/ 
    if (fail)
    {
        printf("allocation failed\n");
        releaseHostMem();
        return -1; 
    }
    return 0;
}

// copy model data into aligned, pinned memory; cast double to float
int HostMemory::fillHostModelMem(Mixture* m, double* grid_acc, uint16_t* pointdim)
{
    if (nullptr==grid_acc)
    {
        printf("fill host model failed, null grid_acc pointer\n");
        return -1;
    }

    // ---------------------------------------------------
    // check and set scale for scaling factor if necessary
    // sometimes the scale of the input double data is out of the 
    // precision limitation of float data.
    // in that case, we just scale to a suitable range, and then scale back
    // ---------------------------------------------------
    double min_scale = 1e20;
    double max_scale = 1e-20;
    for (int i=0;i<m->ncomponents;i++)
    {
        if (m->components[i]->covariance->diag_scaling_factor > max_scale)
        {
            max_scale = m->components[i]->covariance->diag_scaling_factor;
        } 
        if (m->components[i]->covariance->diag_scaling_factor < min_scale)
        {
            min_scale = m->components[i]->covariance->diag_scaling_factor;
        } 
    }
    if (max_scale<1e-5)// single precision: 9 decimal
    {
        scale = 1;
        while (max_scale<0.1)
        {
            max_scale *= 10;
            scale *= 10;
        }
    }
    if (min_scale>1e5)// single precision: 9 decimal
    {
        scale = 1;
        while (min_scale>10)
        {
            min_scale /= 10;
            scale /= 10;
        }
    }

    //printf("m->ndim=%d\n",m->ndim);
    // align model data in host memory
    for (int i=0;i<m->ncomponents;i++)
    {
        pScale[i] = (float)((m->components[i]->covariance->diag_scaling_factor*scale) * m->components[i]->weight);
        for(int j=0;j<spikeDim;j++)
        {
            pMean[i + j*m->ncomponents] = (float)(m->components[i]->mean[pointdim[j]]);
            pCovDiag[i + j*m->ncomponents] = (float)(m->components[i]->covariance->data[pointdim[j]*(m->ndim+1)]);
        }
    }

    // cast double grid_acc to float, store in host memory
    for (int i=0;i<nGrid*m->ncomponents;i++)
    {
        pGridAcc[i] = (float)grid_acc[i];
    }

    nComponents = m->ncomponents;
 
    return 0;
}

int HostMemory::setDoubleResult(int nSpikes)
{
    for (int i=0;i<nGrid*nSpikes;i++)
    {
        pResult_d[i] = (double)pResult[i] / scale;
    }
}

//**************************************
//--- decoding interfacing functions ---
//**************************************

int uploadModelToGpu(Mixture* m, double* grid_acc, uint16_t* pointdim, HostMemory* hostMem, GpuMemory* gpuMem)
{
    if(hostMem->fillHostModelMem(m,grid_acc,pointdim))
    {
        return -1;
    }
    if(gpuMem->uploadModelToGpu(*hostMem))
    {
        return -1;
    }
    return 0;
}


//--------- new wrappers ------------------------------------------------------------------

// add a new tetrode/model to the decoder
int GpuDecoder::addTT(Mixture* m, double* grid_acc, int spikeDim, uint16_t* pointDim)
{
    if (spikeDim<1)
    {
	std::cout<< "addTT failed: invalid spikeDim="<<spikeDim <<std::endl;
	return -1;
    }
    if (m==nullptr)
    {
	std::cout<< "addTT failed: null Mixture pointer"<<std::endl;
	return -1;
    }
    if (gridDim_<=0)
    {
	std::cout << "addTT failed: invalid gridDim="<<gridDim_<<std::endl;
	return -1;
    }

    //printf("maxBatchSize_=%d,batchSize_=%d\n",maxBatchSize_,batchSize_);
    hostMems.push_back(new HostMemory(spikeDim,m->ncomponents,gridDim_,maxBatchSize_));
    gpuMems.push_back(new GpuMemory(spikeDim,m->ncomponents,gridDim_,batchSize_));
    if (0 != hostMems.back()->allocHostMem())
    {
	hostMems.pop_back();
	std::cout<< "addTT failed: host mem alloc failed"<<std::endl;
	return -1;
    }
    if (gpuMems.back()->allocGpuMem()!=0)
    {
	gpuMems.pop_back();
	std::cout<< "addTT failed: gpu mem alloc failed"<<std::endl;
	return -1;
    }
    //printf("upload model\n");
    if (0 != uploadModelToGpu(m,grid_acc,pointDim,hostMems.back(),gpuMems.back()))
    {
	hostMems.pop_back();
	gpuMems.pop_back();
	std::cout << "addTT failed: uploadModelToGpu failed"<<std::endl;
	return -1;
    }
    //printf("upload model done\n");

    cudaStream_t stream_tmp;
    cudaStreamCreate(&stream_tmp);
    stream.push_back(stream_tmp);
    n_tt_++;
    //std::cout << "successfully addTT n_tt="<<n_tt_ <<std::endl;
    return 0;
}

int GpuDecoder::uploadKdeSpikes(const int tt_idx, double* spikes, const int n_spikes)
{
    if (tt_idx<0 || tt_idx > n_tt_-1)
    {
	std::cout<< "uploadKdeSpikes failed: invalid tt_idx="<<tt_idx<<std::endl;
	return -1;
    }

    float* spikes_d = gpuMems[tt_idx]->pSpike;
    float* spikes_h = hostMems[tt_idx]->pSpike;
    int nspikedim = gpuMems[tt_idx]->getSpikeDim();

    cudaError_t cuda_result_code;
    //cast double to float
    for (int i=0;i<n_spikes;i++)
    {
        for (int j=0;j<nspikedim;j++)
	{
            spikes_h[i*nspikedim + j] = (float)spikes[i*(nspikedim) + j];
	}
    }
    /*
    cudaMemcpyAsync(spikes_d, spikes_h, n_spikes*nspikedim*sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) 
    {
	printf("uploadKdeSpikes failed: message cudaMemCpy spike, in upload spikes %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
*/
    return uploadData(spikes_d,spikes_h,n_spikes*nspikedim*sizeof(float),stream[tt_idx]);

}

int GpuDecoder::downloadKdeResults(const int tt_idx, const int n_spikes)
{
    if (tt_idx<0 || tt_idx > n_tt_-1)
    {
	std::cout<< "downloadKdeResults failed: invalid tt_idx="<<tt_idx<<std::endl;
	return -1;
    }
    cudaError_t cuda_result_code;
    float* result_d = gpuMems[tt_idx]->pResult;
    float* result_f = hostMems[tt_idx]->pResult;
    int ngrid = gpuMems[tt_idx]->getNgrid();
    /*cudaMemcpyAsync(result_f, result_d, n_spikes*ngrid*sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) 
    {
	printf("downloadKdeResults failed: message cudaMemCpy spike, in upload spikes %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }*/
    return downloadData(result_f,result_d,n_spikes*ngrid*sizeof(float),stream[tt_idx]);
}

int GpuDecoder::launchKde(const int tt_idx, const int n_spikes)
{
    if (tt_idx<0 || tt_idx > n_tt_-1)
    {
        std::cout<< "launchKde failed: invalid tt_idx="<<tt_idx<<std::endl;
        return -1;
    }

    KDEpt kdept;
    KDEparam kdeparam;
    // memory pointers
    kdept.spikes_d = gpuMems[tt_idx]->pSpike;
    kdept.mean_d = gpuMems[tt_idx]->pMean;
    kdept.cov_diag_d = gpuMems[tt_idx]->pCovDiag;
    kdept.scale_d = gpuMems[tt_idx]->pScale;
    kdept.grid_acc_d = gpuMems[tt_idx]->pGridAcc;
    kdept.component_acc_d = gpuMems[tt_idx]->pComponentAcc;
    kdept.result_d = gpuMems[tt_idx]->pResult;
    // params
    kdeparam.ncomponents = gpuMems[tt_idx]->getNcomponents();
    kdeparam.nspikedim = gpuMems[tt_idx]->getSpikeDim();
    kdeparam.ngrid = gpuMems[tt_idx]->getNgrid();
    kdeparam.mcPitch = gpuMems[tt_idx]->getCpitch();
    kdeparam.gaPitch = gpuMems[tt_idx]->getGpitch();
    kdeparam.bs_ca = bs_ca;
    kdeparam.bs_ev = bs_ev;
    kdeparam.maxSmem = maxSmem;
    kdeparam.n_spikes = n_spikes;
    //std::cout << "nComponents="<<kdeparam.ncomponents  <<std::endl;

    if (0!=launch_component_acc(&kdept,&kdeparam))
    {
        std::cout << "launchKde failed: launch component_acc failed"<<std::endl;
	return -1;
    }
    if (0!=launch_evaluate(&kdept,&kdeparam))
    {
        std::cout << "launchKde failed: launch evaluate failed"<<std::endl;
	return -1;
    }
    return 0;
}

double getElapse(timespec ts_start,timespec ts_end)
{
    return (double) (ts_end.tv_sec-ts_start.tv_sec)*1000.0 + (double) (ts_end.tv_nsec-ts_start.tv_nsec)/1000000.0;
}

// -------------------------------------------------------------------
// spikes and results memory spaces are allocated by the user,
// user should  make sure the memory size/layout are adapted to this function.
// spikes: |--- spikeDim_ ---|--- spikeDim ---| ... |--- spikeDim ---|
//         |<------------------ n_spikes*spikeDim ------------------>|
//
// results: |--- gridDim_ ---|--- gridDim ---| ... |--- gridDim ---|
//          |<----------------- n_spikes*gridDim ----------------->|
// -------------------------------------------------------------------
int GpuDecoder::decodeTT(const int tt_idx, double* spikes, const int n_spikes, double* results)
{
    if (tt_idx<0 || tt_idx > n_tt_-1)
    {
	std::cout<< "decodeTT failed: invalid tt_idx="<<tt_idx<<std::endl;
	return -1;
    }

    //test
    //struct timespec t1;
   // struct timespec t2;
   // struct timespec t3;
    //struct timespec t4;
    //struct timespec t5;

    int n_left_spikes = n_spikes;
    int spikes_to_process = 0;
    int spikeDim = hostMems[tt_idx]->getSpikeDim();
    double scale = hostMems[tt_idx]->getScale();
    while(n_left_spikes>0)
    {
        if (n_left_spikes>batchSize_)
	{
            spikes_to_process = batchSize_;
	}
	else
	{
            spikes_to_process = n_left_spikes;
	}
        //clock_gettime(CLOCK_MONOTONIC, &t1);
        // upload spikes to GPU memory
        if (0 != uploadKdeSpikes(tt_idx,spikes,spikes_to_process))
	{
	    std::cout << "decodeTT failed: uploadKdeSpikes failed"<<std::endl;
	    return -1;
	}
        //clock_gettime(CLOCK_MONOTONIC, &t2);
	// launch kde kernels
        if (0 != launchKde(tt_idx,spikes_to_process))
	{
	    std::cout << "decodeTT failed: uploadKdeSpikes failed"<<std::endl;
	    return -1;
	}

        //clock_gettime(CLOCK_MONOTONIC, &t3);
        // download results to CPU memory
        if (0 != downloadKdeResults(tt_idx,spikes_to_process))
	{
	    std::cout << "decodeTT failed: downloadKdeResults failed"<< std::endl;
	    return -1;
	}
        
        //clock_gettime(CLOCK_MONOTONIC, &t4);
	// type casting & rescale data
	for (int i=0;i<gridDim_*spikes_to_process;i++)
	{
            results[i] = (double)(hostMems[tt_idx]->pResult[i]) / scale;
	}
        //clock_gettime(CLOCK_MONOTONIC, &t5);
/*	printf("time for batchSize=%d:\n",spikes_to_process);
	double total = getElapse(t1,t5);
	double upl = getElapse(t1,t2);
	double kde = getElapse(t2,t3);
	double dl = getElapse(t3,t4);
	double cast = getElapse(t4,t5);
	printf("total=%f,upload=%f,kde=%f,dl=%f,cast=%d unit=ms\n",total,upl,kde,dl,cast);
*/
	spikes += batchSize_*(spikeDim);
	results += batchSize_*gridDim_;
	n_left_spikes -= batchSize_;
    }
    
    return 0;
}
void GpuDecoder::showMemRequirement(const int ncomponents, const int spikeDim) const
{
    //gpu
    double totalGpuBytes = 0;
    // try mallocPitch to get pitch size
    float* p_tmp_c;
    float* p_tmp_g;
    size_t cPitch;
    size_t gPitch;

    cudaError_t cuda_result_code;

    cudaMallocPitch(&p_tmp_c, &cPitch, ncomponents*sizeof(float), 2);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("can't get component pitch size: %s,set to component size\n show minimum gpu memory requirement, actual required gpu memory size can be slightly larger\n",cudaGetErrorString(cuda_result_code));
	cPitch = ncomponents;
    }
    if (p_tmp_c != 0)
    {
	cudaFree(p_tmp_c);
    }

    cudaMallocPitch(&p_tmp_g, &gPitch, gridDim_*sizeof(float), 2);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("can't get grid pitch size: %s, set to grid size\n show minimum gpu memory requirement, actual required gpu memory size can be slightly larger\n",cudaGetErrorString(cuda_result_code));
	gPitch = gridDim_;
    }
    if (p_tmp_g != 0)
    {
	cudaFree(p_tmp_g);
    }

    printf("ncomponents=%d,spikeDim=%d,gridDim=%d,maxBatchSize=%d\n",ncomponents,spikeDim,gridDim_,maxBatchSize_);
    printf("--------------GPU------------\n");
    printf("component pitch size=%d,grid pitch size=%d\n",cPitch,gPitch);

    // spike
    totalGpuBytes += spikeDim*sizeof(float)*maxBatchSize_;
    printf("spike needs %d bytes\n",spikeDim*sizeof(float)*maxBatchSize_);
    // mean of components
    totalGpuBytes += cPitch*sizeof(float)*spikeDim;
    printf("mean of components needs %d bytes\n",cPitch*sizeof(float)*spikeDim);
    // diagnal covariance of components
    totalGpuBytes += cPitch*sizeof(float)*spikeDim;
    printf("diagnal covariance of components needs %d bytes\n",cPitch*sizeof(float)*spikeDim);
    // scale of components
    totalGpuBytes += ncomponents*sizeof(float);
    printf("scale of components needs %d bytes\n",ncomponents*sizeof(float));
    // grid acc
    totalGpuBytes += gPitch*sizeof(float)*ncomponents;
    printf("grid acc needs %d bytes\n",gPitch*sizeof(float)*ncomponents);
    // component acc
    totalGpuBytes += cPitch*sizeof(float)*maxBatchSize_;
    printf("components acc needs %d bytes\n",cPitch*sizeof(float)*maxBatchSize_);
    // results
    totalGpuBytes += gPitch*sizeof(float)*maxBatchSize_;
    printf("results needs %d bytes\n",gPitch*sizeof(float)*maxBatchSize_);
    printf("##total gpu memory requirement is %f.1 kbytes\n",totalGpuBytes/1024);
    //cpu
    printf("--------------CPU------------\n");
    double totalCpuBytes = 0;
    // spike
    totalCpuBytes += spikeDim*sizeof(float)*maxBatchSize_;
    printf("spike needs %d bytes\n",spikeDim*sizeof(float)*maxBatchSize_);
    // mean of components
    totalCpuBytes += ncomponents*sizeof(float)*spikeDim;
    printf("mean of components needs %d bytes\n",ncomponents*sizeof(float)*spikeDim);
    // diagnal covariance of components
    totalCpuBytes += ncomponents*sizeof(float)*spikeDim;
    printf("diagnal covariance of components needs %d bytes\n",ncomponents*sizeof(float)*spikeDim);
    // scale of components
    totalCpuBytes += ncomponents*sizeof(float);
    printf("scale of components needs %d bytes\n",ncomponents*sizeof(float));
    // grid acc
    totalCpuBytes += gridDim_*sizeof(float)*ncomponents;
    printf("grid acc needs %d bytes\n",gridDim_*sizeof(float)*ncomponents);
    // results
    totalCpuBytes += gridDim_*sizeof(float)*maxBatchSize_;
    printf("results needs %d bytes\n",gridDim_*sizeof(float)*maxBatchSize_);
    printf("##total cpu memory requirement is %f.1 kbytes\n",totalCpuBytes/1024);

}

int SignificanceAnalyzer::allocateMem()
{
    printf("alloc gpu significance mem\n");
    bool fail = false;
    cudaError_t cuda_result_code;
    // memory for pax
    //if (n_spatial_bin_>0 && max_spikes_>0 && n_spike_group_>0 && n_time_bin_>0)
    if (n_spatial_bin_>0 && shf_idx_size_>0)
    {
        //cudaMallocPitch(&pax_, &g_pitch_, n_spatial_bin_*sizeof(float), max_spikes_*n_spike_group_*n_time_bin_);
        cudaMallocPitch(&pax_, &g_pitch_, n_spatial_bin_*sizeof(float), shf_idx_size_*2);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message MallocPitch pax: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    //printf("allocated %d Btyes for pax\n",max_spikes_*n_spike_group_*n_time_bin_*g_pitch_);
	    printf("allocated %d Btyes for pax\n",shf_idx_size_*g_pitch_);
	}
    }
    else if (!fail)
    {
        //printf("invalid param: n_spatial_bin_=%d,max_spikes_=%d,n_spike_group_=%d,n_time_bin_=%d\n", n_spatial_bin_,max_spikes_,n_spike_group_,n_time_bin_);
        printf("invalid param: n_spatial_bin_=%d,n_pax__=%d\n", n_spatial_bin_,shf_idx_size_*2);
        fail = true;
    }
    // memory for offsets
    if (n_spatial_bin_>0 && n_spike_group_>0 && max_spikes_>0)
    {
        cudaMallocPitch(&offset_, &g_pitch_, n_spatial_bin_*sizeof(float), n_spike_group_* max_spikes_);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message Malloc offset: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for offset\n",n_spike_group_*max_spikes_*n_spatial_bin_);
	}
    }
    else if (!fail)
    {
        printf("invalid param:n_spatial_bin_=%d n_spike_group_=%d, max_spikes_=%d\n", n_spatial_bin_,n_spike_group_,max_spikes_);
        fail = true;
    }

    // memory for number of spikes of each tt/shank 
    if (!fail && n_time_bin_>0)
    {
        cudaMalloc(&cum_n_spikes_, (1+n_time_bin_)*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message malloc n_spikes: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for n_spikes\n",(1+n_time_bin_)*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_time_bin_=%d\n",n_spike_group_,n_time_bin_);
        fail = true;
    }
    // memory for probability 
    if (!fail && n_spatial_bin_>0 && n_shuffle_>0 && n_time_bin_>0)
    {
        cudaMallocPitch(&prob_, &g_pitch_, n_spatial_bin_*sizeof(float), n_time_bin_*n_shuffle_);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message prob mallocPitch: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for prob\n",n_shuffle_*n_time_bin_*g_pitch_);
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d,n_shuffle_=%d,n_time_bin_=%d\n",n_spatial_bin_,n_shuffle_,n_time_bin_);
        fail = true;
    }
    // memory for shuffle indexes
    //if (!fail && n_spike_group_>0 && max_spikes_ && n_shuffle_>0 && n_time_bin_>0)
    if (!fail && shf_idx_size_>0 && n_shuffle_>0)
    {
        //cudaMallocPitch(&shf_idx_, &s_pitch_, n_spike_group_*max_spikes_*sizeof(float), n_shuffle_*n_time_bin_);
        cudaMallocPitch(&shf_idx_, &s_pitch_, shf_idx_size_*sizeof(float), n_shuffle_);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message shf_idx_ mallocPitch: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for shf_idx_\n", n_shuffle_*s_pitch_);
	}
    }
    else if (!fail)
    {
        //printf("invalid param: max_spikes_=%d,n_spike_group_=%d,n_shuffle_=%d,n_time_bin_=%d\n",max_spikes_,n_spike_group_,n_shuffle_,n_time_bin_);
        printf("invalid param: shf_idx_size_=%d,n_shuffle_=%d\n",shf_idx_size_,n_shuffle_);
        fail = true;
    }

    // memory for memory offsets for indexes
    if (!fail && n_spike_group_>0 && max_spikes_ && n_time_bin_>0)
    {
        cudaMalloc(&shift_idx_, n_time_bin_*n_spike_group_*max_spikes_*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message memory shift malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for memory shift\n", n_time_bin_*n_spike_group_*max_spikes_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: max_spikes_=%d,n_spike_group_=%d,n_time_bin_=%d\n",max_spikes_,n_spike_group_,n_time_bin_);
        fail = true;
    }


    // memory for minus offset 
    //if (!fail && n_spatial_bin_>0 && n_time_bin_>0 )
    if (!fail && n_spatial_bin_>0 )
    {
        //cudaMallocPitch(&mo_, &g_pitch_, n_spatial_bin_*sizeof(float),n_time_bin_);
        cudaMalloc(&mo_, n_spatial_bin_*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message malloc mo: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    //printf("allocated %d btyes for mo\n",n_time_bin_*g_pitch_);
	    printf("allocated %d btyes for mo\n",n_spatial_bin_);
	}
    }
    else if (!fail)
    {
        //printf("invalid param: n_spatial_bin_=%d, n_time_bin_=%d\n",n_spatial_bin_,n_time_bin_);
        printf("invalid param: n_spatial_bin_=%d\n",n_spatial_bin_);
        fail = true;
    }
    // memory for pix 
    if (!fail && n_spatial_bin_>0 )
    {
        cudaMalloc(&pix_, n_spatial_bin_*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message malloc pix: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d btyes for pix\n",n_spatial_bin_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d\n",n_spike_group_);
        fail = true;
    }
    // memory for lx
    if (!fail && n_spatial_bin_>0)
    {
        cudaMalloc(&lx_, n_spatial_bin_*sizeof(float));
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message lx malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for lx\n", n_spatial_bin_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d\n",n_spatial_bin_);
        fail = true;
    }

    // alloc memory for pix/lx on cpu side
    if (!fail && n_spatial_bin_>0)
    {
        cudaHostAlloc(&pix_cpu_, n_spatial_bin_*sizeof(float), cudaHostRegisterDefault);
        cudaHostAlloc(&lx_cpu_, n_spatial_bin_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message pix/lx malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for lx/pix cpu\n", 2*n_spatial_bin_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d\n",n_spatial_bin_);
        fail = true;
    }
    
    // alloc memory for pax on cpu side
    if (!fail && n_spatial_bin_>0 && n_spike_group_>0 && max_spikes_>0)
    {
        cudaHostAlloc(&pax_cpu_, n_spike_group_*max_spikes_*n_spatial_bin_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message pax malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for pax cpu\n", n_spike_group_*max_spikes_*n_spatial_bin_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d,n_spike_group_=%d,max_spikes_=%d\n",n_spatial_bin_,n_spike_group_,max_spikes_);
        fail = true;
    }

    // alloc memory for offset on cpu side
    if (!fail && n_spike_group_>0 && max_spikes_>0)
    {
        cudaHostAlloc(&offset_cpu_, n_spatial_bin_*n_spike_group_*max_spikes_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message pax malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for offset cpu\n", n_spatial_bin_*n_spike_group_*max_spikes_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_spatial_bin_=%d,n_spike_group_=%d,max_spikes_=%d\n",n_spatial_bin_,n_spike_group_,max_spikes_);
        fail = true;
    }

    
    // alloc memory for n_spikes/cum_n_spikes on cpu side
    if (!fail && n_time_bin_>0)
    {
        cudaHostAlloc(&n_spikes_cpu_, n_time_bin_*sizeof(float), cudaHostRegisterDefault);
        cudaHostAlloc(&cum_n_spikes_cpu_, (n_time_bin_+1)*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message n_spikes/cum_n_spikes malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for  n_spikes/cum_n_spikes cpu\n", (n_time_bin_*2+1)*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_time_bin_=%d\n",n_time_bin_);
        fail = true;
    }
   
    // alloc memory for memory offsets on cpu side
    if (!fail && n_spike_group_>0 && max_spikes_ && n_time_bin_>0)
    {
        cudaHostAlloc(&shift_idx_cpu_, n_time_bin_*n_spike_group_*max_spikes_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message memory offsets malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for memory shift\n", n_time_bin_*n_spike_group_*max_spikes_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: max_spikes_=%d,n_spike_group_=%d,n_time_bin_=%d\n",max_spikes_,n_spike_group_,n_time_bin_);
        fail = true;
    }

    // alloc memory for shf idx on cpu side
    //if (!fail && n_spike_group_>0 && max_spikes_ && n_time_bin_>0)
    if (!fail && n_shuffle_>0 && shf_idx_size_>0)
    {
        //cudaHostAlloc(&shf_idx_cpu_, n_time_bin_*n_spike_group_*max_spikes_*sizeof(float), cudaHostRegisterDefault);
        cudaHostAlloc(&shf_idx_cpu_, n_shuffle_*shf_idx_size_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message shuffle index malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for shuffle index cpu\n", n_shuffle_*shf_idx_size_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_shuffle_=%d,shf_idx_size_=%d\n",n_shuffle_,shf_idx_size_);
        fail = true;
    }
    
    // alloc memory for prob on cpu side
    if (!fail && n_shuffle_>0 && n_spatial_bin_>0)
    {
        cudaHostAlloc(&prob_cpu_, n_shuffle_*n_spatial_bin_*sizeof(float), cudaHostRegisterDefault);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
            printf("message prob malloc: %s\n",cudaGetErrorString(cuda_result_code));
            fail = true;
        }
	else
	{
	    printf("allocated %d Btyes for shuffle index cpu\n", n_shuffle_*n_spatial_bin_*sizeof(float));
	}
    }
    else if (!fail)
    {
        printf("invalid param: n_shuffle_=%d,n_spatial_bin_=%d\n",n_shuffle_,n_spatial_bin_);
        fail = true;
    }

    if (fail)
    {
        printf("allocation failed\n");
        releaseMem();
        return -1; 
    }
    //printf("alloc gpu mem done\n");
    return 0;
}

int SignificanceAnalyzer::releaseMem()
{
    if (pax_ != nullptr)
        cudaFree(pax_);
    if (prob_ != nullptr)
        cudaFree(prob_);
    if (shf_idx_ != nullptr)
        cudaFree(shf_idx_);
    if (cum_n_spikes_ != nullptr)
        cudaFree(cum_n_spikes_);
    if (pix_ != nullptr)
        cudaFree(pix_);
    if (lx_ != nullptr)
        cudaFree(lx_);
    if (mo_ != nullptr)
        cudaFree(mo_);
    if (shift_idx_ != nullptr)
        cudaFree(shift_idx_);
    if (lx_cpu_ != nullptr)
        free(lx_cpu_);
    if (pix_cpu_ != nullptr)
        free(pix_cpu_);
    if (n_spikes_cpu_ != nullptr)
        free(n_spikes_cpu_);
    if (cum_n_spikes_cpu_ != nullptr)
        free(cum_n_spikes_cpu_);
    if (shift_idx_cpu_ != nullptr)
        free(shift_idx_cpu_);
    if (shf_idx_cpu_ != nullptr)
        free(shf_idx_cpu_);
    if (prob_cpu_ != nullptr)
        free(prob_cpu_);
    return 0;
}

float* SignificanceAnalyzer::pax(const int spike_idx)
{
    int gr_idx = 1;
    while (cum_n_spikes_[gr_idx]<spike_idx && gr_idx<n_gr_-1)
        gr_idx++;
    int spike_idx_gr = spike_idx-cum_n_spikes_[gr_idx-1];
    int pax_idx = (gr_idx-1)*max_spikes_ + spike_idx_gr;

    float* pax_p = (float*)((char*)(pax_)+g_pitch_*pax_idx);
    return pax_p;
}

float* SignificanceAnalyzer::pax(const int bin, const int gr)
{
    int gr_idx = bin*n_spike_group_+gr;
    gr_idx = (gr_idx>=n_gr_) ? (n_gr_-1):gr_idx;

    int pax_idx = gr_idx*max_spikes_;

    float* pax_p = (float*)((char*)(pax_)+g_pitch_*pax_idx);
    return pax_p;
}

float* SignificanceAnalyzer::prob(const int bin, const int shf)
{
    int prob_idx = bin*shf;
    return (float*)((char*)(prob_)+g_pitch_*prob_idx);
}

float* SignificanceAnalyzer::shf_idx(const int shf)
{
    return (float*)((char*)(shf_idx_)+s_pitch_*shf); 
}

void SignificanceAnalyzer::uploadParam(double* pix, double* lx)
{
    // copy pix/lx to GPU
    for(int i=0 ;i< n_spatial_bin_;i++)
    {
        pix_cpu_[i] = (float)pix[i];
	lx_cpu_[i] = 0;
    }

    for (int i=0;i<n_spatial_bin_;i++)
    {
	for (int j=0;j<n_spike_group_;j++)
	{
	    lx_cpu_[i] += (float)lx[j*n_spatial_bin_+i]; 
	}
    }

    cudaMemcpy(pix_, pix_cpu_, n_spatial_bin_*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lx_, lx_cpu_, n_spatial_bin_*sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("message cuda cudaMemcpy pix/lx: %s\n",cudaGetErrorString(cuda_result_code));
    }

    // generate shuffle index
    int k,tmp;
    int* shf_row;
    for (int i=0;i<n_shuffle_;i++)
    {
        shf_row = shf_idx_cpu_ + i*shf_idx_size_;
        // initialization
        for (int j=0;j<shf_idx_size_;j++)
	{
	    shf_row[j] = j;
	}
	// Fisher-Yates shuffle

	for (int j=shf_idx_size_-1;j>0;j--)
	{
	    k = rand()%j;
	    tmp = shf_row[j];
	    shf_row[j] = shf_row[k];
	    shf_row[k] = tmp;
	}
    }
    // copy to GPU
    cudaMemcpy2D(shf_idx_, s_pitch_, shf_idx_cpu_, shf_idx_size_*sizeof(float), shf_idx_size_*sizeof(float), n_shuffle_, cudaMemcpyHostToDevice); 
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("message mean_d memory cpy 2D shf_idx: %s\n",cudaGetErrorString(cuda_result_code));
    }
    // check mem
    //launchChkMemInt(shf_idx_, s_pitch_, shf_idx_size_, n_shuffle_);
}

void SignificanceAnalyzer::updateBin(double* pax, int* n_spikes_g, double* mu, const int n_spikes, const int n_group)
{
    // copy to pinned memory
    for (int i=0;i<n_group*n_spikes;i++)
    {
        for (int j=0;j<n_spatial_bin_;j++)
	{
	    pax_cpu_[i*n_spatial_bin_+j] = (float)pax[i*n_spatial_bin_+j];
	    //printf("i=%d,j=%d\n",i,j);
	}
    } 
    
    float* p_pax = (float*)((char*)pax_ + head*g_pitch_);
    // pax buffer has shf_idx_size*2 rows, and shf_idx_size should be larger than the #pax from 1 bin
    if (n_spikes*n_group>shf_idx_size_)
    {
        printf("pax buffer size(shf_idx_size_*2) is too small!, new paxes=%d, shf_idx_size_=%d\n",n_spikes*n_group,shf_idx_size_);
	return;
    }
    // if reach the end, fill the mem circularly
    if (head + n_spikes*n_group >= shf_idx_size_*2)
    {
        int l1 = shf_idx_size_*2 - head;
	int l2 = n_spikes*n_group - l1;
        cudaMemcpy2D(p_pax, g_pitch_, pax_cpu_, n_spatial_bin_*sizeof(float), n_spatial_bin_*sizeof(float), l1, cudaMemcpyHostToDevice); 
        cudaMemcpy2D(pax_, g_pitch_, pax_cpu_+l1, n_spatial_bin_*sizeof(float), n_spatial_bin_*sizeof(float), l2, cudaMemcpyHostToDevice); 
        head = l2;
    }
    else
    {
        cudaMemcpy2D(p_pax, g_pitch_, pax_cpu_, n_spatial_bin_*sizeof(float), n_spatial_bin_*sizeof(float), n_spikes*n_group, cudaMemcpyHostToDevice);
	head += n_spikes*n_group;
    }
    //printf("head=%d\n",head);
    // check data
    //launchChkPax(pax_, 90, head, shf_idx_size_*2, n_spatial_bin_, g_pitch_);
    // update shuffle idx, the 1st one is the non-shuffled result
    int id = 0;
    int shift = 0;
    for (int i=0;i<n_spike_group_;i++)
    {
        //printf("n_spikes_g[%d]=%d,shift=%d\n",i,n_spikes_g[i],shift);
        for (int j=0;j<n_spikes_g[i];j++)
	{
	    shf_idx_cpu_[id] = j+shift;
	    shf_idx_cpu_[id] = n_spikes*n_group - shf_idx_cpu_[id]-1;
	    //printf("shf_idx_cpu_=[%d]=%d\n",id,shf_idx_cpu_[id]);
	    id++;
	}
	shift += n_spikes_g[i]+n_spikes;
    }
    cudaMemcpy(shf_idx_,shf_idx_cpu_,n_spikes*sizeof(int),cudaMemcpyHostToDevice);
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("message mean_d memory cpy 2D pax,shift_idx: %s\n",cudaGetErrorString(cuda_result_code));
    }
    // check mem
    //launchChkMemInt(shf_idx_, s_pitch_, n_spikes, 2);

    // update number of spikes(actual)
    //n_spikes_cpu_[bin_head] = n_spikes;// may not actually used

    // update minus
    //launch_mo_kernel(pix_, lx_, mo_, cum_n_spikes_, g_pitch_, n_spatial_bin_, n_time_bin_, bin_size_);
    launch_mo_kernel(pix_, lx_, mo_, n_spikes, g_pitch_, n_spatial_bin_, bin_size_);
    // update offset
    id = 0;
    /*float* p_of;
    for (int i=0;i<n_spike_group_;i++)
    {
        printf("mu_[%d]=%f\n",i,mu[i]);
        for (int j=0;j<n_spikes_g[i];j++)
	{
	    p_of = offset_cpu_ + n_spatial_bin_*id;
	    for (int k=0;k<n_spatial_bin_;k++)
	    {
	        p_of[k] = pix_cpu_[k]*1e-10/mu[i];
	    }
	    printf("of[%d][0]=%f\n",i,p_of[0]);
	    id++; 
	}
    }*/
    
    // update n_pax
    n_pax_ += n_spikes*n_group;
    n_pax_ = n_pax_ % (shf_idx_size_*2);
    // update likelihood optional
    if (n_pax_>=shf_idx_size_)
    {
        launch_ll_bin_kernel(pax_, mo_, offset_, prob_, shf_idx_, g_pitch_, s_pitch_, n_spatial_bin_, n_spikes, bin_head, head, shf_idx_size_*2, n_shuffle_);
	//printf("bin_head=%d,n_shuffle=%d\n",bin_head,n_shuffle_);
	// copy ll to cpu
	float* p_prob = (float*)((char*)prob_ + (bin_head*n_shuffle_)*g_pitch_);
        //downloadData(prob_cpu_,p_prob,n_spatial_bin_*n_shuffle_*sizeof(float),0);
        //cudaMemcpy2D(prob_cpu_, g_pitch_, prob_, n_spatial_bin_*sizeof(float), n_spatial_bin_*sizeof(float), n_shuffle_, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(prob_cpu_, n_spatial_bin_*sizeof(float), p_prob, g_pitch_, n_spatial_bin_*sizeof(float), n_shuffle_, cudaMemcpyDeviceToHost);
        cuda_result_code = cudaGetLastError();
        if (cuda_result_code!=cudaSuccess) {
    	    printf("message memory cpy 2D prob to cpu: %s\n",cudaGetErrorString(cuda_result_code));
        }
	for (int i=0;i<n_spatial_bin_;i++)
	{
	    //printf("prob[0][%d]=%f,",i,prob_cpu_[i]);
	}
	//printf("\n");
	for (int i=0;i<n_spatial_bin_;i++)
	{
	    //printf("prob[1][%d]=%f,",i,prob_cpu_[i+n_spatial_bin_]);
	}
	//printf("\n");
	
	// update head
	bin_head = (bin_head+1)%n_time_bin_;
	if (bin_head==n_time_bin_-1 && full_==false)
	    full_ = true;
	
	// compute significance
	
    }
}
bool SignificanceAnalyzer::getProb(double* prob)
{
    if (n_pax_<shf_idx_size_)
        return false;
    for (int i=0;i<n_shuffle_;i++)
    {
        for (int j=0;j<n_spatial_bin_;j++)
	{
	    prob[j+n_spatial_bin_*i] = prob_cpu_[j+n_spatial_bin_*i];
	}
    }
    return true;
}
/*void SignificanceAnalyzer::updateBin(double* pax, int* n_spikes_g, const int n_spikes, const int n_group)
{
    // copy to pinned memory
    for (int i=0;i<n_group*n_spikes;i++)
    {
        for (int j=0;j<n_spatial_bin_;j++)
	{
	    pax_cpu_[i*n_spatial_bin_+j] = (float)pax[i*n_spatial_bin_+j];
	}
    }
    for (int i=0;i<n_group;i++)
        printf("n_spikes_g[%d]=%d\n",i,n_spikes_g[i]);
    
    //float* p_pax = (float*)((char*)pax_ + head*n_spike_group_*max_spikes_*g_pitch_);
    float* p_pax = (float*)((char*)pax_ + head*g_pitch_);
    // update cummulative number of spikes(actual)
    n_spikes_cpu_[head] = n_spikes;
    cum_n_spikes_cpu_[0] = 0;
    for (int i=0;i<n_time_bin_;i++)
        cum_n_spikes_cpu_[i+1] = cum_n_spikes_cpu_[i]+n_spikes_cpu_[i];

    
    // update shift_idx
    int id = 0;
    for (int i=0;i<n_time_bin_;i++)
    {
        for (int j=0;j<n_spikes_cpu_[i];j++)
	{
	    shift_idx_cpu_[id] = i*max_spikes_;
	}
    }

    // copy to GPU buffer
    cudaMemcpy2D(p_pax, g_pitch_, pax_cpu_, n_spatial_bin_*sizeof(float), n_spatial_bin_*sizeof(float), n_group*n_spikes, cudaMemcpyHostToDevice); 
    cudaMemcpy(cum_n_spikes_,cum_n_spikes_cpu_,(n_time_bin_+1)*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(shift_idx_,shift_idx_cpu_,cum_n_spikes_cpu_[n_time_bin_]*n_spike_group_*sizeof(float),cudaMemcpyHostToDevice);
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
	printf("message mean_d memory cpy 2D pax,cum_n_spikes, shift_idx: %s\n",cudaGetErrorString(cuda_result_code));
    }
    
    // update minus
    launch_mo_kernel(pix_, lx_, mo_, cum_n_spikes_, g_pitch_, n_spatial_bin_, n_time_bin_, bin_size_);

    // update shuffle idx
    launch_update_shuffle_idx_kernel(shf_idx_, s_pitch_, head, n_spikes, n_group, n_shuffle_);

    // generate random numbers on C code
    int* shf_id_host = (int*)malloc((n_shuffle_+1)*n_spike_group_*n_spikes*sizeof(int));
    printf("n_shuffle_=%d,n_spike_group_=%d,n_spikes=%d\n",n_shuffle_,n_spike_group_,n_spikes);
    id = 0;
    for (int i=0;i<n_spike_group_;i++)
    {
        for (int j=0;j<n_spikes_g[i];j++)
	{
	    shf_id_host[id] = i*n_spikes+j;
	    printf("shf_id_host=[%d]=%d\n",id,shf_id_host[id]);
	    id++;
	}
    }

    struct timespec t1;
    struct timespec t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    // Fisher Yates shuffle
    for (int i=0;i<n_shuffle_;i++)
    {
        int* p_shf = shf_id_host + (i+1)*n_spike_group_*n_spikes;
	// initialize
	for (int j=0;j<n_spike_group_*n_spikes-1;j++)
	{
	    p_shf[j] = j;
	}

	for (int j=0;j<n_spike_group_*n_spikes-1;j++)
	{
	    int k = rand()%(n_spike_group_*n_spikes-j) + j;
            int tmp =  p_shf[k];
	    p_shf[k] = p_shf[j];
	    p_shf[j] = tmp;
	}
	if (i==10)
	{
	    for (int j=0;j<n_spike_group_*n_spikes;j++)
	    {
	        printf("p_shf[%d]=%d\n",j,p_shf[j]);
	    }
	}
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    
    double total = getElapse(t1,t2);
    printf("shuffle time=%f\n",total);
      // update head
    head = (head+1)%n_time_bin_;
    if (head==n_time_bin_-1 && full_==false)
        full_ = true;
    printf("head=%d\n",head);
}*/

float SignificanceAnalyzer::significance()
{
    // compute significance based on rwd
    if (n_pax_>=shf_idx_size_)
    {
	float* p_prob = (float*)((char*)prob_ + (bin_head*n_shuffle_)*g_pitch_);
	//launchSigKernel(p_prob, n_shuffle, n_time_bin_);
    }
    return 0;
}
