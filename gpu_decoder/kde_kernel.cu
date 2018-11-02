#include "kde_kernel.h"
#include "math_functions.h"
#include "device_functions.h" 
#include "cuda_fp16.h"
#include <stdlib.h>
//#include "math.h"
//#include <immintrin.h>
const int default_bs = 64;

/* ----------------------------------------------------------------------------
 * compute componnet_acc before launching evaluation kernel
 * this kernel is paralleled on the component axis
 * ----------------------------------------------------------------------------
 * TEMPLATES: NSPIKEDIM - spike feature dimension
 *            BLOCKSIZE - block size, static shared memory size is launched
 *                        based on this number
 *
 * INPUT: spikes_d        - spike buffer pointer
 *        mean_d          - component mean buffer pointer
 *        cov_diag_d      - diagonal covariance buffer pointer
 *        component_acc_d - component accumulator buffer pointer(*OUTPUT*)
 *                          computation results will be written to this buffer
 *        ncomponents     - number of components
 *        mcPitch         - memory pitch size of component axis aligned buffer
 *                          (cov_diag_d,component_acc_d)
 *
 * NOTE: for current version:
 *       NSPIKEDIM can only be 1,2,3,4
 *       BLOCKSIZE can only be 64,128,256,512
 *
 * ----------------------------------------------------------------------------
 * AUHTOR:  Sile Hu, yuehusile@gmail.com
 * DATE  :  2018-02-07
 * VERSION: 0
 * ---------------------------------------------------------------------------- */
template <int NSPIKEDIM,int BLOCKSIZE>
__global__ void component_acc_kernel(float* spikes_d, float* mean_d, float* cov_diag_d, float* component_acc_d, const int ncomponents, const size_t mcPitch)
{
    __shared__ float spike[NSPIKEDIM];
    __shared__ float acc[BLOCKSIZE][NSPIKEDIM];
    float mean;
    float cov;
    float dx,acc_sum;
    int spike_idx = blockIdx.x;
    // find the corresponding component_acc memory space
    component_acc_d = (float*)((char*)component_acc_d + spike_idx*mcPitch);
    //load current spike into shared memory
    if (threadIdx.y ==0 && threadIdx.x<NSPIKEDIM)
    {
        spike[threadIdx.x] = spikes_d[spike_idx*NSPIKEDIM+threadIdx.x];
    }    
    // load mean,covariance of current component into register
    int c_idx = blockIdx.y * blockDim.x + threadIdx.x;//component index, spike idx = threadIdx.y
    float* row_m;
    float* row_c;
    if (c_idx<ncomponents)
    {
        row_m = (float*)((char*)mean_d + threadIdx.y*mcPitch);
        row_c = (float*)((char*)cov_diag_d + threadIdx.y*mcPitch);
        mean = row_m[c_idx];
        cov = row_c[c_idx];
    }
    __syncthreads();
    
    // compute component acc of each spike dimension
    if (c_idx<ncomponents)
    {
        dx = spike[threadIdx.y]-mean;
        dx = __fdividef(dx*dx, cov);
        acc[threadIdx.x][threadIdx.y] = dx;//store results into shared memory;
    }
    __syncthreads();

    //sum all dimensions together, and store to global memory
    if (c_idx<ncomponents)
    {
        if (threadIdx.y==0 && NSPIKEDIM>1)
        {
            acc[threadIdx.x][0] += acc[threadIdx.x][1]; 
        }
        if (threadIdx.y==1 && NSPIKEDIM>3)
        {
            acc[threadIdx.x][2] += acc[threadIdx.x][3];
        }
        if (threadIdx.y==2 && NSPIKEDIM>5)
        {
            acc[threadIdx.x][4] += acc[threadIdx.x][5];
        }
        if (threadIdx.y==3 && NSPIKEDIM>7)
        {
            acc[threadIdx.x][6] += acc[threadIdx.x][7];
        }
        if (threadIdx.y==4 && NSPIKEDIM>9)
        {
            acc[threadIdx.x][8] += acc[threadIdx.x][9];
        }
    }
    __syncthreads();

    if (c_idx<ncomponents) 
    {
	if (threadIdx.y==0 && NSPIKEDIM>2)
	    acc[threadIdx.x][0] += acc[threadIdx.x][2];
	if (threadIdx.y==1 && NSPIKEDIM>6)
	    acc[threadIdx.x][4] +=  acc[threadIdx.x][6];
    }
    __syncthreads();

    if (c_idx<ncomponents) 
    {
	if (threadIdx.y==0 && NSPIKEDIM>4)
	    acc[threadIdx.x][0] += acc[threadIdx.x][4];
    }
    __syncthreads();
    
    if (c_idx<ncomponents) 
    {
	if (threadIdx.y==0 && NSPIKEDIM>8)
	    acc[threadIdx.x][0] += acc[threadIdx.x][8];
	if (threadIdx.y==0)
	    component_acc_d[c_idx] = acc[threadIdx.x][0]>CUTOFF ? -1 : acc[threadIdx.x][0];
    }

/*    if (c_idx<ncomponents && threadIdx.y==0)
    {
        for (int i=1;i<NSPIKEDIM;i++)
	{
	    acc[threadIdx.x][0] += acc[threadIdx.x][i];
	}
	component_acc_d[c_idx] = acc[threadIdx.x][0]>CUTOFF ? -1 : acc[threadIdx.x][0];
    }
*/
}

/* ----------------------------------------------------------------------------
 * wrapper for launching componnet_acc kernel
 * ----------------------------------------------------------------------------
 * INPUT: pt    - pointer to gpu buffer pointer structure
 *        param - pointer to KDE parameter structure
 *
 * NOTE: for current version:
 *       kernels are launched with blocksize = 128 * spike feature dimension
 *
 * ----------------------------------------------------------------------------
 * AUHTOR:  Sile Hu, yuehusile@gmail.com
 * DATE  :  2018-02-07
 * VERSION: 0
 * ---------------------------------------------------------------------------- */

int launch_component_acc(KDEpt* pt, KDEparam* param)
{
    // compute grid & block size of component_acc kernel
    dim3 bsca(param->bs_ev,param->nspikedim);
    int grid_components = (param->ncomponents-1)/param->bs_ev + 1;
    dim3 gridca(param->n_spikes,grid_components);

    cudaError_t cuda_result_code;

    // ************ TODO: launch kernels with different block size ******

    // launch kernel based on spike feature dimension
    switch (param->nspikedim)
    {
    case 1:
        component_acc_kernel<1,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 2:
        component_acc_kernel<2,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 3:
        component_acc_kernel<3,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 4:
        component_acc_kernel<4,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 5:
        component_acc_kernel<5,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 6:
        component_acc_kernel<6,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 7:
        component_acc_kernel<7,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 8:
        component_acc_kernel<8,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 9:
        component_acc_kernel<9,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 10:
        component_acc_kernel<10,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    case 11:
        component_acc_kernel<11,default_bs><<<gridca, bsca>>>(pt->spikes_d, pt->mean_d, pt->cov_diag_d, pt->component_acc_d, param->ncomponents, param->mcPitch);
        break;
    default:
        printf("invalid nspikedim = %d\n",param->nspikedim);
        return -1;
    }
    // synch current stream
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message component acc kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }

    return 0;
}

// main kde kernel function
// max smem size can be set for componennt_acc_scale is 41952-128*4*4=39904bytes
template <int BLOCKSIZE>
__global__ void evaluate_kernel(float* component_acc_d, float* grid_acc_d, float* scale_d, float* result_d, const int ncomponents, const int ngrid, const size_t gaPitch, const int mcPitch, const int c_tile_size, const int c_n_tile)
{
    extern __shared__ half component_acc_scale[];
    __shared__ float tmp_result[BLOCKSIZE][4];

    int g_idx = blockIdx.y*blockDim.x + threadIdx.x;
    int tile_size = blockDim.x*2;
    int spike_idx = blockIdx.x;
    // find the corresponding component_acc memory space
    component_acc_d = (float*)((char*)component_acc_d + spike_idx*mcPitch);
    // find the corresponding result memory space
    result_d += spike_idx*ngrid;
    // number of loops for loading component_acc and scale into shared memory
    int nloadloop = (c_tile_size-1)/tile_size + 1;
    int c_idx, s_idx, t_idx;
    float acc,tmp,sum;
    double result;
    float* row_ga;
    int comp_left = ncomponents;
    int comp_in_tile;
 
    float tile_sum = 0;
    int offset = 0;
    int max_c_idx_in_tile = 0;

    for (int t=0;t<c_n_tile;t++)
    {
        // components to be processed in current COMPONENT tile
	comp_in_tile = comp_left>c_tile_size ? c_tile_size:comp_left;
	// shared memory idx for component in current COMPONENT tile
	t_idx = threadIdx.x + blockDim.x*(threadIdx.y/2);
	// component idx
	c_idx = t_idx + offset;
	// update component index range in current tile
	max_c_idx_in_tile += comp_in_tile;

        /* ----------------- copy to shared memory ---------------- */
	// each thread load nloadloop*2 times from device memory to shared memory
	for (int i=0;i<nloadloop;i++)
	{
	    // correponding scale idx in shared memory    
	    s_idx = t_idx + comp_in_tile;
	    // load component_acc into the first half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (0==threadIdx.y || 2==threadIdx.y))
	    {
		component_acc_scale[t_idx] = __float2half(component_acc_d[c_idx]);
	    }
	    // load scale into the second half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (1==threadIdx.y || 3==threadIdx.y))
	    {
		component_acc_scale[s_idx] = __float2half(scale_d[c_idx]);
	    }
	    c_idx += tile_size;
	    t_idx += tile_size;
	}
	half* component_acc = component_acc_scale;
	half* scale = component_acc_scale + comp_in_tile;
	__syncthreads();

	/* --------------- compute kde -------------------------- */
	// devide all the components into 4 parts and compute in paral
	int range[5];
	range[0] = 0;
	range[1] = comp_in_tile/4;
	range[2] = range[1]*2;
	range[3] = range[1]+range[2];
	range[4] = comp_in_tile;
	if (g_idx<ngrid)
	{
	    result = 0;
	    //#pragma unroll 37
	    // ** potential improvement ** : proper unroll the loop using #pragma unroll Num
	    for (int C=range[threadIdx.y];C<range[threadIdx.y+1];C++)
	    {
		acc = __half2float(component_acc[C]);
		row_ga = (float*)((char*)grid_acc_d + (C+offset)*gaPitch);
		if (acc>0)
		{
		    tmp = row_ga[g_idx];
		    sum = tmp + acc;
		    if (tmp>=0 && sum<CUTOFF)
		    {
			//result += __half2float(scale[C])*__expf(-0.5*sum);
			//result += __fmul_rn(__half2float(scale[C]),__expf(-0.5*sum));
			result = __fmaf_rn(__half2float(scale[C]),__expf(-0.5*sum),result);
		        
		    }  
		}
	    }
	    tile_sum += result;
	}
	offset+=c_tile_size;
	comp_left-=comp_in_tile;
	__syncthreads();
    }
    tmp_result[threadIdx.x][threadIdx.y] = tile_sum;
    __syncthreads();
    /* --------------- 1st sum of results in shared memory ---------------- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        tmp_result[threadIdx.x][0] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][2];
    }
    if (g_idx<ngrid && 1 == threadIdx.y)
    {
        tmp_result[threadIdx.x][1] = tmp_result[threadIdx.x][1] + tmp_result[threadIdx.x][3];
    }
    __syncthreads();

    /* --------------- 2nd sum & copy result back from shared memory ----- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        result_d[g_idx] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][1];
    }
}
// main kde kernel function
// max smem size can be set for componennt_acc_scale is 41952-128*4*4=39904bytes
template <int BLOCKSIZE>
__global__ void evaluate_kernel_e(float* component_acc_d, float* grid_acc_d, float* scale_d, float* result_d, const int ncomponents, const int ngrid, const size_t gaPitch, const int mcPitch, const int c_tile_size, const int c_n_tile)
{
    extern __shared__ half component_acc_scale[];
    //__shared__ float tmp_result[BLOCKSIZE][4];
    __shared__ float tmp_result[4][BLOCKSIZE];

    int g_idx = blockIdx.y*blockDim.x + threadIdx.x;
    //int g_idx = __fmaf_rd(blockIdx.y,blockDim.x,threadIdx.x);
    //int tile_size = blockDim.x*2;
    int tile_size = blockDim.x<<1;
    //int spike_idx = blockIdx.x;
    // find the corresponding component_acc memory space
    //component_acc_d = (float*)((char*)component_acc_d + spike_idx*mcPitch);
    component_acc_d = (float*)((char*)component_acc_d + blockIdx.x*mcPitch);
    // find the corresponding result memory space
    //result_d += spike_idx*ngrid;
    result_d += blockIdx.x*ngrid;
    // number of loops for loading component_acc and scale into shared memory
    int nloadloop = (c_tile_size-1)/tile_size + 1;
    int c_idx, s_idx, t_idx;
    float acc,tmp,sum;
    float result;
    float* row_ga;
    int comp_left = ncomponents;
    int comp_in_tile;
 
    double tile_sum = 0;
    int offset = 0;
    int max_c_idx_in_tile = 0;

    for (int t=0;t<c_n_tile;t++)
    {
        // components to be processed in current COMPONENT tile
	comp_in_tile = comp_left>c_tile_size ? c_tile_size:comp_left;
	// shared memory idx for component in current COMPONENT tile
	t_idx = threadIdx.x + blockDim.x*(threadIdx.y/2);
	//t_idx =__fmaf_rd(threadIdx.x,blockDim.x,(threadIdx.y>>2));
	// component idx
	c_idx = t_idx + offset;
	// update component index range in current tile
	max_c_idx_in_tile += comp_in_tile;

        /* ----------------- copy to shared memory ---------------- */
	// each thread load nloadloop*2 times from device memory to shared memory
	for (int i=0;i<nloadloop;i++)
	{
	    // correponding scale idx in shared memory    
	    s_idx = t_idx + comp_in_tile;
	    // load component_acc into the first half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (0==threadIdx.y || 2==threadIdx.y))
	    {
		component_acc_scale[t_idx] = __float2half(component_acc_d[c_idx]);
	    }
	    // load scale into the second half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (1==threadIdx.y || 3==threadIdx.y))
	    {
		component_acc_scale[s_idx] = __float2half(scale_d[c_idx]);
	    }
	    c_idx += tile_size;
	    t_idx += tile_size;
	}
	half* component_acc = component_acc_scale;
	half* scale = component_acc_scale + comp_in_tile;
	__syncthreads();

	/* --------------- compute kde -------------------------- */
	// devide all the components into 4 parts and compute in paral
	int range[5];
	range[0] = 0;
	//range[1] = comp_in_tile/4;
	range[1] = comp_in_tile>>2;
	range[2] = range[1]*2;
	range[3] = range[1]+range[2];
	range[4] = comp_in_tile;
	if (g_idx<ngrid)
	{
	    result = 0;
            row_ga = (float*)((char*)grid_acc_d + (range[threadIdx.y]+offset)*gaPitch);
	    int gaStep = gaPitch>>2;
	    //#pragma unroll 2
	    // ** potential improvement ** : proper unroll the loop using #pragma unroll Num
	    for (int C=range[threadIdx.y];C<range[threadIdx.y+1];C++)
	    {
		acc = __half2float(component_acc[C]);
		//row_ga = (float*)((char*)grid_acc_d + (C+offset)*gaPitch);
		if (acc>0)
		{
		    tmp = row_ga[g_idx];
		    //sum = tmp + acc;
		    sum = __fadd_rn(tmp,acc);
		    if (tmp>=0 && sum<CUTOFF)
		    {
			result = __fmaf_rn(__half2float(scale[C]),__expf(-0.5*sum),result);
		    }  
		}
		row_ga += gaStep;
	    }
	    //tile_sum += result;
	    tile_sum = __fadd_rn(tile_sum,result);
	}
	offset+=c_tile_size;
	comp_left-=comp_in_tile;
	__syncthreads();
    }
    //tmp_result[threadIdx.x][threadIdx.y] = tile_sum;
    tmp_result[threadIdx.y][threadIdx.x] = tile_sum;
    __syncthreads();
    /* --------------- 1st sum of results in shared memory ---------------- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        //tmp_result[threadIdx.x][0] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][2];
        tmp_result[0][threadIdx.x] = __fadd_rn(tmp_result[0][threadIdx.x],tmp_result[2][threadIdx.x]);
    }
    if (g_idx<ngrid && 1 == threadIdx.y)
    {
        //tmp_result[threadIdx.x][1] = tmp_result[threadIdx.x][1] + tmp_result[threadIdx.x][3];
        tmp_result[1][threadIdx.x] = __fadd_rn(tmp_result[1][threadIdx.x],tmp_result[3][threadIdx.x]);
    }
    __syncthreads();

    /* --------------- 2nd sum & copy result back from shared memory ----- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        //result_d[g_idx] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][1];
        result_d[g_idx] = __fadd_rn(tmp_result[0][threadIdx.x],tmp_result[1][threadIdx.x]);
    }
}

// main kde kernel function
// max smem size can be set for componennt_acc_scale is 41952-128*4*4=39904bytes
template <int BLOCKSIZE>
__global__ void evaluate_kernel_e1(float* component_acc_d, float* grid_acc_d, float* scale_d, float* result_d, const int ncomponents, const int ngrid, const size_t gaPitch, const int mcPitch, const int c_tile_size, const int c_n_tile)
{
    extern __shared__ half component_acc_scale[];
    //__shared__ float tmp_result[BLOCKSIZE][4];
    __shared__ float tmp_result[4][BLOCKSIZE];

    int g_idx = blockIdx.y*blockDim.x + threadIdx.x;
    //int g_idx = __fmaf_rd(blockIdx.y,blockDim.x,threadIdx.x);
    //int tile_size = blockDim.x*2;
    int tile_size = blockDim.x<<1;
    //int spike_idx = blockIdx.x;
    // find the corresponding component_acc memory space
    //component_acc_d = (float*)((char*)component_acc_d + spike_idx*mcPitch);
    component_acc_d = (float*)((char*)component_acc_d + blockIdx.x*mcPitch);
    // find the corresponding result memory space
    //result_d += spike_idx*ngrid;
    result_d += blockIdx.x*ngrid;
    // number of loops for loading component_acc and scale into shared memory
    int nloadloop = (c_tile_size-1)/tile_size + 1;
    int c_idx, s_idx, t_idx;
    float acc,tmp,sum;
    float result;
    float* row_ga;
    int comp_left = ncomponents;
    int comp_in_tile;
 
    double tile_sum = 0;
    int offset = 0;
    int max_c_idx_in_tile = 0;

    for (int t=0;t<c_n_tile;t++)
    {
        // components to be processed in current COMPONENT tile
	comp_in_tile = comp_left>c_tile_size ? c_tile_size:comp_left;
	// shared memory idx for component in current COMPONENT tile
	t_idx = threadIdx.x + blockDim.x*(threadIdx.y/2);
	//t_idx =__fmaf_rd(threadIdx.x,blockDim.x,(threadIdx.y>>2));
	// component idx
	c_idx = t_idx + offset;
	// update component index range in current tile
	max_c_idx_in_tile += comp_in_tile;

        /* ----------------- copy to shared memory ---------------- */
	// each thread load nloadloop*2 times from device memory to shared memory
	for (int i=0;i<nloadloop;i++)
	{
	    // correponding scale idx in shared memory    
	    s_idx = t_idx + comp_in_tile;
	    // load component_acc into the first half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (0==threadIdx.y || 2==threadIdx.y))
	    {
		component_acc_scale[t_idx] = __float2half(component_acc_d[c_idx]);
	    }
	    // load scale into the second half of shared memory
	    if (t_idx<comp_in_tile && c_idx<max_c_idx_in_tile && (1==threadIdx.y || 3==threadIdx.y))
	    {
		component_acc_scale[s_idx] = __float2half(scale_d[c_idx]);
	    }
	    c_idx += tile_size;
	    t_idx += tile_size;
	}
	half* component_acc = component_acc_scale;
	half* scale = component_acc_scale + comp_in_tile;
	__syncthreads();

	/* --------------- compute kde -------------------------- */
	// devide all the components into 4 parts and compute in paral
	int range[5];
	range[0] = 0;
	//range[1] = comp_in_tile/4;
	range[1] = comp_in_tile>>2;
	range[2] = range[1]*2;
	range[3] = range[1]+range[2];
	range[4] = comp_in_tile;

	if (g_idx<ngrid)
	{
	    result = 0;
            //row_ga = (float*)((char*)grid_acc_d + (range[threadIdx.y]+offset)*gaPitch);
	    int gaStep = gaPitch>>2;

	    int c_loop_step = 1264;
	    int c_idx_loop = threadIdx.y*c_loop_step;
            //row_ga = (float*)((char*)grid_acc_d + (c_idx_loop+offset)*gaPitch);

	    while(c_idx_loop<comp_in_tile)
	    {
                row_ga = (float*)((char*)grid_acc_d + (c_idx_loop+offset)*gaPitch);
	        //if (blockIdx.x==0 && g_idx==0 && threadIdx.x==0 && threadIdx.y==1)
		//    printf("c_idx_loop=%d, offset=%d\n",c_idx_loop,offset);
	        #pragma unroll 16
		for (int C=c_idx_loop;C<c_idx_loop+c_loop_step;C++)
		{
	        //    if (blockIdx.x==0 && g_idx==0 && threadIdx.x==0 && threadIdx.y==1)
		//        printf("C=%d\n",C);
		    if (C<comp_in_tile)
		    {
			acc = __half2float(component_acc[C+offset]);
			//row_ga = (float*)((char*)grid_acc_d + (C+offset)*gaPitch);
			if (acc>0)
			{
			    tmp = row_ga[g_idx];
			    //sum = tmp + acc;
			    sum = __fadd_rn(tmp,acc);
			    if (tmp>=0 && sum<CUTOFF)
			    {
				result = __fmaf_rn(__half2float(scale[C+offset]),__expf(-0.5*sum),result);
			    }  
			}
			row_ga += gaStep;
		    }
		}
		c_idx_loop += c_loop_step*4;
	    }
	    //tile_sum += result;
	    tile_sum = __fadd_rn(tile_sum,result);
	}
	offset+=c_tile_size;
	comp_left-=comp_in_tile;
	__syncthreads();
    }
    //tmp_result[threadIdx.x][threadIdx.y] = tile_sum;
    tmp_result[threadIdx.y][threadIdx.x] = tile_sum;
    __syncthreads();
    /* --------------- 1st sum of results in shared memory ---------------- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        //tmp_result[threadIdx.x][0] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][2];
        tmp_result[0][threadIdx.x] = __fadd_rn(tmp_result[0][threadIdx.x],tmp_result[2][threadIdx.x]);
    }
    if (g_idx<ngrid && 1 == threadIdx.y)
    {
        //tmp_result[threadIdx.x][1] = tmp_result[threadIdx.x][1] + tmp_result[threadIdx.x][3];
        tmp_result[1][threadIdx.x] = __fadd_rn(tmp_result[1][threadIdx.x],tmp_result[3][threadIdx.x]);
    }
    __syncthreads();

    /* --------------- 2nd sum & copy result back from shared memory ----- */

    if (g_idx<ngrid && 0 == threadIdx.y)
    {
        //result_d[g_idx] = tmp_result[threadIdx.x][0] + tmp_result[threadIdx.x][1];
        result_d[g_idx] = __fadd_rn(tmp_result[0][threadIdx.x],tmp_result[1][threadIdx.x]);
    }
}

int launch_evaluate(KDEpt* pt, KDEparam* param)
{
    // compute grid & block size of evaluation kernel
    dim3 bsev(param->bs_ca,4);
    int grid_grid = (param->ngrid-1)/param->bs_ca + 1;
    dim3 gridev(param->n_spikes,grid_grid);
    int nSmem = 0;
    int ntile = 1;
    int tile_size = 0;

    if (2*param->ncomponents*sizeof(half)>param->maxSmem)
    {
        nSmem = param->maxSmem;
        tile_size = param->maxSmem/(2*sizeof(half));
        ntile = (param->ncomponents-1)/tile_size + 1;
    }
    else
    {
        nSmem = 2*param->ncomponents*sizeof(half);
        ntile = 1;
        tile_size = param->ncomponents;
    }
    cudaError_t cuda_result_code;

    // ************ TODO: launch kernels with different block size ******
    //evaluate_kernel<128><<<gridev, bsev, nSmem>>>(pt->component_acc_d, pt->grid_acc_d, pt->scale_d, pt->result_d, param->ncomponents, param->ngrid, param->gaPitch, param->mcPitch,tile_size,ntile);
    evaluate_kernel_e<default_bs><<<gridev, bsev, nSmem>>>(pt->component_acc_d, pt->grid_acc_d, pt->scale_d, pt->result_d, param->ncomponents, param->ngrid, param->gaPitch, param->mcPitch,tile_size,ntile);
    // synch current stream
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message evaluate kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
    //printf("CUTOFF=%d\n",CUTOFF);
   return 0;
}

int uploadData(float* buff_d, float* buff_h, const size_t data_size_in_bytes, cudaStream_t stream)
{
    cudaError_t cuda_result_code;
    cudaMemcpyAsync(buff_d, buff_h, data_size_in_bytes, cudaMemcpyHostToDevice,0);
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) 
    {
	printf("uploadData failed: message cudaMemcpyAsync %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }

    return 0;
 
}
int downloadData(float* buff_h, float* buff_d, const size_t data_size_in_bytes, cudaStream_t stream)
{
    cudaError_t cuda_result_code;
    cudaMemcpyAsync(buff_h, buff_d, data_size_in_bytes, cudaMemcpyDeviceToHost,0);
    cudaStreamSynchronize(0);
    cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) 
    {
	printf("download failed: message cudaMemcpyAsync %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
    return 0;
 
}

__global__ void fakeDataKernel(float* pdata, int size, float num)
{
    for (int i=0;i<size;i++)
    {
        pdata[i] = num;
    }
    printf("size=%d,num=%f\n",size,num);
}
__global__ void fakeDataKernelnspike(int* pcum, int size, int num)
{
    int cum = 0;
    for (int i=0;i<size;i++)
    {
	cum+=num;
	pcum[i] = cum;
    }
    printf("size=%d,num=%d,cum=%d\n",size,num,cum);
}
__global__ void fakeDataKernel2d(float* pdata, size_t pitch, int size, int width, float num)
{
    float* p = pdata;
    for (int i=0;i<size;i++)
    {
        p = (float*)((char*)pdata + i*pitch);

	for (int j=0;j<width;j++)
	{
            p[j] = num;
	}
    }
    printf("size=%d,num=%f,width=%d,pitch=%d\n",size,num,width,(int)pitch);
    return;
}

__global__ void fakeDataKernelpax(float* pax, int n_spikes, size_t pitch, int n_pos, int n_tbin, float num)
{
    int pos_id = threadIdx.x;
    float* p;
    if (pos_id<n_pos)
    {
        for (int i=0;i<n_spikes;i++)
        {
	    p = (float*)((char*)pax + i*n_tbin*pitch);
            p[pos_id] = num;
        }
    }
    printf("n_spikes=%d,num=%f,n_pos=%d,pitch=%d,n_tbin\n",n_spikes,num,n_pos,(int)pitch,n_tbin);
}
__global__ void fakeDataKernelshf(int* pdata, size_t pitch, int size, int width)
{
    int* p = pdata;
    for (int i=0;i<size;i++)
    {
        p = (int*)((char*)pdata + i*pitch);

	for (int j=0;j<width;j++)
	{
            p[j] = (j+i)%width;
	}
    }
    printf("size=%dwidth=%d,pitch=%d\n",size,width,(int)pitch);
    return;
}
void launchFakeDataKernel(float* pdata, int size, float num)
{
    fakeDataKernel<<<1,1>>>(pdata,size,num);
    cudaDeviceSynchronize();
}
void launchFakeDataKernelnspike(int* pcum, int size, int num)
{
    fakeDataKernelnspike<<<1,1>>>(pcum,size,num);
    cudaDeviceSynchronize();
}
void launchFakeDataKernel2d(float* pdata, size_t pitch,int size,int width, float num)
{
    fakeDataKernel2d<<<1,1>>>(pdata,pitch,size,width,num);
    cudaDeviceSynchronize();
}
void launchFakeDataKernelpax(float* pax, int n_spikes, size_t pitch, int n_pos, int n_tbin, float num)
{
    fakeDataKernelpax<<<1,n_pos>>>(pax,n_spikes,pitch,n_pos,n_tbin,num);
    cudaDeviceSynchronize();
}
void launchFakeDataKernelshf(int* pdata, size_t pitch,int size,int width)
{
    fakeDataKernelshf<<<1,1>>>(pdata,pitch,size,width);
    cudaDeviceSynchronize();
}

__global__ void ll_kernel(float* pax, float* minus_offset, float* prob, int* shf, int* cum_spikes, size_t g_pitch, int n_pos, int n_bin, int max_spikes)
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int bin_idx = threadIdx.y;
    int shf_idx = blockIdx.y;
    int b_shift = cum_spikes[bin_idx+1];
    int n_spikes = cum_spikes[bin_idx+1] - cum_spikes[bin_idx]; 
    float* p_shf = (float*)((char*)shf + shf_idx*g_pitch) + b_shift;
    float* p_pax;
    float* p_prob = (float*)((char*)prob + (shf_idx*n_bin + bin_idx)*g_pitch);
    float* p_mo = (float*)((char*)minus_offset + bin_idx*g_pitch);
    int pax_idx;
    float sum_log_pax = 0;
    float ll;
    if (g_idx<n_pos)
    {
        // sum log pax
	for (int i=0;i<n_spikes;i++)
	{
	    pax_idx = p_shf[i];
	    p_pax = (float*)((char*)pax + pax_idx*g_pitch);
	    sum_log_pax += __logf(p_pax[g_idx]);
	}
	// minus offset
	prob[g_idx] = sum_log_pax - p_mo[g_idx];
    }
}

int launch_ll_kernel(float* pax, float* minus_offset, float* prob, int* shf, int* cum_spikes, size_t g_pitch, int n_pos, int n_bin, int max_spikes, int n_shuffle)
{
    int bs = 64;
    dim3 blockdim(bs,n_bin);
    dim3 griddim(n_shuffle,(n_pos-1)/bs +1);
    cudaError_t cuda_result_code1 = cudaGetLastError();
    if (cuda_result_code1!=cudaSuccess) {
        printf("message ll kernel3: %s\n",cudaGetErrorString(cuda_result_code1));
	//return -1;
    }
    ll_kernel<<<griddim,blockdim>>>(pax, minus_offset,prob,shf,cum_spikes,g_pitch,n_pos,n_bin,max_spikes);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message ll kernel2: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}

__global__ void minus_offset_kernel(float* pix, float* lx, float* p_mo, int n_spikes, size_t g_pitch, int n_pos, float bin_size)
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (g_idx<n_pos)
    {
        p_mo[g_idx] = n_spikes*__logf(pix[g_idx]) + bin_size*lx[g_idx];
        //p_mo[g_idx] = n_spikes*__logf(pix[g_idx])*12 + bin_size*lx[g_idx];// tmp test 
	//printf("mo[%d]=%f, log(pix)=%f\n",g_idx,p_mo[g_idx],__logf(pix[g_idx]));
    }
    
}

int launch_mo_kernel(float* pix, float* lx, float* minus_offset, int n_spikes, size_t g_pitch, int n_pos, float bin_size)
{
    int bs = 64;
    //dim3 blockdim(bs,n_bin);
    dim3 griddim(1,(n_pos-1)/bs +1);

    //minus_offset_kernel<<<griddim,blockdim>>>(pix, lx, minus_offset,cum_spikes,g_pitch,n_pos, bin_size);
    minus_offset_kernel<<<griddim,bs>>>(pix, lx, minus_offset,n_spikes,g_pitch,n_pos, bin_size);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message mo kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}

__global__ void update_shuffle_idx_kernel(int* shf_idx, size_t s_pitch, int bin_idx, int n_spikes, int n_group)
{
   int bin_shift = bin_idx*n_spikes*n_group;
   int row_idx = blockIdx.x*blockDim.x + threadIdx.x;
   int* p_shf = (int*)((char*)shf_idx + (bin_shift+row_idx)*s_pitch);
   
   //fake
   //int n = rand();
   for (int i=0;i<n_spikes*n_group;i++)
   {
       p_shf[i] = i;
   }
}

int launch_update_shuffle_idx_kernel(int* shf_idx, size_t s_pitch,int bin_idx, int n_spikes, int n_group, int n_shuffle)
{
    int bs = 512;
    int gs = (n_shuffle-1)/bs + 1;

    update_shuffle_idx_kernel<<<gs,bs>>>(shf_idx,s_pitch,bin_idx,n_spikes,n_group);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message update shuffle idx kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}

__global__ void checkMemInt(int* p_, size_t pitch, int width, int size)
{
    int* p;

    if (threadIdx.x==0)
    {
        for (int i=0;i<size;i++)
	{
	    p = (int*)((char*)p_ + pitch*i);
	    for (int j=0;j<width;j++)
	    {
	        printf("p[%d][%d]=%d\n",i,j,p[j]);
	    }
	}
    }
}

int launchChkMemInt(int* p_, size_t pitch, int width, int size)
{
    checkMemInt<<<1,1>>>(p_,pitch,width,size);
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message check mem kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}

__device__ float* getShfPaxRow(float* pax, int idx, int head, int size, size_t pitch)
{
    int p_id = (head + size - idx)%size;
    float* p = (float*)((char*)pax + p_id*pitch);
    return p;
}
/*
__device__ float* getNewPaxRow(float* pax, int* shift, int n_spk_bin, int idx, int head, int size, size_t pitch)
{
    
    int p_id = (head-1 + size - idx)%size;
    //printf("p_id=%d\n",p_id);
    float* p = (float*)((char*)pax + p_id*pitch);
    return p;
}
*/
__global__ void checkPax(float* pax, int n_idx, int head, int size, int n_pos, size_t pitch)
{
    if (threadIdx.x==0)
    {
        for (int i=0;i<n_idx;i++)
	{
	    //float* p = getShfPaxRow(pax,i,head,size,pitch);
	    int p_id = (head-1-i+size)%size;
	    float* p = (float*)((char*)pax + p_id*pitch);
	    printf("pax_id=%d\n",p_id);
	    for (int j=0;j<5;j++)
	    {
	        printf("pax[%d][%d]=%f,",i,j,p[j]);
	    }
	    printf("\n");
	}
    }
    printf("\n");
}

int launchChkPax(float* pax, int n_idx, int head, int size, int n_pos, size_t pitch)
{
    checkPax<<<1,1>>>(pax,n_idx,head,size,n_pos,pitch);
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message check pax kernel: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}


__global__ void ll_bin_kernel(float* pax, float* p_mo, float* offset, float* prob, int* shf, size_t g_pitch, size_t s_pitch, int n_pos, int n_spikes, int bin_idx, int head, int size, int n_shuffle)
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int shf_idx = blockIdx.y;
    int* p_shf = (int*)((char*)shf + shf_idx*s_pitch);
    float* p_pax;
    float* p_prob = (float*)((char*)prob + (shf_idx + n_shuffle*bin_idx)*g_pitch);
    int pax_idx;
    float sum_log_pax = 0;
    float ll;
    if (g_idx<n_pos)
    {
        // sum log pax
        int p_id;
	float* p_pax;
	//for (int i=0;i<n_spikes;i++)
	for (int i=0;i<n_spikes;i++)
	{
	    p_id = (head-1-p_shf[i]+size) % size;
	    p_pax = (float*)((char*)pax + p_id*g_pitch);
	    sum_log_pax += __logf(p_pax[g_idx]+1e-13);
	    //if (g_idx==0)
	    {
	        //printf("i=%d,p_pax[%d]=%f, log(pax)=%f, sum=%f, shf=%d\n",i,g_idx,p_pax[g_idx]+1e-4,__logf(p_pax[g_idx]+1e-4),sum_log_pax,p_shf[i]);
	    }
	}
	// minus offset
	p_prob[g_idx] = sum_log_pax - p_mo[g_idx];
	//printf("id=%d,p_pax[%d]=%f\n",p_id,g_idx,p_pax[g_idx]);
	//if (g_idx==0&&shf_idx<2)
	    //printf("shf=%d,prob[%d]=%f,sum_log_pax=%f,p_mo=%f,pid=%d,bin_idx=%d\n",shf_idx,g_idx,p_prob[g_idx],sum_log_pax,p_mo[g_idx],p_id,bin_idx);
    }
}

int launch_ll_bin_kernel(float* pax, float* p_mo, float* offset, float* prob, int* shf, size_t g_pitch, size_t s_pitch, int n_pos, int n_spikes, int bin_idx, int head, int size,int n_shuffle)
{
    int bs = 64;
    //dim3 blockdim(bs,n_bin);
    dim3 griddim((n_pos-1)/bs +1,n_shuffle);
    cudaError_t cuda_result_code1 = cudaGetLastError();
    if (cuda_result_code1!=cudaSuccess) {
        printf("message ll kernel3: %s\n",cudaGetErrorString(cuda_result_code1));
	//return -1;
    }
    ll_bin_kernel<<<griddim,bs>>>(pax, p_mo, offset, prob, shf, g_pitch, s_pitch, n_pos,n_spikes,bin_idx,head,size,n_shuffle);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message ll kernel2: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
}

// current setting requires n_pos<=BS, for a proper synchronization with block
__global__ void normalize_kernel(float* prob, int n_pos, int n_shuffle, size_t g_pitch, int bin_idx)
{
    int g_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int shf_idx = blockIdx.y;
    float* p_prob = (float*)((char*)prob + (shf_idx + n_shuffle*bin_idx)*g_pitch);
    // step 1 sum all the log_prob for mean
    __shared__ float tmp;
    if (g_idx==0)
    {
        tmp = 0;
	for (int i=0;i<n_pos;i++)
	{
	    tmp += p_prob[i];
	}
        tmp = tmp/n_pos;
    }
    __syncthreads();
    // step 2 exp
    __shared__ float exp_prob[64];
    exp_prob[g_idx]=__expf(p_prob[g_idx]-tmp);
    // step 3 compute sum
    __syncthreads();
    if (g_idx==0)
    {
        tmp = 0;
        for (int i=0;i<n_pos;i++)
	{
	    tmp += exp_prob[i];
	}
    }
    __syncthreads();
    // step 4 normalize
    p_prob[g_idx] = exp_prob[g_idx]/tmp;
}

int launch_normalize_kernel(float* prob, size_t g_pitch, int n_pos, int bin_idx, int n_shuffle)
{
    int bs = 64;
    //dim3 blockdim(bs,n_bin);
    dim3 griddim((n_pos-1)/bs +1,n_shuffle);
    cudaError_t cuda_result_code1 = cudaGetLastError();
    if (cuda_result_code1!=cudaSuccess) {
        printf("message norm kernel3: %s\n",cudaGetErrorString(cuda_result_code1));
	//return -1;
    }
    normalize_kernel<<<griddim,bs>>>(prob, n_pos, n_shuffle, g_pitch, bin_idx);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message norm kernel2: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
    return 0;
}
// current version uses 2 weights
__global__ void rwd_kernel(float* prob, float* dsmat, int n_shf, size_t g_pitch, int n_pos, int n_bin, float* rwd, float dmax, int bin_idx, int bin_buf_sz)
{
    int shf_idx = blockIdx.x*blockDim.x + threadIdx.x;
     
    float* p_prob;
    float* p_dsmat;
    float max2[50][2];
    float B_tmp[50][2];
    int max2_idx[50][2];
    float dist_states[50][50];
    float A[50][50];
    float B[50][50];
    float tmp;

    int actual_bin_idx = (bin_idx-n_bin + bin_buf_sz)%bin_buf_sz;
    if (shf_idx==0)
    {
        //printf("actual_bin_idx=%d,bin_idx=%d,bin_buf_sz=%d\n",actual_bin_idx,bin_idx,bin_buf_sz);
    }

    for (int i=0;i<n_bin;i++)
    {
        // step 1: find the largest two probiblities 
        p_prob = (float*)((char*)prob + (shf_idx + n_shf*actual_bin_idx)*g_pitch);
        actual_bin_idx = (actual_bin_idx + 1)%bin_buf_sz;
	max2[i][0] = 0;
	max2[i][1] = 0;
	// sort for two largest
	for (int k=0;k<2;k++)
	{
	    for (int j=0;j<n_pos;j++)
	    {
		if (p_prob[j]>max2[i][k])
		{
		    max2[i][k] = p_prob[j];
		    max2_idx[i][k] = j;
		}
	    }
	    p_prob[max2_idx[i][k]]=0;
	}
	p_prob[max2_idx[i][0]]=max2[i][0];
	p_prob[max2_idx[i][1]]=max2[i][1];
	tmp = max2[i][0]+max2[i][1];
	if (tmp>0)
	{
	    max2[i][0]=max2[i][0]/tmp;
   	    max2[i][1]=max2[i][1]/tmp;
	}
    // for debugging
	//if (shf_idx==0)
//	{
	    //printf("bin_idx=%d,prob[0]=%f,prob[1]=%f,max[0]=[%d]%f,max[1]=[%d]%f\n",i,p_prob[0],p_prob[1],max2_idx[i][0],max2[i][0],max2_idx[i][1],max2[i][1]);
//	}
    }
    // step 2: compute dist states
    for (int j=0;j<n_bin;j++)
    {
        for (int k=0;k<n_bin;k++)
	{
	    dist_states[j][k] = 0;
	    if (j!=k)
	    {
		for (int n=0;n<2;n++)
		{
		    for(int m=0;m<2;m++)
		    {
			p_dsmat = (float*)((char*)dsmat + (max2_idx[j][n])*g_pitch);
			dist_states[j][k] += (max2[j][n])*(max2[k][m])*(p_dsmat[max2_idx[k][m]]);
		    }
		}
	    }
	}
    }
    // for debugging
    /*
    if(shf_idx==0)
    {
        for (int j=0;j<5;j++)
	{
	    for (int k=0;k<5;k++)
	    {
                printf("dist_states[%d][%d]=%f,",j,k,dist_states[j][k]);
	    }
	    printf("\n");
	}
    }
    */
    /*
    if (shf_idx==0)
    {
        for (int j=0;j<n_bin;j++)
	{
            p_prob = (float*)((char*)prob + (shf_idx + n_shf*j)*g_pitch);
	    for (int k=0;k<n_bin;k++)
	    {
		printf("%f,",p_prob[k]);
	    }
	    printf("\n");
	}
    }*/
    // step 3: compute A and B
    // reuse max2 array
    //compute dist_time
    for (int j=0;j<n_bin;j++)
    {
        for (int k=0;k<n_bin;k++)
	{
	    if (abs(j-k)>=3)
	    {
	        A[j][k] = dmax;
	    }
	    else
	    {
	        A[j][k] = (abs(j-k)*dmax/3.0);
	    }
	}
    }
    /*
    if(shf_idx==0)
    {
        for (int j=0;j<5;j++)
	{
	    for (int k=0;k<5;k++)
	    {
                printf("A[%d][%d]=%f,",j,k,A[j][k]);
	    }
	    printf("\n");
	}
    }
    */

    //compute row and col means
    float ma = 0;
    float mb = 0;
    for (int j=0;j<n_bin;j++)
    {
        max2[j][0]=0;
	B_tmp[j][0]=0;
        max2[j][1]=0;
	B_tmp[j][1]=0;
        for (int k=0;k<n_bin;k++)
	{
	   max2[j][0] += A[j][k];
	   max2[j][1] += A[k][j];
	   B_tmp[j][0] += dist_states[j][k];
	   B_tmp[j][1] += dist_states[k][j];
	}
	ma += max2[j][0];
	mb += B_tmp[j][0];
        max2[j][0] /= n_bin;
        max2[j][1] /= n_bin;
        B_tmp[j][0] /= n_bin;
        B_tmp[j][1] /= n_bin;
    }
    ma /= (n_bin*n_bin);
    mb /= (n_bin*n_bin);
    //for debugging
    /*
    if(shf_idx==0)
    {
        for (int j=0;j<2;j++)
	{
	    for (int k=0;k<5;k++)
	    {
                printf("B_tmp[%d][%d]=%f,",k,j,B_tmp[k][j]);
	    }
	    printf("\n");
	}
	printf("ma=%f,mb=%f\n",ma,mb);
    }
    */
    // compute A,B & covariance and variance
    float dcov=0;
    float dvarx=0;
    float dvary=0;
    for (int j=0;j<n_bin;j++)
    {
        for (int k=0;k<n_bin;k++)
	{
	    A[j][k] = A[j][k] - max2[j][0] - max2[k][1] + ma;
	    dist_states[j][k] = dist_states[j][k] - B_tmp[j][0] - B_tmp[k][1] + mb;
	    dcov += A[j][k]*dist_states[j][k];
	    dvarx += A[j][k]*A[j][k];
	    dvary += dist_states[j][k]*dist_states[j][k];
	}
    }
    //for debugging
    /*
    if(shf_idx==0)
    {
        for (int j=0;j<5;j++)
	{
	    for (int k=0;k<5;k++)
	    {
                printf("dist_states[%d][%d]=%f,",j,k,dist_states[j][k]);
	    }
	    printf("\n");
	}
    }
    */
    dcov /= (n_bin*n_bin);
    dvarx /= (n_bin*n_bin);
    dvary /= (n_bin*n_bin);
    // step 4: compute covariance and variance
    rwd[shf_idx] = sqrt(dcov/sqrt(dvarx*dvary));
    if (dcov<0)
    {
        rwd[shf_idx]=0;
    }
    if (shf_idx==1)
    {
        //printf("rwd=%f,dcov=%f,dvarx=%f,dvary=%f\n",rwd[shf_idx],dcov,dvarx,dvary);
	/*for (int j=0;j<n_bin;j++)
	{
	    for (int k=0;k<n_bin;k++)
	    {
	        printf("%f,",dist_states[j][k]);
	    }
	    printf("\n");
	}*/
    }
}

int launch_rwd_kernel(float* prob, float* dsmat, int n_shuffle, size_t g_pitch, int n_pos, int n_bin, float* rwd, float dmax, int bin_idx, int bin_buf_sz)
{
    int bs = 64;
    //dim3 blockdim(bs,n_bin);
    int griddim =  (n_shuffle-1)/bs +1;
    cudaError_t cuda_result_code1 = cudaGetLastError();
    if (cuda_result_code1!=cudaSuccess) {
        printf("message rwd kernel3: %s\n",cudaGetErrorString(cuda_result_code1));
	//return -1;
    }
    rwd_kernel<<<griddim,bs>>>(prob, dsmat, n_shuffle, g_pitch, n_pos, n_bin, rwd, dmax, bin_idx, bin_buf_sz);
    // synch current stream
    cudaDeviceSynchronize();
    cudaError_t cuda_result_code = cudaGetLastError();
    if (cuda_result_code!=cudaSuccess) {
        printf("message rwd kernel2: %s\n",cudaGetErrorString(cuda_result_code));
	return -1;
    }
    return 0;
}
