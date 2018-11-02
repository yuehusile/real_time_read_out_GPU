#ifndef __KDE_KERNEL_H__
#define __KDE_KERNEL_H__

#include "../gmmcompression/mixture.h"
#include "cuda_runtime.h"

// device memory pointers for KDE
typedef struct KDEpt
{
    float* spikes_d;
    float* mean_d;
    float* cov_diag_d; 
    float* scale_d;
    float* component_acc_d;
    float* grid_acc_d;
    float* result_d;
};

// params for KDE
typedef struct KDEparam
{
    int n_spikes;
    int bs_ca;
    int bs_ev;
    int maxSmem;
    int ngrid; 
    int ncomponents;
    int nspikedim;
    size_t mcPitch;
    size_t gaPitch;
};

#ifdef __cplusplus
extern "C"
#endif
int launch_component_acc(KDEpt* pt, KDEparam* param);

#ifdef __cplusplus
extern "C"
#endif
int launch_evaluate(KDEpt* pt, KDEparam* param);

#ifdef __cplusplus
extern "C"
#endif
int uploadData(float* buff_d, float* buff_h, const size_t data_size_in_bytes, cudaStream_t stream);

#ifdef __cplusplus
extern "C"
#endif
int downloadData(float* buff_h, float* bufff_d, const size_t data_size_in_bytes, cudaStream_t stream);

#ifdef __cplusplus
extern "C"
#endif
void launchFakeDataKernel(float* pdata, int size, float num);
#ifdef __cplusplus
extern "C"
#endif
void launchFakeDataKernelnspike(int* pcum, int size, int num);
#ifdef __cplusplus
extern "C"
#endif
void launchFakeDataKernel2d(float* pdata, size_t pitch,int size, int width, float num);
#ifdef __cplusplus
extern "C"
#endif
void launchFakeDataKernelpax(float* pax, int n_spikes, size_t pitch, int n_pos, int n_tbin, float num);
#ifdef __cplusplus
extern "C"
#endif
void launchFakeDataKernelshf(int* pdata, size_t pitch,int size,int width);
#ifdef __cplusplus
extern "C"
#endif
int launch_ll_kernel(float* pax, float* minus_offset, float* prob, int* shf, int* cum_spikes, size_t g_pitch, int n_pos, int n_bin, int max_spikes, int n_shuffle);
#ifdef __cplusplus
extern "C"
#endif
int launch_mo_kernel(float* pix, float* lx, float* minus_offset, int n_spikes, size_t g_pitch, int n_pos, float bin_size);

#ifdef __cplusplus
extern "C"
#endif
int launch_update_shuffle_idx_kernel(int* shf_idx, size_t s_pitch,int bin_idx, int n_spikes, int n_group, int n_shuffle);
#ifdef __cplusplus
extern "C"
#endif
int launchChkMemInt(int* p_, size_t pitch, int width, int size);
#ifdef __cplusplus
extern "C"
#endif
int launchChkPax(float* pax, int n_idx, int head, int size, int n_pos, size_t pitch);
#ifdef __cplusplus
extern "C"
#endif
int launch_ll_bin_kernel(float* pax, float* p_mo, float* offset, float* prob, int* shf, size_t g_pitch, size_t s_pitch, int n_pos, int n_spikes, int bin_idx, int head, int size,int n_shuffle);
#ifdef __cplusplus
extern "C"
#endif
int launch_normalize_kernel(float* prob, size_t g_pitch, int n_pos, int bin_idx, int n_shuffle);

#ifdef __cplusplus
extern "C"
#endif
int launch_rwd_kernel(float* prob, float* dsmat, int n_shuffle, size_t g_pitch, int n_pos, int n_bin, float* rwd, float dmax, int bin_idx, int bin_buf_sz);
#endif
