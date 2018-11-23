import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t

cdef extern void evaluate_mixed_kde_c( double *data, double *weights, double *x, int32_t *kernels, double *bandwidths, double **distance, int32_t* nlut, int32_t ndim, int32_t npoints, int32_t nx, double cutoff_gauss, double *output )
cdef extern void evaluate_mixed_kde_grid_c( double *data, double *weights, double *x, double *y, int32_t *kernels, double *bandwidths, double **distance, int32_t* nlut, int32_t ndimx, int32_t ndimy, int32_t npoints, int32_t nx, int32_t ny, double cutoff_gauss, double *output )

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn of the wrap around for entire function
def evaluate_mixed_kde_grid( np.ndarray[double, ndim=2, mode="c"] data, np.ndarray[double, ndim=1, mode="c"] w, np.ndarray[double, ndim=2, mode="c"] x, np.ndarray[double, ndim=2, mode="c"] y, np.ndarray[int32_t, ndim=1, mode="c"] kernels, np.ndarray[double, ndim=1, mode="c"] bandwidths, distance, double cutoff_gauss=4.0 ):
    npoints = data.shape[1]
    cdef int32_t nx = x.shape[1]
    cdef int32_t ny = y.shape[1]
    cdef int32_t ndim = data.shape[0]
    cdef int32_t ndimx = x.shape[0]
    cdef int32_t ndimy = y.shape[0]
    
    assert ndimx+ndimy==ndim
    
    cdef np.ndarray[int32_t, ndim=1, mode="c"] nlut
    cdef np.ndarray[double, ndim=2, mode="c"] z
    
    nlut = np.array( [ len(dst) for dst in distance ], dtype=np.int32 )
    
    z = np.zeros( (ny,nx), dtype=np.float64, order='C' )
    
    cdef double **D = <double **>malloc(ndim * sizeof(double*))
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] dist
    
    for k in range(ndim):
        if nlut[k]>0:
            dist = distance[k]
            D[k] = <double*> dist.data
    
    evaluate_mixed_kde_grid_c( &data[0,0], &w[0], &x[0,0], &y[0,0], &kernels[0], &bandwidths[0], D, &nlut[0], ndimx, ndimy, npoints, nx, ny, cutoff_gauss, &z[0,0] )
    
    free(D)
    
    return z

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn of the wrap around for entire function
def evaluate_mixed_kde( np.ndarray[double, ndim=2, mode="c"] data, np.ndarray[double, ndim=1, mode="c"] w, np.ndarray[double, ndim=2, mode="c"] x, np.ndarray[int32_t, ndim=1, mode="c"] kernels, np.ndarray[double, ndim=1, mode="c"] bandwidths, distance, double cutoff_gauss=4.0 ):
    
    cdef int32_t npoints = data.shape[1]
    cdef int32_t nx = x.shape[1]
    cdef int32_t ndim = data.shape[0]
    
    cdef np.ndarray[int32_t, ndim=1, mode="c"] nlut
    cdef np.ndarray[double, ndim=1, mode="c"] z
    
    nlut = np.array( [ len(y) for y in distance ], dtype=np.int32 )
    
    z = np.zeros( nx, dtype=np.float64, order='C' )
    
    #TODO: create D as a list of pointers to the distance matrices
    cdef double **D = <double **>malloc(ndim * sizeof(double*))
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] dist
    
    for k in range(ndim):
        if nlut[k]>0:
            dist = distance[k]
            D[k] = <double*> dist.data
    
    evaluate_mixed_kde_c( &data[0,0], &w[0], &x[0,0], &kernels[0], &bandwidths[0], D, &nlut[0], ndim, npoints, nx, cutoff_gauss, &z[0] )
    
    free(D)
    
    return z

