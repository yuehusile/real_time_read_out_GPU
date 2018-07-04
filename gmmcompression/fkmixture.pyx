import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from libc.stdint cimport uint32_t, uint16_t, uint8_t
from libcpp cimport bool as bool_t

import pdb

cdef extern from "covariance.h":
    ctypedef struct CovarianceMatrix:
        uint32_t refcount
        uint16_t ndim
    
    CovarianceMatrix* covariance_create( uint16_t ndim )
    void covariance_delete( CovarianceMatrix* cov )
    
    void covariance_set_full( CovarianceMatrix* cov, double* values )
    void covariance_print( CovarianceMatrix* cov )
    
cdef extern from "component.h":
    ctypedef struct GaussianComponent:
        uint32_t refcount
        uint16_t ndim
        double weight
        double *mean
        CovarianceMatrix *covariance
    
    GaussianComponent* component_create_empty( uint16_t ndim )
    void component_delete( GaussianComponent* component )
    void component_get_mean( GaussianComponent* component, double* mean)
    void component_get_covariance_array( GaussianComponent* component, double* covariance)

cdef extern from "mixture.h" nogil:
    ctypedef struct Mixture:
        uint32_t refcount
        double sum_n
        double sum_weights
        uint16_t ndim
        uint32_t ncomponents
        uint32_t buffersize
        GaussianComponent **components
        CovarianceMatrix *samplecovariance
        
    Mixture* mixture_create( int )
    void mixture_delete( Mixture* mixture)
    
    int mixture_save_to_file( Mixture* mixture, char* filename )
    Mixture* mixture_load_from_file( char* filename )
    void mixture_update_cache( Mixture* mixture )    
    
    void mixture_set_samplecovariance( Mixture* mixture, CovarianceMatrix* cov)
    CovarianceMatrix* mixture_get_samplecovariance( Mixture* mixture)
    void mixture_addsamples( Mixture* mixture, double* means, uint32_t nsamples, uint16_t ndim )
    
    void mixture_evaluate( Mixture* mixture, double* points, uint32_t npoints, double* result)
    void mixture_evaluategrid( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* points, uint32_t npoints, uint16_t npointdim, uint16_t* pointdim, double* result )
    
    void mixture_evaluate_diagonal( Mixture* mixture, double* points, uint32_t npoints, double* result)
  
    void mixture_evaluategrid_diagonal( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* testpoint, uint16_t npointsdim, uint16_t* pointsdim, double* output )
    void mixture_prepare_grid_accumulator( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* grid_acc )
    void mixture_evaluategrid_diagonal_multi( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output )
    
    Mixture* mixture_compress( Mixture* mixture, double threshold, uint8_t weighted_hellinger )
    void mixture_merge_samples( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    void mixture_merge_samples_constant_covariance( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    void mixture_merge_samples_match_bandwidth( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    
    void mixture_get_means( Mixture* mixture, double* means )
    void mixture_get_scaling_factors( Mixture* mixture, double* scales )
    void mixture_get_weights( Mixture* mixture, double* weights )
    void mixture_get_covariances( Mixture* mixture, double* covariances )
    
    double compute_distance( GaussianComponent* c1, GaussianComponent* c2 )
    void moment_match( GaussianComponent** components, uint32_t ncomponents, GaussianComponent* output, uint8_t normalize )
    
    Mixture* mixture_marginalize( Mixture* mixture, uint16_t ndim, uint16_t* dims )
    
    double hellinger_single( GaussianComponent** components, uint32_t n, GaussianComponent* model, uint8_t weighted)

cdef extern from "../gpu_decoder/gpu_kde.hpp" nogil:
    cdef cppclass GpuDecoder:
        GpuDecoder(int gridDim, int maxBatchSize) except +
        int n_tt()
        int maxBatchSize()
        int batchSize()
        int gridDim()
        int component_acc_block_size()
        int evaluation_block_size()
        void setBatchSize(const int bs)
        void set_component_acc_block_size(const int bs)
        void set_evaluation_block_size(const int bs)
        int addTT(Mixture* m, double*grid_acc, int spikeDim, uint16_t* pointDim)
        int decodeTT(const int tt_idx, double* spikes, const int n_spikes, double* results)
        # to be done decode all
        double result(const int tt_idx, const int spike_idx, const int grid_idx)
        void clear()

cdef extern from "../gpu_decoder/gpu_kde.hpp" nogil:
    cdef cppclass SignificanceAnalyzer:
        SignificanceAnalyzer(const int n_pos, const int n_group, const float bin_size, const int n_shf, const int n_tbin,const int max_sp) except +
        void uploadParam(double* pix, double* lx)
        void updateBin(double* pax, int* n_spikes_g, double* mu, int n_spikes, int n_group )
        bool_t getProb(double* prob)
        void clear()

cdef class CovarianceClass:
    cdef CovarianceMatrix* _c_covariance
    def __init__(self, np.ndarray[double, ndim=2, mode="c"] data):
        nr = data.shape[0]
        nc = data.shape[1]
        assert nr==nc
        assert nr>0
        
        self._c_covariance = covariance_create(nr)
        covariance_set_full( self._c_covariance, &data[0,0] )
        
        if self._c_covariance is NULL:
            raise MemoryError
    
    def __dealloc__(self):
        if self._c_covariance is not NULL:
            #print("deleting covariance matrix")
            covariance_delete(self._c_covariance)
    
    def show(self):
        covariance_print( self._c_covariance )
    
    @property
    def refcount(self):
        return self._c_covariance.refcount
    
    @property
    def ndim(self):
        return self._c_covariance.ndim
    
        
cdef class ComponentClass:
    cdef GaussianComponent* _c_component
    def __init__(self, ndim=1):
        self._c_component = component_create_empty(ndim)
        if self._c_component is NULL:
            raise MemoryError
    
    def __dealloc__(self):
        if self._c_component is not NULL:
            #print("deleting component")
            component_delete(self._c_component)
    
    @property
    def refcount(self):
        return self._c_component.refcount
    
    @property
    def ndim(self):
        return self._c_component.ndim
    
    @property
    def weight(self):
        return self._c_component.weight
    
    @property
    def mean(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ndim), dtype=np.float64, order="C" )
        component_get_mean(self._c_component, &result[0] )
        return result
    
    @property
    def covariance(self):
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.empty( (self.ndim,self.ndim), dtype=np.float64, order="C" )
        component_get_covariance_array(self._c_component, &result[0,0] )
        return result
    


cdef class MixtureClass:
    cdef Mixture* _c_mixture
    def __cinit__(self, ndim=1):
        if ndim==0:
            self._c_mixture = NULL
        else:
            self._c_mixture = mixture_create(ndim) 
            if self._c_mixture is NULL:
                raise MemoryError
        
    def __dealloc__(self):
        if self._c_mixture is not NULL:
            #print("deleting mixture")
            mixture_delete(self._c_mixture)
    
    def marginalize( self, np.ndarray[uint16_t, ndim=1, mode="c"] dims ):
        result = MixtureClass(ndim=0)
        result._c_mixture = mixture_marginalize( self._c_mixture, len(dims), &dims[0] )
        return result
    
    def hellinger( self, ComponentClass c, index=0, n=0, weighted=1):
        index = int(index)
        n = int(n)
        weighted = int(weighted)
        
        assert index>=0 and index<self.ncomponents
        
        if n==0:
            n = self.ncomponents - index
        
        assert n>0 and n<=(self.ncomponents-index)
        
        assert isinstance( c, ComponentClass )
        
        #cdef GaussianComponent* comp = c._c_component
        
        return hellinger_single( &self._c_mixture.components[index], n, c._c_component, weighted )
        
    
    def distance(self, c1, c2 ):
        c1 = int(c1)
        c2 = int(c2)
        
        assert c1>=0 and c1<self.ncomponents
        assert c2>=0 and c2<self.ncomponents
        
        return compute_distance( self._c_mixture.components[c1], self._c_mixture.components[c2] )
    
    def moment_match_components(self, index=0, n=0, normalize=1 ):
        
        index = int(index)
        n = int(n)
        normalize = int(normalize)
        
        assert index>=0 and index<self.ncomponents
        
        if n==0:
            n = self.ncomponents - index
        
        assert n>0 and n<=(self.ncomponents-index)
        
        c = ComponentClass( self.ndim )
        
        moment_match( &self._c_mixture.components[index], n, c._c_component, normalize )
        
        return c
    
    def set_sample_covariance(self, np.ndarray[double, ndim=2, mode="c"] data):
        
        c = CovarianceClass( data )
        mixture_set_samplecovariance( self._c_mixture, c._c_covariance )
        del c
    
    def add_samples(self, np.ndarray[double, ndim=2, mode="c"] means ):
        
        nsamples = means.shape[0]
        ndim = means.shape[1]
        
        mixture_addsamples( self._c_mixture, &means[0,0], nsamples, ndim )
    
    def compress(self,threshold=0.01, weighted_hellinger=True):
        
        m = mixture_compress( self._c_mixture, threshold, weighted_hellinger )
        result = MixtureClass(ndim=0)
        result._c_mixture = m
        
        return result
    
    def merge_samples( self, np.ndarray[double, ndim=2, mode="c"] samples, threshold=1.0, covariance_match = 'full'):
        nsamples = samples.shape[0]
        ndim = samples.shape[1]
        
        assert ndim==self.ndim
        
        if covariance_match == 'constant':
            print "Not there"
#            mixture_merge_samples_constant_covariance( self._c_mixture, &samples[0,0], nsamples, threshold )
        elif covariance_match == 'bandwidth':
            mixture_merge_samples_match_bandwidth( self._c_mixture, &samples[0,0], nsamples, threshold )
        else: #'full'
            mixture_merge_samples( self._c_mixture, &samples[0,0], nsamples, threshold )
    
    
    def evaluate(self, np.ndarray[double, ndim=2, mode="c"] x, diagonal=False):
        
        npoints = x.shape[0]
        ndim = x.shape[1]
        
        assert ndim==self.ndim
        
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.zeros( npoints, dtype=np.float64, order='C' )
        
        if npoints<1:
            return result
        
        if diagonal:
            mixture_evaluate_diagonal( self._c_mixture, &x[0,0], npoints, &result[0])
        else:
            mixture_evaluate( self._c_mixture, &x[0,0], npoints, &result[0])
        
        return result
        
    def build_grid_accumulator(self, np.ndarray[double, ndim=2, mode="c"] grid):
        
        ngrid = grid.shape[0]
        ngriddim = grid.shape[1]
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] griddim
        griddim = np.arange( ngriddim, dtype=np.uint16 )
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] grid_acc
        grid_acc = np.zeros( (self.ncomponents,ngrid), dtype=np.float64, order="C" )
        
        mixture_prepare_grid_accumulator( self._c_mixture, &grid[0,0], ngrid, ngriddim, &griddim[0], &grid_acc[0,0] )
        
        return grid_acc
    
    def evaluate_grid_multi(self, np.ndarray[double, ndim=2, mode="c"] grid_acc, np.ndarray[double, ndim=2, mode="c"] x ):
       
        cdef int ngrid
        ngrid = grid_acc.shape[1]
        
        cdef int npoints
        npoints = x.shape[0]
        cdef int npointsdim
        npointsdim = x.shape[1]
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.zeros( (npoints,ngrid), dtype=np.float64, order="C" )
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] pointsdim
        pointsdim = np.arange( self.ndim-npointsdim, self.ndim, dtype=np.uint16 )

        with nogil:
            mixture_evaluategrid_diagonal_multi( self._c_mixture, &grid_acc[0,0], ngrid, &x[0,0], npoints, npointsdim, &pointsdim[0], &result[0,0] )
        
        return result
    
    def evaluate_grid(self, np.ndarray[double, ndim=2, mode="c"] grid_acc, np.ndarray[double, ndim=2, mode="c"] x):
        
        ngrid = grid_acc.shape[1]
        npointsdim = x.shape[1]
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.zeros( (1,ngrid), dtype=np.float64, order="C" )
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] pointsdim
        pointsdim = np.arange( self.ndim-npointsdim, self.ndim, dtype=np.uint16 )
        
        mixture_evaluategrid_diagonal( self._c_mixture, &grid_acc[0,0], ngrid, &x[0,0], npointsdim, &pointsdim[0], &result[0,0] )
        
        return result
    
    def update_cache( self ):
        mixture_update_cache( self._c_mixture )
    
    @property
    def ncomponents(self):
        return self._c_mixture.ncomponents
    
    @property
    def ndim(self):
        return self._c_mixture.ndim
    
    @property
    def refcount(self):
        return self._c_mixture.refcount
    
    @property
    def sum_n(self):
        return self._c_mixture.sum_n
    
    @property
    def sum_weights(self):
        return self._c_mixture.sum_weights
    
    @property
    def means(self):
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.empty( (self.ncomponents,self.ndim), dtype=np.float64, order="C" )
        mixture_get_means( self._c_mixture, &result[0,0] )
        return result
    
    @property
    def scaling_factors(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ncomponents), dtype=np.float64, order="C" )
        mixture_get_scaling_factors( self._c_mixture, &result[0] )
        return result
    
    @property
    def weights(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ncomponents), dtype=np.float64, order="C" )
        mixture_get_weights( self._c_mixture, &result[0] )
        return result
    
    @property
    def covariances(self):
        cdef np.ndarray[np.double_t, ndim=3, mode="c"] result
        result = np.empty( (self.ncomponents, self.ndim, self.ndim), dtype=np.float64, order="C" )
        mixture_get_covariances( self._c_mixture, &result[0,0,0] )
        return result
   
    def tofile(self, filename):
        return mixture_save_to_file( self._c_mixture, filename )
    
    @classmethod
    def fromfile(cls, filename):
        result = MixtureClass(ndim=0)
        result._c_mixture = mixture_load_from_file( filename )
        # update scaling factors
        result.update_cache()        
        
        return result
    
cdef class GpuDecoderClass:
    cdef GpuDecoder* _cpp_gpudecoder
    def __cinit__(self, int gridDim, int maxBatchSize=8192):
        _cpp_gpudecoder = new GpuDecoder(gridDim, maxBatchSize)
        self._cpp_gpudecoder = _cpp_gpudecoder

    def __deaclloc__(self):
        del self._cpp_gpudecoder

    @property
    def n_tt(self):
        return self._cpp_gpudecoder.n_tt()

    @property
    def maxBatchSize(self):
        return self._cpp_gpudecoder.maxBatchSize()

    @property
    def batchSize(self):
        return self._cpp_gpudecoder.batchSize()

    @property
    def gridDim(self):
        return self._cpp_gpudecoder.gridDim()

    @property
    def component_acc_block_size(self):
        return self._cpp_gpudecoder.component_acc_block_size()

    @property
    def evaluation_block_size(self):
        return self._cpp_gpudecoder.evaluation_block_size()

    def setBatchSize(self, int bs):
        return self._cpp_gpudecoder.setBatchSize(bs)

    def set_component_acc_block_size(self, int bs):
        return self._cpp_gpudecoder.set_component_acc_block_size(bs)

    def set_evaluation_block_size(self, int bs):
        return self._cpp_gpudecoder.set_evaluation_block_size(bs)

    def addTT(self, MixtureClass mixture,\
              np.ndarray[double, ndim=2, mode="c"] grid_acc,\
              n_spikes_features, gridDim):
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] pointsdim
        pointsdim = np.arange( gridDim, n_spikes_features+gridDim, dtype=np.uint16 )
        return self._cpp_gpudecoder.addTT(mixture._c_mixture,&grid_acc[0,0],n_spikes_features,&pointsdim[0])

    def decodeTT(self, int tt_idx,\
                 np.ndarray[double, ndim=2, mode="c"] spikes,\
                 int n_spikes):
        cdef int ret
        ngrid = self.gridDim

        cdef np.ndarray[double, ndim=2, mode="c"] results = np.ones((n_spikes, ngrid),dtype=np.float64, order="C") 
        if n_spikes>0:
            with nogil:
                ret = self._cpp_gpudecoder.decodeTT(tt_idx,&spikes[0,0],n_spikes,&results[0,0])
        return results
    
    def clearMem(self):
        self._cpp_gpudecoder.clear()
        print "memory cleared"

cdef class SignificanceAnalyzerClass:
    cdef SignificanceAnalyzer* _cpp_significanceanalyzer
    cdef int n_pos
    cdef int n_shuffle
    cdef int ngrid

    def __cinit__(self, int n_pos, int n_group, float bin_size, int n_shf=1000, int n_tbin=10,int max_sp=100):
        _cpp_significanceanalyzer = new SignificanceAnalyzer(n_pos,n_group,bin_size,n_shf,n_tbin,max_sp)
        self._cpp_significanceanalyzer = _cpp_significanceanalyzer
        self.n_pos = n_pos
        self.n_shuffle = n_shf
        self.n_pos = n_pos
    
    def uploadParam(self, np.ndarray[double, ndim=1, mode="c"] pix,np.ndarray[double, ndim=2, mode="c"] lx):
        self._cpp_significanceanalyzer.uploadParam(&pix[0],&lx[0,0])
    
    def updateBin(self, pax, np.ndarray[int, ndim=1, mode="c"] n_spikes_g, np.ndarray[double, ndim=1, mode="c"] mu):

        n_spikes = len(pax)*len(pax[0])
        cdef np.ndarray[double, ndim=2, mode="c"] pax2 = np.zeros((n_spikes, self.n_pos),dtype=np.float64, order="C")
        #print "n_spikes_g={}".format(n_spikes_g)
        idx = 0
        for i in range(len(pax)):
            for j in range(len(pax[i])):
                #print "i={},j={}".format(i,j)
                #print "pax[{}]={}".format(i,pax[i])
                pax2[idx] = pax[i][j]
                idx = idx + 1
        #print "pax2={}".format(pax2)
        self._cpp_significanceanalyzer.updateBin(&pax2[0,0],&n_spikes_g[0],&mu[0],len(pax[0]),len(pax))
        cdef np.ndarray[double, ndim=2, mode="c"] prob = np.ones((self.n_shuffle, self.n_pos),dtype=np.float64, order="C") 
        ready = self._cpp_significanceanalyzer.getProb(&prob[0,0])
        
        return prob,ready
        #print len(pax2)
        #print pax[11][34]
        #print pax2[419]

    def __deaclloc__(self):
        del self._cpp_significanceanalyzer
  
