import numpy as np
import scipy as sp
import scipy.special
import scipy.stats

import abc

from mixed_kde_cython import evaluate_mixed_kde, evaluate_mixed_kde_grid
from fklab.gmmcompression.fkmixture import MixtureClass as Mixture
import fklab.segments as seg

def scaling_factor_trunc_gauss( h, cutoff=0.05 ):
    """Returns scaling factor for trunated gaussian kernel.
    
    Parameters
    ----------
    h : 1D array-like
        vector of standard deviations
    cutoff : float
        probability to cut off
    
    Returns
    -------
    scale : float
        scaling factor
    cutoff: float
        cut off in standard deviations
    
    """
    h = np.array( h, copy=False ).ravel()
    n = len(h)
    s = (2.0*np.pi)**(-n/2.0)
    s /= np.product(h)
    s /= (1-cutoff)**n
    cutoff = -sp.stats.norm.ppf( cutoff/2.0 ) #cutoff in st.dev.
    return s, cutoff

def scaling_factor_epanechnikov( h ):
    """Returns scaling factor for epanechnikov kernel.
    
    Parameters
    ----------
    h : 1D array-like
        vector of bandwidths
    
    Returns
    -------
    scale : float
        scaling factor
    
    """
    h = np.array( h, copy=False).ravel()
    n = len(h)
    s = ( np.pi**(n/2.0) ) / sp.special.gamma( n/2.0 + 1 )
    s = (n/2.0 + 1)/s
    s /= np.product(h)
    return s

def scaling_factor_box( h ):
    """Returns scaling factor for box kernel.
    
    Parameters
    ----------
    h : 1D array-like
        vector of bandwidths
    
    Returns
    -------
    scale : float
        scaling factor
    
    """
    h = np.array( h, copy=False).ravel()
    n = len(h)
    s = ( np.pi**(n/2.0) ) / sp.special.gamma( n/2.0 + 1 )
    s = 1.0/s
    s /= np.product(h)
    return s

def scaling_factor_trunc_vonmises( kappa, cutoff=0.05 ):
    """Returns scaling factor for truncated Von Mises kernel.
    
    Parameters
    ----------
    kappa : 1D array-like
        vector of circular concentrations
    cutoff : float
        probability to cut off
    
    Returns
    -------
    scale : float
        scaling factor
    cutoff: float
        cut off in radians
    
    """
    kappa = np.array( kappa )
    s = 1.0 / np.product( 2*np.pi*sp.special.i0(kappa) )
    
    #adjust for cutoff
    #cutoff is the total probability at both tails to cut off
    if cutoff>0:
        cutoff_angle = sp.stats.vonmises.ppf(cutoff/2.0,kappa)
        s /= (1-cutoff)**kappa.size
    else:
        cutoff_angle = np.ones( kappa.shape ) * -np.pi
    
    return s,cutoff_angle

#density class minimum interface:
# property: ndim
# method: evaluate( points )
# method: evaluate_grid( grid, points )
# method: evaluate_marginal( points, dim )

class DensityBase(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def ndim(self):
        pass
    
    @abc.abstractproperty
    def nsamples(self):
        pass
    
    @abc.abstractmethod
    def evaluate(self, points):
        pass
    
    @abc.abstractmethod
    def evaluate_grid(self, grid, points):
        pass
    
    @abc.abstractmethod
    def evaluate_marginal(self, points, dim):
        pass

class MergingCompressionDensity(DensityBase):
    
    def __init__(self, samples=None, ndim=None, sample_covariance = 1.0, method='full', threshold=1.0):
        #samples: (nsamples,ndim) array
        
        if samples is None and ndim is None:
            ndim = 1 #by default one dimensional density
            nsamples = 0
        elif ndim is None:
            samples = np.array( samples )
            if samples.ndim<2:
                samples = samples.reshape( (samples.size,1) )
            elif samples.ndim>2:
                raise ValueError
            nsamples, ndim = samples.shape
        else:
            nsamples = 0
        
        self._ndim = ndim
        
        self._mixture = Mixture(ndim=ndim)
        
        self.method = method
        self.threshold = threshold
        self.sample_covariance = sample_covariance
        
        self._total_sum_weights = 0.0
        self._total_sum_nsamples = 0.0
        
        if nsamples>0:
            self.addsamples( samples )
    
    def addsamples(self, samples):
        
        samples = np.array( samples )
        if samples.ndim<2:
            samples = samples.reshape( (samples.size,1) )
        elif samples.ndim>2:
            raise ValueError
        nsamples, ndim = samples.shape
        
        if ndim!=self._ndim:
            raise ValueError
        
        samples = np.ascontiguousarray( samples )
        
        if self._method in ['full','bandwidth','constant']:
            self._mixture.merge_samples( samples, threshold=self._threshold, covariance_match = self._method )
        else:
            self._mixture.add_samples( samples )
    
    def evaluate(self, points ):
        
        return self._mixture.evaluate( np.ascontiguousarray(points) )
    
    def evaluate_grid( self, grid, points ):
        
        return self._mixture.evaluate_grid(  np.ascontiguousarray(grid), np.ascontiguousarray(points) )
    
    def evaluate_grid_experimental( self, grid, points ):
        
        grid_acc = self._mixture.build_grid_accumulator( np.ascontiguousarray(grid) )
        return self._mixture.evaluate_grid_experimental_multi( grid_acc, np.ascontiguousarray(points) )
    
    def evaluate_marginal(self, points, dims):
        m = self._mixture.marginalize( dims )
        x = m.evaluate( points )
        return x
    
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self,value):
        value = str(value)
        if value not in ['none','full','bandwidth','constant']:
            raise ValueError
        
        self._method = value
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self,value):
        value = float(value)
        if value<0:
            raise ValueError
        
        self._threshold = value
    
    @property
    def ndim(self):
        return self._mixture.ndim
    
    @property
    def nsamples(self):
        return self._mixture.sum_n
    
    @property
    def sample_covariance(self):
        #TODO: add method to get sample covariance as numpy array to mixture class
        return self._sample_covariance
    
    @sample_covariance.setter
    def sample_covariance(self, value):
        value = np.asarray( value, dtype=np.float64 )
        if value.size==1:
            value = float(value) * np.diag( np.ones( self._ndim ) )
        elif value.ndim==1 and len(value)==self._ndim:
            value = np.diag( value )
        elif value.ndim==2 and value.shape[0]==self._ndim and value.shape[1]==self._ndim:
            pass
        else:
            raise ValueError
        
        self._mixture.set_sample_covariance( value )
        
        self._sample_covariance = value

class MixedKDE(DensityBase):
    
    _kernel_list = ['gaussian','epanechnikov','vonmises','box','delta']
    _datatype_kernel_map = {'linear':('gaussian','epanechnikov','box'),
                            'circular':('vonmises',),
                            'categorical':('delta',) }
    
    def __init__(self, data, labels=None, datatypes=None, kernels=None, bandwidths=None, distance=None, weights=None, cutoff_gauss=0.05, cutoff_vonmises=0.05):
        #data, datatypes are immutable
        
        self._data = np.array( data, dtype=np.float64, copy=False )
        
        if self._data.ndim==1:
            self._data = self._data[None,:]
        elif self._data.ndim!=2:
            raise ValueError
        
        self._data = np.ascontiguousarray(self._data)
        
        self._ndim = self._data.shape[0]
        
        self.labels = labels if labels else ['dim'+str(x) for x in xrange(self._ndim)]
        self._datatypes = self._validate_datatypes( datatypes if datatypes else 'linear' )
        self.kernels = kernels if kernels else [self._datatype_kernel_map[x][0] for x in self.datatypes]
        self.bandwidths = bandwidths if bandwidths else 1
        self.distance = distance
        self.weights = weights if weights else 1
        self.cutoff_gauss = cutoff_gauss
        self.cutoff_vonmises = cutoff_vonmises
        
    @property
    def ndim(self):
        return self._ndim
    
    @property
    def nsamples(self):
        return self._data.shape[1]
    
    @property
    def cutoff_gauss(self):
        return self._cutoff_gauss
    
    @cutoff_gauss.setter
    def cutoff_gauss(self,value):
        value = float(value)
        if value<0:
            raise ValueError
        self._cutoff_gauss = value
    
    @property
    def cutoff_vonmises(self):
        return self._cutoff_vonmises
    
    @cutoff_vonmises.setter
    def cutoff_vonmises(self,value):
        value = float(value)
        if value<0:
            raise ValueError
        self._cutoff_vonmises = value
    
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self,value):
        if not isinstance(value,(tuple,list)):
            value = (value,)
        if len(value)!=self._ndim or not all( [isinstance(x,str) for x in value] ) or len(set(value))!=len(value):
            raise ValueError
        self._labels = tuple(value)
    
    @property
    def datatypes(self):
        return self._datatypes
    
    def _validate_datatypes(self,value):
        if not isinstance(value,(tuple,list)):
            value = (value,)
        if len(value)==1:
            value = value*self._ndim
        if len(value)!=self._ndim or not all( x in ['linear', 'circular', 'categorical'] for x in value ):
            raise ValueError
        return tuple(value)
    
    @property
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self,value):
        if not isinstance(value,(tuple,list)):
            value = (value,)
        if len(value)==1:
            value = value * self._ndim
        #if len(value)!=self._ndim or not all( isinstance(x,DistanceLUT) or x is None for x in value ):
        if len(value)!=self._ndim or not all( x is None or (x.ndim==2 and x.shape[0]==x.shape[1]) for x in value ):
            raise ValueError
        
        value = [ np.zeros((0,0),dtype=np.float64) if x is None else x.astype(np.float64,copy=False) for x in value ]
        
        self._distance = tuple(value)
    
    @property
    def bandwidths(self):
        return self._bandwidths
    
    @bandwidths.setter
    def bandwidths(self,value):
        value = np.array(value,dtype=np.float64,copy=False).ravel()
        
        if len(value)==1:
            value = value.repeat(self._ndim)
        
        if len(value)!=self._ndim or not np.all( value>0 ):
            raise ValueError
        
        self._bandwidths = value
    
    @property
    def kernels(self):
        return tuple(self._kernel_list[x] for x in self._kernels)
    
    @kernels.setter
    def kernels(self,value):
        if isinstance(value,str):
            value = [self._kernel_list.index(value)]
        elif isinstance(value,(tuple,list)):
            value = [ self._kernel_list.index(x) if isinstance(x,str) else self._kernel_list[x] for x in value]
        
        value = np.array( value, dtype=np.int32, copy=False ).ravel()
        
        if len(value)==1:
            value = value.repeat(self._ndim)
        
        if len(value)!=self._ndim or not all( self._kernel_list[x] in self._datatype_kernel_map[y] for x,y in zip(value,self.datatypes) ):
            raise ValueError
        
        self._kernels = value
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self,value):
        if value is None:
            self._weights = np.ones( self._data.shape[1], dtype=np.float64 )/self._data.shape[1]
            return
        
        value = np.array( value, dtype=np.float64, copy=False).ravel()
        if len(value)==1:
            value = value.repeat(self._data.shape[1])
        elif len(value)!=self._data.shape[1]:
            raise ValueError
        
        self._weights = value/np.sum(value)
    
    def _labels2index(self,l):
        if not isinstance(l,(tuple,list)):
            l = [l]
        return [self._labels.index(x) if isinstance(x,str) else int(x) for x in l]
    
    def compute_scaling_factor(self,dim=None):
        #compute scaling factor for all kernel types separately
        if dim is None:
            dim = range(self._ndim)
        else:
            dim = self._labels2index(dim)
        
        kernels = self.kernels
        bandwidths = self.bandwidths
        
        s_gauss, cutoff_gauss = scaling_factor_trunc_gauss( [bw for i,(k,bw) in enumerate( zip(kernels,bandwidths) ) if k == 'gaussian'     and i in dim], cutoff=self._cutoff_gauss )
        s_epa = scaling_factor_epanechnikov( [bw for i,(k,bw) in enumerate( zip(kernels,bandwidths) ) if k == 'epanechnikov' and i in dim] )
        s_vm, cutoff_vm = scaling_factor_trunc_vonmises( [bw for i,(k,bw) in enumerate( zip(kernels,bandwidths) ) if k == 'vonmises'     and i in dim], cutoff=self._cutoff_vonmises )
        s_box = scaling_factor_box( [bw for i,(k,bw) in enumerate( zip(kernels,bandwidths) ) if k == 'box'          and i in dim] )
        
        s = s_gauss * s_epa * s_vm * s_box
        
        return s, cutoff_gauss, cutoff_vm
    
    def evaluate(self, x):
        
        #d = [ np.zeros((0,0),dtype=np.float64) ] * self._ndim #HACK: no support for distance LUTs yet
        s, cutoff_gauss, cutoff_vm = self.compute_scaling_factor()
        return s*evaluate_mixed_kde( self._data, self._weights, x, self._kernels, self._bandwidths, self._distance, cutoff_gauss)
    
    def evaluate_grid(self, x, y, dim=None):
        #d = [ np.zeros((0,0),dtype=np.float64) ] * self._ndim #HACK: no support for distance LUTs yet
        s, cutoff_gauss, cutoff_vm = self.compute_scaling_factor()
        return s*evaluate_mixed_kde_grid( self._data, self._weights, x, y, self._kernels, self._bandwidths, self._distance, cutoff_gauss )
    
    def marginal(self, dim=None ):
        #returns MixedKDE object with selected dimensions
        #MixedKDE( self._data[:,dim], labels=self.labels[dim], datatypes=self.datatypes[dim], kernels=self.kernels[dim], bandwidths=self.bandwidths[dim], distance=self.distance[dim], weights=self.weights)
        raise NotImplementedError
    
    def evaluate_marginal(self, x, dim=None):
        #evaluate marginal directly - making use of already cached dataself.
        index = self._labels2index( dim )
        d = [ np.zeros((0,0),dtype=np.float64) ] * len(index) #HACK: no support for distance LUTs yet
        s, cutoff_gauss, cutoff_vm = self.compute_scaling_factor(dim=index)
        return s*evaluate_mixed_kde( self._data[index,:], self._weights, x, self._kernels[index], self._bandwidths[index], d, cutoff_gauss )
    
    def condition(self, y, dim=None):
        #returns MixedKDE object with remaining dimensions and weighted by K(data,y)
        #w = self.evaluate_marginal( y, dim=dim)
        #m = self.marginal( dim=otherdim )
        #m.weights = m.weights*w
        raise NotImplementedError
    
    def evaluate_condition(self, x, y, dim=None):
        #evaluate conditional directly - making use of already cached data
        raise NotImplementedError

class DistanceLUT:
    def __init__(self, index, lut):
        self._index = index
        self._lut = lut
        #TODO: check index vector and lut array
        
    def value2index(self, x, kind='nearest'):
        return sp.interpolate.interp1d( self._index, range(len(self._index)), kind=kind )(x)
    
    def index2value(self, x):
        return self._index[ np.int(x) ]
    
    @property
    def lut(self):
        return self._lut

class KDEDecoder(object):
    def __init__(self, covariatedata, spikedata, trainingtime, covariategrid, offset=0):
        #covariatedata: MixedKDE object
        #spikedata: List of MixeKDE objects, where covariates are positioned in first dimensions
        #spikedata must have ndim >= covariatedata ndim
        #kernels, bandwidths, datatypes, distance for covariate dimensions should be the same for covariatedata and spikedata
        
        if not isinstance(covariatedata,DensityBase):
            raise ValueError
        
        if isinstance(spikedata, DensityBase):
            spikedata = [spikedata]
        
        if not all( isinstance(x,DensityBase) for x in spikedata ):
            raise ValueError
        
        self._covariatedata = covariatedata
        self._spikedata = spikedata
        self._trainingtime = trainingtime
        self._covariategrid = covariategrid
        self._offset = offset
        
        self._occupancy = self._covariatedata.evaluate( self._covariategrid )
        self._mu = [ x.nsamples/self._trainingtime for x in self._spikedata ]
        
        self._spike_rate = [ x.evaluate_marginal( self._covariategrid, range(covariategrid.shape[0]) ) for x in self._spikedata ]
        self._spike_rate = [self._offset + x*y/self._occupancy for x,y in zip(self._mu,self._spike_rate)]
        
        self._spike_rate_sum = sum( self._spike_rate )
        
    
    def decode_single_bin(self, spikedata=None, binsize=1):
        
        p = [x.evaluate_grid(self._covariategrid,y) if len(y)>0 else np.zeros( (0,len(self._covariategrid)) ) for x,y in zip(self._spikedata,spikedata)]
        p = [np.sum( np.log( x + self._offset*self._occupancy/y ), axis=0 ) for x,y in zip( p,self._mu) ]
        
        nspikes = np.sum( [ x.shape[0] for x in spikedata ] )
        
        P = sum( p )
        
        logposterior = P - np.log(self._occupancy)*nspikes - binsize*self._spike_rate_sum
        
        posterior = np.exp( logposterior - np.max(logposterior) )
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    def decode(self, spikedata, spiketime, bins):
        
        posterior = np.zeros( (len(self._covariategrid),len(bins)), dtype=np.float64 )
        
        bb = [bins.contains( x )[2] for x in spiketime]
        
        binsizes = bins.duration
        
        #iterate over test bins
        for k in range( len(bins) ):
            
            #for each tetrode get spike amp in test bin
            spike_amp_selection = [ x[b[k,0]:(b[k,1]+1)] for x,b in zip(spikedata,bb) ]
            
            #collect decoder.decode( grid, spike_amp, bin_size ) in array
            posterior[:,k] = self.decode_single_bin( spike_amp_selection, binsize=binsizes[k] )
    
        return posterior
    
    def decode2(self, spikedata, spiketime, bins, experimental=False):
        
        #spikedata is list of arrays with size Nspikes x Ndim
        #spiketime is list of vectors of size Nspikes
        #bins is segment like
        
        bins = seg.Segment( bins )
        if bins.hasoverlap():
            raise ValueError("Bins have overlap")
        
        posterior = np.zeros( (self._covariategrid.shape[1], len(bins)), dtype=np.float64 )
        
        total_n = 0.
        
        #loop through each source
        for t,data,density,mu in zip(spiketime,spikedata,self._spikedata,self._mu):
            
            #select all data in requested time bins
            #`inseg` is True(False) for each spike time that is (not) contained within a bin
            #`n` is for each bin the number of spike times it contains
            inseg, n, _ = bins.contains( t )
            
            total_n += n
            
            d = data[:,inseg]
            
            n_total_spikes = np.sum(n)
            cumulative_spikes = np.cumsum(n)
            
            #process spikes in blocks
            maxblocksize = round(5000000. / len(self._covariategrid)) #limit the size of the output array to 5M elements
            
            startbin = 0
            n_processed_spikes = 0
            
            offset = self._offset * self._occupancy / mu
            
            while startbin<len(bins):
                
                #find next bin, such that number of spike from start to end bin is <= maxblocksize
                endbin = np.searchsorted( cumulative_spikes, maxblocksize + n_processed_spikes, 'right' )
                
                start_spike_index = n_processed_spikes
                end_spike_index = cumulative_spikes[ endbin - 1 ]
                
                if experimental:
                    p = density.evaluate_grid_experimental( self._covariategrid, np.ascontiguousarray(d[:,start_spike_index:end_spike_index]) )
                else:
                    p = density.evaluate_grid(self._covariategrid, np.ascontiguousarray(d[:,start_spike_index:end_spike_index]))
                
            
                p = np.split( p, cumulative_spikes[startbin:endbin]-n_processed_spikes, axis=0 )
                
                p = [ np.sum( np.log( x + offset ), axis=0 ) if len(x)>0 else 0. for x in p[:-1] ]
                
                for idx,kk in enumerate( range(startbin,endbin) ):
                    posterior[:,kk] += p[idx]
                
                startbin = endbin
                n_processed_spikes = end_spike_index
                
            
        posterior = posterior - np.log(self._occupancy[:,None])*total_n[None,:] - self._spike_rate_sum[:,None]*bins.duration[None,:]
        
        posterior = np.exp( posterior - np.max(posterior,axis=0) )
        posterior = posterior / np.sum(posterior,axis=0)
        
        return posterior

