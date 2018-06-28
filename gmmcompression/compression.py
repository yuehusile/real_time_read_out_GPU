
from fkmixture import MixtureClass as Mixture
import math

def compute_clustered_model(data,covariance=None,samplesize=100,threshold=0.05,model=None,weighted_hellinger=True):
    ndata,ndim = data.shape
    N = int(math.floor(ndata/samplesize))
    
    if covariance is None:
        covariance = np.diag( np.ones( ndim, dtype=np.float ) )
    
    if model is None:
        model = Mixture(ndim)
    
    model.set_sample_covariance( covariance ) 
    
    for k in xrange(N):
        model.add_samples( data[k*samplesize:(samplesize+k*samplesize)])
        model = model.compress( threshold=threshold, weighted_hellinger=weighted_hellinger )
    
    return model

def compute_compressed_model(data,covariance=None,threshold=2,model=None,covariance_match = 'full'):
    ndata,ndim = data.shape
    
    if covariance is None:
        covariance = np.diag( np.ones( ndim, dtype=np.float ) )
    
    if model is None:
        model = Mixture(ndim)
    
    model.set_sample_covariance( covariance ) 
    
    model.merge_samples( data, threshold=threshold, covariance_match = covariance_match )
    
    return model
