import numpy as np
import math

LOG2PI = 1.83787706640935

def compute_compressed_model(data,covariance=None,samplesize=500,costThreshold=0.01):
    ndata,ndim = data.shape
    N = int(math.floor(ndata/samplesize))
    
    if covariance is None:
        covariance = np.array( [[1,0],[0,1]], dtype=np.float )
    
    model = compressed_model()
    
    for k in xrange(N):
        model.add_samples_and_compress( means=data[k*samplesize:(samplesize+k*samplesize)], covariances=covariance, costThreshold=costThreshold)
    
    return model

class compressed_model(object):
    sum_weights = 0.0 #running sum of weights of all samples
    sum_n = 0 #running sum of number of samples
        
    def __init__(self):
        
        self.weights = None
        self.means = None
        self.covariances = None
        
        self._cache_iS = []
        self._cache_logConstant = []
        self._cache_sigma_points = []
        self._cache_icov = []
    
    def add_samples(self, means=None, covariances=None, weights=None):
        if means is None:
            return
        
        ndim = self.ndim
        
        means = np.asarray(means, dtype=np.float)
        if means.ndim!=2 or (ndim is not None and means.shape[1]!=ndim):
            raise ValueError
        
        ncomp,ndim = means.shape
        
        if covariances is None:
            cov = np.diag( np.zeros( ndim ) )
        else:
            cov = np.asarray(covariances,dtype=np.float)
            if cov.ndim<2 or cov.ndim>3 or (cov.ndim==2 and cov.shape!=(ndim,ndim)) or (cov.ndim==3 and cov.shape[0]!=ncomp and cov.shape[1:]!=(ndim.ndim)):
                raise ValueError
        
        if weights is None:
            weights = np.array( [1.0] )
        else:
            weights = np.asarray( weights, dtype=np.float ).ravel()
            if weights.ndim!=1 or (len(weights)!=1 and len(weights)!=ncomp):
                raise ValueError
        
        if len(weights)==1:
            weights = weights * np.ones( ncomp )
        
        attenuation = 1.0
        
        attenuated_sum_weights = self.sum_weights * attenuation
        sum_sample_weights = np.sum(weights)
        self.sum_weights = attenuated_sum_weights + sum_sample_weights
        
        attenuated_sum_n = self.sum_n * attenuation
        self.sum_n = attenuated_sum_n + ncomp
        
        mixing_factor_old = attenuated_sum_weights/self.sum_weights
        mixing_factor_new = sum_sample_weights/self.sum_weights
        
        weights = weights / sum_sample_weights       
        
        if cov.ndim==2:
            icov = np.linalg.inv( cov )
            iS = np.linalg.cholesky( icov )
            logConstant = np.sum( np.log( np.diag( iS ) ) ) - 0.5*ndim*LOG2PI
            
            self._cache_iS.extend( [iS]*ncomp )
            self._cache_logConstant.extend( [logConstant]*ncomp )
            self._cache_icov.extend( [icov]*ncomp )
            
            cov = np.repeat( cov.reshape((1,ndim,ndim)), repeats=ncomp, axis=0 )
            
        else:
            for idx in xrange(ncomp):
                icov = np.linalg.inv( cov[idx] )
                iS = np.linalg.cholesky( icov )
                logConstant = np.sum( np.log( np.diag( iS ) ) ) - 0.5*ndim*LOG2PI
                
                self._cache_iS.append( iS )
                self._cache_logConstant.append( logConstant )
                self._cache_icov.append( icov )
        
        maxv = 3
        k = max(0,maxv-ndim)
        for idx in xrange(ncomp):
            self._cache_sigma_points.append( get_sigma_points( means[idx], cov[idx], k ) )
        
        if self.ndim is None:
            self.weights = weights
            self.means = means
            self.covariances = cov
        else:
            self.weights = np.concatenate( (self.weights*mixing_factor_old, weights*mixing_factor_new) )
            self.means = np.concatenate( (self.means, means ) )
            self.covariances = np.concatenate( (self.covariances,cov ) )
    
    @staticmethod
    def fromfile(filename):
        #load means, covariances, weights
        #load sum_weights, sum_n
        #load cached values
        
        #create compressed model object
        #m = compressed_model
        #m.means = 
        #m.covariances = 
        #m.weights = 
        #m.sum_weights = 
        #etc.
        
        #return m
        
        raise NotImplementedError
    
    def tofile(filename):
        #save means, covariances, weights
        #save sum_weights, sum_n
        #save cached values
        
        raise NotImplementedError
        
    
    @property
    def ncomponents(self):
        if self.weights is None:
            return 0
        else:
            return len(self.weights)
    
    @property
    def ndim(self):
        if self.means is None:
            return None
        else:
            return self.means.shape[1]
    
    def evaluate(self,x,subset=None):
        
        ndata,ndim = x.shape
        if subset is None:
            comp = range(self.ncomponents)
        else:
            comp = subset
        
        if ndim!=self.ndim:
            raise ValueError
        
        p = np.zeros( ndata )
        
        if ndata==0:
            return p
        
        for idx in comp:
            dx = x - self.means[idx:idx+1,:]
            dx = np.dot(dx,self._cache_iS[idx])
            p = p + self.weights[idx] * np.exp(self._cache_logConstant[idx]-0.5*np.sum(dx*dx,axis=1))
        
        return p
    
    def evaluate_marginal(self,x,dim=None,subset=None):
        ndata,ndim = x.shape
        if subset is None:
            comp = range(self.ncomponents)
        else:
            comp = subset
        
        if dim is None:
            dim = range(self.ndim)
            
        if ndim!=len(dim):
            raise ValueError
        
        p = np.zeros( ndata )
        
        if ndata==0:
            return p
        
        for idx in comp:
            dx = x - self.means[np.ix_([idx],dim)]
            #select dimensions in cached iS
            #iS = self._cache_iS[np.ix_([idx],dim,dim)]
            iS = self._cache_iS[idx][np.ix_(dim,dim)]
            #we have to recompute logConstant - we could cache log(diag(iS))-0.5*LOG2PI instead
            logConstant = np.sum( np.log( np.diag( iS ) ) ) - 0.5*ndim*LOG2PI
            dx = np.dot(dx,iS)
            p = p + self.weights[idx] * np.exp( logConstant-0.5*np.sum(dx*dx,axis=1) )
        
        return p
    
    def add_samples_and_compress(self, means=None, covariances=None, weights=None, costThreshold=0.01):
        self.add_samples( means=means, covariances=covariances, weights=weights )
        self.compress( costThreshold=costThreshold )
    
    def hellinger_distance_single(self, other, subset=None, weighted=False):
        
        if subset is None:
            subset = range(self.ncomponents)
        
        if not weighted:
            w1_sum = np.sum(self.weights[subset])
            w1 = self.weights[subset] / w1_sum
            w2_sum = other[2]
            w2 = other[2] / w2_sum
        else:
            w1 = self.weights[subset]
            w2 = other[2]
        
        w0 = np.concatenate( (w1*0.5,w2*0.5) ,axis=0)
        w0_sum = np.sum(w0)
        w0 = w0 / w0_sum
        
        ndim = self.ndim
        if ndim!=other[0].shape[1]:
            raise ValueError
        
        maxv = 3
        k = max(0,maxv-ndim)
        
        sigma_points = [get_sigma_points( other[0][0], other[1][0], k )]
        sigma_points = np.concatenate( [self._cache_sigma_points[idx] for idx in subset] + sigma_points, axis=0 )
        
        if k==0:
            w = np.ones(2*ndim)/(2*(ndim+k))
        else:
            w = np.ones(2*ndim+1)/(2*(ndim+k))
            w[0] = 1.0*k / (ndim+k)
        
        W = w0[:,None] * w[None,:]
        
        pdf1 = self.evaluate(sigma_points, subset=subset)
        
        ndata,ndim = sigma_points.shape
        pdf2 = np.zeros( ndata )
        
        if ndata>0:
            dx = sigma_points - other[0][0]
            iS = np.linalg.cholesky( np.linalg.inv( other[1][0] ) )
            logConstant = np.sum( np.log( np.diag( iS ) ) ) - 0.5*ndim*LOG2PI
            dx = np.dot(dx,iS)
            pdf2 = pdf2 + w2 * np.exp(logConstant-0.5*np.sum(dx*dx,axis=1))
            
        
        if not weighted:
            pdf1 /= w1_sum
            pdf2 /= w2_sum
        
        pdf0 = (0.5*pdf1+0.5*pdf2)/w0_sum #TODO: check if correct for weighted=False/True
        
        g = (np.sqrt(pdf1)-np.sqrt(pdf2))**2
        H = np.sqrt(np.abs(np.sum(W.ravel()*g/pdf0)/2.0))
        
        return H
    
    def hellinger_distance(self, other, weighted=False):
        
        if not weighted:
            w1_sum = np.sum(self.weights)
            w1 = self.weights / w1_sum
            w2_sum = np.sum(other.weights)
            w2 = other.weights / w2_sum
        else:
            w1 = self.weights
            w2 = other.weights
        
        w0 = np.concatenate( (w1*0.5,w2*0.5) ,axis=0)
        w0_sum = np.sum(w0)
        w0 = w0 / w0_sum
        
        ndim = self.ndim
        if ndim!=other.ndim:
            raise ValueError
        
        maxv = 3
        k = max(0,maxv-ndim)
        
        sigma_points = np.concatenate( self._cache_sigma_points + other._cache_sigma_points, axis=0 )
        
        if k==0:
            w = np.ones(2*ndim)/(2*(ndim+k))
        else:
            w = np.ones(2*ndim+1)/(2*(ndim+k))
            w[0] = 1.0*k / (ndim+k)
        
        W = w0[:,None] * w[None,:]
        
        pdf1 = self.evaluate(sigma_points)
        pdf2 = other.evaluate(sigma_points)
        
        if not weighted:
            pdf1 /= w1_sum
            pdf2 /= w2_sum
        
        pdf0 = (0.5*pdf1+0.5*pdf2)/w0_sum #TODO: check if correct for weighted=False/True
        
        g = (np.sqrt(pdf1)-np.sqrt(pdf2))**2
        H = np.sqrt(np.abs(np.sum(W.ravel()*g/pdf0)/2.0))
        
        return H
    
    def moment_match(self):
        #mu,cov,w = moment_match_gaussian( self.means, self.covariances, self.weights )
        #return compressed_model( weights=w, means=mu, covariances=cov )
        return moment_match_gaussian( self.means, self.covariances, self.weights )
    
    def two_gaussian_approximation(self, subset=None):
        
        if subset is None:
            subset = xrange(self.ncomponents)
            mu, cov, w = self.means, self.covariances, self.weights
        else:
            mu, cov, w = self.means[subset], self.covariances[subset], self.weights[subset]
        
        ncomp,ndim = mu.shape
        
        if ncomp<2:
            return (mu,cov,w), None, np.zeros( ncomp, dtype=np.int )
        elif ncomp==2:
            return (mu[:1],cov[:1],w[:1]), (mu[1:],cov[1:],w[1:]), np.array( [0,1], dtype=np.int )
        
        #to initialize: fit with single Gaussian using moment matching and split in two
        model_mu, model_cov, model_w = moment_match_gaussian( mu, cov, w )
        model_mu, model_cov, model_w = split_gaussian( model_mu, model_cov, model_w )
        
        previous_responsibilities = np.ones(ncomp)
        previous_solutions = []
        
        responsibilities = np.zeros( ncomp, dtype=np.int )
        distances = np.zeros( ncomp, dtype=np.float )
            
        D = np.zeros(2)
        
        done=False
        while not done:
            
            #1. compute responsibilities
            for c,idx in enumerate(subset):
                #icov = np.linalg.inv(mixture_cov[c])
                icov = self._cache_icov[idx]
                for m in xrange(2):
                    #D[m] = _compute_component_distance( model_mu[m], model_cov[m], mixture_mu[c], mixture_cov[c] )
                    delta = (model_mu[m]-mu[c]).reshape( (1,ndim) )
                    cov1 = model_cov[m]
                    cov2 = cov[c]
                    D[m] = 0.5*( np.log( np.linalg.det(cov2) / np.linalg.det(cov1) ) + np.trace( np.dot( icov, cov1 ) ) + np.dot( np.dot( delta, icov ), delta.T ) - ndim )
            
                responsibilities[c] = np.argmin(D)
                distances[c] = D[responsibilities[c]]
        
            tmp = np.sum(responsibilities)
            if tmp==0 or tmp==ncomp: #all sub mixture components assigned to single model component
                idx = np.argmax(distances) #find mixture component with largest distance
                responsibilities[idx] = 1-responsibilities[idx] #assign to the other model component
                done =True
            
            #2. refit
            model_mu, model_cov, model_w = _refit(mu,cov,w,responsibilities)
            
            #3. same solution as before?
            done = np.sum(responsibilities==previous_responsibilities)==ncomp
            
            #test for cycles - does this ever happen?
            if not done:
                tmp = hashlib.sha1(responsibilities).hexdigest()
                if tmp in previous_solutions:
                    print "cycle!"
                    done =True #we are in a cycle
                else:
                    previous_solutions.append(tmp)
            
            previous_responsibilities = responsibilities
        
        return (model_mu[:1],model_cov[:1],model_w[:1]), (model_mu[1:],model_cov[1:],model_w[1:]), responsibilities
    
    def compress(self, costThreshold=0.05):
        
        model = [self.moment_match()]
        
        component_distances = [self.hellinger_distance_single( model[0], weighted=True )]
        component_indices = [ np.arange(self.ncomponents) ]

        niterations=0
        while True: # and niterations<100:

            #find model component with largest Hellinger distance
            val = max( component_distances )
            idx = component_distances.index(val)
            
            if val <= costThreshold:
                break
            
            #split selected model component
            reference_indices = component_indices[idx]
            submodel1, submodel2, responsibilities = self.two_gaussian_approximation(subset=reference_indices)
            
            #remove original component
            del model[idx]
            del component_indices[idx]
            del component_distances[idx]
            
            #add submodels to model
            model.append( submodel1 )
            component_indices.append( reference_indices[responsibilities==0] )
            component_distances.append( self.hellinger_distance_single( submodel1, subset=component_indices[-1], weighted=True ) )
            
            if submodel2 is not None:
                model.append( submodel2 )
                component_indices.append( reference_indices[responsibilities==1] )
                component_distances.append( self.hellinger_distance_single( submodel2, subset=component_indices[-1], weighted=True ) )
            
            niterations+=1
        
        #replace object's weights, means and covariances with new model
        self.weights = np.concatenate( [component[2] for component in model], axis=0 )
        self.means = np.concatenate( [component[0] for component in model], axis=0 )
        self.covariances = np.concatenate( [component[1] for component in model], axis=0 )
        
        ndim = self.ndim
        
        #recompute cache
        self._cache_iS = []
        self._cache_logConstant = []
        self._cache_icov = []
        for idx in range( len(model) ):
            icov = np.linalg.inv( model[idx][1][0] )
            iS = np.linalg.cholesky( icov )
            logConstant = np.sum( np.log( np.diag( iS ) ) ) - 0.5*ndim*LOG2PI
            
            self._cache_iS.append( iS )
            self._cache_logConstant.append( logConstant )
            self._cache_icov.append( icov )
        
        
        maxv = 3
        k = max(0,maxv-ndim)
        self._cache_sigma_points = []
        for idx in range( len(model) ):
            self._cache_sigma_points.append( get_sigma_points( model[idx][0][0], model[idx][1][0], k ) )
        
        return model
    
    

class mixmodel(object):
    def __init__(self, weights=None, means=None, covariances=None):
        if weights is None:
            self.weights = np.array([],dtype=float)
        else:
            self.weights = np.asarray(weights,dtype=np.float)
            if self.weights.ndim!=1:
                raise ValueError
        
        #self.ncomponents = len(self.weights)
        
        if means is None:
            self.means = np.zeros( (self.ncomponents,2), dtype=np.float )
        else:
            self.means = np.asarray(means,dtype=np.float)
            if self.means.ndim!=2 or self.means.shape[0]!=self.ncomponents:
                raise ValueError
        
        #self.ndim = self.means.shape[1]
        
        if covariances is None:
            self.covariances = np.repeat( np.diag( np.ones( self.ndim ) ).reshape( (1,self.ndim, self.ndim) ), repeats=self.ncomponents, axis=0 )
        else:
            self.covariances = np.asarray(covariances,dtype=np.float)
            if self.covariances.ndim!=3 or self.covariances.shape[0]!=self.ncomponents or self.covariances.shape[1:]!=(self.ndim,self.ndim):
                raise ValueError
        
    def evaluate(self,x):
        return normmixpdf(self,x)
    
    def evaluate_grid(self, x, y, dim=None):
        pass
    
    def evaluate_marginal(self, x, dim=None):
        pass
    
    def add_observations(self, obs, uncertainty=None):
        pass
    
    def compress(self,*args,**kwargs):
        return compress_mixture(self,*args,**kwargs)
    
    def moment_match(self):
        mu,cov,w = moment_match_gaussian( self.means, self.covariances, self.weights )
        return mixmodel( weights=w, means=mu, covariances=cov )
    
    def extract(self,key):
        return mixmodel( weights=self.weights[key], means=self.means[key], covariances=self.covariances[key] )
    
    def subtract(self,key):
        m = mixmodel( weights=self.weights[key], means=self.means[key], covariances=self.covariances[key] )
        self.weights = np.delete(self.weights,key,axis=0)
        self.means = np.delete(self.means,key,axis=0)
        self.covariances = np.delete(self.covariances,key,axis=0)
        return m
    
    def add(self,other):
        self.weights = np.concatenate( (self.weights,other.weights) )
        self.means = np.concatenate( (self.means,other.means) )
        self.covariances = np.concatenate( (self.covariances,other.covariances) )
    
    def add_points(self,means,covariances=None,weights=None):
        if means.ndim != 2:
            raise ValueError
        
        n,ndim = means.shape
        if n<1:
            return
        
        if ndim != self.ndim:
            raise ValueError
            
        self.means = np.concatenate( (self.means, means) )
        
        if covariances is None:
            covariances = np.zeros( (n,ndim,ndim) )
        elif covariances.shape==(ndim,ndim):
            covariances = np.repeat( covariances.reshape( (1,ndim,ndim) ), repeats=n, axis=0 )
        elif covariances.ndim!=3 or covariances.shape[0]!=n or covariances.shape[1:]!=(ndim,ndim):
            raise ValueError
        
        self.covariances = np.concatenate( (self.covariances,covariances) )
        
        if weights is None:
            weights = np.ones(n,dtype=np.float)/n
        elif weights.ndim != 1 or len(weights)!=n:
            raise ValueError
        
        self.weights = np.concatenate( (self.weights, weights) )
        
    @property
    def ncomponents(self):
        return len(self.weights)
    
    @property
    def ndim(self):
        return self.means.shape[1]
    
    def two_gaussian_approximation(self):
        mu,cov,w,resp = two_gaussian_approximation( self.means, self.covariances, self.weights )
        return mixmodel(weights=w,means=mu,covariances=cov), resp
    
    

def unscented_hellinger( p1, p2, weighted=False ):
    if not weighted:
        p1.weights = p1.weights / np.sum(p1.weights)
        p2.weights = p2.weights / np.sum(p2.weights)

    
    #merge distributions
    p0 = mixmodel()
    p0.means = np.concatenate( (p1.means, p2.means), axis=0 )
    p0.covariances = np.concatenate( (p1.covariances, p2.covariances), axis=0 )
    p0.weights = np.concatenate( (p1.weights*0.5,p2.weights*0.5), axis=0 )

    p0.weights = p0.weights / np.sum(p0.weights)
    
    #presumably all weights are > 0 and if not:
    #idx = p0.weights>0
    #p0.weights = p0.weights[idx]
    #p0.means = p0.means[idx]
    #p0.covariances = p0.covariances[idx]

    maxv = 3
    [x,n,w,k] = compute_sigma_points( p0, maxv )
    
    W = p0.weights[:,None] * w[None,:]
    
    pdf_p1 = normmixpdf(p1, x)
    pdf_p2 = normmixpdf(p2, x)
    
    pdf_p1 = pdf_p1*(pdf_p1 > 0)
    pdf_p2 = pdf_p2*(pdf_p2 > 0)
    
    pdf_p0 = normmixpdf(p0, x)
    
    g = (np.sqrt(pdf_p1)-np.sqrt(pdf_p2))**2
    H = np.sqrt(np.abs(np.sum(W.ravel()*g/pdf_p0)/2))
    
    return H

def compute_sigma_points( p, maxv ):
    
    ncomp,ndim = p.means.shape
    
    k = maxv - ndim
    #prevent negative weights
    if k<0:
        k=0
        maxv=ndim
    
    nsigmapoints = 2*ndim + int(k!=0)
    
    x = np.zeros( (nsigmapoints*ncomp, ndim) )
    current = 0
    for idx in range(ncomp):
        x[current:current+nsigmapoints,:] = get_sigma_points( p.means[idx], p.covariances[idx], k )
        current+=nsigmapoints
    
    if k==0:
        w = np.ones(2*ndim)/(2*(ndim+k))
    else:
        w = np.ones(2*ndim+1)/(2*(ndim+k))
        w[0] = 1.0*k / (ndim+k)
    
    if np.abs(np.sum(w)-1) > 1e-5:
        raise ValueError
    
    return x, nsigmapoints, w, k

def get_sigma_points(mu,cov,k):
    ndim = len(mu)
    u,s,v = np.linalg.svd(cov)
    S = np.dot(u,np.sqrt(np.diag(s)))*np.sqrt(ndim+k)
    S = np.concatenate((S,-S),axis=0).T.reshape(ndim*2,ndim)
    
    x = S + mu[None,:]
    
    if k!=0:
        x = np.concatenate( (mu[None,:],x), axis=0 )
    
    return x



def normmixpdf( model, x, minerr=None ):
    if minerr is not None:
        th_err = np.log(minerr)
    
    log2pi = 1.83787706640935
    
    ndata,ndim = x.shape
    
    ncomp = len(model.weights)
    
    p = np.zeros( ndata )
    ptmp = np.zeros( ndata )
    
    for idx in range(ncomp):
        iS = np.linalg.cholesky( np.linalg.inv(model.covariances[idx]) )
        logdetiS = np.sum( np.log( np.diag( iS ) ) )
        logConstant = (logdetiS - 0.5*ndim*log2pi )
        
        dx = x - model.means[idx:idx+1,:]
        dx = np.dot(dx,iS)
        pl = logConstant - 0.5*np.sum(dx*dx,axis=1)
        
        if minerr is not None:
            sel = pl>th_err
            ptmp[:]=0
            ptmp[sel] = np.exp(pl[sel])
        else:
            ptmp = np.exp(pl)
        
        p = p + model.weights[idx]*ptmp
    
    return p



def moment_match_gaussian( means, covariances, weights, normalize=True):
    """Convert a mixture of Gaussians to a single Gaussian by moment matching"""
    
    ncomp,ndim = means.shape
    
    if ncomp<2:
        return means, covariances, weights
    
    weights_new = np.sum(weights,keepdims=True)
    if normalize:
        weights = weights/weights_new
    
    means_new = np.dot( weights, means ).reshape( (1,ndim) )
    
    covariances_new = np.sum( weights[:,None,None] * (covariances + means[:,:,None]*means[:,None,:]), axis=0, keepdims=True ) - np.dot(means_new.T,means_new)[None,:,:]
    
    return means_new, covariances_new, weights_new

def split_gaussian( mu, cov, weight=1.0):
    
    u,s,v = np.linalg.svd(cov)
    idx = np.argmax(s)
    
    delta = np.sqrt(s[0,idx])*0.5*v[:,idx]
    means_new = mu + np.array([[1],[-1]])*delta
    cov_new = cov + mu*mu.T - 0.5*np.sum(means_new[:,:,None]*means_new[:,None,:],axis=0,keepdims=True)
    cov_new = np.repeat( cov_new, repeats=2, axis=0 )
    weight_new = np.ones(2,dtype=np.float)*0.5*weight
    
    return means_new, cov_new, weight_new
    


import hashlib

def two_gaussian_approximation(mixture_mu, mixture_cov, mixture_w):
    
    ncomp,ndim = mixture_mu.shape
    
    #if number of components in mixture <= 2, then return
    if ncomp<=2:
        return mixture_mu, mixture_cov, mixture_w
    
    #to initialize: fit with single Gaussian using moment matching and split in two
    model_mu, model_cov, model_w = moment_match_gaussian( mixture_mu, mixture_cov, mixture_w )
    model_mu, model_cov, model_w = split_gaussian( model_mu, model_cov, model_w )
    
    
    previous_responsibilities = np.ones(ncomp)
    previous_solutions = []
    
    #repeatedly:
    done = False
    while not done:
        #1. compute responsibilities
        responsibilities, distances = _compute_responsibilities( model_mu, model_cov, mixture_mu, mixture_cov )
        
        tmp = np.sum(responsibilities)
        if tmp==0 or tmp==ncomp: #all sub mixture components assigned to single model component
            idx = np.argmax(distances) #find mixture component with largest distance
            responsibilities[idx] = 1-responsibilities[idx] #assign to the other model component
            done =True
        
        #2. refit
        model_mu, model_cov, model_w = _refit(mixture_mu,mixture_cov,mixture_w,responsibilities)
        
        #3. same solution as before?
        done = np.sum(responsibilities==previous_responsibilities)==ncomp
        
        #test for cycles
        if not done:
            tmp = hashlib.sha1(responsibilities).hexdigest()
            if tmp in previous_solutions:
                done =True #we are in a cycle
            else:
                previous_solutions.append(tmp)
        
        previous_responsibilities = responsibilities
    
    return model_mu, model_cov, model_w, responsibilities

def _compute_responsibilities( model_mu, model_cov, mixture_mu, mixture_cov ):
    
    ncomp,ndim = mixture_mu.shape
    
    responsibilities = np.zeros( ncomp, dtype=np.int )
    distances = np.zeros( ncomp, dtype=np.float )
    
    D = np.zeros(2)
    
    for c in xrange(ncomp):
        #icov = np.linalg.inv(mixture_cov[c])
        for m in xrange(2):
            D[m] = _compute_component_distance( model_mu[m], model_cov[m], mixture_mu[c], mixture_cov[c] )
            #delta = (model_mu[m]-mixture_mu[c]).reshape( (1,ndim) )
            #cov1 = model_cov[m]
            #cov2 = mixture_cov[c]
            #D[m] = 0.5*( np.log( np.linalg.det(cov2) / np.linalg.det(cov1) ) + np.trace( np.dot( icov, cov1 ) ) + np.dot( np.dot( delta, icov ), delta.T ) - ndim )
        responsibilities[c] = np.argmin(D)
        distances[c] = D[responsibilities[c]]
    
    return responsibilities, distances

def _compute_component_distance( mu1, cov1, mu2, cov2 ):
    dm = len(mu1)
    icov2 = np.linalg.inv(cov2)
    delta = (mu1-mu2).reshape( (1,dm) )
    return 0.5*( np.log( np.linalg.det(cov2) / np.linalg.det(cov1) ) + np.trace( np.dot( icov2, cov1 ) ) + np.dot( np.dot( delta, icov2 ), delta.T ) - dm )

def _refit( mixture_mu, mixture_cov, mixture_w, responsibilities ):
    
    ndim = mixture_mu.shape[1]
    
    mu = np.zeros( (2,ndim) )
    cov = np.zeros( (2,ndim,ndim) )
    w = np.zeros(2)
    
    b = responsibilities==0
    w[0] = np.sum( mixture_w[b] )
    mu[0], cov[0], _ = moment_match_gaussian( mixture_mu[b], mixture_cov[b], mixture_w[b]/w[0] )
    
    b = responsibilities==1
    w[1] = np.sum( mixture_w[b] )
    mu[1], cov[1], _ = moment_match_gaussian( mixture_mu[b], mixture_cov[b], mixture_w[b]/w[1] )
    
    w = w * np.sum(mixture_w) / np.sum(w)
    
    return mu, cov, w



def compress_mixture(mixture, minNumberComponents=5, maxNumberComponents=100, costThreshold=0.05):
    
    if mixture.ncomponents<minNumberComponents:
        return mixture
    
    #initialize compressed approximate with a single Gaussian
    model = mixture.moment_match()
    
    #compute Hellinger distance
    Dhell = unscented_hellinger( model, mixture, weighted=True )
    
    component_indices = [ np.arange(mixture.ncomponents) ]
    component_distances = [ Dhell ]
    
    #cmp_data.idxToref = {[1:length(pdf.w)]} ;
    #cmp_data.hells = [Hell] ;
    #cmp_data.cantbrake = [0] ;
    
    while mixture.ncomponents >= minNumberComponents:
        
        #find model component with largest Hellinger distance
        val = max( component_distances )
        idx = component_distances.index(val)
        
        if val <= costThreshold: # and len(component_indices)>minNumberComponents:
            break
        
        #split selected model component
        reference_indices = component_indices[idx]
        submixture = mixture.extract( reference_indices )
        model_component = model.subtract( [idx] )
        
        submodel,responsibilities = submixture.two_gaussian_approximation()
        
        #add submodel to model
        model.add(submodel)
        
        #update component_indices
        del component_indices[idx]
        component_indices.extend( [ reference_indices[responsibilities==0], reference_indices[responsibilities==1] ] )
        
        #compute Hellinger distance for two new sub components
        del component_distances[idx]
        submixture = mixture.extract( component_indices[-2] )
        submodel = model.extract([-2])
        component_distances.append( unscented_hellinger( submodel, submixture, weighted=True ) )
        
        submixture = mixture.extract( component_indices[-1] )
        submodel = model.extract([-1])
        component_distances.append( unscented_hellinger( submodel, submixture, weighted=True ) )
        
        if model.ncomponents > maxNumberComponents:
            break
    
    return model
    



#add_input
#1. observations weights: np.ones(nobservations)
#2. 
