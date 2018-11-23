"""
=========================================================
Independent Component Analysis (:mod:`fklab.signals.ica`)
=========================================================

.. currentmodule:: fklab.signals.ica

Tools for indepedent component analysis (ICA).

.. autosummary::
    :toctree: generated/
    
    FastICA
    ncomp
    explained_variance
    reconstruct
    inspect_component
    
"""

import numpy as np
import matplotlib.pyplot as plt

import fklab.signals.multitaper
import fklab.statistics.correlation
import fklab.plot

from sklearn.decomposition import FastICA

__all__ = ['ncomp', 'explained_variance', 'FastICA', 'reconstruct', 'inspect_component']


def explained_variance( components, mix, normalize=False ):
    
    v = np.sum( mix**2, axis=0) * np.var( components, axis=0 )
    if normalize:
        v = v / np.sum(v)
    
    return v

def ncomp( signals, p=99.9, plot=False ):
    """Determine number of components that explain desired percentage of variance.
    
    Parameters
    ----------
    signals : 2d array with shape (nsamples, nsignals)
    p : float
        Desired percentage of explained variance
    plot : bool
        Will plot explained variance for all PCA components.
    
    Returns
    -------
    ncomp : int
        Number of PCA components that explain at least `p`% of the signal variance.
    
    """
    
    n = signals.shape[1]
    
    # compute covariance of data
    cov = np.cov( signals, rowvar=False)
    # compute eigenvalues and eigenvectors
    w,v = np.linalg.eig( cov )
    
    # explained variance for eignevectors (sorted from largest to smallest)
    explained_variance = 100*np.sort(w)[::-1] / np.sum(w)
    
    # find number of PCA components that explain at least p% of the variance
    ncomp = np.searchsorted( np.cumsum(explained_variance), p ) + 1
    
    if plot:
        fig, ax = plt.subplots( 1,1 )
        
        ax.plot( range(n), explained_variance)
        ax.plot( range(ncomp), explained_variance[0:ncomp], 'ro')
        ax.plot( range(ncomp,n), explained_variance[ncomp:], 'k', linestyle='none', marker='x', markeredgewidth=2, markersize=10)
        ax.set_xlim(-1,n)
        ax.set_ylim(-10,110)
        ax.set_xticks(range(n))
        ax.set_ylabel("explained variance")
        ax.set_xlabel("principal component")
        ax.set_title("{0} PCA components explain >{1}% ({2:.1f}%) of the variance.".format(ncomp,p,np.sum(explained_variance[:ncomp])))
    
    return ncomp

def reconstruct( ica, data, components=None, samples=None, sensors=None):
    """Reconstruct mixtures from independent components
    
    Parameters
    ----------
    ica : FastICA object
    data : (nsamples, ncomponents) array
    components, samples, sensors : int, slice, int array, boolean array
        Selections for which samples, components and sensors to reconstruct.
    
    Returns
    --------
    (nsamples, nsensors) array
    
    """
    
    if components is None:
        components = slice(None)
    elif isinstance(components,int):
        components = [components,]
        
    if samples is None:
        samples = slice(None)
        
    if sensors is None:
        sensors = slice(None)
        
    return np.dot( data[samples,components], ica.mixing_[sensors,components].T ) + ica.mean_[sensors]

def inspect_component( t, data, ica, component=0, order=None, time_window=None, upper_freq=None, fs=None ):
    """Plot independent component characteristics.
    
    Parameters
    ----------
    t : 1d array
        Time vector
    data : (nsamples, ncomponents) array
        indepdent components time courses
    ica : FastICA object
    component : int, optional
    order : 1d int array, optional
    time_window : [float, float]
    upper_freq: float
    fs : float
    
    Returns
    -------
    figure, axes
    
    """
    
    if time_window is None:
        time_window = [ t[0], t[-1] ]
    
    if order is None:
        order = np.arange(data.shape[0])
    
    if fs is None:
        fs = 1./np.median( np.diff(t) )
    
    if upper_freq is None:
        upper_freq = fs/2.
    
    nchannels, ncomp = ica.mixing_.shape
    
    fig, ax = plt.subplots(2,3, gridspec_kw={'width_ratios':[1,4,1], 'height_ratios':[1,5]}, figsize=[16,10])
    ax[-1,-1].axis('off')
    
    # plot spatial loading
    loading = ica.mixing_[:,order[component]]
    scale = 1.1 * np.max(np.abs(loading))
    ax[0,0].plot( loading )
    ax[0,0].set_title('spatial loading')
    ax[0,0].set_xlabel('channel')
    ax[0,0].set_yticks([0])
    ax[0,0].set_ylim(-scale,scale)
    ax[0,0].set_xlim(-1, nchannels)
    
    # plot time courses
    ax[0,1].plot( t, data[:,order[component]] )
    ax[0,1].set_title('time course')
    ax[0,1].set_xlim(time_window)
    ax[0,1].set_yticks([])
    ax[0,1].set_xlabel('time [s]')
    
    reconstruction = reconstruct( ica, data, components=order[component] )
    spacing = -6 * np.max((np.var( reconstruction, axis=0 )**0.5))
    fklab.plot.plot_signals( t, reconstruction, spacing = spacing, colormap='Dark2', axes=ax[1,1] );
    ax[1,1].set_title('time course')
    ax[1,1].set_xlim(time_window)
    ax[1,1].set_ylim( nchannels*spacing, -spacing );
    ax[1,1].set_xlabel('time [s]')
    ax[1,1].set_title('reconstructed signals')
    
    # plot spectrum
    spectrum, freq, _, _ = fklab.signals.multitaper.mtspectrum( data[:,order[component]], fs=fs, window_size=1., bandwidth=10., fpass=upper_freq )
    
    ax[0,2].fill_between( freq, 0., spectrum )
    ax[0,2].set_xlabel('Frequency [Hz]')
    ax[0,2].set_ylabel('Power')
    ax[0,2].set_title('spectrum')
    ax[0,2].set_yticklabels([])
    
    # plot cross correlation
    cc = [fklab.statistics.correlation.xcorrn( data[:,order[component]], data[:,order[k]], scale='pearson', lags=[-500,500]) for k in range(ncomp)]
    lags = cc[0][1][0]/fs
    cc = [ x[0] for x in cc ]
    
    fklab.plot.plot_signals( lags, cc, spacing=-0.5, colormap='Dark2', axes=ax[1,0])
    ax[1,0].set_title('cross-correlation')
    ax[1,0].set_xlabel('lag [s]')
    ax[1,0].set_ylabel('component')
    ax[1,0].set_ylim(-0.5*ncomp, 0.5)
    ax[1,0].set_xlim( lags[0], lags[-1])
    ax[1,0].set_yticks( -0.5*np.arange(ncomp) )
    ax[1,0].set_yticklabels( np.arange(ncomp));

    return fig, ax

#mix = []
#for k in range(25):
    #_ = ica.fit_transform( mixture[ np.random.choice(mixture.shape[0], mixture.shape[0]), :] )
    #mix.append( ica.mixing_.copy() )
               
#mix = np.concatenate( mix, axis=1 )

#d = np.corrcoef( mix, rowvar=False )
#d = 1 - np.abs(d)
#d[np.tril_indices(len(d))] = d.T[np.tril_indices(len(d))]

#plt.figure()
#sns.heatmap( d )

#import sklearn.manifold
##model = sklearn.manifold.TSNE( n_components=2, random_state=0, metric='precomputed')
#model = sklearn.manifold.MDS( n_components=2, dissimilarity='precomputed')
#m = model.fit_transform( np.sqrt(d) )
#plt.figure()

#for i1 in range(len(m)):
    #for i2 in range(len(m)):
        #if np.sqrt(d[i1,i2])<0.4:
            #plt.plot( m[[i1,i2],0], m[[i1,i2],1], color=np.ones(3) * 2.5 * np.sqrt(d[i1,i2]) )

#plt.plot( m[:,0], m[:,1], 'r.')

#d[np.arange(len(d)), np.arange(len(d))] = 0.
#Z = scipy.cluster.hierarchy.average(scipy.spatial.distance.squareform(np.sqrt(d)))
#coph, _ = scipy.cluster.hierarchy.cophenet( Z, scipy.spatial.distance.squareform(np.sqrt(d) ))
#print(coph)
#plt.figure()
#dendro = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=0.35)



#def horn_parallel_analysis( signals, alpha=0.05, nshuffle=100):
    
    #nsamples, nsignals = signals.shape
    
    ## compute covariance of data
    #cov = np.cov( signals, rowvar=False)
    ## compute eigenvalues and eigenvectors
    #w,v = np.linalg.eig( cov )
    
    #shuffle = signals.copy()
    #eig = [None,] * nshuffle
    
    #for k in range(nshuffle):
        #for d in range(nsignals):
            #shuffle[:,d] = shuffle[ np.random.permutation(nsamples), d ]
        #cov = np.cov( shuffle, rowvar=False )
        #eig[k], _ = np.linalg.eig( cov )
    
    #return w, np.vstack(eig)

