"""
==================================================================
Multi-taper spectral analysis (:mod:`fklab.signals.multitaper`)
==================================================================

.. currentmodule:: fklab.signals.multitaper

Functions for multi-taper spectral analysis, including spectrum, 
spectrogram, coherence and coherogram. The multitaper implementation is
based on the open-source Chronux Matlab toolbox (http://www.chronux.org).

Spectral analysis
=================

.. autosummary::
    :toctree: generated/
    
    mtspectrum
    mtspectrogram
    mtcoherence
    mtcoherogram
    
Plots
=====

.. autosummary::
    :toctree: generated/
    
    plot_spectrum
    plot_spectrogram
    plot_coherence
    plot_coherogram

Utilities
=========

.. autosummary::
    :toctree: generated/
    
    nextpow2
    mtfft
    mtoptions

"""

import warnings
import math
#import functools
import numpy as np
import scipy as sp
import scipy.stats
import spectrum
import fklab.segments as seg
from fklab.utilities import inrange
from .basic_algorithms import generate_windows, extract_data_windows, extract_trigger_windows


__all__ = ['nextpow2', 'mtfft', 'mtspectrum', 'mtspectrogram', 
           'mtcoherence', 'mtcoherogram', 'plot_spectrum', 'plot_spectrogram',
           'plot_coherence', 'plot_coherogram', 'mtoptions']

def nextpow2(n):
    """
    Compute the first power of 2 larger than a given value.
    
    Parameters
    ----------
    n : number
    
    Returns
    -------
    exponent : integer
        The smallest integer such that 2**exponent is larger than `n`.
    
    """
    n = np.abs( n )
    val, p = 1, 0
    while val < n:
        val = val * 2
        p += 1
    return p

def _compute_nfft( n, pad=0 ):
    """Compute padded number of samples for FFT.
    
    Parameters
    ----------
    n : int
        number of samples to pad
    pad : int, optional
        amount of padding to a higher power of 2. Zero means no padding.
    
    Returns
    -------
    n : int
        padded number of samples
    
    """
    
    return max( 2**(nextpow2(n+1) + pad - 1), n )

def _allowable_bandwidth( n, fs=1.):
    """Multitape bandwidth limits.
    
    Parameters
    ----------
    n : int
        Number of samples in signal
    fs : float, optional
        Sampling frequency of signal
    
    Returns
    -------
    [min, max]
        minimum and maximum allowable bandwidths
    
    """
    
    return [fs/n, 0.5*fs*(n-1)/n]

#@functools.lru_cache(maxsize=2)
def _check_bandwidth( bw, n, fs=1.0, correct=False ):
    """Check and correct multitaper bandwidth.
    
    Parameters
    ----------
    bw : float
        Requested bandwidth
    n : int
        Number of samples in signal
    fs : float, optional
        Sampling frequency
    correct : bool, optional
        Correct bandwidth if requested value is out of range.
    
    Returns
    -------
    bw : float
        Multitaper bandwidth
        
    """
    if bw is None:
        TW = min(3, (n-1)/2.0)
        bw = TW * fs / n
    else:
        bw = float(bw)
        limits = _allowable_bandwidth(n, fs)
        
        if bw<limits[0] or bw>limits[1]:
            if correct:
                bw = max( min(limits[1],bw), limits[0] )
            else:
                raise ValueError('Bandwidth out of range. Minimum bandwidth = {minbw} Hz. Maximum bandwidth = {maxbw} Hz.'.format(minbw=limits[0], maxbw=limits[-1]))
    
    return bw

#@functools.lru_cache(maxsize=2)
def _check_ntapers( ntapers, n, bw=None, fs=1.0, correct=False ):
    """Check and correct number of tapers.
    
    Parameters
    ----------
    ntapers : int
        Requested number of tapers
    n : int
        Number of samples in signal
    bw : float, optional
        Requested bandwidth
    fs : float, optional
        Sampling frequency
    correct : bool
        Correct number of tapers and bandwidth if requested values are
        out of range.
    
    Returns
    -------
    ntapers : int
        Number of tapers
    bw : float
        Multitaper bandwidth
    
    """
    
    bw = _check_bandwidth( bw, n, fs, correct )
    
    TW = bw*n/fs
    maxtapers = int( math.floor( 2*TW-1 ) )
    
    if ntapers is None:
        ntapers = maxtapers
    else:
        ntapers = int(ntapers)
        if ntapers<1 or ntapers>maxtapers:
            if correct:
                ntapers = max( min(maxtapers, ntapers), 1)
            else:
                raise ValueError('Invalid number of tapers. Maximum number of tapers = {maxtapers}'.format(maxtapers=maxtapers))
    
    return ntapers, bw

#@functools.lru_cache(maxsize=2)
def _compute_tapers( bw, n, fs, ntapers, correct=False ):
    """Compute tapers.
    
    Parameters
    ----------
    bw : float
        Requested bandwidth
    n : int
        Number of samples in signal
    fs : float
        Sampling frequency
    ntapers : int
        Requested number of tapers
    correct : bool
        Correct number of tapers and bandwidth if requested values are
        out of range.
        
    Returns
    -------
    tapers : ndarray
    
    """
    
    ntapers, bw = _check_ntapers( ntapers, n, bw, fs, correct )
    tapers = spectrum.dpss( int(n), int( bw*n/fs ), int(ntapers) )[0] * np.sqrt(fs)
    return tapers


class mtoptions(object):
    """Class to manage multitaper options.
    
    Parameters
    ----------
    bandwidth : scalar
    fpass : scalar or [min, max]
    error : {'none', 'theory', 'jackknife'}
    pvalue : scalar
    pad : int
    ntapers : int
    
    Attributes
    ----------
    bandwidth
    fpass
    error
    pvalue
    pad
    ntapers
    
    Methods
    -------
    bandwidth_range(nsamples, fs)
    nfft(nsamples)
    frequencies(nsamples, fs)
    validate(nsamples, fs, correct)
    tapers(nsamples, fs, correct)
    
    """
    
    def __init__(self, bandwidth=None, fpass=None, error='none', pvalue=0.05, pad=0, ntapers=None):
        self.bandwidth = bandwidth
        self.fpass = fpass
        self.error = error
        self.pvalue = pvalue
        self.pad = pad
        self.ntapers = ntapers
    
    def keys(self):
        return ['bandwidth','fpass','error','pvalue','pad','ntapers']
    
    def __getitem__(self,key):
        if key in self.keys():
            return object.__getattribute__(self,key)
        else:
            raise KeyError('Unknown key')
    
    def bandwidth_range(self, nsamples, fs=1.):
        """Permissible range of bandwidths.
        
        Parameters
        ----------
        nsamples : int
            Number of samples in signal
        fs : float, optional
            Sampling frequency of signal
        
        Returns
        -------
        [min, max]
            minimum and maximum allowable bandwidths
        
        """
        return _allowable_bandwidth( nsamples, fs )
    
    def nfft(self, nsamples):
        """Compute padded number of samples for FFT.
        
        Parameters
        ----------
        n : int
            number of samples to pad

        Returns
        -------
        n : int
            padded number of samples
        
        """
        return _compute_nfft( nsamples, self._pad )
    
    def frequencies(self, nsamples, fs=1.0):
        """Compute frequency vector.
        
        Parameters
        ----------
        nsamples : int
            Number of samples
        fs : scalar
            Sampling frequency
        
        Returns
        -------
        f : 1d array
            Frequencies
        fidx : 1d array
            Indices of selected frequencies that fall within the `fpass`
            setting.
        
        """
        nfft = self.nfft(nsamples)
        f = fs*np.arange(nfft)/nfft
        
        if self._fpass is None:
            fpass = [0., fs/2.]
        else:
            fpass = self._fpass
        
        fidx = np.logical_and( f>=fpass[0], f<=fpass[1] )
        
        return f, fidx
    
    def validate(self, nsamples, fs=1., correct=False):
        """Validate multitaper options.
        
        Parameters
        ----------
        nsamples : int
            Number of samples
        fs : scalar
            Sampling frequency
        correct : bool
            Correct bandwidth and tapers if needed.
            
        Returns
        -------
        dict
            Validated multitaper options and pre-computed tapers.
        
        """
        
        nsamples = int(nsamples)
        if nsamples<3:
            raise ValueError('Number of samples should be at least 3.')
        
        fs = float(fs)
        if fs<=0.:
            raise ValueError('Sampling frequency should be larger than zero.')
        
        d = dict(
            sampling_frequency=fs,
            nsamples=nsamples,
            error=self._error,
            pvalue=self._pvalue,
            pad=self.pad,
            nfft=self.nfft( nsamples ),
            )
        
        d['ntapers'], d['bandwidth'] = _check_ntapers( self._ntapers, nsamples, bw=self._bw, fs=fs, correct=correct )
        d['frequencies'], d['fpass'] = self.frequencies( nsamples, fs )
        
        d['tapers'] = self.tapers( nsamples, fs, correct )
        
        return d
    
    def tapers(self, nsamples, fs=1., correct=False ):
        """Compute tapers.
        
        Parameters
        ----------
        nsamples : int
            Number of samples in signal
        fs : float
            Sampling frequency
        correct : bool
            Correct number of tapers and bandwidth if requested values are
            out of range.
            
        Returns
        -------
        tapers : ndarray
        
        """
        return _compute_tapers( self._bw, nsamples, fs, self._ntapers, correct)
    
    @property
    def bandwidth(self):
        """Bandwidth for tapers."""
        return self._bw
    
    @bandwidth.setter
    def bandwidth(self, val):
        if not val is None:
            val = float(val)
            if val<=0.:
                raise ValueError()
        self._bw = val
    
    @property
    def fpass(self):
        """Selection of frequency band of interest."""
        return self._fpass
    
    @fpass.setter
    def fpass(self, val):
        if not val is None:
            val = list(np.array(val,dtype=np.float64).ravel())
            if len(val)==1:
                val = [0., val[0]]
            elif not len(val)==2:
                raise ValueError('FPass should be a scalar or 2-element sequence.')
            
            if val[0]<0. or val[0]>val[1]:
                raise ValueError('FPass values should be larger than zero and strictly monotonically increasing.')
        
        self._fpass = val
    
    @property
    def error(self):
        """Type of error to compute ('none', 'theory', 'jackknife')"""
        return self._error
    
    @error.setter
    def error(self, val):
        if val is None or not val:
            val = 'none'
        elif not val in ('none','theory','jackknife'):
            raise ValueError("Error should be one of 'none', 'theory', 'jackknife'.")
        self._error= val
    
    @property
    def pvalue(self):
        """P-value for error computation."""
        return self._pvalue
    
    @pvalue.setter
    def pvalue(self, val):
        val = float(val)
        if val<=0. or val>=1:
            raise ValueError('p-Value should be between zero and one.')
        
        self._pvalue=val
    
    @property
    def pad(self):
        """Amount of padding."""
        return self._pad
    
    @pad.setter
    def pad(self, val):
        if not val is None:
            val = int(val)
            if val<0:
                raise ValueError('Pad should be equal to or larger than zero.')
        
        self._pad = val
    
    @property
    def ntapers(self):
        """Number of tapers."""
        return self._ntapers
    
    @ntapers.setter
    def ntapers(self,val):
        if not val is None:
            val=int(val)
            if val<1:
                raise ValueError('Number of tapers should be larger than zero.')
        
        self._ntapers = val

def mtfft(data, tapers, nfft, fs ):
    """Multi-tapered FFT.
    
    Parameters
    ----------
    data : 2d array
        data array with samples along first axis and signals along the second axis
    tapers: 2d array
        tapers with samples along first axis and tapers along the second axis
    nfft : integer
        number of points for FFT calculation
    fs : float
        sampling frequency of data
    
    Returns
    -------
    J : 3d array
        FFT of tapered signals. Shape of the array is (samples, tapers, signals)
    
    """
    #TODO: check shape of m and tapers
    
    data = np.array( data )
    tapers = np.array( tapers )
    
    nfft = int( nfft )
    fs = float( fs )
    
    data_proj = data[:,None,:] * tapers[:,:,None]
    
    J = np.fft.fft( data_proj, nfft, axis=0 ) / fs
    
    return J


def _spectrum_error(S, J, errtype, pval, avg, numsp=None):
    
    if errtype == 'none' or errtype is None:
        return None
    
    nf, K, C = J.shape # freq x tapers x channels
    
    if S.ndim==1:
        S = S[:,None]
    
    if numsp is not None:
        numsp = np.array(numsp).ravel()
        if len(numsp)!=C:
            raise ValueError('Invalid value for numsp')
    
    pp = 1. - float(pval)/2.
    qq = 1. - pp
    
    if avg:
        dim = K * C
        C = 1
        dof = 2*dim*np.ones(1) #degrees of freedom
        
        if not numsp is None:
            dof = np.fix( 1. / (1./dof + 1./(2.*np.sum(numsp))) )
        
        J = np.reshape( J, (nf,dim,C) )
        
    else:
        
        dim = K
        dof = 2.*dim*np.ones(C)
        
        if not numsp is None:
            for ch in range(C):
                dof[ch] = np.fix( 1./ (1. / dof + 1./(2.*numsp[ch])) )
    
    Serr = np.zeros( (2,nf,C) )
    
    if errtype=='theory':
        Qp = sp.stats.chi2.ppf( pp, dof )
        Qq = sp.stats.chi2.ppf( qq, dof )
        
        #check size of dof and Qp, Qq
        #either could be scalar of vector
        
        Serr[0,:,:] = dof[None,:] * S / Qp[None,:]
        Serr[1,:,:] = dof[None,:] * S / Qq[None,:]
    
    else: #errtype == 'jackknife'
        tcrit = sp.stats.t.ppf( pp, dim-1 )
        
        Sjk = np.zeros( (dim,nf,C) )
        
        for k in range(dim):
            Jjk = J[:,np.setdiff1d( range(dim), [k] ),:] #1-drop projection
            eJjk = np.sum( Jjk * np.conjugate(Jjk), axis=1 )
            Sjk[k,:,:] = eJjk / (dim-1) #1-drop spectrum
        
        sigma = np.sqrt( dim-1 ) * np.std( np.log(Sjk), axis=0 )
        
        #if C==1; sigma=sigma'; end
        #
        #conf=repmat(tcrit,nf,C).*sigma;
        #conf=squeeze(conf); 
    
        conf = tcrit * sigma
        Serr[0,:,:] = S * np.exp( -conf )
        Serr[1,:,:] = S * np.exp( conf )
        
    
    #Serr=shiftdim( squeeze(Serr), 1 );
    
    return Serr

def _mtspectrum_single( data, fs=1.0, average=False, **kwargs ):
    """Compute multi-tapered spectrum of vectors.
    
    Parameters
    ----------
    data : 2d array
        data array with samples along first axis and channels along the second axis
    fs : float, optional
        sampling frequency
    average : bool
        compute averaged spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers
    
    Returns
    -------
    S : vector or 2d array
        spectral density, with shape (frequencies, signals) or
        (frequencies,) if average==True.
    f : 1d array
        vector of frequencies
    Serr : None or 3d array
        lower and upper error estimates. The shape of the array is
        (2, frequencies, signals), where the first axis contains the
        lower and upper error estimates.
    options : dict
    
    """
    
    #TODO: check data
    data = np.array(data)
    N = data.shape[0]
    
    options = mtoptions( **kwargs )
    options = options.validate( N, fs )
    
    f = options['frequencies'][ options['fpass'] ]
    
    J = mtfft( data, options['tapers'], options['nfft'], options['sampling_frequency'] )
    J = J[ options['fpass'] ]
    S = 2*np.mean( np.real(np.conjugate(J) * J), axis=1 ) #factor two because we combine positive and negative frequencies
    
    if average:
        S = np.mean( S, axis=1 )
    
    Serr = _spectrum_error( S, J, options['error'], options['pvalue'], average )
    
    return S, f, Serr, options

def mtspectrum( data, fs=1., start_time=0., window_size=None,
                epochs=None, average=True, triggers=None, **kwargs ):
    """Compute windowed multi-tapered spectrum.
    
    Parameters
    ----------
    data : 1d array
        data vector
    fs : float, optional
        sampling frequency
    starttime : float, optional
        time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool, optional
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    
    Returns
    -------
    S : vector or 2d array
        spectral density, with shape (frequencies, signals) or
        (frequencies,) if average==True.
    f : 1d array
        vector of frequencies
    Serr : None or 3d array
        lower and upper error estimates. The shape of the array is
        (2, frequencies, signals), where the first axis contains the
        lower and upper error estimates.
    options : dict
    
    """
    
    # check data
    data = np.array( data )
    if data.ndim!=1:
        raise ValueError('Only vector data is supported')
    
    if triggers is None:
        _, data = extract_data_windows( data, window_size, start_time=start_time, fs=fs, epochs=epochs )
    else:
        _, data = extract_trigger_windows( data, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs )
        
    return _mtspectrum_single( data, fs=fs, average=average, **kwargs )

def mtspectrogram( data, fs=1., start_time=0., window_size=None, 
                   window_overlap=0., epochs=None, average=True,
                   triggers=None, trigger_window=1., **kwargs ):
    """Compute multi-tapered spectrogram.
    
    Parameters
    ----------
    data : 1d array
        data vector
    fs : scalar, optional
        sampling frequency
    start_time : float, optional
        time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    window_overlap : float, optional
        Fraction of overlap between neighbouring windows
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    trigger_window : float or [left, right]
        Window around trigger for which to compute spectrogram
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers
        
    Returns
    -------
    S : vector or 2d array
        spectral density, with shape (frequencies, signals) or
        (frequencies,) if average==True.
    t : 1d array
        vector of times
    f : 1d array
        vector of frequencies
    Serr : None or 4d array
        lower and upper error estimates. The shape of the array is
        (time, 2, frequencies, signals), where the second axis contains the
        lower and upper error estimates.
    options : dict
    
    """
    
    data = np.array(data)
    
    if triggers is None:
        if data.ndim==1:
            data = np.reshape(data, (len(data),1))
        elif data.ndim!=2:
            raise ValueError("Data should be vector or 2d array.")
    elif data.ndim!=1:
        raise ValueError("For triggered spectrogram, data should be a vector.")
    
    if not triggers is None:
        _, data = extract_trigger_windows( data, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs )
    
    n, ch = data.shape
    
    if triggers is None:
        s = start_time
    else:
        ## hack: copy of code from _generate_trigger_windows
        trigger_window = np.array( trigger_window, dtype=np.float ).ravel()
        if len(trigger_window)==1:
            trigger_window = np.abs(trigger_window) * [-1,1]
        elif len(trigger_window)!=2 or np.diff(trigger_window)<=0.:
            raise ValueError('Invalid window')
        s = trigger_window[0]
    
    nwin, t, idx = generate_windows( n, window_size, window_overlap=window_overlap, fs=fs, start_time=s, center=True )
    
    options = mtoptions( **kwargs )
    options = options.validate( np.round( np.float(window_size)*fs ), fs )
    
    fpass = options['fpass']
    f = options['frequencies'][fpass]
    numfreq = np.sum(fpass)
    
    Serr = None
    compute_error = options['error'] not in [None, 'none']
    
    if average:
        S = np.zeros( (nwin,numfreq) )
        if compute_error:
            Serr = np.zeros( (nwin,numfreq,2) )
    else:
        S = np.zeros( (nwin,numfreq,ch) )
        if compute_error:
            Serr = np.zeros( (nwin,numfreq,ch,2) )
    
    for k,indices in enumerate(idx()):
        #idx = np.arange( winstart[k], winstart[k] + window_size, dtype=np.int )
        J = mtfft( data[indices,:], options['tapers'], options['nfft'], options['sampling_frequency'])
        J = J[fpass,:]
        
        if average:
            #factor two because we combine positive and negative frequencies
            S[k] = 2*np.mean( np.mean( np.real(np.conjugate(J)*J), axis=1 ), axis=1 )
        else:
            #factor two because we combine positive and negative frequencies
            S[k] = 2*np.mean( np.real(np.conjugate(J)*J), axis=1 )
        
        if compute_error:
            Serr[k,:,:,:] = _spectrum_error(S[k], J, options['error'], options['pvalue'], average)
       
    return S,t,f,Serr,options


def _coherence_error(c, j1, j2, errtype, pval, avg, numsp1=None, numsp2=None):
    
    if errtype == 'none' or errtype is None:
        return None, None, None
    
    nf, K, Ch = j1.shape   # freq x tapers x channels
    
    if c.ndim==1:
        c = c[:,None]
    
    if numsp1 is not None:
        numsp1 = np.array(numsp1).ravel()
        if len(numsp1)!=Ch:
            raise ValueError('Invalid value for numsp1')
    
    if numsp2 is not None:
        numsp2 = np.array(numsp2).ravel()
        if len(numsp2)!=Ch:
            raise ValueError('Invalid value for numsp2')
    
    pp = 1-float(pval)/2.
    
    # find the number of degress of freedom
    if avg:
        dim = K*Ch
        
        if not numsp1 is None:
            dof1 = np.fix( 2*np.sum(numsp1)*2*dim / (2*np.sum(numsp1) + 2*dim) )
        else:
            dof1 = 2*dim
        
        if not numsp2 is None:
            dof2 = np.fix( 2*np.sum(numsp2)*2*dim / (2*np.sum(numsp2) + 2*dim) )
        else:
            dof2 = 2*dim
        
        dof = np.array(min(dof1, dof2)).ravel()
        Ch = 1
        
        j1 = j1.reshape( (nf,dim,1) )
        j2 = j2.reshape( (nf,dim,1) )
    else:
        dim = K
        dof = 2*dim
        
        if not numsp1 is None:
            dof1 = np.fix( 2*numsp1*2*dim / (2*numsp1 + 2*dim) )
        else:
            dof1 = 2*dim
        
        if not numsp2 is None:
            dof2 = np.fix( 2*numsp2*2*dim / (2*numsp2 + 2*dim) )
        else:
            dof2 = 2*dim
        
        dof = np.minimum( dof1, dof2 )
    
    # theoretical, asymptotic confidence level
    df = np.ones(dof.shape, dtype=np.float)
    df[dof>2] = 1./((dof/2.)-1)
    confC = np.sqrt( 1. - pval**df )
    
    # phase standard deviation (theoretical and jackknife) and
    # jackknife confidence intervals for c
    if errtype=='theory':
        phistd = np.sqrt( (2./dof[None,:]*(1./(c**2)-1)) )
        Cerr = None
    else: # errtype=='jackknife'
        tcrit = sp.stats.t.ppf( pp, dof-1 )
        atanhCxyk = np.empty( (dim,nf,Ch) )
        phasefactorxyk = np.empty( (dim,nf,Ch), dtype=np.complex )
        Cerr = np.empty( (2,nf,Ch) )
        for k in range(dim):  # dim is the number of 'independent estimates'
            indxk = np.setdiff1d( range(dim), k )
            j1k = j1[:,indxk,:]
            j2k = j2[:,indxk,:]
            ej1k = np.sum(j1k*np.conjugate(j1k),1)
            ej2k = np.sum(j2k*np.conjugate(j2k),1)
            ej12k = np.sum(j1k*np.conjugate(j2k),1)
            Cxyk = ej12k/np.sqrt(ej1k*ej2k)
            absCxyk = np.abs(Cxyk)
            atanhCxyk[k] = np.sqrt(2*dim-1)*np.arctanh(absCxyk) # 1-drop estimate of z
            phasefactorxyk[k] = Cxyk/absCxyk
        
        atanhC = np.sqrt(2*dim-2)*np.arctanh(c)
        sigma12 = np.sqrt(dim-1)*np.std(atanhCxyk,axis=0) # jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimate
        
        Cu = atanhC + tcrit[None,:]*sigma12
        Cl = atanhC - tcrit[None,:]*sigma12
        
        Cerr[0] = np.maximum(np.tanh(Cl/np.sqrt(2*dim)),0.)
        Cerr[1] = np.tanh(Cu/np.sqrt(2*dim-2))
        
        phistd = np.sqrt( (2*dim-2)*(1-np.abs(np.mean(phasefactorxyk, axis=0))))
    
    return confC, phistd, Cerr

def _mtcoherence_single( x, y, fs=1.0, average=False, **kwargs ):
    """Compute multi-tapered coherence of two vectors.
    
    Parameters
    ----------
    x : 2d array
    y : 2d array
    fs : float, optional
        Sampling frequency
    average : bool
        compute averaged spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers
    
    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval.
    options : dict
    
    """
    #TODO: check shape of x and y
    
    x = np.array( x )
    y = np.array( y )
    
    N = x.shape[0]
    
    options = mtoptions( **kwargs )
    options = options.validate( N, fs )
    
    f = options['frequencies'][ options['fpass'] ]
    
    J1 = mtfft( x, options['tapers'], options['nfft'], options['sampling_frequency'] )
    J2 = mtfft( y, options['tapers'], options['nfft'], options['sampling_frequency'] )
    J1 = J1[options['fpass']]
    J2 = J2[options['fpass']]
    
    S12 = np.mean( np.conjugate(J1)*J2, axis=1 )
    S1  = np.mean( np.conjugate(J1)*J1, axis=1 )
    S2  = np.mean( np.conjugate(J2)*J2, axis=1 )
    
    if average:
        S12 = np.mean( S12, axis=1 )
        S1  = np.mean( S1 , axis=1 )
        S2  = np.mean( S2 , axis=1 )
    
    C12 = S12 / np.sqrt( S1 * S2 )
    c = np.abs( C12 )
    phi = np.angle( C12 )    
    
    err = _coherence_error(c, J1, J2, options['error'], options['pvalue'], average )
    
    return c, phi, f, err, options

def mtcoherence( x, y, fs=1., start_time=0, window_size=None, 
                 epochs=None, average=True, triggers=None, **kwargs ):
    """Compute windowed multi-tapered coherence of two vectors.
    
    Parameters
    ----------
    x : 1d array
    y : 1d array
    fs : float, optional
        Sampling frequency
    start_time : float
       time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval.
    options : dict
    
    """
    
    # check data
    x = np.array( x )
    y = np.array( y )
    
    if x.ndim!=1:
        raise ValueError('Only vector data is supported')
    
    if triggers is None:
        _, x = extract_data_windows( x, window_size, start_time=start_time, fs=fs, epochs=epochs )
        _, y = extract_data_windows( y, window_size, start_time=start_time, fs=fs, epochs=epochs )
    else:
        _, x = extract_trigger_windows( x, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs )
        _, y = extract_trigger_windows( y, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs )
        
    return _mtcoherence_single( x, y, fs=fs, average=average, **kwargs )
    
def mtcoherogram( x, y, fs=1., start_time=0., window_size=None, 
                   window_overlap=0., epochs=None, average=True,
                   triggers=None, trigger_window=1., **kwargs ):
    """Compute multi-tapered coherogram.
    
    Parameters
    ----------
    x : 1d array
    y : 1d array
    fs : float, optional
        Sampling frequency
    start_time : float
       time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    window_overlap : float, optional
        Fraction of overlap between neighbouring windows
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    trigger_window : float or [left, right]
        Window around trigger for which to compute spectrogram
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. `pad`=0 means no
        padding, `pad`=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers
    
    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    t : 1d array
        vector of times
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval. Note: currently error calculations are not supported
        for coherograms
    options : dict
    
    """
    
    x = np.array(x)
    y = np.array(y)
    
    if x.shape!=y.shape:
        raise ValueError('Input signals need to have the same size.')
    
    if triggers is None:
        if x.ndim==1:
            x = np.reshape(x, (len(x),1))
            y = np.reshape(y, (len(y),1))
        elif x.ndim!=2:
            raise ValueError("Data should be vector or 2d array.")
    elif x.ndim!=1:
        raise ValueError("For triggered spectrogram, data should be a vector.")
    
    if not triggers is None:
        _, x = extract_trigger_windows( x, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs )
        _, y = extract_trigger_windows( y, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs )
    
    n, ch = x.shape
    
    if triggers is None:
        s = start_time
    else:
        ## hack: copy of code from _generate_trigger_windows
        trigger_window = np.array( trigger_window, dtype=np.float ).ravel()
        if len(trigger_window)==1:
            trigger_window = np.abs(trigger_window) * [-1,1]
        elif len(trigger_window)!=2 or np.diff(trigger_window)<=0.:
            raise ValueError('Invalid window')
        s = trigger_window[0]
    
    nwin, t, idx = generate_windows( n, window_size, window_overlap=window_overlap, fs=fs, start_time=s, center=True )
       
    options = mtoptions( **kwargs )
    options = options.validate( np.round( np.float(window_size)*fs ), fs )
    
    fpass = options['fpass']
    f = options['frequencies'][fpass]
    numfreq = np.sum(fpass)
    
    Serr = None
    compute_error = options['error'] not in [None, 'none']
    
    if average:
        coh = np.zeros( (nwin,numfreq) )
        phi = np.zeros( (nwin,numfreq) )
        #if compute_error:
        #    Serr = np.zeros( (nwin,numfreq,2) )
    else:
        coh = np.zeros( (nwin,numfreq,ch) )
        phi = np.zeros( (nwin,numfreq,ch) )
        #if compute_error:
        #    Serr = np.zeros( (nwin,numfreq,ch,2) )
    
    for k,indices in enumerate(idx()):
        
        J1 = mtfft( x[indices,:], options['tapers'], options['nfft'], options['sampling_frequency'] )
        J2 = mtfft( y[indices,:], options['tapers'], options['nfft'], options['sampling_frequency'] )
        J1 = J1[fpass,:]
        J2 = J2[fpass,:]
        
        S12 = np.mean( np.conjugate(J1)*J2, axis=1 )
        S1  = np.mean( np.conjugate(J1)*J1, axis=1 )
        S2  = np.mean( np.conjugate(J2)*J2, axis=1 )
        
        if average:
            S12 = np.mean( S12, axis=1 )
            S1  = np.mean( S1 , axis=1 )
            S2  = np.mean( S2 , axis=1 )
    
        C12 = S12 / np.sqrt( S1 * S2 )
        coh[k] = np.abs( C12 )
        phi[k] = np.angle( C12 )
        
        #if compute_error:
        #    pass
    
    return coh,phi,t,f,(None,None,None),options


import matplotlib.pyplot as plt

def plot_spectrum( data, t=None, axes=None, units=None, db=True, color='black', **kwargs ):
    """Plot power spectral density of data vector.
    
    Parameters
    ----------
    data : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes.
    units : str, optional
        Units of the data (e.g. mV)
    db : bool, optional
        Plot density in dB.
    color : any matplotlib color, optional
        Color of plot.
    kwargs : mtspectrum parameters
    
    Returns
    -------
    axes : matplotlib Axes object
    artists : list of plot elements
    (s,f,err,options) : output of mtspectrum
    
    """
    
    if not t is None:
        kwargs['start_time'] = kwargs.get('start_time',t[0])
        kwargs['fs'] = kwargs.get('fs',1./np.mean(np.diff(t)))
    
    S, f, err, options = mtspectrum( data, **kwargs )
    
    if db:
        S = 10. * np.log10(S)
        err = 10. * np.log10(err)
    
    if axes is None:
        axes = plt.gca()
    
    artists = []
    
    if not err is None:
        artists.append( axes.fill_between( f, err[0,:,0], err[1,:,0], facecolor=color, alpha=0.2 ) )
    
    artists.extend( plt.plot( f, S, axes=axes, color=color ) )
    
    if units is None or units=='':
        units = '1'
    else:
        units = str(units)
        units = units + '*' + units
    
    plt.xlabel('frequency [Hz]')
    plt.ylabel('power spectral density [{units}/Hz] {db}'.format(units=units, db='in db' if db else ''))
    
    return axes, artists, (S,f,err,options)

def plot_spectrogram( data, t=None, axes=None, units=None, db=True, **kwargs ):
    """Plot spectrogram of data vector.
    
    Parameters
    ----------
    data : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes.
    units : str, optional
        Units of the data (e.g. mV)
    db : bool, optional
        Plot density in dB.
    kwargs : mtspectrogram parameters
    
    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (s,t,f,err,options) : output of mtspectrogram
    
    """
    
    if not t is None:
        kwargs['start_time'] = kwargs.get('start_time',t[0])
        kwargs['fs'] = kwargs.get('fs',1./np.mean(np.diff(t)))
    
    winsize = kwargs.get('window_size')
    
    S, t, f, err, options = mtspectrogram( data, **kwargs )
    
    if db:
        S = 10. * np.log10(S)
    
    if axes is None:
        axes = plt.gca()
    
    artists = []
    
    artists.append( plt.imshow( S.T, axes=axes, cmap='YlOrRd', aspect='auto', 
                        origin='lower', extent=[t[0,0],t[-1,1], f[0], f[-1]],
                        interpolation='nearest' ) )
    
    plt.ylabel('frequency [Hz]')
    plt.xlabel('{label} [s]'.format(label='time' if kwargs.get('triggers',None) is None else 'latency'))
    
    cbar = plt.colorbar()
    artists.append(cbar)
    
    if units is None or units=='':
        units = '1'
    else:
        units = str(units)
        units = units + '*' + units
        
    cbar.set_label('power spectral density [{units}/Hz] {db}'.format(units=units, db='in db' if db else ''))
    plt.draw()
    
    return axes, artists, (S,t,f,err,options)

def plot_coherence( signal1, signal2, t=None, axes=None, color='black', **kwargs ):
    """Plot coherence between two data vectors.
    
    Parameters
    ----------
    signal1 : 1d array
    signal2 : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes
    color : any matplotlib color, optional
        Color of plot.
    kwargs : mtcoherence parameters
    
    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (coh,phi,f,err,options) : output of mtcoherence
    
    """
    
    if not t is None:
        kwargs['start_time'] = kwargs.get('start_time',t[0])
        kwargs['fs'] = kwargs.get('fs',1./np.mean(np.diff(t)))
    
    coh, phi, f, err, options = mtcoherence( signal1, signal2, **kwargs )
    
    if axes is None:
        axes = plt.gca()
    
    artists = []
    
    if not err[2] is None:
        artists.append( axes.fill_between( f, err[2][0,:,0], err[2][1,:,0], facecolor=color, alpha=0.2 ) )
        
    artists.extend( plt.plot( f, coh, axes=axes, color=color ) )
    
    if not err[0] is None:
        artists.append( plt.axhline(y=err[0], color='red', linestyle=':') )
    
    plt.xlabel('frequency [Hz]')
    plt.ylabel('coherence')
    
    plt.ylim(0,1)
    
    return axes, artists, (coh,phi,f,err,options)
    
def plot_coherogram( signal1, signal2, t=None, axes=None, **kwargs ):
    """Plot coherogram of two data vectors.
    
    Parameters
    ----------
    signal1 : 1d array
    signal2 : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes
    kwargs : mtcoherogram parameters
    
    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (coh,phi,t,f,err,options) : output of mtcoherence  
    
    """
    
    if not t is None:
        kwargs['start_time'] = kwargs.get('start_time',t[0])
        kwargs['fs'] = kwargs.get('fs',1./np.mean(np.diff(t)))
    
    winsize = kwargs.get('window_size')
    
    coh, phi, t, f, err, options = mtcoherogram( signal1, signal2, **kwargs )
    
    if axes is None:
        axes = plt.gca()
    
    artists = []
    
    artists.append( plt.imshow( coh.T, axes=axes, cmap='YlOrRd', aspect='auto', 
                        origin='lower', extent=[t[0,0],t[-1,1], f[0], f[-1]],
                        interpolation='nearest' ) )
    plt.clim(0.,1.)
    
    plt.ylabel('frequency [Hz]')
    plt.xlabel('{label} [s]'.format(label='time' if kwargs.get('triggers',None) is None else 'latency'))
    
    cbar = plt.colorbar()
    cbar.set_label('coherence')
    
    artists.append(cbar)
    
    plt.draw()
    
    return axes, artists, (coh,phi,t,f,err,options)
