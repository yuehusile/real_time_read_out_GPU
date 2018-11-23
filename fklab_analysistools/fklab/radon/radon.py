"""
==========================================
Radon transform (:mod:`fklab.radon.radon`)
==========================================

.. currentmodule:: fklab.radon.radon


.. autosummary::
    :toctree: generated/
    
    RadonTransform
    build_theta_vector
    build_rho_vector
    local_integrate
    bresenham
    test_line_fit
    
"""

import radonc
import numpy as np
import scipy as sp
import scipy.stats
import enum

import fklab.signals.kernelsmoothing

__all__ = ['PadMethod', 'Constraint', 'Interpolation', 'IntegralMethod', 
           'RadonTransform', 'build_rho_vector', 'build_theta_vector',
           'local_integrate', 'bresenham', 'test_line_fit']

class PadMethod(enum.Enum):
    No=0
    Median=1
    Mean=2
    GeoMean=3
    Random=4

Constraint = radonc.Constraint
Interpolation = radonc.Interpolation
IntegralMethod = radonc.IntegralMethod

def build_rho_vector( nx, ny, dx=1., dy=1., thetarange=None, intercept=False, oversampling=1. ):
    """Construct rho vector for sampling radon transform.
    
    Parameters
    ----------
    nx, ny : integer
        size of image
    dx, dy: float
        size of image pixels
    thetarange : (float, float) sequence
        lower and upper bounds of the angle range
    intercept : bool
        whether rho is interpreted as the intercept with the horizontal center line
    oversampling : scalar
        oversampling factor
    
    Returns
    -------
    rho : array
        vector of rho values
    (dr, range, n) : (float, (float,float), integer)
        rho step size, range of rho values, number of rho values
    
    """
    
    import fklab.statistics.circular
    
    if thetarange is None or (thetarange[0]==-0.5*np.pi and thetarange[1]==0.5*np.pi):
        thetarange = None
        val = np.sqrt( (dy/dx)**2 + 1 )
    else:
        thetarange = fklab.circular.wrap(thetarange, -0.5*np.pi, 0.5*np.pi )
        if len(thetarange)!=2:
            pass
        if thetarange[0]>thetarange[1]:
            thetarange = fklab.circular.wrap(thetarange, 0, np.pi )
            
        f = lambda x: -np.minimum( 1.0/np.abs( np.sin(x) ), dy/(dx*np.abs(np.cos(x))) )
        _, val, _, _ = sp.optimize.fminbound( f, thetarange[0], thetarange[1], full_output=True )
        val = -val
    
    drho = dy/val
    
    xmin = -dx*(nx-1)/2.0
    ymin = -dy*(ny-1)/2.0
    
    if intercept:
        rhomax = np.abs(xmin)
        nr = 2*rhomax/drho + 1
        rhorange = [-drho*(nr-1)/2.0, rhomax]
    else:
        if thetarange is None: # default range of [-0.5pi, 0.5pi]
            rhomax = np.sqrt( xmin**2 + ymin**2 )
            nr = 2*rhomax/drho + 1
            rhorange = [-drho*(nr-1)/2.0, rhomax]
        else:
            minrho = np.inf
            maxrho = -np.inf
            
            f = lambda theta, scale, x, y: scale*(x*np.cos(theta) + y*np.sin(theta))
            
            for x in [xmin, -xmin]:
                for y in [ymin, -ymin]:
                    _, tmp, _, _ = sp.optimize.fminbound( f, thetarange[0], thetarange[1], (-1,x,y), full_output=True )
                    maxrho = np.maximum( maxrho, -tmp )
                    _, tmp, _, _ = sp.optimize.fminbound( f, thetarange[0], thetarange[1], (+1,x,y), full_output=True )
                    minrho = np.minimum( minrho, tmp )
            
            rhorange = [minrho, maxrho]
            nr = np.ceil(np.diff( rhorange ) / drho + 1)
                
    rho = np.linspace( rhorange[0], rhorange[1], nr*oversampling )
    
    return rho, (drho, rhorange, nr)

def build_theta_vector( nx, ny, dx=1., dy=1., thetarange=None, oversampling=1. ):
    """Construct theta vector for sampling radon transform.
    
    Parameters
    ----------
    nx, ny : integer
        size of image
    dx, dy: float
        size of image pixels
    thetarange : (float, float) sequence
        lower and upper bounds of the angle range
    oversampling : scalar
        oversampling factor
    
    Returns
    -------
    theta : array
        vector of theta values
    (dt, range, n) : (float, (float, float), integer)
        theta step size, range of theta values and number of theta values
    
    """
    
    import fklab.statistics.circular
    
    if thetarange is None:
        thetarange = [-0.5*np.pi, 0.5*np.pi]
        val = np.sqrt( (dy/dx)**2 + 1 )
    else:
        thetarange = fklab.circular.wrap(thetarange, -0.5*np.pi, 0.5*np.pi )
        if len(thetarange)!=2:
            pass
        if thetarange[0]>thetarange[1]:
            thetarange = fklab.circular.wrap(thetarange, 0, np.pi )
            
        f = lambda x: -np.minimum( 1.0/np.abs( np.sin(x) ), dy/(dx*np.abs(np.cos(x))) )
        _,val,_,_ = sp.optimize.fminbound( f, thetarange[0], thetarange[1], full_output=True )
        val = -val
    
    xmin = -dx*(nx-1)/2.0
    ymin = -dy*(ny-1)/2.0
    
    dtheta = dy / (val * np.sqrt(xmin**2 + ymin**2) )
    nt = np.ceil( np.diff( thetarange ) / dtheta )
    dtheta = np.diff( thetarange ) / nt
    
    theta = np.linspace( thetarange[0], thetarange[1], nt * oversampling )
    
    return theta, (dtheta, thetarange, nt)


class RadonTransform(radonc.Radon):
    """Radon transform class
    
    Parameters
    ----------
    dx : positive scalar
    dy : positive scalar
    interpolation : string or Interpolation
    constraint : string or Constraint
    integral_method : string or IntegralMethod
    pad : string or PadMethod
    oversampling : scalar
    valid : bool
    intercept : bool
        
    """
    
    def __init__( self, *args, **kwargs ):
        self.pad = kwargs.pop('pad',None)
        self.oversampling = kwargs.pop('oversampling', 1.)
        radonc.Radon.__init__( self, *args, **kwargs )
    
    def __iter__(self):
        
        props = ['dx', 'dy', 'interpolation', 'constraint',
                 'integral_method', 'valid', 'intercept', 'pad',
                 'oversampling']
        
        for x in props:
            yield x, self.__getattribute__(x)
    
    def __repr__(self):
        return dict(self).__repr__()
    
    @property
    def pad(self):
        return self.pad_
    
    @pad.setter
    def pad(self, val):
        if isinstance(val, str):
            val = PadMethod[val]
        elif isinstance(val, PadMethod):
            pass
        elif val is None:
            val = PadMethod.No
        elif val is not None:
            val = float(val)
        
        self.pad_ = val
    
    @property
    def oversampling(self):
        return self.oversampling_
    
    @oversampling.setter
    def oversampling(self, val):
        val = float(val)
        if val<=0.:
            raise Exception("Oversampling factor needs to be larger than zero.")
        self.oversampling_ = val
    
    def padded_transform(self, data, theta, rho):
        """Radon transform with padding
        
        Parameters
        ----------
        data : 2d array
        theta : 1d array
            vector of angles for sampling radon transform
        rho : 1d array
            vector of offsets for sampling radon transform
        
        Returns
        -------
        radon : 1d array
        n : 3d array
        
        """
        
        if self.pad==PadMethod.No:
            return self.transform( data, theta, rho )
        
        if self.constraint == Constraint.No:
            raise Exception("Padded radon transform only supported with X/Y constraint.")
        
        # temporary set integral method to sum 
        method = method_orig = self.integral_method
        if method in (IntegralMethod.Integral, IntegralMethod.Mean) :
            self.integral_method = method = IntegralMethod.Sum
        
        try:
            r, nn = self.transform( data, theta, rho )
        finally:
            # reset original method
            self.integral_method = method
        
        ax = 1 if self.constraint == Constraint.X else 0
        
        if self.pad==PadMethod.Median:
            val = np.median( data, axis=ax )
        elif self.pad==PadMethod.Mean:
            val = np.mean( data, axis=ax )
        elif self.pad==PadMethod.GeoMean:
            val = sp.stats.mstats.gmean( data, axis=ax )
        elif self.pad==PadMethod.Random:
            for k in range( data.shape[1-ax] ):
                idx = np.logical_or( k<nn[:,:,0] , k>nn[:,:,1] )
                if method==IntegralMethod.Sum:
                    r[ idx ] += np.random.choice( np.take( data, k, axis=1-ax ), size=(np.sum(idx)) )
                elif method==IntegralMethod.LogSum:
                    r[ idx ] += np.log( np.random.choice( np.take( data, k, axis=1-ax ), size=(np.sum(idx)) ) )
                elif method==IntegralMethod.Product:
                    r[ idx ] *= np.random.choice( np.take( data, k, axis=1-ax ), size=(np.sum(idx)) )
                else:
                    # should not happen 
                    raise Exception("Internal error: unrecognized integral method.")
        else:
            pad = 0 if self.pad is np.NaN else self.pad
            val = np.full( data.shape[1-ax], float(pad) )
        
        if self.pad!=PadMethod.Random:
            
            if method==IntegralMethod.Product:
                a = np.concatenate( ([0], np.cumprod( val[0:-1]  )))
                b = np.concatenate( ( np.cumprod( val[-1:0:-1] )[::-1], [0]))
                r = r * a[ nn[:,:,0 ]] * b[ nn[:,:,1] ]
            else: # Sum or LogSum
                if method == IntegralMethod.LogSum:
                   val = np.log(val)
                a = np.concatenate( ([0], np.cumsum( val[0:-1]  )))
                b = np.concatenate( ( np.cumsum( val[-1:0:-1] )[::-1], [0]))
                r = r + a[ nn[:,:,0 ]] + b[ nn[:,:,1] ]
            
        if method_orig == IntegralMethod.Mean:
            r /= data.shape[1-ax]
        elif method_orig == IntegralMethod.Integral:
            if self.constraint==Constraint.X:
                r *= self.dx / np.abs(np.sin(theta[:,None]))
            else:
                r *= self.dy / np.abs(np.cos(theta[:,None]))
                
        return r, nn
    
    def padded_slice(self, data, theta, rho):
        """Slice through image with padding
        
        Parameters
        ----------
        data : 2d array
        theta : 1d array
            vector of angles for sampling radon transform
        rho : 1d array
            vector of offsets for sampling radon transform
        
        Returns
        -------
        slice : 1d array
        
        """
        
        if self.pad==PadMethod.No:
            val, _= self.slice( data, theta, rho )
            return val
        
        if self.constraint == Constraint.No:
            raise Exception('Padded slice only supported with X/Y constraint.')
        
        r, nn = self.slice( data, theta, rho )
        
        ax = 1 if self.constraint == Constraint.X else 0
       
        if self.pad==PadMethod.Median:
            val = np.median( data, axis=ax ) 
        elif self.pad==PadMethod.Mean:
            val = np.mean( data, axis=ax )
        elif self.pad==PadMethod.GeoMean:
            val = sp.stats.mstats.gmean( data, axis=ax )
        elif self.pad==PadMethod.Random:
            idx = [None,]*2
            idx[ax] = np.random.randint(data.shape[ax], size=(data.shape[1-ax],))
            idx[1-ax] = np.arange(data.shape[1-ax], dtype=np.int)
            val = data[ tuple(idx) ]
        else:
            val = np.full( data.shape[1-ax], float(self.pad) )
        
        val[ nn[0] : nn[1] ] = r
    
        return val
    
    def fit_line(self, data, x=None, y=None, theta=None, rho=None, form='slope-intercept'):
        """Best line fit using radon transform.
        
        Parameters
        ----------
        data : 2d array
        x : 1d array
            regularly spaced vector of x-values (first dimension of data array)
        y : 1d array
            regularly spaced vector of y-values (second dimension of data array)
        theta : 1d array
            vector of angles for sampling radon transform
        rho : 1d array
            vector of offsets for sampling radon transform
        form : {'slope-intercept', 'normal'}
            returns line parameters in either slope-intercept form or
            normal (theta, rho) form.
            
        Returns
        -------
        line parameters : (float, float)
            either (slope, intercept) or (theta, rho)
        score : float
        (radon, theta, rho, thetamax, rhomax) : (2d array, 1d array, 1d array, scalar, scalar)
        projection : 1d array
        
        """
        
        if x is None:
            x = np.arange( data.shape[0] )
        elif len(x)!=data.shape[0]:
            raise Exception("Incorrect length x-vector.")
        
        if y is None:
            y = np.arange( data.shape[1] )
        elif len(y)!=data.shape[1]:
            raise Exception("Incorrect length y-vector.")
        
        if rho is None:
            rho, _ = build_rho_vector( len(x), len(y), oversampling=self.oversampling )
        if theta is None:
            theta, _ = build_theta_vector( len(x), len(y), oversampling=self.oversampling )
        
        r, n = self.padded_transform( data, theta, rho)
        
        idx = np.nanargmax( r )
        idx = np.unravel_index( idx, r.shape )
        
        thetamax = theta[ idx[0] ]
        rhomax = rho[ idx[1] ]
        
        proj = self.padded_slice( data, thetamax, rhomax )
        
        score = r[idx[0],idx[1]]
        
        if self.constraint==Constraint.X:
            score = score / data.shape[0]
        elif self.constraint==Constraint.Y:
            score = score / data.shape[1]
        
        # apply scaling
        dx = np.mean( np.diff( x ) )
        dy = np.mean( np.diff( y ) )
        
        theta_ = np.arctan( np.tan( thetamax ) * dx/dy )
        if np.abs(np.cos(thetamax))<0.5:
            rho_ = rhomax*dy*np.sin(theta_)/np.sin(thetamax)
        else:
            rho_ = rhomax*dx*np.cos(theta_)/np.cos(thetamax)
        
        # convert to different line form
        if form=='slope-intercept':
            line_parameters = normal_to_slope_intercept( theta_, rho_, origin=( 0.5*( x[0] + x[-1] ), 0.5*( y[0] + y[-1] ) ) )
        else:
            line_parameters = (theta_, rho_)
        
        return line_parameters, score, (r, theta, rho, thetamax, rhomax), proj


def normal_to_slope_intercept( theta, rho, origin=(0,0) ):
    
    slope = -1./np.tan( theta )
    intercept = rho/np.sin(theta) - slope*origin[0] + origin[1]
    
    return slope, intercept

def local_integrate( m, n=3, axis=0 ):
    """Perform local integration (summing) along array axis.
    
    Parameters
    ----------
    m : nd array
    n : int
        Number of neighboring bins to integrate over
    axis : int
        Array axis along which to perform integration
    
    """
    return fklab.signals.kernelsmoothing.smooth1d( m, axis=axis, kernel='uniform', bandwidth=n/2., normalize=False )

def bresenham( x0, y0, x1, y1 ):
    """Bresenham's algorithm for line digitization
    
    Parameters
    ----------
    x0, y0, x1, y0 : int
        Line start and end coordinates
    
    Returns
    -------
    
    """
    
    steep = abs(y1-y0) > abs(x1-x0)
    
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    
    if x0>x1:
        x0, x1, y0, y1 = x1, x0,y1, y0
    
    deltax, deltay = x1-x0, abs(y1-y0)
    
    error = 0
    y = y0
    
    ystep = 1 if y0<y1 else -1
    
    cx, cy = [], []
    
    for x in range(x0,x1+1):
        
        if steep:
            cx.append(y)
            cy.append(x)
        else:
            cx.append(x)
            cy.append(y)
        
        error = error + deltay;
        if 2*error >= deltax:
            y = y + ystep;
            error = error - deltax
    
    return cx, cy

def test_line_fit( size=(30,30), interpolation='Linear', constraint='None', pad='No', integral_method='Sum' ):
    """Tests radon based line fitting.
    
    Plots the results of radon transform and line fitting.
    
    Parameters
    ----------
    size : (int, int)
    interpolation : {'Linear', 'Nearest'}
    constraint : {'None', 'X', 'Y'}
    pad : {'No', 'Median', 'Mean', 'GeoMean', 'Random'}
    integral_method : {'Sum', 'Integra;', 'Mean', 'LogSum', 'Product'}
        
    """
    
    import matplotlib.pyplot as plt
    import scipy.sparse
    
    dx, dy = 10., 0.1
    xstart, ystart = 7.5, 25.

    # (x0, y0, x1, y1)
    lines = [ ( 0, size[1]-1, int(0.8*(size[0]-1)),  int(0.2*(size[1]-1)) ),
              ( 0, int(0.3*(size[1]-1)), size[0]-1, int(0.3*(size[1]-1))),
              ( int(0.2*(size[0]-1)),  int(0.2*(size[1]-1)), int(0.9*(size[0]-1)), int(0.7*(size[1]-1))),
              (int(0.45*(size[0]-1)),  0, int(0.45*(size[0]-1)), size[1]-1)
            ]
    
    x = np.arange(size[0])*dx + xstart
    y = np.arange(size[1])*dy + ystart
    
    extent = [x[0]-0.5*dx, x[-1]+0.5*dx, y[0]-0.5*dy, y[-1]+0.5*dy]
    
    fig, ax = plt.subplots( 3, len(lines), figsize=(16,10), sharex='row', sharey='row' )

    radon_transform = RadonTransform( interpolation=interpolation, constraint=constraint, oversampling=4, pad=pad, integral_method=integral_method )
    
    for L in range(len(lines)):
        cx, cy = bresenham( *lines[L] )
        m = scipy.sparse.coo_matrix( (np.ones(len(cx)), (cx, cy)), shape=size ).toarray()
        m += np.random.uniform( 0, 0.8, size=size )
        
        (slope, intercept), score, r, proj = radon_transform.fit_line( m, x=x,y=y )
        
        ax[0,L].imshow( r[0], aspect='auto', interpolation='none', origin='lower',
                        extent= [ r[2][0], r[2][-1], r[1][0], r[1][-1] ], vmin=0, vmax=50 )
        
        ax[0,L].plot( r[4], r[3], 'ko', markerfacecolor='None', markersize=15, markeredgewidth=1 )
        
        mi, ma = np.min( r[2] ), np.max( r[2] )
        ax[0,L].set_xlim( mi-0.1*(ma-mi), ma + 0.1*(ma-mi) )
        mi, ma = np.min( r[1] ), np.max( r[1] )
        ax[0,L].set_ylim( mi-0.1*(ma-mi), ma + 0.1*(ma-mi) )
        
        ax[1,L].imshow( m.T, interpolation='none', aspect='auto', origin='lower' , extent=extent, cmap='Blues')
        ax[1,L].plot( x, slope*x + intercept, 'r' )
        ax[1,L].set_xlim( extent[:2] )
        ax[1,L].set_ylim( extent[2:] )
        
        ax[2,L].plot( proj )
        ax[2,L].set_ylim(0,2)
        
        
    
