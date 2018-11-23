"""
=================================================
Correlation (:mod:`fklab.statistics.correlation`)
=================================================

.. currentmodule:: fklab.statistics.correlation

Functions for correlation analysis.

.. autosummary::
    :toctree: generated/
    
    xcorrn

"""

import numpy as np
import scipy as sp
import scipy.fftpack

__all__ = ['xcorrn']

def xcorrn( x, y=None, lags=None, scale='none', remove_mean=False ):
    """Computes n-dimensional auto/cross correlation.
    
    Parameters
    ----------
    x : nd array
    y : nd array, optional
        If not present, then the auto-correlation of `x` will be computed.
    lags : int, (int,), (from, to), ((from1,to1), (from2,to2), ... )
        The lags at which the correlation is computed. If a single integer
        is provided, then the correlation will be computed for -lags...lags
        for all dimensions. Otherwise, the lags will range between from...to,
        which can be provided for each dimension separately, or for all dimensions
        together.
    scale : {'none', 'biased', 'unbiased', 'coeff', 'pearson'}
        Scaling option. `none`, returns the raw correlation (sum( x*y )); 
        `biased`, correlation normalized by total number of data points;
        `unbiased`, correlation normalized by the actual number of data 
        points used at each lag; `coeff`, correlation normalized by
        sqrt( sum(x**2) * sum(y**2) ), which corresponds to the Pearson 
        correlation coeffiient if `x` and `y` have zero mean; `pearson`,
        Pearson correlation coefficient that only takes into account the
        actual number of contributing data points at each lag (unlike `coeff`).
    remove_mean : bool
        Will remove the mean prior to computing the correlation.
    
    Returns
    -------
    cc : array
        Cross correlation result.
    lags : tuple of arrays
    
    """
    
    x = np.array( x, copy=True )
    y = x if y is None else np.array(y, copy=True)
    
    ndim = x.ndim
    
    if x.ndim != y.ndim:
        raise ValueError('Arrays x and y should have same dimensionality')
    
    if remove_mean:
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
    
    shape = np.maximum( np.array( x.shape ), np.array( y.shape ) )
    
    if lags is None:
        lags = np.array([ (-yshape+1,xshape-1) for xshape, yshape in zip( x.shape, y.shape ) ], dtype=np.int)
    else:
        lags = np.atleast_1d( lags )
        
        if lags.ndim==1 and lags.size==1:
            lags = np.array( [ [-1,1] ] * ndim, dtype=np.int) * abs(int(lags))
        elif lags.ndim==1 and lags.size==2:
            lags = np.ones(ndim, dtype=np.int)[:,None] * lags[None,:]
        elif lags.ndim!=2 or lags.shape!=(ndim,2):
            raise ValueError('Invalid lags')
    
    if np.any( np.abs(lags[:,0])>=y.shape ) or np.any( np.abs(lags[:,1])>=x.shape ):
        raise ValueError('Lag values are too high.')
    
    index_vectors = [ np.arange( a, b+1 ) for a,b in lags ]
    lags = np.ix_( *index_vectors )
       
    #check scale parameter
    if scale not in ('none','biased','unbiased','coeff','pearson'):
        raise ValueError('Invalid scale option.')
    
    hasNaN = False
    xnan = np.isnan(x)
    ynan = np.isnan(y)
    
    if np.any( xnan ) or np.any( ynan ):
        hasNaN = True
    
    if hasNaN or scale=='pearson':
        validx = np.logical_not(xnan)
        validy = np.logical_not(ynan)
        x[ np.logical_not(validx) ] = 0 
        y[ np.logical_not(validy) ] = 0
    
    #compute amount of padding
    fftsize = (2**np.ceil( np.log2( 2*shape-1 ) )).astype(np.int)
    
    #compute raw correlation (i.e. sum of x*y at all lags)
    fx = sp.fftpack.fftn( x, shape=fftsize )
    fy = sp.fftpack.fftn( y, shape=fftsize )
    
    c = np.real( sp.fftpack.ifftn( fx * np.conjugate(fy) ) )
    
    #select region corresponding to desired lags
    c = c[ lags ]
    
    if scale =='none':
        # raw, non-normalized cross-correlation (i.e. SUM[x*y])
        pass
    elif scale == 'biased':
        # cross-covariance, normalized by the number of valid data points
        # i.e. E[X*Y] = SUM[X+Y]/N
        if hasNaN:
            c = c/np.sum( np.logical_or(validx,validy) )
        else:
            c = c/np.prod(shape)
    elif scale == 'unbiased':
        # cross-covariance, normalized by the actual number of valid
        # data points that contribute to the correlation at each lag
        if hasNaN:
            fvalidx = sp.fftpack.fftn( validx, shape=fftsize )
            fvalidy = sp.fftpack.fftn( validy, shape=fftsize )
            tmp = np.real( sp.fftpack.ifftn( fvalidx * np.conjugate( fvalidy ) ) )
            tmp = tmp[ np.ix_( *index_vectors ) ]
            c = c/np.round(tmp)
        else:
            tmp = reduce( np.multiply, [ np.minimum( np.minimum(sx,sy), np.minimum(sx - l, sy + l) ) for sx,sy,l in zip(x.shape, y.shape,lags)] )
            tmp[ tmp<=0 ] = 1
            c = c/tmp
    elif scale == 'coeff':
        # normalized cross-correlation
        # corresponds to the pearson correlation coefficient if
        # the input signals have zero mean.
        # however, it is biased in the sense that the total number of valid data points
        # is used, rather than the actual number of data points at each lag
        c = c/np.sqrt(np.sum(x**2)*np.sum(y**2))
    elif scale == 'pearson':
        # unbiased Pearson correlation coefficient
        # (i.e. at each lag, the correlation coefficient is computed using
        # the actual number of valid data points)
        
        fvalidx = sp.fftpack.fftn( validx, shape=fftsize )
        fvalidy = sp.fftpack.fftn( validy, shape=fftsize )
        n = np.round( np.real( sp.fftpack.ifftn( fvalidx*np.conjugate(fvalidy) ) ) )
        
        xn = np.real(sp.fftpack.ifftn( fx*np.conjugate(fvalidy) ))
        yn = np.real(sp.fftpack.ifftn( fvalidx*np.conjugate(fy) ))
        
        fx2 = sp.fftpack.fftn( x**2, shape=fftsize )
        fy2 = sp.fftpack.fftn( y**2, shape=fftsize )
        
        xs = np.real(sp.fftpack.ifftn( fx2*np.conjugate(fvalidy) ))
        ys = np.real(sp.fftpack.ifftn( fvalidx*np.conjugate(fy2) ))
        
        n = n[ np.ix_( *index_vectors ) ]
        xn = xn[ np.ix_( *index_vectors ) ]
        yn = yn[ np.ix_( *index_vectors ) ]
        xs = xs[ np.ix_( *index_vectors ) ]
        ys = ys[ np.ix_( *index_vectors ) ]
        
        den = np.abs( np.sqrt( n*xs - xn**2 ) * np.sqrt( n*ys - yn**2 ) )
        valid = den>np.spacing(1);
        
        xy = np.copy(c)
        
        c = np.zeros(n.shape) + np.nan;
        c[valid] = ( n[valid]*xy[valid] - xn[valid]*yn[valid] ) / den[valid]
        c[n<=1] = np.nan
    
    return c, lags

#def pdfxcorr2( x, y=None, lags=None, timelags=None ):
    
    #if isinstance( x, np.ndarray ):
        #x = [x,]
    
    #if y is None:
        #y = x
    #elif isinstance( y, np.ndarray ):
        #y = [y,]
    
    #assert len(x)==len(y) and len(x)>0
    
    #ndimx = np.array( [ xi.ndim for xi in x ] )
    #ndimy = np.array( [ yi.ndim for yi in y ] )
    
    #assert np.all( np.logical_and( ndimx>1, ndimx<4 ) )
    #assert np.all( np.logical_and( ndimy>1, ndimy<4 ) )
    
    #shapex = np.array( [ [xi.shape[0], xi.shape[1], 1 if ndx==2 else xi.shape[2]] for xi,ndx in zip( x, ndimx ) ] )
    #shapey = np.array( [ [yi.shape[0], yi.shape[1], 1 if ndy==2 else yi.shape[2])]for yi,ndy in zip( y, ndimy ) ] )
    
    #shape = shapex[0,0:2]
    
    #assert np.all( np.logical_and( shapex[:,0:2]==shape, shapey[:,0:2]==shape ) )
    #assert np.all( shapex[:,2] == shapey[:,2] )
    
    #maxlag = shape-1;
    
    ##scalar, [iterable]
    #if lags is not None:
        #lags = np.ones(2) * np.array( lags, dtype=np.int ).ravel()
        
        #assert np.all( np.logical_or( lags>=0, np.isnan(lags) ) )
        #maxlag[ np.logical_not( np.isnan(lags) ) ] = lags[ np.logical_not( np.isnan(lags) ) ]
    
    #if timelags is not None:
        #timelags = np.array(timelags,dtype=np.int).ravel()
        #assert np.all( timelags>=0 )
    #else:
        #timelags = np.array([0],dtype=np.int)
    
    #lags = np.ix_( *[ np.arange( -r,r+1 ) for r in maxlag ] )
    
    #fftsize = 2**np.ceil( np.log2( 2*shape-1 ) )
    
    ##select region corresponding to desired lags
    #index_vectors = [ np.arange( -l, l+1 ) for l in maxlag ]
    
    #C = np.zeros(  )
    #n = np.zeros( )
    
    

#fcn = @(a,b) freqxcorr(a,b,fftsize,idx);
#%fcn = @(a,b) xcorr2(a,b);

#C = zeros( numel(lags{1}), numel(lags{2}), numel(timelags) );
#n = zeros(numel(timelags),1);

#valid = ~isnan( sum( cat(3,x{:}, y{:} ), 3 ) );
#Cvalid = round( fcn( double(valid), double(valid) ) );

#LagData = cell(ncells,numel(timelags));


#function [C,n,lags,timelags,LagData] = xcorr_pdf2( x, varargin )
#%XCORR_PDF2 cross correlation between series of 2D PDFs
#%
#%  c=XCORR_PDF2(x) Given a 3D array x (where 3rd dimension is time), this
#%  function will compute the average 2D autocorrelation over all slices in
#%  x. If x is a cell array, it should contain arrays with the same numbers
#%  of rows and columns (but the size of the 3rd dimension may vary). In
#%  this case the averaged autocorrelation will be computed over all slices
#%  in all arrays.
#%
#%  c=XCORR_PDF2(x,y) This will compute the cross correlation between x and
#%  y. Both arguments can be either 3D arrays or cell arrays containing 3D
#%  arrays. Regardless, x and y should have the same size.
#%
#%  c=XCORR_PDF2(...,parm1,val1,...) Specify optional parameter/value pairs:
#%   Lags - the spatial lags of the 2D correlation. Can be either a scalar
#%          (for equal number of lags in row and column dimensions) or a
#%          2-element vector. If empty, the correlation will be computed
#%          over the maximum number of lags. (default=[])
#%   TimeLags - The 2D correlation will be computed over the specified time
#%              lags. If empty, the auto correlation (i.e. time lag=0) will
#%              be computed. Otherwise, a vector of desired time lags should
#%              be specified. (default=0). The output array c will contain
#%              the cross correlations for each time lag along the 3rd
#%              dimension.
#%
#%  [c,n]=XCORR_PDF2(...) Also returns for each time lag the number of
#%  correlations that contributed to the average correlations in c.
#%
#%  [c,n,lags,timelags]=XCORR_PDF2(...) Also returns a cell array with
#%  spatial lags and a vector with time lags.
#%
#%  [c,n,lags,timelags,l]=XCORR_PDF2(...) Also returns for each separate
#%  array in the cell arrays x and y and for each time lag, array with raw
#%  correlation between slices.
#%

#%  Copyright 2009 Fabian Kloosterman

#%% Check arguments
#if nargin<1
    #help(mfilename)
    #return
#end

#options = struct('Lags', [], 'TimeLags', 0 );
#[options,other] = parseArgs( varargin, options );

#if isnumeric(x)
    #x = {x};
#elseif ~iscell(x)
    #error('xcorr_pdf2:invalidArguments', 'A numeric array or cell array is needed')
#end

#ncells = numel(x);

#if isempty(other)
    #y = x;
#elseif isscalar(other)
    #if isempty(other{1})
        #y = x;
    #elseif ~iscell(other{1})
        #y=other;
    #else
        #y=other{1};
    #end
#end

#if ncells~=numel(y) || ~all( cellfun( 'isclass', x, 'double' ) ) || ...
        #~all( cellfun( 'isclass', y, 'double' ) )
    #error('xcorr_pdf2:invalidArguments', 'This function requires double arrays or cell arrays with double arrays');
#end

#nd = cellfun( 'ndims', x );

#if ~isequal( nd, cellfun( 'ndims', y ) ) || any(nd)>3
    #error('xcorr_pdf2:invalidArguments', 'Arrays need to have at most 3 dimensions')
#end

#sz = cellfun( @(a) size(a), x, 'UniformOutput', false );

#if ~isequal( sz, cellfun( @(a) size(a), y, 'UniformOutput', false ) )
    #error('xcorr_pdf2:invalidArguments', 'X and Y arrays need to have the same dimensions')
#end

#for k=1:ncells
    #sz{k} = cat( 2, sz{k}, ones(1,3-nd(k)) );
#end

#sz = vertcat( sz{:} );

#maxlag = max(sz(:,1:2),[],1)-1;
#%maxtimelag = max(sz(:,3))-1;

#if isempty(options.Lags)
    #%pass
#elseif ~isnumeric(options.Lags) || ~all((options.Lags)>=0 | isnan(options.Lags)) || (~isscalar(options.Lags) && (~isvector(options.Lags) || numel(options.Lags)~=2))
    #error('xcorr_pdf2:invalidArgument', 'Invalid spatial lags vector')
#else
    #if isscalar(options.Lags)
        #options.Lags = ones(1,2).*options.Lags;
    #end
    
    #maxlag( ~isnan(options.Lags) ) = round( options.Lags(~isnan(options.Lags)));
    
#end

#if isempty(options.TimeLags)
    #timelags = 0;
#elseif ~isnumeric(options.TimeLags) || ~isvector(options.TimeLags) || any(options.TimeLags<0)
    #error('xcorr_pdf2:invalidArgument', 'Invalid time lags vector');
#else
    #timelags = round( options.TimeLags(:)' );
#end

#lags = { (-maxlag(1):maxlag(1))' -maxlag(2):maxlag(2) };

#[f, fftsize] = log2( 2*max(sz(:,1:2),[],1)-1 );
#fftsize( f==0.5) = fftsize( f==0.5 ) -1; %exact power of two
#fftsize = 2.^fftsize;

#idx = cell(1,2);
#for k=1:2
   #idx{k} = [ (fftsize(k) - maxlag(k) + 1):fftsize(k) 1:(maxlag(k)+1)];
#end

#fcn = @(a,b) freqxcorr(a,b,fftsize,idx);
#%fcn = @(a,b) xcorr2(a,b);

#C = zeros( numel(lags{1}), numel(lags{2}), numel(timelags) );
#n = zeros(numel(timelags),1);

#valid = ~isnan( sum( cat(3,x{:}, y{:} ), 3 ) );
#Cvalid = round( fcn( double(valid), double(valid) ) );

#LagData = cell(ncells,numel(timelags));

#for b = 1:ncells

    #tmp = repmat( ~valid, [1 1 sz(b,3)] );
    #x{b}(tmp) = 0;
    #y{b}(tmp) = 0;
    
    #for L = 1:numel(timelags)
        
        #lag = timelags(L);
        
        #if nargout>4
            #LagData{b,L} = zeros(numel(lags{1}), numel(lags{2}), max(0,sz(b,3)-lag) );
        #end
        
        #if (lag+1)>sz(b,3)
            #continue;
        #end

        #for S = 1:(sz(b,3)-lag)
            
            #tmp = ( fcn( x{b}(:,:,lag+S), y{b}(:,:,S) ) );
            
            #if nargout>4
                #LagData{b,L}(:,:,S) = tmp;
            #end
            
            #C(:,:,L) = C(:,:,L) + tmp;
            
            #n(L) = n(L) + 1;
            
        #end
        
    #end
    
    
    
#end

#C = bsxfun( @rdivide, C, shiftdim( n, -2 ) );

#C( repmat(Cvalid, [1, 1, size(C,3)])==0 ) = NaN;

#end

#function c = freqxcorr(x,y,fftsize,idx)
#fx = fftn( x, fftsize );
#fy = fftn( y, fftsize );
#c = real( ifftn( fx .* conj(fy) ) );
#c = c(idx{:});
#end

#def _freq_xcorr( x, y=None, fftsize=None, selection=None ):
    #fx = sp.fftpack.fftn( x, shape=fftsize )
    
    #if y is None:
        #fy = fx
    #else:
        #fy = sp.fftpack.fftn( y, shape=fftsize )
    
    #c = np.real( sp.fftpack.ifftn( fx * np.conjugate(fy) ) )
    
    #if selection is not None:
        #c = c[ np.ix_( *selection ) ]
    
    #return c
