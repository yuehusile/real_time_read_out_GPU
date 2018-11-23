"""
========================================================
General utilities (:mod:`fklab.utilities.general`)
========================================================

.. currentmodule:: fklab.utilities.general

General utiltity functions.

.. autosummary::
    :toctree: generated/
    
    check_vector
    check_vector_list
    
    issorted
    isascending
    isdescending
    
    partition_vector
    partitions
    blocks
    slices
    slicedarray
    
    inrange
    natural_sort
    
    randomize

"""


from __future__ import division

import re

import numpy as np
import numba

__all__ = ['check_vector', 'check_vector_list', 'issorted','isascending',
           'isdescending','partition_vector','partitions','blocks','slices',
           'slicedarray','inrange','natural_sort', 'randomize']

def check_vector(x, copy=True, real=True):
    """Convert to numpy vector if possible.
    
    Parameters
    ----------
    x : array-like
    copy : bool
        the output vector will always be a copy of the input
    real : bool
        check if vector is real-valued
    
    Returns
    -------
    1d array
    
    """
    
    try:
        x = np.array(x,copy=copy)
    except TypeError:
        raise ValueError('Cannot convert data to numpy array')
    
    if x.size==1 or x.size==max( x.shape ):
        x = x.ravel()
    else:
        raise ValueError('Data is not arranged as a vector')
    
    # The array needs to contain real values only.
    if real and not np.isrealobj(x):
        raise ValueError('Values are not real numbers')
    
    return x

def check_vector_list(x, copy=True, same_shape=False, real=True):
    """Convert to sequence of numpy vectors.
    
    Parameters
    ----------
    x : array-line or sequence of array-likes
    copy : bool
        the output vectors will always be copies of the inputs
    same_shape : bool
        the output vectors will have the same shape
    real : bool
        check if vectors are real-valued
    
    Returns
    -------
    tuple of 1d arrays
    
    """
    
    if isinstance( x, np.ndarray ):
        x = ( check_vector( x, copy=copy, real=real ), )
    else:
        x = tuple( [ check_vector( y, copy=copy, real=real ) for y in x ] )
    
    if same_shape and len(x)>1:
        if not all( [ x[0].shape == y.shape for y in x ] ):
            raise ValueError('Vectors have different shapes.')
    
    return x


def issorted(x,strict=False):
    """
    Tests if vector is sorted.
    
    Parameters
    ----------
    x : array_like
        will be converted into a vector before testing if it is sorted.
    strict : bool, optional
        values in `x` should be strictly monotonically increasing or
        decreasing (i.e. vectors with equal values are not considered
        strictly sorted).
    
    Returns
    -------
    bool
        whether input vector is sorted.
    
    See Also
    --------
    isascending, isdescending
    
    """
    x = np.asarray(x).ravel()
    return _issorted(x,strict)

def isascending(x,strict=False):
    """
    Tests if values in vector are in ascending order.
    
    Parameters
    ----------
    x : array_like
        will be converted into a vector before testing the ordering.
    strict : bool, optional
        values in `x` should be strictly monotonically increasing
        (i.e. vectors with equal values are not considered strictly
        sorted).
    
    Returns
    -------
    bool
        whether input vector is sorted in ascending order.
    
    See Also
    --------
    issorted, isdescending
    
    """
    x = np.asarray(x).ravel()
    return _isascending(x,strict)

def isdescending(x,strict=False):
    """
    Tests if values in vector are in descending order.
    
    Parameters
    ----------
    x : array_like
        will be converted into a vector before testing the ordering.
    strict : bool, optional
        values in `x` should be strictly monotonically decreasing
        (i.e. vectors with equal values are not considered strictly
        sorted).
    
    Returns
    -------
    bool
        whether input vector is sorted in descending order.
    
    See Also
    --------
    issorted, isascending
    
    """
    x = np.asarray(x).ravel()
    return _isdescending(x,strict)

@numba.jit(nopython=True, nogil=True)
def _issorted(x,strict):
    n = x.shape[0]
    flag = True
    
    if n<3:
        flag=True
    else:
        if strict:
            if x[1]>x[0]:
                for k in range(1,n):
                    if x[k+1]<=x[k]:
                        flag=False
                        break
            elif x[1]<x[0]:
                for k in range(1,n):
                    if x[k+1]>=x[k]:
                        flag=False
                        break
            else:
                flag=False
        else:
            for k in range(0,n-1):
                if x[k]!=x[k+1]:
                    pos = x[k]<x[k+1]
                    for j in range(k+1,n-1):
                        if (x[j]>x[j+1] and pos) or (x[j]<x[j+1] and not pos):
                            flag=False
                            break
                    break
    
    return flag

@numba.jit(nopython=True, nogil=True)
def _isascending(x,strict):
    n = x.shape[0]
    flag = True
    
    if n<2:
        flag=True
    else:
        if strict:
            for k in range(n-1):
                if (x[k+1]-x[k])<=0:
                    flag=False
                    break
        else:
            for k in range(n-1):
                if (x[k+1]-x[k])<0:
                    flag=False
                    break
    
    return flag

@numba.jit(nopython=True, nogil=True)
def _isdescending(x,strict):
    n = x.shape[0]
    flag=True
    
    if n<2:
        flag=True
    else:
        if strict:
            for k in range(n-1):
                if (x[k+1]-x[k])>=0:
                    flag=False
                    break
        else:
            for k in range(n-1):
                if (x[k+1]-x[k])>0:
                    flag=False
                    break
    
    return flag


def partition_vector( v, **kwargs ):
    """Partitions vector into subvectors.
    
    See :func:`partitions` for detailed help on keyword arguments.
    
    Parameters
    ----------
    v : array_like, can be indexed with numpy array
    
    Returns
    -------
    tuple of subvectors
    
    See Also
    --------
    partitions 
    
    """
    kwargs['size'] = len(v)
    p = partitions( **kwargs )
    return ( v[idx] for idx in p )

def partitions( size=None, partsize=None, nparts=None, method='block', keepremainder=True ):
    """Partition elements in multiple groups.
    
    Parameters
    ----------
    size : int, optional
        Number of elements to partition. If not given, `size` will be
        calculated as `nparts` * `partsize`
    partsize : int, optional
        Number of elements in a partition. If not given, 'partsize' will
        be calculated as ceil( `size` / `nparts` ).
    nparts : int, optional
        Number of partitions. If not given, 'nparts' will be calculated
        as ceil( `size` / `partsize` ).
    method : {'block','random','sequence'}, optional
        Partitioning method. 'block': first `partsize` elements are 
        assigned to partition 1, second `partsize` elements to partition
        2, etc. 'random': each elements is randomly assigned to a
        partition. 'sequence': elements are distributed to partitions in
        order.
    keepremainder : bool, optional
        Whether or not to keep remaining elements that are not part of a
        partition (default is True).
    
    Returns
    -------
    tuple of 1D arrays
        Each array contains the indices of the elements in a partition
    
    """
    args = [size is None, partsize is None, nparts is None]
    if all(args) or not any(args):
        raise ValueError
    
    if size is None:
        if partsize is None:
            partsize = 1
        elif nparts is None:
            nparts = 1
        size = nparts * partsize
    elif partsize is None:
        if nparts is None:
            partsize = 1
            nparts = size
        else:
            partsize = np.int( np.ceil( size/nparts ) )
    else:
        nparts = np.int( np.ceil( size/partsize ) )
    
    if nparts*partsize>size and not keepremainder:
        if args[1]: #partsize was None
            partsize = np.int( np.floor(size/nparts) )
        elif args[2]: #nparts was None
            nparts = np.int( np.floor( size/partsize ) )
        else:
            raise InternalError
        size = nparts*partsize
    
    if method == 'block':
        idx = np.floor(np.arange(size)/partsize)
    elif method == 'random':
        idx = np.floor(np.arange(size)/partsize)
        np.random.shuffle(idx)
    elif method == 'sequence':
        idx = np.remainder(np.arange(size),nparts)
    else:
        raise TypeError('Method argument should be one of block, random, sequence')
    
    return (np.nonzero(idx==k)[0] for k in np.arange(nparts))

def blocks(nitems=1,blocksize=1):
    """Iterates over a smaller blocks of a large number of items.
    
    Parameters
    ----------
    nitems : integer
        number of items to iterate over in blocks
    blocksize : integer
        size of the block
    
    Returns
    -------
    start : integer
        start index of the block
    n : integer
        number of items in the block
    
    Examples
    --------
    Compute the averages of adjacent non-overlapping blocks of 10 values
    in a length 100 vector filled with random numbers.
    
    >>> data = np.random.uniform(low=0,high=1,size=100)
    >>> m = np.array( [np.mean( data[start:(start+n)] ) for start,n in blocks( nitems=len(data), blocksize=10 )] )
    array([ 0.45686263,  0.52700117,  0.50307317,  0.44052573,  0.68276929,
            0.56265324,  0.58711927,  0.59307625,  0.58343556,  0.56201659]) #random
    
    """
    
    start=0
    n = blocksize if blocksize<nitems else nitems
    
    while n>0:
        yield start,n
        start+=n
        n = blocksize if start+blocksize<=nitems else nitems-start

def slices( n, size=1, start=0, shift=None, step=None, strictsize=False ):
    """ Iterates over slices in a range.
    
    Parameters
    ----------
    n : integer or iterable, required
        total number of items to iterate over. If an iterable is given, 
        the length of the iterable is used.
    size : integer
        size of the slice (default = 1)
    start : integer
        starting index (default = 0)
    shift : integer
        the number of items by which each slice is shifted, allowing for
        overlap or gaps between adjacent slices (default = size)
    step : integer
        step argument for returned slice object (default = None)
    strictsize : bool
        if True, the slice size is not adjusted at the end of the range 
        and the final items may be skipped (default = False)
    
    Returns
    -------
    slice
        a slice object that can be used for indexing
    
    Examples
    --------
    The following example select slices of length 5 from a length 20
    vector. The first slice starts at index 5 and each next slice is
    shifted by 3 elements (i.e. slices overlap). `strictsize` is set to
    True, which means that the last elements in data will be skipped if
    a slice with length 10 cannot be created.
    
    >>> data = range(20)
    >>> [data[selection] for selection in slices( data, size=5, start=5, shift=3, strictsize=True )]
    [[5, 6, 7, 8, 9], [8, 9, 10, 11, 12], [11, 12, 13, 14, 15], [14, 15, 16, 17, 18]]
    
    """
    
    try:
        n = len(n)
    except TypeError:
        n = int(n)
    
    start = int(start)
    
    size = size if size<n else n
    
    if shift is None:
        shift = size
    else:
        shift = int(shift)
        if shift <=0:
            raise ValueError( "Invalid value for shift" )
    
    while size>0:
        
        yield slice(start,start+size,step)
        
        start += shift
        
        if start+size>n:
            if strictsize:
                break
            else:
                size = n-start
            
def slicedarray( x, size=1, axis=0, **kwargs ):
    """Iterates over sub-arrays.
    
    Parameters
    ----------
    x : array
        numpy array to iterate over
    size : integer
        size of the sub-array (default = 1)
    axis : integer
        axis along which to iterate (default = 0)
    start : integer
        starting index (default = 0)
    shift : integer
        the number of items by which each slice is shifted, allowing for
        overlap or gaps between adjacent slices (default = size)
    step : integer
        step argument for returned slice object (default = None)
    strictsize : bool
        if True, the slice size is not adjusted at the end of the range 
        and the final items may be skipped (default = False)
    
    Returns
    -------
    array
        sub array of original array
    
    See Also
    --------
    slices
    
    """
    
    x = np.asarray( x )
    
    indices = [ slice(None) ] * x.ndim
    
    for s in slices( x.shape[axis], size, **kwargs ):
        indices[ axis ] = s
        yield x[ tuple(indices) ]


def inrange(x,low=None,high=None,include_boundary=True):
    """Tests if values are in range.
    
    The range is defined by a lower (`low`) and upper (`high`) boundary.
    A value of None indicates no boundary. If the upper boundary is
    smaller than the lower boundary, the the range is inverted. For example,
    if low=10 and high=5, then all values in x that are smaller than 5 or
    larger than 10 are within range.
    
    Parameters
    ----------
    x : array-like
        data values to test
    low : scalar number
        lower boundary of the range, default is `None` (no lower boundary)
    high : scalar number
        upper boundary of the range, default is `None` (no upper boundary)
    include_boundary : bool
        whether or not the boundaries are included, default is `True`
    
    Returns
    -------
    bool array
        True for all values in x that are within range
    
    Examples
    --------
    >>> b = inrange( [1,2,3,4,5] )
    array([ True,  True,  True,  True,  True], dtype=bool)
    
    >>> b = inrange( [1,2,3,4,5], low=3 )
    array([ False,  False,  True,  True,  True], dtype=bool)
    
    >>> b = inrange( [1,2,3,4,5], high=2 )
    array([ True,  True, False, False, False], dtype=bool)
    
    >>> b = inrange( [1,2,3,4,5], low=2, high=4 )
    array([False,  True,  True,  True, False], dtype=bool)
    
    >>> b = inrange( [1,2,3,4,5], low=4, high=2 )
    array([ True,  True, False,  True,  True], dtype=bool)
    
    """
    
    x = np.asarray(x)
    
    if include_boundary:
        op_low = np.greater_equal
        op_high = np.less_equal
    else:
        op_low = np.greater
        op_high = np.less
    
    if low is None:
        if high is None:
            return np.ones( x.shape, dtype=np.bool8 )
        else:
            return op_high( x, high )
    
    if high is None:
        return op_low( x, low )
    
    if high>=low:
        return np.logical_and( op_low(x, low), op_high(x, high) )
    else:
        return np.logical_or( op_low(x, low), op_high(x, high) )

def natural_sort( iter, reverse=False ):
    """Sorts iterable with strings in natural order.
    
    Parameters
    ----------
    iter : iterable
    reverse : bool, optional
    
    Returns
    -------
    sorted list
    
    """
    
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(iter, key=alphanum_key, reverse=reverse)


def randomize( a, axis=0, group=None, method='shuffle' ):
    """Randomize array along axis.
    
    Parameters
    ----------
    a : ndarray
    axis : int
        array dimension along which to perform randomization.
    group : int or sequence
        array dimensions that will be randomized coherently with `axis`.
    method : 'shuffle' or 'roll'
        Either randomly permute or roll (with wrapping) array values
        along `axis`.
    
    Returns
    -------
    out : ndarray
        array with randomized values
    index : list of index arrays
        such that a.flat[ np.ravel_multi_index( index, a.shape, mode='wrap' )]
        returns the randomized array.
    
    """
    
    if group is None:
        group = []
    elif not isinstance( group, (list, tuple) ):
        group = [ int(group) ]
    
    axis = int(axis) % a.ndim
    group = [ int(x) % a.ndim for x in group ]
    
    if axis in group:
        raise ValueError("Group axes have to be different from rolling axis.")
    
    if not method in ('shuffle', 'roll'):
        raise ValueError('Invalid randomization method')
    
    shapes = np.ones( (a.ndim, a.ndim), dtype=np.int )
    shapes[ np.diag_indices( a.ndim ) ] = a.shape
    
    indices = [ np.arange(n).reshape( shapes[d] ) for d,n in enumerate(a.shape) ]
    
    if method=='roll':
        random_offset = np.random.randint( a.shape[axis], size=[ 1 if (d==axis or d in group) else n for d,n in enumerate(a.shape) ] )
        indices[ axis ] = indices[ axis ] - random_offset
    else: # 'shuffle'
        indices[ axis ] = np.argsort( np.random.uniform(0,1, size=[ 1 if d in group else n for d,n in enumerate(a.shape) ]), axis=axis )
    
    return a.flat[np.ravel_multi_index( indices, a.shape, mode='wrap' )], indices
