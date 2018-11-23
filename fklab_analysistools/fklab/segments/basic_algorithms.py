"""
===========================================================
Segment algorithms (:mod:`fklab.segments.basic_algorithms`)
===========================================================

.. currentmodule:: fklab.segments.basic_algorithms

Provides basic algorithms for lists of segments.

.. autosummary::
    :toctree: generated/
    
    segment_sort
    segment_has_overlap
    segment_remove_overlap
    segment_invert
    segment_exclusive
    segment_union
    segment_difference
    segment_intersection
    segment_scale
    segment_concatenate
    segment_contains
    segment_count
    segment_overlap
    segment_asindex
    segment_join
    segment_split
    segment_applyfcn
    segment_uniform_random

"""

from __future__ import division

import operator
import numpy as np

__all__ = ['segment_sort','segment_has_overlap', 
           'segment_remove_overlap','segment_invert','segment_exclusive',
           'segment_union','segment_difference','segment_intersection',
           'segment_scale','segment_concatenate','segment_contains',
           'segment_count','segment_overlap','segment_asindex','segment_join',
           'segment_split','segment_applyfcn','segment_uniform_random',
           'check_segments']


def check_segments(x, copy=False):
    """Convert to segment array.
    
    Parameters
    ----------
    x : 1d array-like or (n,2) array-like
    copy : bool
        the output will always be a copy of the input
    
    Returns
    -------
    (n,2) array
    
    """
    
    try:
        x = np.array(x,copy=copy)
    except TypeError:
        raise ValueError('Cannot convert data to numpy array')
    
    # The array needs to contain real values only.
    # Is this a proper general test for numbers?    
    if not np.isrealobj(x):
        raise ValueError('Values are not real numbers')
    
    # The array has to have two dimensions of shape(X,2), where X>=0.
    # As a special case, a one dimensional vector of at least length two is considered
    # a valid list of segments, e.g. when data is specified as a list [0,2,3].
    
    if x.shape == (0,):
        x = np.zeros([0,2])
    elif x.ndim == 1 and len(x) > 1:
        x = np.vstack( (x[0:-1],x[1:]) ).T
    elif x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Incorrect array size')
    
    # Negative duration segments are not allowed.
    if np.any( np.diff( x, axis=1 ) < 0 ):
        raise ValueError('Segment durations cannot be negative')
    
    return x

def segment_sort(segments):
    """Sort segments by start time.
    
    Parameters
    ----------
    segments : segment array
    
    Returns
    -------
    sorted segments
    
    """
    segments = check_segments(segments,copy=False)
    if segments.shape[0]>1:
        idx = np.argsort(segments[:,0])
        segments = segments[idx,:]
    return segments

def segment_has_overlap(segments):
    """Check for overlap of segments.
    
    Parameters
    ----------
    segments : segment array
    
    Returns
    -------
    bool
        True if any of the segments overlap.
    
    """
    segments = check_segments(segments,copy=True)
    segments = segment_sort(segments)
    
    if np.any(segments[1:,0]<segments[:-1,1]):
        return True
    else:
        return False
    
def segment_remove_overlap(segments,strict=True):
    """Remove overlap between segments.
    
    Segments that overlap are merged.
    
    Parameters
    ----------
    segments : segment array
    strict : bool
        Only merge two segments if the end time of the first is stricly
        larger than (and not equal to) the start time of the second segment.
    
    Returns
    -------
    segments without overlap
    
    """
    
    
    segments = check_segments(segments,copy=False)
    
    n = segments.shape[0]
    if n==0:
        return segments
    
    segments = segment_sort(segments)
    
    s = segments[:1,:]
    
    if strict:
        fcn = operator.lt
    else:
        fcn = operator.le
    
    for k in range(1,n):
        if fcn( s[-1,1], segments[k,0] ):
            s = np.concatenate([s,segments[k:k+1,:]])
        else:
            s[-1,1] = np.maximum(segments[k,1],s[-1,1])
    
    return s

def segment_invert(segments):
    """Invert segments.
    
    Constructs segments from the inter-segment intervals.
    
    Parameters
    ----------
    segments : segment array
    
    Returns
    -------
    segments
    
    """
    segments = check_segments(segments,copy=False)
    segments = segment_remove_overlap( segments )
    n = len(segments)
    seg = np.concatenate( ([-np.Inf], segments.ravel(), [np.Inf]) ).reshape((n+1,2))
    if np.all(seg[0,:]==[-np.Inf,-np.Inf]):
        seg = np.delete(seg,0,0)
    
    if np.all(seg[-1,:]==[np.Inf,np.Inf]):
        seg = np.delete(seg,-1,0)
        
    return seg

def segment_exclusive(segments,*others):
    """Exclusive operation.
    
    Extracts parts of segments that do not overlap with any other segment.
    
    Parameters
    ----------
    segments : segment array
    *others : segment arrays
    
    Returns
    -------
    segments
    
    """
    #nothing to do if no other segments are provided
    segments = check_segments(segments,copy=False)
    if len(others)==0:
        return segments
    
    #combine all other segment lists and invert
    others = segment_union( *others )
    others = segment_invert( others )
    
    return segment_intersection( segments, others )

def segment_union(*args):
    """Combine segments (logical OR).
    
    Parameters
    ----------
    *args : segment arrays
    
    Returns
    -------
    segments
    
    """
    
    data = np.zeros( (0,2) )
    
    for obj in args:
        data = np.concatenate( (data,check_segments(obj,copy=False)) )
    
    data = segment_remove_overlap(data)
    
    return data

def segment_difference(*args):
    """Difference between segments (logical XOR).
    
    Parameters
    ----------
    *args : segment arrays
    
    Returns
    -------
    segments
    
    """
    tmp = segment_invert( segment_intersection(*args) )
    return segment_intersection( segment_union(*args), tmp )

def segment_intersection(*args):
    """Intersection between segments (logical AND).
    
    Parameters
    ----------
    *args : segment arrays
    
    Returns
    -------
    segments
    
    """
    if len(args)==0:
        return np.zeros( (0,2) )
    
    segment_list = [ segment_remove_overlap(x) for x in args ]
    
    if len(segment_list)==1:
        return segment_list[0]
    
        
    segment_stack = segment_list[0]
    
    for iseg in segment_list:
        
        overlap = np.zeros([0,2])
        
        for k in range(segment_stack.shape[0]):
            
            b = np.logical_and(segment_stack[k,0] <= iseg[:,0], segment_stack[k,1] > iseg[:,0] )
            b = np.logical_or(b, np.logical_and( iseg[:,0] <= segment_stack[k,0], iseg[:,1] > segment_stack[k,0] ) )
            if sum(b)>0:
                overlap_start = np.maximum( segment_stack[k,0], iseg[b,0] )
                overlap_stop = np.minimum( segment_stack[k,1], iseg[b,1] )
                overlap_new = np.vstack([overlap_start, overlap_stop]).T
                overlap = np.concatenate([overlap, overlap_new])
        
        if overlap.shape[0] == 0:
            break
        
        segment_stack = overlap
    
    return overlap

def segment_scale(segments, value, reference=0.5):
    """Scale segment durations.
    
    Parameters
    ----------
    segments : segment array
    value : scalar or 1d array
        Scaling factor
    reference: scalar or 1d array
        Relative reference point in segment used for scaling. A value of
        0.5 means symmetrical scaling around the segment center. A value
        of 0. means that the segment duration will be scaled without
        altering the start time.
        
    Returns
    -------
    segments
    
    """
    value = np.array(value,dtype=np.float64).squeeze()
    reference = np.array(reference,dtype=np.float64).squeeze()
    segments = check_segments(segments,copy=False)
    
    if value.ndim==0:
        value = value.reshape([1])
        
    if value.ndim == 1:
        value = value.reshape([len(value),1])
        value = np.diff(segments, axis=1) * (value-1)
        value = np.concatenate([-reference*value,(1-reference)*value], axis=1)
        segments = segments + value
    else:
        raise ValueError('Invalid shape of scaling value')
        
    return segments

def segment_concatenate(*args):
    """Concatenate segments.
    
    Parameters
    ----------
    *args : segment arrays
    
    Returns
    -------
    segments
    
    """
    if len(args)==0:
        return np.zeros( (0,2) )
    
    segments = np.concatenate( [ check_segments(x,copy=True) for x in args ], axis=0 )
    return segments
        
def segment_contains( segment, x, issorted=True, expand=None ):
    """Test if values are contained in segments.
    
    Segments are considered left closed and right open intervals. So, 
    a value x is contained in a segment if start<=x and x<stop. 
    
    Parameters
    ----------
    segment : segment array
    x : 1d array
    issorted : bool
        Assumes vector `x` is sorted and will not sort it internally.
        Note that even if `issorted` is False, the third output argument
        will still return indices into the (internally) sorted vector.
    expand : bool
        Will expand the last output to full index arrays into 'x' for
        each segment. The default is True if `issorted` is False and
        vice versa. Note that for non-sorted data (`issorted` is False) and
        `expand`=False, the last output argument will contain start and stop
        indices into the (internally) sorted input array.
    
    Returns
    -------
    ndarray
        True for each value in `x` that is contained within any segment.
    ndarray
        For each segment the number of values in `x` that it contains.
    ndarray
        For each segment, the start and end indices of values in SORTED
        vector `x` that are contained within that segment.
    
    """
    
    x = np.array(x).ravel()
    
    if expand is None:
        expand = not issorted
    
    if not issorted:
        sort_indices = np.argsort( x )
        x = x[sort_indices]
    
    segment = check_segments(segment, copy=False)
    nseg = segment.shape[0]
    nx = len(x)
    
    xp = 0 #index of current x value
    xfillp = 0 #index of last x value that has been tested
    
    isinseg = np.zeros(x.shape, dtype=np.bool) #True for each x inside any of the segments
    contains = -1 * np.ones(segment.shape,dtype=np.int) #for each segment start and end index of x-values that are contained within the segment
    ninseg = np.zeros(segment.shape[0], dtype=np.int) #for each segment number of x-values it contains
    
    if nx>0:
        
        #loop through all segments
        for sp in range(nseg):
            
            if x[xp]<segment[sp,0]: #current x is before segment start
                # find first x inside segment
                idx = np.searchsorted( x[xp:], segment[sp,0], side='left' ) #switch to 'right' to make left open interval
                if (idx+xp)>=nx: #no x inside segment found
                    break
                if x[idx+xp]>=segment[sp,1]: #Use '>' to make right closed interval
                    continue
                xp += idx #update current x
                contains[sp,0] = xp #mark first x for this segment
            elif x[xp]>=segment[sp,1]: #x-value is past current segment, go to next segment. Use '>' to make right closed interval
                continue
            else:
                contains[sp,0] = xp #mark first x for this segment
            
            
            #find last x in segment and mark it
            xlastp = xp + np.searchsorted( x[xp:], segment[sp,1], side='left' ) - 1  #switch to 'right' to make right closed interval
            contains[sp,1] = xlastp
            
            #count number of x values in segment
            ninseg[sp] = contains[sp,1] - contains[sp,0] + 1
            
            #fast forward current fill index if needed
            if xp>xfillp:
                xfillp = xp
            
            #mark x-values as contained
            isinseg[xfillp:(xlastp+1)] = 1
            
            # set new current fill index
            if xfillp < xlastp:
                xfillp = xlastp
    
    if not issorted:
        isinseg = isinseg[ np.argsort(sort_indices) ]
    
    if expand:
        if issorted:
            contains = [ np.arange(start,stop+1) for start,stop in contains ]
        else:
            contains = [ sort_indices[start:stop+1] for start,stop in contains ]
    
    return isinseg, ninseg, contains

def segment_count(segments,x):
    """Count number of segments.
    
    Parameters
    ----------
    segments : segment array
    x : ndarray
    
    Returns
    -------
    ndarray
        For each value in `x` the number of segments that contain that value.
    
    """
    
    segments = check_segments(segments,copy=False)
    x = np.array(x)
    x_shape = x.shape
    x = x.ravel()
    n = segments.shape[0]
    nx = len(x)
    
    tmp = np.concatenate( [np.vstack([segments[:,0],np.zeros(n)]), np.vstack( [x, np.ones(nx)] )], axis=1 )
    tmp = tmp[:,tmp[0,:].argsort(kind='mergesort')]
    idx = tmp[1,:].nonzero()
    tmp_cs = np.cumsum(tmp[1,:], axis=0)
    seg_start = idx - tmp_cs[idx]
    
    tmp = np.concatenate( [np.vstack([segments[:,1],np.zeros(n)]), np.vstack( [x, np.ones(nx)] )], axis=1 )
    tmp = tmp[:,tmp[0,:].argsort(kind='mergesort')]
    idx = tmp[1,:].nonzero()
    tmp_cs = np.cumsum(tmp[1,:], axis=0)
    seg_end = idx - tmp_cs[idx]
    
    return (seg_start - seg_end).reshape(x_shape)

def segment_overlap(segments,other=None):
    """Returns absolute and relative overlaps between segments.
    
    Parameters
    ----------
    segments : segment array
    other : segment array, optional
        If `other` is not provided, then overlaps within `segments` are
        analyzed.
    
    Returns
    -------
    ndarray
        absolute overlap between all combinations of segments
    ndarray
        overlap relative to duration of first segment
    ndarray
        overlap relative to duration of second segment
    
    """
    
    segments = check_segments(segments,copy=False)
    
    if other is None:
        other = segments
    else:
        other = check_segments(other,copy=False)
    
    nA = len(segments)
    nB = len(other)
    
    LA = np.diff( segments, axis=1 ).reshape([nA,1])
    LB = np.diff( other, axis=1).reshape([1,nB])
    
    delta = np.mean(other,axis=1).reshape([1,nB]) - np.mean(segments,axis=1).reshape([nA,1])
    
    out1 = np.maximum( 0, np.minimum( -np.abs(delta) + 0.5*np.abs(LB-LA), 0 ) + np.minimum(LA,LB) )
    
    out2 = out1 / LA
    out3 = out1 / LB
    
    return out1, out2, out3

def segment_asindex(segments,x):
    """Convert segments to indices into vector.
    
    Parameters
    ----------
    segments : segment array
    x : ndarray
    
    Returns
    -------
    segments (indices)
    
    """
    x = np.array(x).squeeze()
    b = segment_contains(segments,x)[0]
    
    b = np.diff(np.concatenate(([0], b, [0])));
    
    seg = np.vstack( ((b==1).nonzero()[0], (b==-1).nonzero()[0] - 1 ) ).T
    
    return seg

def segment_join(segments,gap=0):
    """Join segments with small inter-segment gap.
    
    Parameters
    ----------
    segments : segment array
    gap : scalar
        Segments with an interval equal to or smaller than `gap` will be
        merged.
    
    Returns
    -------
    segments
    
    """
    segments = segment_remove_overlap(segments)
    intervals = segments[1:,0] - segments[:-1,1]
    idx = (intervals<=gap).nonzero()[0]
    if len(idx)>0:
        combiseg = np.concatenate( (segments[idx,:1], segments[idx+1,1:]), axis=1 )
        segments = segment_union( segments, combiseg )
        
    return segments

def segment_split(segments, size=1, overlap=0, join=True, tol=1e-7):
    """Split segments into smaller segments with optional overlap.
    
    Parameters
    ----------
    segments : segment array
    size : scalar
        Duration of split segments.
    overlap : scalar
        Relative overlap (>=0. and <1.) between split segments.
    join : bool
        Join all split segments into a single segment array. If `join` is
        False, a list is returned with split segments for each original
        segment separately.
    tol : scalar
        Tolerance for determining number of bins.
        
    Returns
    -------
    segments or list of segments
    
    """
    segments = check_segments(segments,copy=False)
    nbins = (np.diff(segments,1,axis=1).ravel() - overlap*size) / ((1-overlap)*size)
    idx = (np.ceil(nbins)-nbins) < tol
    
    nbins[idx] = np.ceil(nbins[idx]);
    nbins[~idx] = np.floor(nbins[~idx]);
    
    nbins = nbins.astype(np.int)

    n = len(nbins)
    
    seg = []
    if overlap==0:
        for k in range(0,n):
            tmp = np.arange(0,nbins[k]+1).reshape((nbins[k]+1,1))*size;
            seg.append( segments[k,0]+np.concatenate((tmp[0:-1],tmp[1:]),axis=1) )
    else:
        for k in range(0,n):
            tmp = np.arange(0,nbins[k]).reshape((nbins[k],1))*(1-overlap)*size + segments[k,0]
            seg.append(np.concatenate((tmp,tmp+size),axis=1))
    
    if join:
        if len(seg)==0:
            seg = np.zeros( (0,2) )
        else:
            seg = np.concatenate(seg, axis=0)
    
    return seg

def segment_applyfcn(segments,x,*args,**kwargs):
    """Apply function to segmented data.
    
    Parameters
    ----------
    segments : segment array
    x : ndarray
        The function is applied to values in this array that lie within
        the segments.
    separate : bool
        Apply function to data in each segment separately
    function : callable
        Function that takes one or more data arrays.
    default : any
        Default value for segments that do not contain data (only used
        when separate is True)
    *args : ndarray-like
        Data arrays that are segmented (along first dimension) according
        to the corresponding values in `x` that lie within the segments,
        and passed to `function`. 
    
    Returns
    -------
    ndarray or [ ndarray, ]
        Result of applying function to segmented data.
    
    """
    
    b,nn,b2 = segment_contains( segments, x )
    
    separate = bool( kwargs.get( 'separate', False ) )
    function = kwargs.get( 'function', len )
    default = kwargs.get( 'default', None )
    
    if len(args)==0:
        if not separate:
            data = function( x[b] )
        else:
            data = [function(x[ii[0]:(ii[1]+1)]) if ii[0]>=0 and ii[0]<=ii[1] else default for ii in b2]
    else:
        if not separate:
            data = function(*[y[b] for y in args])
        else:
            data = [function(*[y[ii[0]:(ii[1]+1)] for y in args]) if ii[0]>=0 and ii[0]<=ii[1] else default for ii in b2]
    
    return data

def segment_uniform_random(segments,size=(1,)):
    """Sample values uniformly from segments.
    
    Parameters
    ----------
    segments : segment array
    size : tuple of ints
        Shape of returned array
    
    Returns
    -------
    ndarray
    
    """
    
    segments = check_segments(segments,copy=False)
    
    #calculate segment durations and cumulative sum
    d = np.diff( segments, axis=1 ).squeeze()
    cs = np.concatenate( ([0],np.cumsum(d)) )
    
    #concatenate segments
    s = np.sum(d)
    
    #draw randomly from concatenated segments
    rtemp = np.random.uniform(low=0.0,high=s,size=size)
    r = np.zeros( rtemp.shape )
    
    #undo concatenation
    for k in range(len(segments)):
        idx = np.logical_and( rtemp>=cs[k], rtemp<cs[k+1] )
        r[idx] = rtemp[idx] + segments[k,0] - cs[k]
    
    return r
