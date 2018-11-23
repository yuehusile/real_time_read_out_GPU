"""
=============================================
Segment class (:mod:`fklab.segments.segment`)
=============================================

.. currentmodule:: fklab.segments.segment

Provides a container class for a list of segments. Each segment is defined by
a start and end point (usually in units of time). Basic operations on lists
of segments are implemented as methods of the class.

Class
=====

.. autosummary::
    :toctree: generated/
    
    Segment

Exceptions
==========

.. autosummary::
    :toctree: generated/
    
    SegmentError

"""

from __future__ import division

import numpy as np

from fklab.utilities import issorted, partition_vector
from .basic_algorithms import ( check_segments, segment_sort, segment_has_overlap,
  segment_remove_overlap, segment_invert, segment_exclusive, segment_union, 
  segment_difference, segment_intersection, segment_scale, segment_concatenate,
  segment_contains, segment_count, segment_overlap, segment_asindex, segment_join,
  segment_split, segment_applyfcn, segment_uniform_random )


__all__ = ['SegmentError','Segment']


class SegmentError(Exception):
    """Exception raised if array does not represent segments"""
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)


class Segment(object):
    """Segment container class.
    
    Parameters
    ----------
    data : Segment, array_like
    
    """
    
    def __init__(self,data=[],copy=True):
        if isinstance(data,Segment):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else: 
            self._data = check_segments(data,copy)
    
    @classmethod
    def issegment(cls,x):
        """Test is `x` is valid segment array."""
        
        if isinstance(x, Segment):
            return True
        
        try:
            check_segments(x)
        except ValueError:
            return False
        
        return True
    
    @classmethod
    def fromarray(cls,data):
        """Construct Segment from array.
        
        Parameters
        ----------
        data : (n,2) array
        
        Returns
        -------
        Segment
        
        """
        return Segment(data)
    
    @classmethod
    def fromlogical(cls,y,x=None):
        """Construct Segment from logical vector.
        
        Parameters
        ----------
        y : 1d logical array
            Any sequenceof True values that is flanked by False values is
            converted into a segment.
        x : 1d array like, optional
            The segment indices from `y` will be used to index into `x`.
        
        Returns
        -------
        Segment
        
        """
        
        y = np.asarray( y==True, dtype=np.int8)
        
        if len(y)==0 or np.all(y==0):
            return Segment([])
        
        d = np.diff( np.concatenate( ([0],y,[0]) ) )
        segstart = np.nonzero( d[0:-1]==1 )[0]
        segend = np.nonzero( d[1:]==-1 )[0]
        
        seg = np.vstack( (segstart,segend) ).T
        
        if x is not None:
            seg = x[seg]
        
        return Segment(seg)
    
    @classmethod
    def fromindices(cls,y,x=None):
        """Construct segments from vector of indices.
        
        Parameters
        ----------
        y : 1d array like
            Vector of indices. Segments are created from all neighboring
            pairs of values in y (as long as the difference is positive).
        x : 1d array like, optional
            The segment indices from `y` will be used to index into `x`.
        
        Returns
        -------
        Segment
        
        """
        
        if len(y)==0:
            return Segment([])
         
        d = np.nonzero( np.diff(y)>1 )[0]
        segstart = y[ np.concatenate(([0],d+1)) ]
        segend = y[ np.concatenate((d,[len(y)-1]))]
        
        seg = np.vstack( (segstart,segend) ).T
        
        if x is not None:
            seg = x[seg]
           
        return Segment(seg)
            
    @classmethod
    def fromevents(cls,on,off,greedyStart=False,greedyStop=False):
        """Construct segments from sequences of start and stop values.
        
        Parameters
        ----------
        on : 1d array like
            segment start values.
        off : 1d array like
            segment stop values.
        greedyStart : bool
            If multiple start values precede a stop value, then the first
            start value is used. 
        greedyStop : bool
            If multiple stop values follow a start value, then the last
            stop value is used.
        
        Returns
        -------
        Segment
        
        """
        
        on = np.array(on,dtype=np.float64).ravel()
        off = np.array(off,dtype=np.float64).ravel()
        
        events = np.concatenate( (on,off) )
        eventid = np.concatenate( ( np.ones( len(on) ), -np.ones( len(off) ) ) )
        
        isort = np.argsort( events, kind='mergesort' ) #mergesort keeps items with same key in same relative order
        events = events[isort]
        eventid = eventid[isort]
        
        diff_eventid = np.diff( eventid )
        
        #if greedyStart = True, remove all on-events in blocks (except first one)
        if greedyStart:
            invalid = np.nonzero( np.logical_and( diff_eventid==0, eventid[1:]==1 ) )[0] + 1
        else:
            invalid = np.nonzero( np.logical_and( diff_eventid==0, eventid[0:-1]==1 ) )[0]
        
        #if greedyStop = True, remove all off-events in blocks (except last one)
        if greedyStop:
            invalid = np.concatenate( (invalid, np.nonzero( np.logical_and( diff_eventid==0, eventid[0:-1]==-1))[0] ) )
        else:
            invalid = np.concatenate( (invalid, np.nonzero( np.logical_and( diff_eventid==0, eventid[1:]==-1))[0] + 1 ) )
        
        events = np.delete( events, invalid )
        eventid = np.delete( eventid, invalid )
        
        s = np.nonzero( np.diff( eventid ) == -2 )[0]
        s = np.vstack( (events[s], events[s+1]) ).T
        
        return Segment(s)
        
    @classmethod
    def fromduration(cls,anchor,duration,reference=0.5):
        """Construct segments from anchor points and durations.
        
        Parameters
        ----------
        anchor : scalar or 1d array like
            Anchoring points for the new segments. If `reference` is not
            given, then the anchor determines the segment center.
        duration : scalar or 1d array like
            Durations of the new segments
        reference : scalar or 1d array like, optional
            Relative reference point of the anchor in the segment.
            If `reference` is 0., the anchor defines the segment start, 
            if `reference` is 1., the anchor defines the segment stop.
        
        Returns
        -------
        Segment
        
        """
        
        #anchor + duration*[-reference (1-reference)]
        anchor = np.array(anchor,dtype=np.float64).ravel()
        duration = np.array(duration,dtype=np.float64).ravel()
        reference = np.array(reference,dtype=np.float64).ravel()
        
        start = anchor - reference*duration
        stop = anchor + (1-reference)*duration
        
        return Segment( np.vstack( (start,stop) ).T )
        
        pass
    
    def __array__(self,*args):
        return self._data.__array__(*args)
    
    def asarray(self):
        """Return numpy array representation of Segment object data."""
        return self._data #should we return a copy here?
    
    def __repr__(self):
        """Return string representation of Segment object."""
        return "Segment(" + repr( self._data ) + ")"
        
    def __str__(self):
        """Return string representation of Segment object data."""
        return "Segment(" + str(self._data) + ")"
    
    @property
    def start(self):
        """Return a vector of segment start values."""
        # this returns a copy
        return self._data[:,0].copy()
    
    @start.setter
    def start(self,value):
        """Set segment start values."""
        #Should we re-order after changing start points?
        if np.any( self._data[:,1]<value ):
            raise SegmentError('Segment start times should be <= stop times')
            
        self._data[:,0] = value
    
    @property
    def stop(self):
        """Return a vector of segment stop values."""
        # this returns a copy        
        return self._data[:,1].copy()
    
    @stop.setter
    def stop(self,value):
        """Set segment stop values."""
        #TODO: check if values are not beyond start points
        if np.any( self._data[:,0]>value ):
            raise SegmentError('Segment stop times should be >= start times')
            
        self._data[:,1] = value
    
    @property
    def duration(self):
        """Return a vector of segment durations."""
        return np.diff(self._data,axis=1).ravel()
    
    @duration.setter
    def duration(self,value):
        """Set new duration of segments."""
        value = np.array(value,dtype=np.float64).ravel()
        ctr = np.mean(self._data,axis=1)
        self._data[:,0] = ctr - 0.5*value
        self._data[:,1] = ctr + 0.5*value
    
    @property
    def center(self):
        """Return a vector of segment centers."""
        return np.mean(self._data,axis=1)
    
    @center.setter
    def center(self,value):
        """Set new centers of segments."""
        
        value = np.array(value,dtype=np.float64).ravel()
        dur = np.diff(self._data,axis=1).squeeze()
        self._data[:,0] = value - 0.5*dur
        self._data[:,1] = value + 0.5*dur
    
    def __len__(self):
        """Return the number segments in the container."""
        return int(self._data.shape[0])
    
    def issorted(self):
        """Check if segment starts are sorted in ascending order."""
        return issorted( self._data[:,0] )
    
    def isort(self):
        """Sort segments (in place) in ascending order according to start value."""
        if self._data.shape[0]>1:
            idx = np.argsort(self._data[:,0])
            self._data = self._data[idx,:]
            
        return self
    
    def sort(self):
        """Sort segments in ascending order according to start value.
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.isort()
        return s
    
    def argsort(self):
        """Argument sort of segment start value.
        
        Returns
        -------
        ndarray
            Indices that will sort the segment array.
        
        """
        return np.argsort(self._data[:,0])
    
    @property
    def intervals(self):
        """Duration of intervals between segments."""
        return self._data[1:,0] - self._data[:-1,1]
    
    def hasoverlap(self):
        """Check if any segments are overlapping."""
        return segment_has_overlap( self._data )
    
    def removeoverlap(self,strict=True):
        """Remove overlap between segments through merging.
        
        This method will sort segments as a side effect.
        
        Parameters
        ----------
        strict : bool
            Only merge two segments if the end time of the first is stricly
            larger than (and not equal to) the start time of the second segment.
        
        Returns
        -------
        Segment
        
        """
        self._data = segment_remove_overlap( self._data, strict=strict )
        return self
        
    def __iter__(self):
        """Iterate through segments in container."""
        idx = 0
        while idx<self._data.shape[0]:
            yield self._data[idx,0], self._data[idx,1]
            idx+=1
    
    def not_(self):
        """Test if no segments are defined."""
        return self._data.shape[0]==0
    
    def truth(self):
        """Test if one or more segments are defined."""
        return self._data.shape[0]>0

    
    def exclusive(self,*others):
        """Exclude other segments.
        
        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.
        
        Parameters
        ----------
        *others : segment arrays
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.iexclusive(*others)
        return s
    
    def iexclusive(self,*others):
        """Exclude other segments (in place).
        
        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.
        
        Parameters
        ----------
        *others : segment arrays
        
        """
        self._data = segment_exclusive( self._data, *others )
        return self        
    
    def invert(self):
        """Invert segments.
        
        Constructs segments from the inter-segment intervals.
        This method Will remove overlap as a side effect.
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.iinvert()
        return s
        
    def iinvert(self):
        """Invert segments (in place).
        
        Constructs segments from the inter-segment intervals.
        This method Will remove overlap as a side effect.
        
        """
        self._data = segment_invert( self._data )
        return self
    
    __invert__ = invert
    
    def union(self,*others):
        """Combine segments (logical OR).
        
        This method Will remove overlaps as a sife effect.
        
        Parameters
        ----------
        *others : segment arrays
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.iunion(*others)
        return s
        
    def iunion(self,*others):
        """Combine segments (logical OR) (in place).
        
        This method Will remove overlaps as a sife effect.
        
        Parameters
        ----------
        *others : segment arrays
        
        """
        self._data = segment_union( self._data, *others )
        return self
    
    __or__ = union
    __ror__ = __or__
    __ior__ = iunion
    
    def difference(self,*others):
        """Return non-overlapping parts of segments (logical XOR).
        
        Parameters
        ----------
        *others : segment arrays
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.idifference(*others)
        return s
        
    def idifference(self,*others):
        """Return non-overlapping parts of segments (logical XOR) (in place).
        
        Parameters
        ----------
        *others : segment arrays
        
        """
        self._data = segment_difference( self._data, *others )
        return self
    
    def __xor__(self,other):
        """Return non-overlapping parts of segments (logical XOR).
        
        Parameters
        ----------
        *others : segment arrays
        
        Returns
        -------
        Segment
        
        """
        return self.difference(other)
    
    __rxor__ = __xor__
    __ixor__ = idifference
    
    def intersection(self,*others):
        """Return intersection (logical AND) of segments.
        
        Parameters
        ----------
        *others : segment arrays
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.iintersection(*others)
        return s
    
    def iintersection(self,*others):
        """Return intersection (logical AND) of segments (in place).
        
        Parameters
        ----------
        *others : segment arrays
        
        """
        self._data = segment_intersection( self._data, *others )
        return self 

    __and__ = intersection
    __rand__ = __and__
    __iand__ = iintersection

    
    def __eq__(self,other):
        """Test if both objects contain the same segment data.
        
        Parameters
        ----------
        other : segment array
        
        Returns
        -------
        bool
        
        """
        if not isinstance(other,Segment):
            other = Segment(other)
        return (self._data.shape == other._data.shape) and np.all(self._data == other._data)
    
    def __ne__(self,other):
        """Test if objects contain dissimilar segment data.
        
        Parameters
        ----------
        other : segment array
        
        Returns
        -------
        bool
        
        """
        if not isinstance(other,Segment):
            other = Segment(other)
        return (self._data.shape != other._data.shape) and np.any( self._data != other._data)
    
    def __getitem__(self,key):
        """Slice segments.
        
        Parameters
        ----------
        key : slice or indices
        
        Returns
        -------
        Segment
        
        """
        return Segment(self._data[key,:]) #does not return a view!
    
    def __setitem__(self,key,value):
        """Set segment values.
        
        Parameters
        ----------
        key : slice or indices
        value : scalar or ndarray
        
        """
        self._data[key,:] = value
        return self
    
    def __delitem__(self,key):   
        """Delete segments (in place).
        
        Parameters
        ----------
        key : array like
            Index vector or boolean vector that indicates which segments
            to delete.
            
        """
        #make sure we have a np.ndarray
        key = np.array(key)
        
        #if a logical vector with length equal number of segments, then find indices
        if key.dtype == np.bool and key.ndim == 1 and len(key)==self._data.shape[0]:
            key = np.nonzero(key)[0]
            
        self._data = np.delete( self._data, key, axis=0 )
        return self
    
    def offset(self, value):
        """Add offset to segments.
        
        Parameters
        ----------
        value : scalar or 1d array
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.ioffset(value)
        
        return s
    
    def ioffset(self,value):
        """Add offset to segments (in place).
        
        Parameters
        ----------
        value : scalar or 1d array
        
        """
        value = np.array(value,dtype=np.float64).squeeze()
                    
        if value.ndim == 1:
            value = value.reshape([len(value),1])
        elif value.ndim!=0:
            raise ValueError('Invalid shape of offset value')

        self._data = self._data + value
        return self
    
    def scale(self, *args, **kwargs):
        """Scale segment durations.
    
        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.iscale(*args,**kwargs)
        return s
        
    def iscale(self, value, reference=0.5):
        """Scale segment durations (in place).
    
        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.
        
        """
        self._data = segment_scale( self._data, value, reference=reference )
        return self
    
    def concat(self,*others):
        """Concatenate segments.
        
        Parameters
        ----------
        *others : segment arrays
            
        Returns
        -------
        Segment
            
        """
        s = Segment(self)
        s.iconcat(*others)
        return s
        
    def iconcat(self,*others):
        """Concatenate segments (in place).
        
        Parameters
        ----------
        *others : segment arrays
            
        """
        self._data = segment_concatenate( self._data, *others )
        return self
    
    def __iadd__(self,value):
        """Concatenates segments or adds offset (in place).
        
        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If `value` is a Segment object, then its segments are
            concatenated to this Segment. Otherwise,
            `value` is added as an offset to the segments.
            
        """
        if isinstance(value,Segment):
            return self.iconcat(value)
        
        return self.ioffset(value)
    
    def __add__(self,value):
        """Concatenate segments or adds offset.
        
        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If `value` is a Segment object, then a new Segment object
            with a concatenated list of segment is returned. Otherwise,
            `value` is added as an offset to the segments.
            
        Returns
        -------
        Segment
        
        """
        if isinstance(value,Segment):
            return self.concat(value)
        
        return self.offset(value)    
    
    __radd__ = __add__
    
    def __sub__(self,value):
        """Subtract value.
        
        Parameters
        ----------
        value : scalar or 1d array
        
        Returns
        -------
        Segment
        
        """
        return self.offset(-value)
    
    __rsub__ = __sub__
    
    def __isub__(self,value):
        """Subtract value (in place).
        
        Parameter
        ---------
        value : scalar or 1d array
        
        """
        return self.ioffset(-value)
    
    __mul__ = scale
    __rmul__ = __mul__
    __imul__ = iscale
    
    def __truediv__(self,value):
        """Divide segment durations.
        
        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.
        
        Returns
        -------
        Segment
        
        """
        return self.scale(1.0/value)
    
    def __rtruediv__(self,value):
        return NotImplemented
    
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    
    def __itruediv__(self,value):
        """Divide segment durations (in place).
        
        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.
        
        """
        return self.iscale(1.0/value)
    
    __idiv__ = __itruediv__
    
    def contains(self,value,issorted=True,expand=None):
        """Test if values are contained in segments.
        
        Segments are considered left closed and right open intervals. So, 
        a value x is contained in a segment if start<=x and x<stop. 
        
        Parameters
        ----------
        value : sorted 1d array
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
            For each segment, the start and end indices of values in `x` 
            that are contained within that segment.
        
        """
        #TODO: test if self is sorted
        #TODO: test if value is sorted
        #TODO: support scalars and nd-arrays for value?
        return segment_contains(self._data, value, issorted, expand)

    def __contains__(self,value):
        return self.contains(value)[0]
    
    def count(self,x):
        """Count number of segments.
        
        Parameters
        ----------
        x : ndarray
        
        Returns
        -------
        ndarray
            For each value in `x` the number of segments that contain that value.
        
        """
        return segment_count(self._data,x)
        
    def overlap(self,other=None):
        """Returns absolute and relative overlaps between segments.
        
        Parameters
        ----------
        other : segment array, optional
            If `other` is not provided, then auto-overlaps are analyzed.
        
        Returns
        -------
        ndarray
            absolute overlap between all combinations of segments
        ndarray
            overlap relative to duration of first segment
        ndarray
            overlap relative to duration of second segment
        
        """
        return segment_overlap(self._data,other=other)
    
    def asindex(self,x):
        """Convert segments to indices into vector.
        
        Parameters
        ----------
        x : ndarray
        
        Returns
        -------
        Segment (indices)
        
        """
        return Segment(segment_asindex( self._data,x))
    
    def ijoin(self,gap=0):
        """Join segments with small inter-segment gap (in place).
        
        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than `gap` will be
            merged.
        
        """
        self._data = segment_join(self._data,gap=gap)
        return self
    
    def join(self,*args,**kwargs):
        """Join segments with small inter-segment gap.
        
        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than `gap` will be
            merged.
        
        Returns
        -------
        Segment
        
        """
        s = Segment(self)
        s.ijoin(*args,**kwargs)
        return s
        
    def split(self, size=1, overlap=0, join=True, tol=1e-7):
        """Split segments into smaller segments with optional overlap.
        
        Parameters
        ----------
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
        Segment or list of Segments
        
        """
    
        seg = segment_split( self._data, size=size, overlap=overlap, join=join, tol=tol )
        if len(seg)==0:
            return Segment([])
        elif isinstance( seg, list ): #we have a list of segments
            return [Segment(x) for x in seg]
        else:
            return Segment( seg )
    
    def applyfcn(self,x,*args,**kwargs):
        """Apply function to segmented data.
        
        Parameters
        ----------
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
        return segment_applyfcn( self._data, x, *args, **kwargs )
    
    def partition(self,**kwargs):
        """Partition segments into groups.
        
        See `fklab.general.partitions` and `fklab.general.partition_vector'
        for more information.
        
        Parameters
        ----------
        nparts : int
            Number of partitions
        method: 'block', 'random', 'sequence'
            Method of assigning segments to partitions.
        
        Returns
        -------
        Segment object
            partitioned subset of segments
        
        """
        return partition_vector( self, **kwargs )
        #kwargs['size'] = self._data.shape[0]
        #return (self[idx] for idx in partitions( **kwargs ))
    
    def uniform_random(self,size=(1,)):
        """Sample values uniformly from segments.
        
        Parameters
        ----------
        size : tuple of ints
            Shape of returned array.
        
        Returns
        -------
        ndarray
        
        """
        return segment_uniform_random(self._data, size=size)

