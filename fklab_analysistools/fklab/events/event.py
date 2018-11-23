"""
=======================================
Event class (:mod:`fklab.events.event`)
=======================================

.. currentmodule:: fklab.events.event

Provides a container class for a vector of event times. Event vectors
can be created and manipulated through the class. Basic event algorithms
are implemented as methods of the class.

.. autosummary::
    :toctree: generated/
    
    Event

"""

from __future__ import division

import numpy as np

import fklab.utilities as util
from .basic_algorithms import ( event_count, event_intervals, event_bin,
  event_rate, peri_event_histogram, event_bursts, filter_bursts,
  filter_intervals, complex_spike_index, check_events )

__all__ = ['Event']


class AttributeMixIn:
    _attributes = {}
    def set_attribute(self, key, val ):
        if len(val)!=len(self):
            raise ValueError
        self._attributes[key] = val
    
    def get_attribute(self, key):
        return self._attributes[key]
    
    def list_attributes(self):
        return self._attributes.keys()
    
    def del_attribute(self,key):
        del self._attributes[key]


class Event(object, AttributeMixIn):
    """Event class.
    
    Parameters
    ----------
    data : 1d array like
        Vector of event times.
    copy : bool
        Create copy of input.
    
    Attributes
    ----------
    
    
    """
    
    def __init__(self,data=[],copy=True):
        
        if isinstance(data,Event):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else:
            self._data = check_events(data,copy)
        
        super(Event,self).__init__()
        
    @classmethod
    def isevent(cls,x):
        """Check if `x` is event-like"""

        if isinstance(x, Event):
            return True
        
        try:
            check_events(x)
        except ValueError:
            return False
        
        return True
    
    def __array__(self,*args):
        return self._data.__array__(*args)
    
    def asarray(self):
        """Return numpy array representation of Event object data."""
        return self._data #should we return a copy here?
    
    def __repr__(self):
        """Return string representation of Event object."""
        return "Event(" + repr( self._data ) + ")"
        
    def __str__(self):
        """Return string representation of Event object data."""
        return "Event(" + str(self._data) + ")"
    
    def __len__(self):
        """Return the number of events in the container."""
        return int(self._data.__len__())
    
    def issorted(self):
        """Check if events are sorted in ascending order."""
        return util.issorted(self._data)
    
    def sort(self):
        """Sort events in ascending order."""
        if self._data.__len__()>1:
            idx = np.argsort(self._data)
            self._data = self._data[idx]
            #TODO: sort attributes
            
        return self
    
    def __iter__(self):
        """Iterate through events in container."""
        idx = 0
        while idx<self._data.__len__():
            yield self._data[idx]
            idx+=1
    
    def not_(self):
        """Test if no events are defined."""
        return self._data.__len__()==0
    
    def truth(self):
        """Test if one or more events are defined"""
        return self._data.__len__()>0
    
    def __eq__(self,other):
        """Test if both objects contain the same event data"""
        if not isinstance(other,Event):
            other = Event(other)
        return (self._data.shape == other._data.shape) and np.all(self._data == other._data)
    
    def __ne__(self,other):
        """Test if objects contain dissimilar event data"""
        if not isinstance(other,Event):
            other = Event(other)
        return (self._data.shape != other._data.shape) and np.any( self._data != other._data)
    
    def __getitem__(self,key):
        """Slice events."""
        return Event(self._data[key])
    
    def __setitem__(self,key,value):
        self._data[key] = value
        return self
    
    def __delitem__(self,key):       
        #make sure we have a np.ndarray
        key = np.array(key)
        
        #if a logical vector with length equal number of segments, then find indices
        if key.dtype == np.bool and key.ndim == 1 and len(key)==self._data.__len__():
            key = np.nonzero(key)[0]
            
        self._data = np.delete( self._data, key, axis=0 )
        return self
    
    def offset(self, value):
        """Add offset to events."""
        e = Event(self) #copy
        e.ioffset(value) #apply offset to copy
        return e
    
    def ioffset(self,value):
        """Add offset to events (in place)."""
        value = np.array(value,dtype=np.float64).squeeze()
                    
        if value.ndim == 1:
            value = value.reshape([len(value),1])
        elif value.ndim!=0:
            raise ValueError('Invalid shape of offset value')

        self._data = self._data + value        
        
        return self
    
    def __iadd__(self,value):
        """Concatenates events or adds offset (in place)."""
        if isinstance(value,Event):
            return self.iconcat(value)
        
        return self.ioffset(value)
    
    def __add__(self,value):
        """Concatenate events or adds offset."""
        if isinstance(value,Event):
            return self.concat(value)
        
        return self.offset(value)    
    
    __radd__ = __add__
    
    def __sub__(self,value):
        """Add negative offset."""
        return self.offset(-value)
    
    __rsub__ = __sub__
    
    def __isub__(self,value):
        """Add negative offset (in place)."""
        return self.ioffset(-value)
    
    def concat(self,*others):
        """Concatenate events.
        
        Parameters
        ----------
        *others : event vectors
        
        Returns
        -------
        Event
        
        """
        e = Event(self)
        e.iconcat(*others)
                
        return e
        
    def iconcat(self,*others):
        """Concatenate events (in place).
        
        Parameters
        ----------
        *others : event vectors
        
        """
        
        if len(others)==0:
            return self
        
        #make sure all inputs are Events
        tmp = [x if isinstance(x,Event) else Event(x) for x in others]
        
        #TODO: check attribute compatibility
        
        data = [self._data]
        data.extend( [x._data for x in tmp] )
        data = np.concatenate( data ,axis=0)
        self._data = data        
        
        return self
    
    def count(self, x):
        """Returns the cumulative count of events.
        
        Parameters
        ----------
        x : 1d array
            times at which to evaluate cumulative event count
        
        Returns
        -------
        count : 1d array
            event counts
        
        """
        return event_count( self._data, x )
    
    def intervals(self,other=None,kind='post'):
        """Return inter-event intervals.
        
        Parameters
        ----------
        other : 1d array, optional
            vector of sorted event times (in seconds)
        kind : {'pre', '<', 'post', '>', 'smallest', 'largest'}
            type of interval to return. 'pre' or '<': interval to previous event, 
            'post' or '>': interval to next event, 'smallest' or 'largest':
            smallest/largest of the intervals to the previous and next events.
        
        Returns
        -------
        intervals : 1d array
            the requested interval for each event in the input vector `events`.
            Intervals to events in the past have a negative sign.
        index : 1d array
            index of the event to which the interval was determined
        
        """
    
        return event_intervals( self._data, other=other, kind=kind )
    
    def bin(self, bins, kind='count'):
        """Count number of events in bins.
        
        Parameters
        ----------
        bins : ndarray
            array of time bin start and end times (in seconds). Can be either
            a vector of sorted times, or a (n,2) array of bin start and
            end times.
        kind : {'count','binary','rate'}, optional
            determines what count to return for each bin. This can be the
            number of events (`count`), the presence or absence of events
            (`binary`) or the local event rate in the time bin (`rate`).
        
        Returns
        -------
        counts : 1d array
            event counts for each bin
            
        """
        
        return event_bin(self._data, bins, kind=kind)
    
    def meanrate(self,segments=None):
        """Return mean rate of events
        
        Parameters
        ----------
        segments : (n,2) array or Segment, optional
            array of time segment start and end times
        separate : bool, optional
            compute event rates for all segments separately
        
        Returns
        -------
        rate : array
            Mean firing rate for each of the input event time vectors.
            If `separate`=True, then a 2d array is returned, where `rate[i,j]`
            represents the mean firing rate for event vector `i` and
            segment `j`.
        
        """
    
        return event_rate( self._data, segments=segments, separate=False )
    
    def peri_event_histogram(self, reference=None, lags=None, segments=None, normalization=None, unbiased=False, remove_zero_lag=False ):
        """Compute peri-event time histogram.
        
        Parameters
        ----------
        reference : 1d array or sequence of 1d arrays, optional
            vector(s) of sorted reference event times (in seconds).
            If not provided, then `events` are used as a reference.
        lags : 1d array, optional
            vector of sorted lag times that specify the histogram time bins
        segments : (n,2) array or Segment, optional
            array of time segment start and end times
        normalization : {'none', 'coef', 'rate', 'conditional mean intensity',
                         'product density', 'cross covariance',
                         'cumulant density', 'zscore'}, optional
            type of normalization
        unbiased : bool, optional
            only include reference events for which data is available at all lags 
        remove_zero_lag : bool, optional
            remove zero lag event counts
        
        Returns
        -------
        3d array
            peri-event histogram of shape (lags, events, references)
        
        """
    
        return peri_event_histogram( self._data, reference=reference, lags=lags, segments=segments, normalization=normalization, unbiased=unbiased, remove_zero_lag=remove_zero_lag)
    
    def detectbursts(self, intervals=None, nevents=None, amplitude=None, attribute=False ):
        """Detect bursts of events.
        
        Parameters
        ----------
        intervals : 2-element sequence, optional
            minimum and maximum inter-event time intervals to consider two
            events as part of a burst
        nevents : 2-element sequence, optional
            minimum and maximum number of events in a burst
        amplitude : 1d array, optional
            vector of event amplitudes
        
        Returns
        -------
        1d array
            vector with burst indicators: 0=non-burst event, 1=first event
            in burst, 2=event in burst, 3=last event in burst
        
        """
        
        return event_bursts( self._data, intervals=intervals, nevents=nevents, amplitude=amplitude )
        #TODO: automatically add attribute with output
        
    def filterbursts(self, bursts=None, method='reduce', **kwargs):
        """Filter events in place based on participation in bursts.
        
        Parameters
        ----------
        bursts : 1d array, optional
            burst indicator vector as returned by `event_bursts` function.
            If not provided, it will be computed internally (parameters to
            the event_bursts function can be provided as extra keyword arguments)
        method : {'none', 'reduce', 'remove', 'isolate', 'isolatereduce'}
            filter method to be applied. 'none': remove all events,
            'reduce': only keep non-burst events and first event in bursts, 
            'remove': remove all burst events, 'isolate': remove all non-burst
            events, 'isolatereduce': only keep first event in bursts.
        
        """
    
        events,idx = filter_bursts( self._data, bursts=bursts, method=method, **kwargs )
        self._data = events
        return self
        #TODO: filter attributes, allow bursts='attribute'
    
    def filterintervals(self, mininterval=0.003):
        """Filter out events in place based on interval to previous event.
        
        Parameters
        ----------
        mininterval : scalar, optional
        
        """
        
        events, idx = filter_intervals( self._data, mininterval=mininterval )
        self._data = events
        return self
        #TODO: filter attributes
    
    def density(self, x=None, kernel='gaussian', bandwidth=0.1, rtol = 0.05, **kwargs):
        """Kernel density estimation.
        
        Parameters
        ----------
        x : ndarray
            Time points at which to evaluate density. If `x` is None, then 
            a KernelDensity object is returned.
        kernel : str
        bandwidth : scalar
            Kernel bandwidth
        rtol, **kwargs : extra arguments for sklearn.neighbor.kde.KernelDensity. 
        
        Returns
        -------
        ndarray or KernelDensity object
        
        """
        from sklearn.neighbors.kde import KernelDensity
        #create density function
        #TODO test speed of KernelDensity - roll our own?
        kde = KernelDensity( kernel=kernel, bandwidth=bandwidth, rtol=rtol, **kwargs ).fit(self._data[:,None])
        if x is not None:
            #evaluate density function
            x = np.array(x,copy=False)
            kde = np.exp( kde.score_samples(x[:,None]) )
        return kde
    
    def complex_spike_index(self, amplitude=None, intervals=None):
        """Compute complex spike index.
        
        Parameters
        ----------
        amplitude : 1d array
            vector of spike amplitudes
        intervals : 2-element sequence
            minimum and maximum inter-spike time intervals to consider two
            spikes as part of a burst
        
        Returns
        -------
        scalar
        
        """
        
        return float( complex_spike_index( self._data, spike_amp=amplitude, intervals=intervals ) )
    
    def average(self, time, data, lags=None, fs=None, interpolation='linear', function=None):
        """Event triggered average.
        
        Parameters
        ----------
        time : 1d array
            vector of sample times for data array
        data : ndarray
            array of data samples. First dimension should be time.
        lags : 2-element sequence, optional
            minimum and maximum lags over which to compute average
        fs : float, optional
            sampling frequency of average. If not provided, will be calculated
            from time vector `t`
        interpolation : string or integer, optional
            kind of interpolation. See `scipy.interpolate.interp1d` for more
            information.
        function : callable, optional
            function to apply to data samples (e.g. to compute something else
            than the average)
        
        Returns
        -------
        ndarray
            event triggered average of data
        
        """
        from .basic_algorithms import event_average
        return event_average( self._data, time, data, lags=lags, fs=fs, interpolation=interpolation, function=function)


#annotations - dict of container-level metadata - initiate with read-only keys
#attributes - dict of item-level metadata (len(attr)==nitems)

#obj.annotations[key] -> returns annotation

#obj.attributes[key] -> returns attribute
#obj.attributes[key][idx] -> should be valid and return subset of attributes
#len(obj.attributes[key]) -> should be equal to len(obj)

#obj[idx] -> returns subset of items and their attributes

#split_by_attribute(attribute) -> returns list of objects with subset of original items, split by attribute value ( equality, ranges, custom function ) in case of equality, optionally turn attribute into annotation
#select_by_attribute(s) -> returns subset of items where fcn( attribute value, *args ) is True

#at object collection level
#select_by_annotation( ... )

#setting/concatenation of containers should also check attributes and merge annotations

#class ValidationDict(dict):

    #def __init__(self, reference, *args, **kwargs):
        #self._reference = reference
        #try:
            #len(self._reference)
        #except:
            #raise TypeError
        
        #self.update(*args, **kwargs)

    ##def __setitem__(self, key, value):
    ##    if key in self._readonlykeys:
    ##        raise KeyError("%(key)s is a read-only key" % {'key':str(key)})
    ##    super(ValidationDict, self).__setitem__(key, value)

    #def __setitem__(self, key, value):
        #if len(value)!=len(self._reference):
            #raise ValueError("Incorrect len of value")
        #super(ValidationDict, self).__setitem__(key, value)

    #def update(self, *args, **kwargs):
        #if args:
            #if len(args) > 1:
                #raise TypeError("update expected at most 1 arguments, "
                                #"got %d" % len(args))
            #other = dict(args[0])
            #for key in other:
                #self[key] = other[key]
        #for key in kwargs:
            #self[key] = kwargs[key]

    #def setdefault(self, key, value=None):
        #if key not in self:
            #self[key] = value
        #return self[key]

#class ContainerBase(object):
    
    #def __init__(self):
        #self.attributes = ValidationDict( self )
        #self.annotations = {}
    
    #def __len__(self):
        #raise NotImplementedError
