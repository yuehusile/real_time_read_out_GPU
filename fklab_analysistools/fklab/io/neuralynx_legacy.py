"""
=================================================
Neuralynx file import (:mod:`fklab.io.neuralynx`)
=================================================

.. currentmodule:: fklab.io.neuralynx

Description

Constants
=========

==================  ================================
`NLXHEADERSIZE`     Length in bytes of file header
`NLXCSCBUFFERSIZE`  Size of CSC buffers in samples
==================  ================================

Functions
=========

.. autosummary::
    :toctree: generated/
    
    NlxOpen
    NlxTimestamp2Seconds
    NlxSeconds2Timestamp

File classes
============

.. autosummary::
    :toctree: generated/
    
    NlxFileSpike
    NlxFileTahiti
    NlxFileCSC
    NlxFileEvent
    NlxFileVideo

Auxillary classes
=================

.. autosummary::
    :toctree: generated/
    
    NlxHeader
    NlxVideoPoints

Exceptions
==========

.. autosummary::
    :toctree: generated/
    
    NeuralynxError
    NeuralynxIOError

"""

import os
import warnings
import numbers
import numpy as np

from fklab.io.binary_legacy import BinaryFileReader
from fklab.segments import Segment

__all__ = ['NeuralynxError','NeuralynxIOError','NlxTimestamp2Seconds','NlxSeconds2Timestamp','NLXHEADERSIZE','NLXCSCBUFFERSIZE','NlxHeader',
           'NlxFileSpike','NlxFileTahiti','NlxFileCSC','NlxFileEvent','NlxFileVideo','NlxVideoPoints','NlxOpen']

#==============================================================================
# Exceptions for this module
#==============================================================================

class NeuralynxError( Exception ):
    """Base class for exceptions in this module."""
    pass

class NeuralynxIOError( NeuralynxError ):
    """Exception raised when attempting to read from non-Neuralynx type data file."""
    def __init__(self, filename, msg):
        self.filename = filename
        self.msg = msg

    def __str__(self):
        return self.msg + " ( file: " + self.filename + " )"

#==============================================================================
# Helper functions
#==============================================================================

def NlxTimestamp2Seconds(timestamp):
    """Convert Neuralynx timestamps to seconds."""
    return np.asarray(timestamp*0.000001,dtype=np.float64)

def NlxSeconds2Timestamp(time):
    """Convert seconds to Neuralynx timestamps."""
    return np.asarray(time*1000000,dtype=np.uint64)

#==============================================================================
# Define functions to parse header value strings
#==============================================================================

def _NlxBoolHandler(value):
    """Convert string with true/false to bool"""
    return str(value).lower() == 'true'

def _NlxFloatArrayHandler(value):
    """Convert string to array of floats."""
    return np.fromstring( value, dtype=float, sep=' ')

def _NlxIntArrayHandler(value):
    """Convert string to array of ints."""
    return np.fromstring( value, dtype=int, sep=' ')

def _NlxStringHandler(value):
    """Strip whitespace from strings."""
    return str(value).strip()

def _NlxFeatureHandler(value):
    """Return input unmodified"""
    return value

def _NlxEnableHandler(value):
    """Convert enabled/disabled strings to boolean."""
    return value=='Enabled'

def _NlxColorTrackingHandler(value):
    """Convert color tracking string to (boolean,threshold)"""
    val = np.fromstring( value, dtype=int, sep=' ')
    return (val[0]==1, val[1])

#==============================================================================
# Define constants
#==============================================================================

NLXHEADERSIZE = 16384 #fixed length of header as defined by Neuralynx
NLXCSCBUFFERSIZE = 512 #number of samples in CSC data record

#dictionary of file extension and file type relationships
_NLXFILEEXT = {
    '.ncs':'CSC',
    '.nev':'Event',
    '.nse':'Spike', #single electrode
    '.nst':'Spike', #stereotrode
    '.ntt':'Spike', #tetrode
    '.nvt':'Video',
    '.moz':'MOZ',
}

#a list of tuples, where the first item defines
#a handler function and the second item is a list of parameters that
#are processed by the handler function
_NLXVALUEHANDLERS = [
    (_NlxBoolHandler, [
        'DSPHighCutFilterEnabled',
        'DSPLowCutFilterEnabled',
        'DualThresholding',
        'InputInverted',
        ]
    ),
    (int, [
        'ADMaxValue',
        'AlignmentPt',
        'DspHighCutNumTaps',
        'DspLowCutNumTaps',
        'MinRetriggerSamples',
        'NumADChannels',
        'RecordSize',
        'WaveformLength',
        'SamplesPerRecord',
        ]
    ),
    (float, [
        'DspFilterDelay_\xb5s',
        'SamplingFrequency',
        'SpikeRetriggerTime',
        'DspHighCutFrequency',
        'DspLowCutFrequency',        
        ]
    ),
    (_NlxFloatArrayHandler,[
        'ADBitVolts',
        'InputRange',
        'ThreshVal',
        ]
    ),
    (_NlxIntArrayHandler, [
        'ADChannel',
        'Resolution',
        ]
    ),
    (_NlxStringHandler, [
        'AcqEntName',
        'CheetahRev',
        'DspHighCutFilterType',
        'DspLowCutFilterType',
        'FileType',
        'FileVersion',
        'HardwareSubSystemName',
        'HardwareSubSystemType',
        ]
    ),
    (_NlxFeatureHandler, [
        'Feature',
        ]
    ),
    (_NlxEnableHandler, [
        'DspDelayCompensation',
        ]
    ),
    (_NlxColorTrackingHandler, [
        'IntensityThreshold',
        'RedThreshold',
        'GreenThreshold',
        'BlueThreshold',
        ]
    ),
]

#==============================================================================
# Neuralynx data file header functions and classes
#==============================================================================

_units_scale_factor = {'V':1, 'mV':1000, 'uV':1000000}

def readheader(file):
    """Reads Neuralynx data file header and determines file type.
    
    Neuralynx data files contain a fixed length text header. This function will
    read the header and determine the file type based on the FileType header
    parameter or the file extension.
    
    Parameters
    ----------
    file : str
        file name
    
    Returns
    -------
    hdr : list of str
        header lines
    filetype : str
        One of ``CSC``, ``Spike``, ``Event`` or ``Video`` or ``MOZ``.
    
    """
    
    #read fixed size text header
    with open(file,'r') as f:
        hdr = f.read(NLXHEADERSIZE)

    #check if this is a valid Neuralynx data file
    if len(hdr)!=NLXHEADERSIZE: #or hdr[0:35]!="######## Neuralynx Data File Header":
        raise NeuralynxIOError(file,'Unrecognizable data file header')

    #remove unused part of header (filled with chr(0)) and separate lines
    hdr = hdr.strip(chr(0)).splitlines()

    #look for FileType header parameter
    filetype = None
    for line in hdr:
        if len(line)>9 and line[0:9]=="-FileType":
            filetype = line[10:]
            break
    
    #alternatively, check file extension to determine file type
    if filetype==None:
        #let's try the file extension
        import os
        fileroot,fileext = os.path.splitext(file)
        try:
            filetype = _NLXFILEEXT[fileext]
        except KeyError:
            raise NeuralynxIOError(file, 'Unrecognized or unsupported data file type')

    return hdr, filetype

def parseheader(hdr):
    """Parse Neuralynx data file header and extract parameter/value pairs.
    
    Neuralynx data files contain a text header, where each line defines a
    parameter and its value. Value strings are automatically converted to the
    desired data type using value handler functions as defined in
    `_NLXVALUEHANDLERS`. 
    
    Parameters
    ----------
    hdr : list of str
        Header lines as returned by `readheader`.
        
    Returns
    -------
    parameters : dict
        Header parameter/value pairs.
    
    """

    import re
    
    #define regular expression to find parameter/value pairs
    expr = "-(?P<parameter>[\w\\\\]+) ?(?P<value>.*)"
    expr = re.compile(expr, re.UNICODE)

    #create empty dictionary to hold parameter/value pairs
    parameters = dict()

    for line in hdr:
        #does line contain valid parameter/value pair?
        match = expr.match(line)
        if match is None:
            continue
        
        #extract parameter and value
        param = match.group('parameter')
        val = match.group('value')
        
        #convert value string using appropriate value handler function
        for handler in _NLXVALUEHANDLERS:
            if param in handler[1]:
                try:
                    val = handler[0](val)
                    break
                except:
                    import warnings
                    warnings.warn('Unable to process parameter ' + param + ' in header')
                    param = None

        #collect all `Feature` header parameters in single `Features' parameter
        if param=='Feature':
            param = 'Features'
            if param in parameters.keys():
                parameters[param].append(val)
            else:
                parameters[param] = [val]

        elif param!=None:
            parameters[param] = val

    return parameters

class NlxHeader( object ):
    """Neuralynx data file header object.
    
    NlxHeader(filename)
    
    Returns a `NlxHeader` object that represents the header parameter/value 
    pairs in a Neuralynx data file.
    
    Parameters
    ----------
    filename : str
        Name and path of the data file.
    
    """

    _parameters = dict() #dictionary of parameter/value pairs in header

    def __init__(self, filename):
        import os
        self._path, self._file = os.path.split( os.path.abspath(filename) )
        self._rawheader, self.filetype = readheader(filename)
        self._parameters = parseheader(self._rawheader)
    
    def __repr__(self):
        return self._parameters.__repr__()
    
    __str__ = __repr__

    @property
    def fullpath(self):
        """Return full path to file."""
        import os
        return os.path.join(self._path, self._file)

    @property
    def parameters(self):
        """Return sorted list of header parameters."""
        p = self._parameters.keys()
        p.sort()
        return p

    def getvalue(self,param,default=None):
        """Return value of header parameter."""
        return self._parameters.get(param,default)
        
    @property
    def values(self):
        """Return copy of parameter/value dictionary."""
        import copy
        return copy.deepcopy( self._parameters )

class NlxFileBase(object):
    """Base class for all Neuralynx data files.
    
    NlxFileBase(source)
    
    Parameters
    ----------
    source : NlxHeader or str
        Either the name and path of the file or an existing `NlxHeader`
        instance.
    
    Attributes
    ----------
    data
    filesize
    nrecords
    recordsize
    header
    fullpath
    starttimestamp
    endtimestamp
    starttime
    endtime
    
    Methods
    -------
    info()
        Display file information.
    timeslice(start=0.0,stop=1.0,delta=None)
        Create slice object for extended array slicing based on time values.  
    
    """
    
    def __init__(self,source,dtype,mmap=False, field_aliases={}, field_groups={} ):
        #read file header
        if isinstance(source,NlxHeader):
            self._header = source;
        else:
            self._header = NlxHeader(source)
        
        #get size of records in file
        self._recordsize = self._header.getvalue('RecordSize')
        
        self._record_dtype = dtype
        
        #define field aliases and groups
        self._field_aliases = {'original_time':('timestamp', NlxTimestamp2Seconds)}
        self._field_aliases['original_timestamp'] = ('timestamp',None)
        self._field_aliases['time'] = ('timestamp', NlxTimestamp2Seconds)
        self._field_groups = {}
        
        #merge in field aliases and groups specified as arguments
        self._field_aliases.update( field_aliases )
        self._field_groups.update( field_groups )
        
        self.data = BinaryFileReader( self.fullpath, self._record_dtype, offset=NLXHEADERSIZE, alias=self._field_aliases, groups=self._field_groups, mmap=mmap )
        
        if self.nrecords>0:
            self._starttimestamp = self.data.original_timestamp[0]
            self._endtimestamp = self.data.original_timestamp[self.nrecords-1]
        else:
            self._starttimestamp = 0
            self._endtimestamp = 0
        
    def __str__(self):
        return self.__class__.__name__ +  " ( " + self._header.fullpath + " )"
    
    __repr__ = __str__
    
    def info(self):
        s = str(self) + "\n"
        s += "record size = " + str(self._recordsize) + " bytes\n"
        s += "record dtype = " + str(self._record_dtype) + "\n"
        s += "# records = " + str(self.nrecords) + " records\n"
        print(s)
        
    @property
    def recordsize(self):
        return self._recordsize

    @property
    def filesize(self):
        import os
        return os.path.getsize( self.fullpath )

    @property
    def nrecords(self):
        return (self.filesize - NLXHEADERSIZE)/self.recordsize

    @property
    def header(self):
        return self._header.values

    @property
    def fullpath(self):
        """Get the full path to data file."""
        return self._header.fullpath
    
    def _convert_time_to_timestamp(self, t ):
        return NlxSeconds2Timestamp(t)
    
    def _convert_timestamp_to_time(self, t ):
        return NlxTimestamp2Seconds(t)
    
    def _search_time_indices(self, start=0.0, stop=1.0, extra=0):
        import bisect
        
        #convert times to timestamps (in microseconds)
        start = self._convert_time_to_timestamp(start)
        stop = self._convert_time_to_timestamp(stop)

        #find first index
        start = max( bisect.bisect_left( self.data.original_timestamp, start ) - extra, 0 )
        stop = min( bisect.bisect_right( self.data.original_timestamp, stop ) + extra, self.nrecords ) 
        
        return start, stop

    def timeslice(self, start=0.0, stop=1.0, delta=None, extra=0 ):
        """Return slice based on time values."""
        
        start, stop = self._search_time_indices( start, stop, extra )

        if stop>=start:
            if delta is None:
                return slice(start,stop)
            else:
                avg_delta = (self.data.original_time[stop]-self.data.original_time[start])/(stop-start-1)
                delta = max( (np.round(delta/avg_delta),1) )
                return slice(start,stop,delta)
                
        else:
            raise IndexError('Time epoch out of range or too small')
    
    def readdata(self,start=None,stop=None):
        if start is None:
            start = self.starttime
        
        if stop is None:
            stop = self.endtime
        
        tslice = self.timeslice( start, stop )
        
        data = self.data.default[tslice]
        
        return data
    
    @property
    def starttime(self):
        return self._convert_timestamp_to_time(self.starttimestamp)

    @property
    def endtime(self):
        return self._convert_timestamp_to_time(self.endtimestamp)

    @property
    def starttimestamp(self):
        return self._starttimestamp

    @property
    def endtimestamp(self):
        return self._endtimestamp

class NlxFileTimedBuffers(NlxFileBase):
    
    def __init__(self,source,dtype,**kwargs):
        
        #call base class __init__
        NlxFileBase.__init__(self, source, dtype, mmap=kwargs.pop('mmap',False), field_aliases=kwargs.pop('field_aliases',{}), field_groups=kwargs.pop('field_groups',{}))
        
        self._nchan = int(kwargs.pop('nchannels',1))
        self._nsamples = int(kwargs.pop('nsamples',1))
    
        self.correct_inversion = kwargs.pop('correct_inversion', False)
        self.scale_data = kwargs.pop('scale_data', True)
        self.expand_time = kwargs.pop('expand_time', False)
        self.unwrap_data = kwargs.pop('unwrap_data', True)
        self.units = kwargs.pop('units', 'uV' )
        self.clip = kwargs.pop('clip', True)
        
        if self.aliasing():
            warnings.warn("High cut filter frequency is higher than half the sampling frequency. Data in this file is possibly affected by aliasing.", RuntimeWarning )
        
    def sample2record(self, sample):
        return np.floor( sample/self.nsamples )
    
    def record2sample(self,record):
        return np.int( record*self.nsamples )
    
    def aliasing(self):
        fs = self.header['SamplingFrequency']
        hc = self.header['DspHighCutFrequency']
        return (not self.header["DSPHighCutFilterEnabled"]) or (2*hc > fs)
    
    @property
    def nchannels(self):
        return self._nchan
    
    @property
    def nsamples(self):
        return self._nsamples
        
    def expand_timestamp(self, timestamp, scale=None):
        
        if self._expand_time:
            if scale is None:
                scale = np.uint64( self._convert_time_to_timestamp(1) / self.header['SamplingFrequency'] )
            ts = np.arange(self._nsamples).reshape(1,self._nsamples) * scale
            ts = ts + timestamp.reshape( timestamp.size, 1)
        else:
            ts = timestamp
        
        if self._unwrap_data:
            ts = ts.flatten()
        
        return ts
    
    def convert_timestamp(self, timestamp):
        return self.expand_timestamp( self._convert_timestamp_to_time( timestamp ), scale=1.0/self.header['SamplingFrequency'] )
    
    def convert_data(self, data):
        
        #expected shape of 3D data array: nbuffers x nsamples x nchannels
        nbuffers, nsamples, nchannels = data.shape
        
        #transpose dimensions
        #data = data.transpose((1,2,0)) # nsamples x nchannels x nbuffers
        
        scale=1
        if self._correct_inversion and self._header.getvalue('InputInverted',False):
            scale = -scale
        
        if self._scale_data:
            scale = scale * self._header.getvalue('ADBitVolts').reshape(1,1,self._nchan)*_units_scale_factor[self._signal_units]
        
        if np.any(scale!=1):
            data = scale * data
        
        if self._clip:
            clip_value = self.header["ADMaxValue"] * scale
            data = data.astype(np.float, copy=False) # force cast to float
            data[np.logical_or( data >= clip_value, data <= -clip_value )] = np.nan
        
        if self._unwrap_data:
            data = data.reshape( (nbuffers*nsamples,nchannels) )
        
        return data
    
    @property
    def correct_inversion(self):
        return self._correct_inversion
    
    @correct_inversion.setter
    def correct_inversion(self,value):
        self._correct_inversion = bool(value)
    
    @property
    def scale_data(self):
        return self._scale_data
    
    @scale_data.setter
    def scale_data(self,value):
        self._scale_data = bool(value)
    
    @property
    def expand_time(self):
        return self._expand_time
    
    @expand_time.setter
    def expand_time(self,value):
        self._expand_time = bool(value)
    
    @property
    def unwrap_data(self):
        return self._unwrap_data
    
    @unwrap_data.setter
    def unwrap_data(self, value):
        self._unwrap_data = bool(value)
    
    @property
    def clip(self):
        return self._clip
    
    @clip.setter
    def clip(self, value):
        self._clip = bool(value)
    
    @property
    def units(self):
        return self._signal_units
    
    @units.setter
    def units(self,value):
        if not value in _units_scale_factor.keys():
            raise ValueError
        self._signal_units = value


class NlxFileSpike(NlxFileTimedBuffers):

    def __init__(self,source,*args,**kwargs):
        
        #we need to figure out the number of channels first
        #read header
        if not isinstance( source, NlxHeader ):
            source = NlxHeader( source )
        nchan = source.getvalue('NumADChannels')
        nsamples = source.getvalue('WaveformLength')
        
        dtype = np.dtype( [
            ('timestamp',np.uint64),
            ('scnumber',np.uint32),
            ('cellnumber',np.uint32),
            ('params',np.uint32,8),
            ('waveform',np.int16,(nsamples,nchan))
            ] )
        
        field_aliases = {'time': ('timestamp',self.convert_timestamp)}
        field_aliases['timestamp'] = ('timestamp',self.expand_timestamp)
        field_aliases['waveform'] = ('waveform',self.convert_data)
        field_groups = {'spikes': ['time','waveform']}
        field_groups['default'] = ['time','waveform']
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('expand_time',False)
        kwargs.setdefault('unwrap_data',False)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, mmap=kwargs.pop('mmap',False), field_aliases=field_aliases, field_groups=field_groups, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "Spike":
            raise NeuralynxIOError(self._header.file, 'Not a valid Spike file [tetrode/stereotrode/electrode]')
    
    def convert_data(self,waveform):
        #if waveform.ndim==2:
        #    waveform = np.expand_dims( waveform, axis=0 )
        
        waveform = super(NlxFileSpike,self).convert_data( waveform )
        
        return waveform


class NlxFileTahiti(NlxFileTimedBuffers):
    
    def __init__(self, source='', *args, **kwargs):
        
        #we need to figure out the number of channels and samples first
        #read header
        if not isinstance( source, NlxHeader ):
            source = NlxHeader( source )
        nchan = source.getvalue('NumADChannels')
        nsamples = source.getvalue('SamplesPerRecord')
        
        dtype = np.dtype( [
        ('seconds',np.int32), 
        ('nanoseconds',np.int32),
        ('dt',np.int32),
        ('data',np.int16,(nsamples,nchan))
        ] )
        
        field_aliases = {'original_time':(['seconds','nanoseconds'], self.convert_timestamp)}
        field_aliases['original_timestamp'] = field_aliases['original_time']
        field_aliases['time'] = (['seconds','nanoseconds'],self.convert_and_expand_timestamp)
        field_aliases['data'] = ('data',self.convert_data)
        field_groups = {'default': ['time','data']}
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('expand_time',True)
        kwargs.setdefault('unwrap_data',True)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, mmap=kwargs.pop('mmap',False), field_aliases=field_aliases, field_groups=field_groups, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "MOZ":
            raise NeuralynxIOError(self._header.file, 'Not a valid Tahiti MOZ file')
    
    def _search_time_indices(self, start=0.0, stop=1.0, extra=0):
        import bisect

        #find first index
        start = max( bisect.bisect_left( self.data.original_timestamp, start ) - extra, 0 )
        stop = min( bisect.bisect_right( self.data.original_timestamp, stop ) + extra, self.nrecords ) 
        
        return start, stop
    
    def _convert_time_to_timestamp(self, t ):
        return t
    
    def _convert_timestamp_to_time(self, t ):
        return t
    
    def convert_timestamp(self,seconds,nanoseconds):
        ts = np.double(seconds) + nanoseconds * 10**-9
        return ts
    
    def convert_and_expand_timestamp(self,seconds,nanoseconds):
        ts = self.convert_timestamp(seconds,nanoseconds)
        return self.expand_timestamp( ts, scale=1.0/self.header['SamplingFrequency'] )
    
    def readdata(self,start=None,stop=None):
        #reimplement readdata to only return samples within [start,stop] window
        
        if start is None:
            start = self.starttime
        
        if stop is None:
            stop = self.endtime
        
        if not (self._expand_time and self._unwrap_data) : #slice on records
            tslice = self.timeslice( start, stop, extra=0 )
            data = self.data.default[tslice]
            return data
        
        tslice = self.timeslice( start, stop, extra=1 )
        data = self.data.default[tslice]
        
        try:
            seg = Segment( [start,stop] )
            
            if self._field_groups['default']==['time']:
                valid = seg.contains(data)
                data = data[valid]
            elif isinstance(data,dict) and 'time' in data.keys():
                valid = seg.contains(data['time'])[0]
                data['time'] = data['time'][valid]
                
                if 'data' in data.keys():
                    data['data'] = data['data'][valid]
        except:
            raise
        
        return data


class NlxFileCSC(NlxFileTimedBuffers):

    def __init__(self,source='',*args,**kwargs):
        
        if not isinstance( source, NlxHeader ):
            source = NlxHeader( source )
        nchan = 1
        nsamples = NLXCSCBUFFERSIZE
        
        dtype = np.dtype( [
        ('timestamp',np.uint64),
        ('channelnumber',np.uint32),
        ('samplefreq',np.uint32),
        ('numvalidsamples',np.uint32),
        ('signal',np.int16,nsamples)
        ] )
        
        field_aliases = {'time': (['timestamp','numvalidsamples'],self.convert_timestamp)}
        field_aliases['timestamp'] = (['timestamp','numvalidsamples'],self.expand_timestamp)
        field_aliases['signal'] = (['signal','numvalidsamples'],self.convert_data)
        field_groups = {'default': ['time','signal']}
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('expand_time',True)
        kwargs.setdefault('unwrap_data',True)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, mmap=kwargs.pop('mmap',False), field_aliases=field_aliases, field_groups=field_groups, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "CSC":
            raise NeuralynxIOError(self._header.file, 'Not a valid CSC file')
    
    def readdata(self,start=None,stop=None):
        
        if start is None:
            start = self.starttime
        
        if stop is None:
            stop = self.endtime
        
        if not (self._expand_time and self._unwrap_data) : #slice on records
            tslice = self.timeslice( start, stop, extra=0 )
            data = self.data.default[tslice]
            return data
        
        tslice = self.timeslice( start, stop, extra=1 )
        data = self.data.default[tslice]
        
        try:
            seg = Segment( [start,stop] )
            
            if self._field_groups['default']==['time']:
                valid = seg.contains(data)
                data = data[valid]
            elif isinstance(data,dict) and 'time' in data.keys():
                valid = seg.contains(data['time'])[0]
                data['time'] = data['time'][valid]
                
                if 'signal' in data.keys():
                    data['signal'] = data['signal'][valid]
        except:
            raise
        
        return data
    
    def convert_data(self,signal,numvalidsamples=None):
        
        if self._nchan==1:
            signal = np.expand_dims( signal, axis=2 )
        
        data = super(NlxFileCSC,self).convert_data( signal )
        
        #deal with invalid samples
        if self._unwrap_data:
            
            invalid_records = np.flatnonzero( numvalidsamples<self._nsamples )
            if len(invalid_records)>0:
                invalid_indices = np.concatenate( [ x*self._nsamples + numvalidsamples[x] + np.arange(self._nsamples - numvalidsamples[x]) for x in invalid_records ] )
                data = np.delete( data, invalid_indices )
        
        return data
    
    def expand_timestamp(self,timestamp=None,numvalidsamples=None,scale=None):
        
        ts = super(NlxFileCSC,self).expand_timestamp( timestamp, scale )
        
        if self._unwrap_data:
            
            invalid_records = np.flatnonzero( numvalidsamples<self._nsamples )
            if len(invalid_records)>0:
                invalid_indices = np.concatenate( [ x*self._nsamples + numvalidsamples[x] + np.arange(self._nsamples - numvalidsamples[x]) for x in invalid_records ] )
                ts = np.delete( ts, invalid_indices )
        
        return ts
    
    def convert_timestamp(self,timestamp,numvalidsamples):
        
        return self.expand_timestamp( timestamp=NlxTimestamp2Seconds( timestamp ), numvalidsamples=numvalidsamples, scale=1.0/self.header['SamplingFrequency'] )


class NlxFileEvent(NlxFileBase):

    def __init__(self,source='',*args,**kwargs):
        
        dtype = np.dtype( [
        ('nstx',np.int16),
        ('npkt_id',np.int16),
        ('npkt_data_size',np.int16),
        ('timestamp',np.uint64),
        ('nevent_id',np.int16),
        ('nttl',np.int16),
        ('ncrc',np.int16),
        ('ndummy1',np.int16),
        ('ndummy2',np.int16),
        ('extra',np.int32,8),
        ('eventstring','S128'),
        ] )
        
        field_groups = {'default': ['time','eventstring']}
        
        #call base class __init__
        NlxFileBase.__init__(self, source,dtype,*args, field_groups=field_groups, **kwargs)

        #check if correct file type
        if self._header.filetype != "Event":
            raise NeuralynxIOError(self._header.file, 'Not a valid Event file')


class NlxVideoPoints:
    #bit 31: reserved
    #bit 30: pure red
    #bit 29: pure green
    #bit 28: pure blue
    #bit 16-27: y
    #bit 15: intensity
    #bit 14: raw red
    #bit 13: raw green
    #bit 12: raw blue
    #bit 0-11: x
    
    RAWBLUE = 12
    RAWGREEN = 13
    RAWRED = 14
    PUREBLUE = 28
    PUREGREEN = 29
    PURERED = 30
    INTENSITY = 15
    
    def __init__(self,data):
        self.x = np.bitwise_and(data,2**12-1)
        self.y = np.bitwise_and(np.right_shift(data,16),2**12-1)
        self.intensity = np.bitwise_and(data,2**15)>0
        self.blue = np.bitwise_and(data,2**28)>0
        self.green = np.bitwise_and(data,2**29)>0
        self.red = np.bitwise_and(data,2**30)>0
        self.rgb = np.concatenate( (self.red[:,:,np.newaxis], self.green[:,:,np.newaxis], self.blue[:,:,np.newaxis]), axis=2 )
        self.bits = np.uint8( np.bitwise_and(np.right_shift(data,self.RAWBLUE-3),120) + np.bitwise_and( np.right_shift( data, self.PUREBLUE ), 7 ) )

class NlxFileVideo(NlxFileBase):
    
    def __init__(self,source='',*args,**kwargs):
        
        dtype = np.dtype( [
        ('swstx',np.uint16),
        ('swid',np.uint16),
        ('swdata_size',np.uint16),
        ('timestamp',np.uint64),
        ('points',np.uint32,400),
        ('sncrc',np.int16),
        ('x',np.int32),
        ('y',np.int32),
        ('angle',np.int32),
        ('targets',np.int32,50),
        ] )
        
        field_aliases = {'points': ('points',self.convert_points)}
        field_aliases['targets'] = ('targets',lambda targets: self.convert_points(points=targets))
        field_groups = {'position': ['time','x','y']}
        field_groups['default'] = ['time','x','y']
        
        #call base class __init__
        NlxFileBase.__init__(self, source,dtype,*args, field_aliases=field_aliases, field_groups=field_groups,**kwargs)

        #check if correct file type
        if self._header.filetype != "Video":
            raise NeuralynxIOError(self._header.file, 'Not a valid Video Tracker file')
        
        # double check sampling frequency
        # based on first 1000 records
        if self.nrecords>1:
            fs = 1./np.mean( np.diff( self.data.time[0:min(self.nrecords,1000)] ) )
            # more than 1% difference
            if np.abs(fs-self.header['SamplingFrequency'])/self.header['SamplingFrequency'] > 0.01:
                self._header._parameters['SamplingFrequency'] = fs
                warnings.warn('Sampling frequency in NVT file header does not match sampling frequency in file. Corrected to ' + str(fs) + ' Hz.') 
    
    def convert_points(self,points):
        return NlxVideoPoints(points)


def NlxOpen(filename='',*args,**kwargs):
    hdr = NlxHeader( filename )
    if hdr.filetype == "Spike":
        return NlxFileSpike(hdr,*args,**kwargs)
    elif hdr.filetype == "CSC":
        return NlxFileCSC(hdr,*args,**kwargs)
    elif hdr.filetype == "Event":
        return NlxFileEvent(hdr,*args,**kwargs)
    elif hdr.filetype == "Video":
        return NlxFileVideo(hdr,*args,**kwargs)
    elif hdr.filetype == "MOZ":
		return NlxFileTahiti(hdr,*args,**kwargs)
    else:
        raise NotImplementedError()


def Nlx2Mwl(filename):
    pass

def main():
    pass

if __name__ == '__main__':
    main()
