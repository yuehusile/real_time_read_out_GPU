"""
=================================================
Neuralynx file import (:mod:`fklab.io.neuralynx`)
=================================================

.. currentmodule:: fklab.io.neuralynx

Classes and functions to read Neuralynx data files.

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
    NlxEventTimes

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

from fklab.io.binary import BinaryFileReader, accessor, signal_indexer
from fklab.segments import Segment

__all__ = ['NeuralynxError','NeuralynxIOError','NlxTimestamp2Seconds','NlxSeconds2Timestamp','NLXHEADERSIZE','NLXCSCBUFFERSIZE','NlxHeader',
           'NlxFileSpike','NlxFileTahiti','NlxFileCSC','NlxFileEvent','NlxFileVideo','NlxVideoPoints','NlxOpen',
           'NlxEventTimes']

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
    ts = timestamp*0.000001
    return ts

def NlxSeconds2Timestamp(time):
    """Convert seconds to Neuralynx timestamps."""
    t = time * 1000000
    
    if isinstance(t, numbers.Number):
        t = np.uint64( t )
    else:
        t = np.asarray(time*1000000, dtype=np.uint64)
    
    return t
    

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
        'GreenThreshold'
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
    readdata(start,stop)
        Read data from file between start and stop time.
    
    """
    
    def __init__(self,source,dtype,accessors={} ):
        #read file header
        if isinstance(source,NlxHeader):
            self._header = source;
        else:
            self._header = NlxHeader(source)
        
        #get size of records in file
        self._recordsize = self._header.getvalue('RecordSize')
        
        self._record_dtype = dtype
        
        #define accessors
        self._accessors = dict( original_time = (accessor(dtype,'timestamp', label='time', record_shape=(), record_processor=NlxTimestamp2Seconds), None),
                                original_timestamp = (accessor(dtype,'timestamp'),None),
                                time = (accessor(dtype,'timestamp', label='time', record_shape=(), record_processor=NlxTimestamp2Seconds), None),
                              )
        self._accessors['default_time'] = self._accessors['time']
              
        #merge in accessors specified as arguments
        self._accessors.update( accessors )
        
        self.data = BinaryFileReader( self.fullpath, self._record_dtype, offset=NLXHEADERSIZE, accessors=self._accessors )
        
        if self.nrecords>0:
            self._starttimestamp = self.data.original_timestamp[0][0]
            self._endtimestamp = self.data.original_timestamp[self.nrecords-1][0]
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
    def record_dtype(self):
        return self._record_dtype
    
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
    
    def _convert_time_to_timestamp(self, time ):
        return NlxSeconds2Timestamp(time)
    
    def _convert_timestamp_to_time(self, timestamp ):
        return NlxTimestamp2Seconds(timestamp)
    
    def timeslice(self, start=None, stop=None, source=None):
        import bisect
        
        if start is None:
            start = self.starttime
        
        if stop is None:
            stop = self.endtime
        
        if source is None:
            source = self.data.default_time
        
        start = max( bisect.bisect_left( source, start ), 0 )
        stop = min( bisect.bisect_right( source, stop ), source.shape[0] )
        
        return slice( start, stop )
        
    def readdata(self, start=None, stop=None):
        return self.data.default[ self.timeslice(start,stop) ] 
    
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
    """Base class for Neuralynx data files with timestamped data records.
    
    Parameters
    ----------
    source : filename or NlxHeader
    dtype : numpy dtype
        Numpy type description of data in file.
    nchannels, nsamples : int
        Number of channels and samples in data record
    correct_inversion : bool
        Undo any input inversion applied during acquisition.
    scale_data : bool
        Scale data from AD values to V, mV or uV.
    units : { 'V', 'mV', 'uV' }
        Selection of units for scaling.
    clip : bool
        Set clipped valued to NaN
    
    Attributes
    ----------
    nchannels
    nsamples
    correct_inversion
    scale_data
    units
    clip
    
    """
    
    def __init__(self,source,dtype,**kwargs):
        
        #call base class __init__
        NlxFileBase.__init__(self, source, dtype, accessors=kwargs.pop('accessors',{}))
        
        self._nchan = int(kwargs.pop('nchannels',1))
        self._nsamples = int(kwargs.pop('nsamples',1))
    
        self.correct_inversion = kwargs.pop('correct_inversion', False)
        self.scale_data = kwargs.pop('scale_data', True)
        self.units = kwargs.pop('units', 'uV' )
        self.clip = kwargs.pop('clip', True)
        
        if self.aliasing():
            warnings.warn("High cut filter frequency is higher than half the sampling frequency. Data in this file is possibly affected by aliasing.", RuntimeWarning )
            
    def sample2record(self, sample):
        """Convert sample indices to record indices.
        """
        return np.floor( sample/self.nsamples )
    
    def record2sample(self,record):
        """Convert record indices to sample indices.
        """
        return np.int( record*self.nsamples )
    
    def aliasing(self):
        """Check if data suffers from potential aliasing.
        """
        fs = self.header['SamplingFrequency']
        hc = self.header['DspHighCutFrequency']
        return self.header["DSPHighCutFilterEnabled"] and (2*hc > fs)
    
    @property
    def nchannels(self):
        return self._nchan
    
    @property
    def nsamples(self):
        return self._nsamples
        
    def _expand_timestamp(self, timestamp, scale=None):
        
        #if self._expand_time:
        if scale is None:
            scale = np.uint64( self._convert_time_to_timestamp(1) / self.header['SamplingFrequency'] )
        
        ts = np.arange(self._nsamples).reshape(1,self._nsamples) * scale
        ts = ts + timestamp.reshape( timestamp.size, 1)
        #else:
        #    ts = timestamp
        
        return ts
    
    def _convert_timestamp(self, timestamp):
        return self._expand_timestamp( self._convert_timestamp_to_time( timestamp ), scale=1.0/self.header['SamplingFrequency'] )
    
    def _convert_data(self, data):
        
        #expected shape of 3D data array: nbuffers x nsamples x nchannels
        nbuffers, nsamples, nchannels = data.shape
        
        scale=1
        if self._correct_inversion and self._header.getvalue('InputInverted',False):
            scale = -scale
        
        if self._scale_data:
            scale = scale * self._header.getvalue('ADBitVolts').reshape(1,1,self._nchan)*_units_scale_factor[self._signal_units]
        
        if np.any(scale!=1):
            data = scale * data
        
        if self._clip:
            clip_value = self.header["ADMaxValue"] * np.abs(scale)
            data = data.astype(np.float, copy=False) # force cast to float
            data[np.logical_or( data >= clip_value, data <= -clip_value )] = np.nan
        
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
    """Neuralynx spike data file (*.ntt)
    
    Parameters
    ----------
    source : filename or NlxHeader
    correct_inversion : bool
        Undo any input inversion applied during acquisition (default: True)
    scale_data : bool
        Scale data from AD values to V, mV or uV (default: True)
    units : { 'V', 'mV', 'uV' }
        Selection of units for scaling (default: uV)
    clip : bool
        Set clipped valued to NaN (default: True)
    
    Attributes
    ----------
    nchannels
    nsamples
    correct_inversion
    scale_data
    units
    clip
    
    """
    
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
        
        spike_time_accessor = accessor(dtype,'timestamp', label='time', record_shape=(), record_processor=self._convert_timestamp_to_time)
        waveform_accessor = accessor(dtype,'waveform',record_processor=self._convert_data)
        
        accessors = dict( sample_time = ( accessor(dtype,'timestamp', label='time', record_shape=(), record_processor=self._convert_timestamp), None), 
                          sample_timestamp = ( accessor(dtype,'timestamp', record_processor=self._expand_timestamp), None),
                          waveform = ( waveform_accessor, None ),
                          spike_time = ( spike_time_accessor, None),
                          spikes = ( [spike_time_accessor, waveform_accessor], None ),
                          default = ( [spike_time_accessor, waveform_accessor], None )
                    )
        
        accessors['default_time'] = accessors['spike_time']
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, accessors=accessors, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "Spike":
            raise NeuralynxIOError(self._header.file, 'Not a valid Spike file [tetrode/stereotrode/electrode]')
        
        indexer = signal_indexer(nsamples,self.nrecords)
        
        time_by_sample_accessor = accessor(dtype,'timestamp', label='time', record_shape=(nsamples,), sample_dim=0, record_processor=self._convert_timestamp)
        waveform_by_sample_accessor = accessor(dtype,'waveform',record_shape=(nsamples,nchan), sample_dim=0, record_processor=self._convert_data)
        
        self.data.add_accessors( dict( time_by_sample = ( time_by_sample_accessor, indexer ),
                                       waveform_by_sample = ( waveform_by_sample_accessor, indexer ) ) )
        
    def _convert_data(self,waveform):

        waveform = super(NlxFileSpike,self)._convert_data( waveform )
        
        return waveform


class NlxFileTahiti(NlxFileTimedBuffers):
    """Tahiti data file."""
    
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
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, accessors={}, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "MOZ":
            raise NeuralynxIOError(self._header.file, 'Not a valid Tahiti MOZ file')
        
        indexer = signal_indexer(nsamples,self.nrecords,None)
        time_by_sample_accessor = accessor( dtype, ['seconds','nanoseconds'], label='time', record_shape=(nsamples,), sample_dim=0, record_processor=self._convert_and_expand_timestamp )
        data_by_sample_accessor = accessor(dtype,'data',sample_dim=0,record_processor=self._convert_data)
        
        accessors = dict( original_time = ( accessor( dtype, ['seconds','nanoseconds'], label='time', record_shape=(), record_processor=self._convert_timestamp ), None ), 
                          original_timestamp = ( accessor( dtype, ['seconds','nanoseconds'], label='timestamp', record_shape=(), record_processor=self._convert_timestamp ), None ),
                          time = ( accessor( dtype, ['seconds','nanoseconds'], label='time', record_shape=(), record_processor=self._convert_timestamp ), None ),
                          time_by_sample = ( time_by_sample_accessor, indexer ),
                          data = ( accessor(dtype,'data',record_processor=self._convert_data), None),
                          data_by_sample = ( data_by_sample_accessor, indexer ), 
                          default = ( [time_by_sample_accessor, data_by_sample_accessor], indexer )
                        )
        
        accessors['default_time'] = accessors['time_by_sample']
        
        self.data.add_accessors( accessors )
        
        
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
    
    def _convert_timestamp(self,seconds,nanoseconds):
        ts = np.double(seconds) + nanoseconds * 10**-9
        return ts
    
    def _convert_and_expand_timestamp(self,seconds,nanoseconds):
        ts = self._convert_timestamp(seconds,nanoseconds)
        return self._expand_timestamp( ts, scale=1.0/self.header['SamplingFrequency'] )
    

class NlxFileCSC(NlxFileTimedBuffers):
    """Neuralynx continuous signal data file (*.ncs)
    
    Parameters
    ----------
    source : filename or NlxHeader
    correct_inversion : bool
        Undo any input inversion applied during acquisition (default: True)
    scale_data : bool
        Scale data from AD values to V, mV or uV (default: True)
    units : { 'V', 'mV', 'uV' }
        Selection of units for scaling (default: uV)
    clip : bool
        Set clipped valued to NaN (default: True)
    
    Attributes
    ----------
    nchannels
    nsamples
    correct_inversion
    scale_data
    units
    clip
    
    """
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
        
        #set default options
        kwargs.setdefault('correct_inversion',True)
        kwargs.setdefault('scale_data',True)
        kwargs.setdefault('units','uV')
        
        #call base class __init__
        NlxFileTimedBuffers.__init__(self, source, dtype, nchannels=nchan, nsamples=nsamples, **kwargs)
        
        #check if correct file type
        if self._header.filetype != "CSC":
            raise NeuralynxIOError(self._header.file, 'Not a valid CSC file')
    
        validsamples = self.data.numvalidsamples[:]
        invalid = np.flatnonzero( validsamples<NLXCSCBUFFERSIZE )
        
        self._ninvalid = len(invalid)
        self._invalid_samples = np.vstack( [invalid, validsamples[invalid]] ).T
        
        indexer = signal_indexer(NLXCSCBUFFERSIZE,self.nrecords,self._invalid_samples )
        time_by_sample_accessor = accessor(dtype,'timestamp', label='time', record_shape=(nsamples,), sample_dim=0, record_processor=self._convert_timestamp)
        signal_by_sample_accessor = accessor(dtype,'signal',record_shape=(nsamples,), sample_dim=0, record_processor=self._convert_data)
        accessors = dict( time_by_sample = ( time_by_sample_accessor, indexer),
                          timestamp_by_sample = ( accessor(dtype,'timestamp',record_shape=(nsamples,), sample_dim=0, record_processor=self._expand_timestamp), indexer),
                          signal_by_sample = ( signal_by_sample_accessor, indexer ),
                          signal = ( accessor(dtype,'signal',record_processor=self._convert_data), None ),
                          default = ( [time_by_sample_accessor, signal_by_sample_accessor], indexer )
                    )
        
        accessors['default_time'] = accessors['time_by_sample']
        
        self.data.add_accessors( accessors )
        
    
    def _convert_data(self,signal):
        
        if self._nchan==1:
            signal = np.expand_dims( signal, axis=2 )
        
        data = super(NlxFileCSC,self)._convert_data( signal )
        
        if self._nchan==1:
            data = data.squeeze(axis=2)
        
        return data
    
    def _expand_timestamp(self,timestamp=None,scale=None):
        
        ts = super(NlxFileCSC,self)._expand_timestamp( timestamp, scale )
        
        return ts
    
    def _convert_timestamp(self,timestamp):
        
        return self._expand_timestamp( timestamp=NlxTimestamp2Seconds( timestamp ), scale=1.0/self.header['SamplingFrequency'] )


class NlxFileEvent(NlxFileBase):
    """Neuralynx event data file (*.nev)
    
    Parameters
    ----------
    source : filename or NlxHeader
    
    """

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
        
        accessors = dict( default = ( [accessor(dtype,'timestamp', label='time', record_shape=(),record_processor=NlxTimestamp2Seconds), accessor(dtype,'eventstring')], None ) )
        
        #call base class __init__
        NlxFileBase.__init__(self, source,dtype,*args, accessors=accessors, **kwargs)

        #check if correct file type
        if self._header.filetype != "Event":
            raise NeuralynxIOError(self._header.file, 'Not a valid Event file')

def NlxEventTimes( path, eventstring= 'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0080).' ):
    """Retrieve times for a specific event from a Neuralynx nev file.
    
    Parameters
    ----------
    path : str
        Either the location of a Neuralynx event file, or a valid path that
        contains a Events.nev file.
    eventstring : str
        Target event string.
    
    Returns
    -------
    float array
        Event times for specified event string.
    
    """
    
    if os.path.isdir( filename ):
        filename = os.path.join( filename, 'Events.nev' )
    
    f = NlxFileEvent( filename )
    
    b = f.data.eventstring[:] == eventstring
    
    return f.data.time[b]

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
    """Neuralynx video tracking data file (*.nvt)
    
    Parameters
    ----------
    source : filename or NlxHeader
    
    """

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
        
        time_accessor = accessor(dtype,'timestamp', label='time', record_shape=(), record_processor=NlxTimestamp2Seconds)
        x_accessor = accessor( dtype, 'y' )
        y_accessor = accessor( dtype, 'x' )
        
        accessors = dict( points = ( accessor(dtype,'points',record_processor=self._convert_points), None),
                          targets = ( accessor(dtype,'targets',record_processor=lambda targets: self._convert_points(points=targets)), None),
                          position = ( [time_accessor, x_accessor, y_accessor], None),
                          default = ( [time_accessor, x_accessor, y_accessor], None)
                        )
        
        #call base class __init__
        NlxFileBase.__init__(self, source,dtype,*args, accessors=accessors,**kwargs)

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
    
    def _convert_points(self,points):
        return NlxVideoPoints(points)


def NlxOpen(filename='',*args,**kwargs):
    """Open Neuralynx data file.
    
    This function will determine the type of Neuralynx data file and 
    construct the appropriate Neuralynx file object. See `NlxFileCSC`, 
    `NlxFileEvent`, `NlxFileVideo`, and `NlxFileSpike`.
    
    Parameters
    ----------
    filename : str
    *args, **kwargs : additional arguments that are passed to Neuralynx
                      file class constructor.
    
    """
    
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
    raise NotImplementedError()

def main():
    pass

if __name__ == '__main__':
    main()
