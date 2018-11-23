"""
=====================================
MWL file import (:mod:`fklab.io.mwl`)
=====================================

.. currentmodule:: fklab.io.mwl

Classes and function to read MWL data files used in the Wilson Lab at MIT.

Functions
=========

.. autosummary::
    :toctree: generated/
    
    MwlOpen
    MwlTimestamp2Seconds
    MwlSeconds2Timestamp

File classes
============

.. autosummary::
    :toctree: generated/
    
    MwlFileDiode
    MwlFileEeg
    MwlFileEvent
    MwlFileFeature
    MwlFileCluster
    MwlFileWaveform

"""

import os
import re
import ast

import numpy as np

from fklab.io.binary_legacy import BinaryFileReader
from .common import *

__all__ = ['MwlOpen','MwlTimestamp2Seconds','MwlSeconds2Timestamp','MwlFileDiode','MwlFileEeg','MwlFileEvent','MwlFileFeature','MwlFileCluster','MwlFileWaveform']

_VERSION = "0.1"

_units_scale_factor = {'V':1, 'mV':1000, 'uV':1000000}

#define mwl file header specific handlers
class MwlFieldHandler(ValueHandler):
    @staticmethod
    def validate(value):
        if isinstance(value,str):
            value = mwl2dtype(value)
        elif isinstance( value, np.dtype ):
            dtype2mwl(value) #test if value can be converted to string
        else:
            raise ValueError
        
        return value
    
    @staticmethod
    def stringify(value):
        return dtype2mwl(value)
            
class MwlFilterHandler(ValueHandler):
    @staticmethod
    def validate(value):
        return code2filter(int(value))
    
    @staticmethod
    def stringify(value):
        return str(filter2code(value))

#mwl file header specific validation object
MwlHeaderValidator = HeaderValidator(default=DefaultHandler)
MwlHeaderValidator.set_handler( IntHandler, ['Argc',
                                             'Probe',
                                             'nelectrodes',
                                             'nchannels',
                                             'nelect_chan',
                                             'errors',
                                             'disk_errors',
                                             'dma_bufsize',
                                             'spikelen',
                                             'spikesep',
                                             'spike_size',
                                             'channel [0-7] (ampgain|adgain|threshold|color|offset|contscale)'
                                             ] )
MwlHeaderValidator.set_handler( FloatHandler, ['rate'] )
MwlHeaderValidator.set_handler( StringEnumHandler(['Binary','Ascii']), ['File type'] )
MwlHeaderValidator.set_handler( MwlFieldHandler, ['Fields'] )
MwlHeaderValidator.set_handler( MwlFilterHandler, ['channel [0-7] filter'] )

#mappings between numpy data types and mwl field codes
_mwl_type_map = { np.uint8:1,
                  np.int16:2,
                  np.int32:3,
                  np.float32:4,
                  np.float64:5,
                  np.uint32:8,
                  np.int8:9,
                  np.str_:10,
                  np.uint16:11,
                  np.int64:12,
                  np.uint64:13 }

_mwl_type_map_inverse = dict((v,k) for k, v in _mwl_type_map.iteritems())

#functions for conversion between numpy dtype and header Fields parameter
def mwl2dtype(s):
    """convert a fields string from mwl file header to numpy dtype."""
    
    #assume new style field descriptors: FIELD1 [tab] FIELD2 [tab] FIELD3
    #where FIELDn: name,type_code,type_size,shape
    #where shape: integer>0 or [p,q,r] or [p q r]
    
    #split fields on tabs
    fields = s.strip().split('\t')
    
    dt = []
    
    for f in fields:
        #get field attributes
        #name,type_code,type_size,shape
        attr = f.strip().split(',')
        
        if len(attr)!=4:
            raise ValueError
        
        np_type = _mwl_type_map_inverse[int(attr[1])]
        np_shape = ast.literal_eval( re.sub(" +",",",attr[3].strip()) )
        
        if isinstance(np_shape,list):
            np_shape = tuple(np_shape)
        
        assert np_type().itemsize==int(attr[2])
        
        dt.append( (attr[0], np_type, np_shape) )
        
    return np.dtype( dt )

def dtype2str(dt,name=None):
    """Convert simple dtype to mwl field string."""
    
    if dt.fields is not None:
        raise ValueError
    
    if dt.type == np.void:
        type_code = _mwl_type_map[dt.subdtype[0].type]
    else:
        type_code = _mwl_type_map[dt.type]
    
    if dt.subdtype:
        type_size = dt.subdtype[0].itemsize
    else:
        type_size = dt.itemsize
    
    if len(dt.shape)==0:
        shape = str(1)
    elif len(dt.shape)==1:
        shape = str(dt.shape[0])
    else:
        shape = str(list(dt.shape)).replace(' ','')
        shape = shape.replace(',',' ')
        
    name = dt.name if name is None else name
        
    s = "{name},{code},{size},{shape}".format(name=name, code=type_code, size=type_size, shape=shape)
    return s

def dtype2mwl(dt):
    """convert numpy dtype to fields string in mwl file header."""
    s = []
    
    if dt.fields is None:
        s.append( dtype2str(dt) )
    else:
        fields = dt.fields.items() #get a list of all fields
        fields.sort(key=lambda x:x[1][1]) #sort by byte offset
        for name,(t,offset) in fields:
            s.append( dtype2str(t,name=name) )
    
    s = '\t'.join(s)
    
    return s

#mappings for filter codes
_LOWFILTERMAP = { 0:0.1, 1:1, 2:10, 4:100, 8:300, 16:600, (8|16):900 }
_HIGHFILTERMAP = { 0:50, 32:100, 256:200, 512:250, (32|256):275, (32|512):325, (256|512):400, (32|256|512):475, 64:3000, 128:6000, (64|128):9000 }
_LOWFILTERMAP_INV = dict((v,k) for k, v in _LOWFILTERMAP.iteritems())
_HIGHFILTERMAP_INV = dict((v,k) for k, v in _HIGHFILTERMAP.iteritems())

#functions for parsing and creating filter codes
def code2filter(f):
    """Extract high-pass and low-pass filter cut-offs from filter code."""
    return _LOWFILTERMAP[f & 31], _HIGHFILTERMAP[f & 481]

def filter2code(f):
    """Convert tuple of high-pass and low-pass filter cut-offs to filter code."""
    return _LOWFILTERMAP_INV[f[0]] + _HIGHFILTERMAP_INV[f[1]]

#utility functions
def header2filetype(h):
    """Determines the MWL file type from file header.
    
    The type is determined as follows: If the header contains the
    parameter 'File Format', then the file type is the value of this
    parameter. Otherwise the type is determined by checking the values
    of the parameters 'Program' and 'Extraction type' (in case of
    adextract). If there is no 'Program' parameter but there is a
    'adversion' parameter it is assumed that the header is from a raw ad
    data file. Possible header types returned: 'event', 'eeg', 'rawpos',
    'waveform', 'diode', 'feature', 'cluster', 'clbound', 'ad', 'unknown'
    
    """
    
    if h.has_parameter('File Format'):
        return h['File Format']
    
    program = h.get_parameter('Program', default=None)
    if not program:
        adversion = h.get_parameter('adversion',default=None)
        if not adversion:
            return 'unknown'
        else:
            return 'ad '+adversion
    else:
        if program=='adextract':
            extraction = h.get_parameter('Extraction type', default=None)
            if not extraction:
                return 'unknown'
            elif extraction=='event strings':
                return 'event'
            elif extraction=='continuous data':
                return 'eeg'
            elif extraction=='extended dual diode position':
                return 'rawpos'
            elif extraction=='tetrode waveforms':
                return 'waveform'
            else:
                return 'unknown'
        elif program=='posextract':
            return 'diode'
        elif program=='spikeparms' or program=='spikeparms2':
            return 'feature'
        elif program=='crextract':
            return 'feature'
        elif program.lower() in ['xclust','xclust3']:
            if h.has_parameter('Cluster'):
                return 'cluster'
            else:
                return 'clbound'
        else:
            return 'unknown'

def MwlTimestamp2Seconds(timestamp):
    """Convert MWL timestamps to seconds."""
    return np.asarray(timestamp*0.0001,dtype=np.float64)

def MwlSeconds2Timestamp(time):
    """Convert seconds to MWL timestamps."""
    return np.asarray(time*10000,dtype=np.uint32)

def ad_bit_volts(gain=1):
    """Computes bit-to-volt scaling factor."""
    vrange = 10.0 #amplifier voltage range
    nlevels = 2048 #number of bit levels (12)
    
    x = (vrange / nlevels) / gain
    
    return x

#functions and class to deal with mwl file text headers
def readheader(filename):
    """Reads text header of mwl file."""
    
    header_lines = []
    header_size = 0
    magic_start = '%%BEGINHEADER'
    magic_end = '%%ENDHEADER'

    with open(filename, 'rb') as fid:
        if fid.read(len(magic_start))!=magic_start:
            raise ValueError
        
        eol = len(fid.readline(3))
        if eol<1 or eol>2:
            raise ValueError
        
        header_size += len(magic_start) + eol
        
        hdr_end = False
        
        while True:
            line = fid.readline()
            
            if len(line)==0:
                break
            
            header_size += len(line)
            
            line = line.rstrip('\n\r')
            
            if line==magic_end:
                hdr_end=True
                break
            
            line = line.strip('% ')
            
            header_lines.append(line)
    
    if not hdr_end:
        raise ValueError #no end of header found!
    
    return header_lines, header_size

def parseheader(h):
    """Parses parameter/value pairs in mwl text header."""
    
    pattern = re.compile( '(?P<param>[A-Za-z]([A-Za-z0-9 \[\]_.])*):( |\t)+(?P<value>[A-Za-z0-9 \-_:;/,.\t()\[\]]+)' )
    
    hdr = [[]]
    
    if len(h)==0:
        return
    
    next_line_length = map( len, h[1:] )
    next_line_length.append(0)
    
    state = 'new'
    ncomments = 0
    
    for line,next_line in zip(h,next_line_length):
        #match regular expression
        match = pattern.match( line ) 
        
        if match:
            d = match.groupdict()
            value = d['value']
            hdr[-1].append( (d['param'],value) )
            state = 'inside'
        elif len(line)==0:
            if next_line>0 and state!='new':
                state = 'new'
                hdr.append([])
            else:
                pass #skip empty line
        else:
            hdr[-1].append( ('__comment' + str(ncomments),line) )
            ncomments+=1
            state='inside'
    
    return hdr
    
class mwlheader(header):
    _parameter_format='% {parameter}:\t{value}\n'
    _comment_format='% {comment}\n'
    _begin_header='%%BEGINHEADER\n'
    _end_header='%%ENDHEADER\n'
    _validator = MwlHeaderValidator
    
    @classmethod
    def fromfile(cls,filename):
        h,hsize = readheader(filename)
        h = parseheader(h)
        return cls( *h )

#classes that represent mwl data files
class MwlFileBase(object):
    """Base class for MWL data files."""
    
    _expected_file_type = None
    _default_extension = 'mwl'
    
    def __init__(self, filename):
        
        #check if file exists
        p = os.path.abspath(filename)
        if not os.path.isfile(p):
            raise IOError('File does not exist')
        
        self._path, self._filename = os.path.split(p)
        
        #read and parse file header
        h, self._headersize = readheader(p)
        self._header = mwlheader( *parseheader(h) )
        
        #determine if file contains Binary or Ascii data
        self._file_format = self._header.get_parameter('File type', default='Binary')
        #determine specific file type
        self._file_type = header2filetype( self._header )
        
        if self._expected_file_type is not None and self._expected_file_type!=self._file_type:
            raise IOError('File is not a valid "' + self._expected_file_type + '" file')
    
    @property
    def fullpath(self):
        return os.path.join(self._path, self._filename )
    
    @property
    def filesize(self):
        return os.path.getsize( self.fullpath )
    
    @property
    def headersize(self):
        return self._headersize
    
    @property
    def fileformat(self):
        return self._file_format
    
    @property
    def filetype(self):
        return self._file_type
    
    @property
    def header(self):
        #TODO: return dict or list of dicts with parameter/values
        #TODO: needs to be a copy, so that values cannot be changed
        pass
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        header = kwargs.get('header', mwlheader() )
        
        if not isinstance(header,mwlheader):
            raise TypeError('Not a valid mwlheader object')
        
        if len(header)==0:
            header.add_subheader()
        
        header[0]['File type'] = kwargs.get('filetype','Binary')
        header[0]['Program'] = 'fklab.io.mwl'
        header[0]['Program version'] = _VERSION
        
        return header
        
    @classmethod
    def create(cls, filename, **kwargs):
        
        filename = os.path.abspath(filename)
        
        root,ext = os.path.splitext(filename)
        if len(ext)==0:
            ext = '.' + cls._default_extension
        filename = root + ext
        
        #does file exist?
        if os.path.isfile(filename) and os.path.exists(filename) and not kwargs.get('overwrite',False):
            raise IOError('File already exists. Use overwrite=True to overwrite existing file')
        
        header = cls._prepare_header( **kwargs )
        header = header.serialize()
        
        with open( filename, 'w' ) as fid:
            fid.write(header)
        
        return cls( filename )

class MwlFileFixedRecord(MwlFileBase):
    """Base class for MWL data files with fixed record sizes."""
    
    _expected_record_dtype = None
    _cast_record_dtype = None
    
    def __init__(self,filename,mmap=False, field_aliases={}, field_groups={}):
        super(MwlFileFixedRecord,self).__init__(filename)
        
        self._field_aliases = {}
        self._field_groups = {}
        
        self._field_aliases.update( field_aliases )
        self._field_groups.update( field_groups )
        
        self._record_dtype = self._header['Fields']
        
        self._prepare_data_access()
        self._setup_data_access(mmap=mmap)
    
    def _prepare_data_access(self):
        pass
    
    def _setup_data_access(self, mmap=False):
        
        if self._expected_record_dtype is not None and self._expected_record_dtype!=self._record_dtype:
            raise ValueError
        
        if self._cast_record_dtype is not None:
            assert self._cast_record_dtype.itemsize==self._record_dtype.itemsize
            self._record_dtype = self._cast_record_dtype
        
        if self._file_format!='Binary':
            raise NotImplementedError
        
        self.data = BinaryFileReader( self.fullpath, self._record_dtype, offset=self._headersize, alias=self._field_aliases, groups=self._field_groups, mmap=mmap )
    
    @property
    def recordsize(self):
        return self._record_dtype.itemsize
    
    @property
    def nrecords(self):
        return (self.filesize - self._headersize)/self.recordsize
    
    @property
    def dtype(self):
        return self._record_dtype
    
    def append_data(self,data):
        if (self._expected_record_dtype and self._expected_record_dtype!=data.dtype) or self._record_dtype!=data.dtype:
            raise TypeError('Data has wrong data type')
        
        if self._file_format=='Binary':
            sep=""
        else:
            sep="\t"
        
        with open(self.fullpath,'ab') as fid:
            fid.seek(0,2) #move to end
            data.tofile( fid, sep=sep)
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        header = super(MwlFileFixedRecord,cls)._prepare_header(**kwargs)
        header[0]['Fields'] = kwargs.get('dtype')
        return header
    
class MwlFileDiode(MwlFileFixedRecord):
    """MWL diode file (*.p)"""
    
    _expected_record_dtype = np.dtype([('timestamp',np.uint32),('xfront',np.int16),('yfront',np.int16),('xback',np.int16),('yback',np.int16)])
    _cast_record_dtype = None
    
    _expected_file_type = 'diode'
    _default_extension = 'p'
    
    def __init__(self, filename, field_aliases={}, field_groups={}, **kwargs):
        
        field_aliases = dict( [ ('time',('timestamp',MwlTimestamp2Seconds)) ], **field_aliases )
        field_groups = dict( [ ('front',['xfront','yfront']), ('back',['xback','yback']) ], **field_groups )
        
        super(MwlFileDiode,self).__init__(filename, field_aliases=field_aliases, field_groups=field_groups, **kwargs)
        
    @classmethod
    def _prepare_header(cls, **kwargs):
        kwargs['dtype'] = self._expected_record_dtype
        header = super(MwlFileDiode,cls)._prepare_header(**kwargs)
        header[0]['File Format'] = MwlFileDiode._expected_file_type
        header[0]['Extract type'] = 'extended dual diode position'
        return header

class MwlFileEeg(MwlFileFixedRecord):
    """MWL eeg file (*.eeg)"""
    
    _expected_file_type = 'eeg'
    _default_extension = 'eeg'
    
    def __init__(self, filename, field_aliases={}, field_groups={},**kwargs):
        
        field_aliases = dict( [ ('time',('timestamp',self.convert_timestamp)), ('data',('data',self.convert_signal)) ], **field_aliases )
        #field_groups = dict(**field_groups)
        
        super(MwlFileEeg,self).__init__(filename, field_aliases=field_aliases, field_groups=field_groups, mmap=kwargs.pop('mmap',False))
        
        self.correct_inversion = kwargs.pop('correct_inversion', True)
        self.scale_signal = kwargs.pop('scale_signal', True)
        self.expand_time = kwargs.pop('expand_time', True)
        self.unwrap_data = kwargs.pop('unwrap_data', True)
        self.units = kwargs.pop('units', 'uV' )
    
    def _prepare_data_access(self):
        nsamples = self._record_dtype.fields['data'][0].itemsize / 2
        
        self._nchannels = self._header.get_parameter('nchannels')
        self._nsamples = nsamples / self._nchannels
        
        self._gains = np.array([self._header['channel ' + str(k) + ' ampgain'] for k in range(self._nchannels)])
        self._sampling_rate = self._header['rate'] / self._nchannels
        
        self._expected_record_dtype = np.dtype([('timestamp',np.uint32),('data', np.int16, self._nchannels*self._nsamples)])
        self._cast_record_dtype = np.dtype([('timestamp',np.uint32),('data', np.int16, (self._nsamples,self._nchannels) )])
    
    def convert_timestamp(self,timestamp):
        
        return self.expand_timestamp( timestamp=MwlTimestamp2Seconds( timestamp  ), scale=1.0/self._sampling_rate )

    def expand_timestamp(self,timestamp,scale=None):
        if self._expand_time:
            if scale is None:
                scale = np.uint32( MwlSeconds2Timestamp(1) / self._sampling_rate )
            ts = np.arange( self._nsamples ).reshape(1,self._nsamples) * scale
            ts = ts + timestamp.reshape( timestamp.size, 1 )
        else:
            ts = timestamp
        
        if self._unwrap_data:
            ts = ts.flatten()
            
        return ts
    
    def convert_signal(self,data):
        
        #apply scaling
        scale = 1
        if self._correct_inversion:
            scale = -scale

        if self._scale_signal:
            scale = scale * ad_bit_volts(self._gains)  * _units_scale_factor[self._signal_units]

        if scale!=1:
            data = scale * data.flatten() #TODO: reshape rather than flatten
        
        return data
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        nchannels = int(kwargs.get('nchannels',4))
        nsamples = int(kwargs.get('nsamples',32))*nchannels
        kwargs['dtype'] = np.dtype([('timestamp',np.uint32),('data', np.int16, nchannels*nsamples)])
        
        header = super(MwlFileEeg,cls)._prepare_header(**kwargs)
        
        header[0]['File Format'] = MwlFileEeg._expected_file_type
        
        return header
    
    @property
    def nsamples(self):
        return self._nsamples
    
    @property
    def nchannels(self):
        return self._nchannels
    
    @property
    def gains(self):
        return self._gains.copy()
    
    @property
    def sampling_rate(self):
        return self._sampling_rate
    
    @property
    def correct_inversion(self):
        return self._correct_inversion
    
    @correct_inversion.setter
    def correct_inversion(self,value):
        self._correct_inversion = bool(value)
    
    @property
    def scale_signal(self):
        return self._scale_signal
    
    @scale_signal.setter
    def scale_signal(self,value):
        self._scale_signal = bool(value)
    
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
    def units(self):
        return self._signal_units
    
    @units.setter
    def units(self,value):
        if not value in _units_scale_factor.keys():
            raise ValueError
        self._signal_units = value

class MwlFileEvent(MwlFileFixedRecord):
    """MWL event file (*.evt)"""
    
    _expected_file_type = 'event'
    _default_extension = 'evt'
    
    def __init__(self, filename, field_aliases={}, field_groups={},**kwargs):
        
        field_aliases = dict( [ ('time',('timestamp',MwlTimestamp2Seconds)) ], **field_aliases )
        #field_groups = dict( **field_groups )
        
        super(MwlFileEvent,self).__init__(filename, field_aliases=field_aliases, field_groups=field_groups, **kwargs)
    
    def _prepare_data_access(self):
        self._string_size = int( kwargs.get('string_size',80) )
        
        self._expected_record_dtype = np.dtype([('timestamp',np.uint32),('string',np.uint8,self._string_size)])
        self._cast_record_dtype = np.dtype([('timestamp',np.uint32),('string',np.str_,self._string_size)])
    
    @property
    def string_size(self):
        return self._string_size
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        string_size = kwargs.get('string_size',80)
        kwargs['dtype'] = np.dtype([('timestamp',np.uint32),('string',np.uint8,string_size)])
        
        header = super(MwlFileEvent,cls)._prepare_header(**kwargs)
        
        header[0]['File Format'] = MwlFileEvent._expected_file_type
        
        return header

class MwlFileFeature(MwlFileFixedRecord):
    """MWL spike feature file (*.pxyabw)"""
    
    _expected_file_type = 'feature'
    _default_extension = 'pxyabw'
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        header = super(MwlFileFeature,cls)._prepare_header(**kwargs)
        header[0]['File Format'] = MwlFileFeature._expected_file_type
        return header

class MwlFileCluster(MwlFileFixedRecord):
    """MWL spike cluster file (*.cluster)"""
    
    _expected_file_type = 'cluster'
    _default_extension = 'cluster'
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        header = super(MwlFileCluster,cls)._prepare_header(**kwargs)
        header[0]['File Format'] = MwlFileCluster._expected_file_type
        return header

class MwlFileWaveform(MwlFileFixedRecord):
    """MWL spike waveform file (*.tt)"""
    
    _expected_file_type = 'waveform'
    _default_extension = 'tt'
    
    def __init__(self, filename, field_aliases={}, field_groups={},**kwargs):
        
        field_aliases = dict( [ ('time',('timestamp',MwlTimestamp2Seconds)), ('waveform',('waveform',self.convert_waveform)) ], **field_aliases )
        field_groups = dict( **field_groups )
        
        super(MwlFileWaveform,self).__init__(filename, field_aliases=field_aliases, field_groups=field_groups, mmap=kwargs.pop('mmap',False))
        
        self.correct_inversion = kwargs.pop('correct_inversion', False)
        self.scale_waveform = kwargs.pop('scale_waveform', True)
        self.expand_time = kwargs.pop('expand_time', False)
        self.units = kwargs.pop('units', 'uV' )
        
    def _prepare_data_access(self):
        nsamples = self._record_dtype.fields['waveform'][0].itemsize / 2
        
        for sh in self._header:
            if sh.has_parameter('nelect_chan'):
                self._nchannels = sh['nelect_chan']
                break
            elif sh.has_parameter('nchannels'):
                self._nchannels = sh['nchannels']
                break        
        
        self._nsamples = nsamples / self._nchannels
        
        self._probe = self._header['Probe']
        self._gains = np.array( [self._header['channel ' + str(k+self._probe*self._nchannels) + ' ampgain'] for k in range(self._nchannels) ] )
        
        self._sampling_rate = self._header['rate'] / self._header['nchannels']
        
        self._expected_record_dtype = np.dtype([('timestamp',np.uint32),('waveform', np.int16, self._nchannels*self._nsamples)])
        self._cast_record_dtype = np.dtype([('timestamp',np.uint32),('waveform', np.int16, (self._nsamples,self._nchannels) )])
    
    def convert_waveform(self,waveform):
        if waveform.ndim==2:
            waveform = np.expand_dims( waveform, axis=0 )
        waveform = waveform.transpose((1,2,0))
        
        scale=1
        if self._correct_inversion:
            scale = -scale
        
        if self._scale_waveform:
            scale = scale * ad_bit_volts(self._gains).reshape(1,self._nchannels,1) * _units_scale_factor[self._signal_units]
        
        if np.any(scale!=1):
            waveform = scale * waveform
        
        return waveform
        
    def convert_timestamp(self,timestamp):
        
        return self.expand_timestamp( timestamp=MwlTimestamp2Seconds( timestamp  ), scale=1.0/self._sampling_rate )

    def expand_timestamp(self,timestamp,scale=None):
        if self._expand_time:
            if scale is None:
                scale = np.uint32( MwlSeconds2Timestamp(1) / self._sampling_rate )
            ts = np.arange( self._nsamples ).reshape(1,self._nsamples) * scale
            ts = ts + timestamp.reshape( timestamp.size, 1 )
        else:
            ts = timestamp
        
        if self._unwrap_data:
            ts = ts.flatten()
            
        return ts
    
    @classmethod
    def _prepare_header(cls, **kwargs):
        nchannels = int(kwargs.get('nchannels',4))
        nsamples = int(kwargs.get('nsamples',32))*nchannels
        kwargs['dtype'] = np.dtype([('timestamp',np.uint32),('waveform', np.int16, nchannels*nsamples)])
        
        header = super(MwlFileWaveform,cls)._prepare_header(**kwargs)
        
        header[0]['File Format'] = MwlFileWaveform._expected_file_type
        
        return header
    
    @property
    def nsamples(self):
        return self._nsamples
    
    @property
    def nchannels(self):
        return self._nchannels
    
    @property
    def gains(self):
        return self._gains.copy()
    
    @property
    def sampling_rate(self):
        return self._sampling_rate
    
    @property
    def probe(self):
        return self._probe
    
    @property
    def correct_inversion(self):
        return self._correct_inversion
    
    @correct_inversion.setter
    def correct_inversion(self,value):
        self._correct_inversion = bool(value)
    
    @property
    def scale_waveform(self):
        return self._scale_waveform
    
    @scale_waveform.setter
    def scale_waveform(self,value):
        self._scale_waveform = bool(value)
    
    @property
    def expand_time(self):
        return self._expand_time
    
    @expand_time.setter
    def expand_time(self,value):
        self._expand_time = bool(value)
    
    @property
    def units(self):
        return self._signal_units
    
    @units.setter
    def units(self,value):
        if not value in _units_scale_factor.keys():
            raise ValueError
        self._signal_units = value


#convience function for opening mwl files
def MwlOpen(filename='',*args,**kwargs):
    """Open MWL data file.
    
    This function will determine the type of MWL data file and 
    construct the appropriate MWL file object. See `MwlFileWaveform`, 
    `MwlFileEvent`, `MwlFileFeature`, and `MwlFileEeg`, `MwlFileDiode`,
    `MwlFileFixedRecord`, 'MwlFileCluster`.
    
    Parameters
    ----------
    filename : str
    *args, **kwargs : additional arguments that are passed to MWL
                      file class constructor.
    
    """
    
    hdr = mwlheader.fromfile(filename)
    filetype = header2filetype( hdr )
    
    if filetype == "waveform":
        return MwlFileWaveform(filename,*args,**kwargs)
    elif filetype == "feature":
        return MwlFileFeature(filename,*args,**kwargs)
    elif filetype == "event":
        return MwlFileEvent(filename,*args,**kwargs)
    elif filetype == "eeg":
        return MwlFileEeg(filename,*args,**kwargs)
    elif filetype =="diode":
        return MwlFileDiode(filename,*args,**kwargs)
    elif filetype == 'fixedrecord':
        return MwlFileFixedRecord(filename,*args,**kwargs)
    elif filetype == 'cluster':
        return MwlFileCluster(filename,*args,**kwargs)
    else:
        raise NotImplementedError

