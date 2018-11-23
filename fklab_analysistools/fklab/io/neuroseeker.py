import os
import numpy as np

# define type of each field (unsigned 32 bit) in a data packet
# notice that order is big endian ('>')
FIELDTYPE = '>u4'

# number of counter values in packet
NCOUNTERS = 20
# total number of channels on probe
NCHANNELS = 1440
# how many channels are packed into a single field in a data packet
NCHANNELS_PER_FIELD = 3
# number of fields in a data packet that contain channel data
NCHANNEL_FIELDS = NCHANNELS / NCHANNELS_PER_FIELD

NREGIONS = 12
NCHANNELS_PER_REGION = NCHANNELS / NREGIONS

REF_OFFSET = 56
NREF_PER_REGION = 8

NCOLUMNS = 4
NROWS = NCHANNELS / NCOLUMNS

BITS_PER_CHANNEL = 10

PACKETTYPE = np.dtype( [('sync',FIELDTYPE), ('counter',(FIELDTYPE, NCOUNTERS)), ('samples',(FIELDTYPE,NCHANNEL_FIELDS))] )

def unpack_prefix( data, out=None ):
    """Retrieve 2-bit prefix from NeuroSeeker packet fields.
    
    Parameters
    ----------
    data : array of unsigned int32
    out : array of integer type, optional
        If given, result will be stored in `out` array.
    
    Returns
    -------
    out : array
        2-bit prefix
    
    """
    
    
    if out is None:
        out = np.empty( data.shape, dtype=np.uint8 )
    
    np.right_shift( data, 30, out, casting='unsafe' )
    
    return out

def remove_prefix( data, out=None ):
    """Remove 2-bit prefix from NeuroSeeker packet fields.
    
    Parameters
    ----------
    data : array of unsigned int32
    out : array of unsigned int32, optional
        If given, result will be stored in `out` array.
    
    Returns
    -------
    out : array
        Data with 2-bit prefix set to zero.
    
    """
    
    if out is None:
        out = np.bitwise_and( data, 2**30 -1 )
    else:
        np.bitwise_and( data, 2**30 -1, out )
        
    return out

def unpack_samples( data, out=None ):
    """Unpack NeuroSeeker probe data samples.
    
    Parameters
    ----------
    data : (nsamples, nfields) array of unsigned int32
        Array of packed data samples for `nsamples` packets.
    out : (nsamples, nchannels) array of signed int16, optional
        If given, result will be stored in `out` array.
    
    Returns
    -------
    out : (nsamples, nchannels) array
    
    """
    
    nsamples, ndata = data.shape
    nchannels = ndata * NCHANNELS_PER_FIELD
    
    if out is None:
        #pre-allocate output
        #note that we use signed int16
        out = np.empty( (nsamples,nchannels) , dtype=np.int16 )
    
    # perform in-place bit shifting
    for k in range(NCHANNELS_PER_FIELD):
        np.right_shift( data, k*BITS_PER_CHANNEL, out[:,k::NCHANNELS_PER_FIELD] )
    
    # only keep the relevant bits
    np.bitwise_and( out, 2**BITS_PER_CHANNEL - 1, out )
    
    return out


# functions for defining channel masks
def region_mask( regions=None ):
    """Define channel mask for selected probe regions.
    
    Parameters
    ----------
    regions : int, sequence of ints
        Selection of probe regions (first region = 1)
        
    Returns
    -------
    b : bool array
        True for each channel on the probe that is part of the selected
        regions.
    
    """
    
    out = np.full( NCHANNELS, False, dtype=np.bool )
    
    if regions is None:
        regions = range(1,NREGIONS+1)
    
    regions = np.array( regions, dtype=np.int ).ravel()
    
    if np.any( regions<1 ) or np.any( regions>12 ):
        raise ValueError('Probe region can only take on values from 1 to 12.')
    
    for r in regions:
        offset = (r-1) * NCHANNELS_PER_REGION
        out[ offset:(offset+NCHANNELS_PER_REGION) ] = True
    
    return out

def reference_mask( regions=None ):
    """Define channel mask for local references in selected probe regions.
    
    Parameters
    ----------
    regions : int, sequence of ints
        Selection of probe regions (first region = 1)
        
    Returns
    -------
    b : bool array
        True for each channel on the probe that is a local reference in
        any the selected regions.
    
    """
    
    out = np.full( NCHANNELS, False, dtype=np.bool )
    
    if regions is None:
        regions = range(1,NREGIONS+1)
        
    regions = np.array( regions, dtype=np.int ).ravel()
    
    if np.any( regions<1 ) or np.any( regions>NREGIONS ):
        raise ValueError('Probe region can only take on values from 1 to {0}.'.format(NREGIONS))
    
    for r in regions:
        offset = (r-1) * NCHANNELS_PER_REGION + REF_OFFSET
        out[ offset:(offset+NREF_PER_REGION) ] = True
    
    return out

def column_mask( columns=None ):
    """Define mask for columns of probe channels.
    
    Note: indices for columns are 1-based!
    
    Parameters
    ----------
    columns : int, sequence of ints
        Selection of columns (first column = 1)
        
    Returns
    -------
    b : bool array
        True for each channel on the probe that is in the selected columns.
    
    """
    
    out = np.full( NCHANNELS, False, dtype=np.bool )
    
    if columns is None:
        columns = range(1,NCOLUMNS+1)
        
    columns = np.array( columns, dtype=np.int ).ravel()
    
    if np.any( columns<1 ) or np.any( columns>NCOLUMNS ):
        raise ValueError('Column can only take on values from 1 to {0}.'.format(NCOLUMNS))
    
    for c in columns:
        out[2*(c-1)::8] = True
        out[2*(c-1)+1::8] = True
    
    return out

def row_mask( rows=None ):
    """Define mask for rows of probe channels.
    
    Note: indices for rows are 1-based!
    
    Parameters
    ----------
    rows : int, sequence of ints
        Selection of rows (first row = 1)
        
    Returns
    -------
    b : bool array
        True for each channel on the probe that is in the selected rows.
    
    """
    
    out = np.full( NCHANNELS, False, dtype=np.bool )
    
    if rows is None:
        rows = range(1,NROWS+1)
        
    rows = np.array( rows, dtype=np.int ).ravel()
    
    if np.any( rows<1 ) or np.any( rows>NROWS ):
        raise ValueError('Row can only take on values from 1 to {0}.'.format(NROWS))
    
    for r in rows:
        index = 8 * ((r-1)/2) + (r-1)%2
        out[index:(index+8):2] = True
    
    return out

def unravel_channel_index( channels ):
    """Convert a channel index to row and column.
    
    Note: indices for rows, columns and channels are 1-based!
    
    Parameters
    ----------
    channels : array of ints
        Probe channel indices
    
    Returns
    -------
    row, column : array of int16
        Row and column indices for the corresponding channels.
    
    """
    
    channels = np.asarray( channels, dtype=np.int16 )
    
    if np.any( channels<1 ) or np.any( channels>NCHANNELS ):
        raise ValueError('Channel can only take values from 1 to {0}.'.format(NCHANNELS))
    
    #row = 1 + 2 * ( (channels-1) / 8 ) + (channels-1)%2
    #column = 1 + ( (channels-1-(channels-1)%2) / 2 )%4
    
    a,column,c = np.unravel_index( channels-1, (180,4,2) )
    row = a*2 + c + 1
    column = column + 1
    
    return row, column

def ravel_channel_index( rows, columns ):
    """Convert a row and column indices to channel index.
    
    Note: indices for rows, columns and channels are 1-based!
    
    Parameters
    ----------
    rows, columns : array of ints
        Row and column indices.
    
    Returns
    -------
    channel : array of int16
        Flat channel index.
    
    """
    rows = np.asarray( rows, dtype=np.int16 )
    columns = np.asarray( columns, dtype=np.int16 )
    
    if np.any( rows<1 ) or np.any( rows>NROWS ):
        raise ValueError('Row can only take on values from 1 to {0}.'.format(NROWS))
        
    if np.any( columns<1 ) or np.any( columns>NCOLUMNS ):
        raise ValueError('Column can only take on values from 1 to {0}.'.format(NCOLUMNS))
    
    c = (rows-1)%2
    a = (rows-1-c)/2
    
    #return 1 + 8*((rows-1)/2) + (rows-1)%2 + 2*(columns-1)
    return np.ravel_multi_index( (a,columns-1,c), (180,4,2) ) + 1

# translate NeuroSeeker data file
def convert( infile, outfile, channel_mask=None, nsamples=None, overwrite=False, blocksize=100000 ):
    """Convert NeuroSeeker data to simple binary file.
    
    Parameters
    ----------
    infile : str
        Neuroseeker file
    outfile : str
        Destination file
    channel_mask : array, optional
        Boolean array to select channels that will be saved to `outfile`
    nsamples : int, optional
        Number of samples to convert (by default, all samples in `infile`
        will be converted)
    overwrite : bool
        Overwrite `outfile` if it already exists (by default, raise exception)
    blocksize : int
        Number of NeuroSeeker data packets that should be processed at a time
    
    """
    
    if channel_mask is None:
        channel_mask = np.full( NCHANNELS, True, dtype=np.bool )
    
    # open input file, assume it is a .nsk file
    f_in = np.memmap( infile, dtype=PACKETTYPE, mode='r' )
    
    nsamples = f_in.shape[0] if nsamples is None else min( int(nsamples), f_in.shape[0] )
    nchannels = np.sum( channel_mask )
    
    # open output file, raise exception if it already exists
    if os.path.exists( outfile ) and not overwrite:
        raise IOError("Output file {0} already esists.".format(outfile))
    
    f_out = np.memmap( outfile, dtype=np.int16, shape=(nsamples,nchannels), mode='w+' )
    
    # read blocks of packets from input file
    # unpack, select channels and write to output file
    nblocks = int(nsamples / blocksize)
    
    # pre-allocate intermediate array
    temp = np.empty( (blocksize,NCHANNELS), dtype=np.int16 )
    
    for k in range(nblocks):
        index = slice( k*blocksize, (k+1)*blocksize )
        unpack_samples( f_in[ index ]['samples'], temp )
        f_out[index,:] = temp[:,channel_mask]
    
    # process last block
    if nblocks*blocksize<nsamples:
        index = slice( nblocks*blocksize, nsamples )
        n = nsamples - nblocks*blocksize
        unpack_samples( f_in[ index ]['samples'], temp[:n,:] )
        f_out[index,:] = temp[:n,channel_mask]

import fklab.io.binary_new as binary

def NskOpen(filename, mask=None):
    
    sync_accessor = binary.accessor( dtype=PACKETTYPE.fields['sync'][0], fcn=lambda data,out: out.__setitem__( slice(None), remove_prefix( data['sync'] ) ) , concatenate=True )
    counter_accessor = binary.accessor( dtype=PACKETTYPE.fields['counter'][0], fcn=lambda data,out: out.__setitem__( slice(None), remove_prefix( data['counter'] ) ), concatenate=False )
    
    if mask is None:
        mask = slice(None)
        nchan = NCHANNELS
    else:
        nchan = np.sum(mask)
    
    sample_accessor = binary.accessor( dtype = np.dtype( (np.int16,(nchan,)) ), fcn=lambda data,out,mask=mask: out.__setitem__( slice(None), unpack_samples(data['samples'])[:,mask] ), concatenate=False, cache=True )
    
    accessors = dict( sync = sync_accessor,
                      counter = counter_accessor,
                      samples = sample_accessor,
                    )
    
    return binary.BinaryFileReader( filename, PACKETTYPE, offset=0, accessors=accessors )

