Different file types:

Continuously Sampled Record [.NCS]
Event Record [.NEV]
Single Electrode Spike Record [.NSE]
Stereotrode Spike Record [.NST]
Tetrode Spike Record [.NTT]
Video Tracker Record [.NVT]
Raw Data A/D Record [.NRD]


Each file has two parts:
1. header (ASCII)
2. data (binary)


The header has a fixed size (16kb) and contains metadata (such as the AD channel, sampling frequency, etc.)
A parser needs to be written that extracts the header fields and interprets the values in each field

The data is subdivided into multiple records of fixed length (except raw data files which can contain variable length records - although I am not sure this happens in practice)


specifications for importer:
provides information about the file
allows access to header fields and values
allows access to individual and ranges of records by ID
allows access to ranges of records by timestamp
supports iterator access to records [start:step:end]
provides easy access to data
allows importing of selected fields in records


implementation:
NLX_HEADER_PARSER <- can parse header, returns dictionary?
NLX_FILE_BASE class <- reads header, supports searching records on ID and timestamp, iterator support, 
NLX_FILE_XXX derived class <- knows about specific record format

NLX_OPEN( file ) -> determines file format and returns NLX_FILE_XXX( file )


example:
f = NLX_OPEN( file );
f.header -> dictionary
f.header['NumADChannels'] -> access header field
f.data[0:-1] -> returns array of records
f.data_by_idrange( START=xx, END=xx | RANGE=(xx,xx) )
f.data_by_timerange( START=xx, END=xx | RANGE=(xx,xx) )
f.data_iterator( START=xx, END=xx, STEP=xx )


optional post-processing for NCS files: flattening of timestamps and data


define file conversion functions:
NLX2MWL( file ) -> NLX_OPEN( file ) -> iterate through records -> save to corresponding MWL file
options for flattening NCS files and for NTT -> spike feature detection -> PXYABW?


