import os
import sys
import numpy as np
from fklab.utilities.general import blocks

class BinaryFileReader(object):
    def __init__(self,filename,dtype,offset=0,alias={},groups={},mmap=None):
        
        if mmap is None:
            # test if python interpreter is running 64 bit
            if sys.maxsize > 2**32:
                mmap = True
            else:
                mmap = False
                
        if mmap:
            self._memorymap = np.memmap(filename, mode = 'r', dtype=dtype, offset=offset )
            #self._memorymap_view = self._memorymap.view(np.recarray)
        
        #save file information
        filesize = os.path.getsize( filename )
        recordsize = dtype.itemsize
        nrecords = (filesize - offset)//recordsize
        self._fileinfo = {'filename':filename,'dtype':dtype,'offset':offset,'recordsize':recordsize,'nrecords':nrecords}
        
        #save group and alias dictionaries
        self._groups = groups
        self._alias = alias
        
        #add any existing field in the file that is not listed in alias
        for fld in self._fileinfo['dtype'].fields:
            if not fld in self._alias:
                self._alias[fld] = (fld,None)
        
        if mmap:
            src = self._memorymap
            klass = _DataProxy_MemoryMap
        else:
            src = self._fileinfo
            klass = _DataProxy_Direct
        
        #set up generic access to file records
        self._data = klass(src, None)
        
        #set up access to individual fields
        for k,v in self._alias.iteritems():
            self.__setattr__(k, klass(src, {k:v}))
        
        #set up access to field groups
        for name,grp in self._groups.iteritems():
            self.__setattr__(name, klass(src, {k:self._alias[k] for k in grp} ) )
    
    def __getitem__(self,key):
        return self._data.__getitem__(key)
    
    def __len__(self):
        return self._fileinfo['nrecords']

class _DataProxy_MemoryMap(object):
    def __init__(self, memorymap, fields=None):
        
        self._memorymap = memorymap
        self._fields = fields
    
    def __len__(self):
        return self._memorymap.__len__()
    
    def __getitem__(self,key):
        
        if self._memorymap.dtype.fields is None or self._fields is None:
            return np.array(self._memorymap.__getitem__(key))
        
        #determine which data fields to read
        fields_to_read = set()
        for alias, (flds, func) in self._fields.iteritems():
            if isinstance(flds,str):
                fields_to_read.add( flds )
            else:
                fields_to_read.update( flds )
                    
        in_data = { fld:np.array(self._memorymap[fld][key]) for fld in fields_to_read }
        
        #apply post-processing functions
        out_data = dict()
        
        for alias,(flds,func) in self._fields.iteritems():
            if isinstance(flds,str):
                flds = [flds]
            
            selected_data = { k:in_data[k] for k in flds }
            
            if func is not None:
                out_data[alias] = func( **selected_data )
            elif len(flds)==1:
                out_data[alias] = in_data[flds[0]]
            else:
                out_data[alias] = selected_data
            
        #do not return dictionary when only one field is requested
        if len(out_data)==1:
            out_data = out_data.values()[0]

        return out_data
        

class _DataProxy_Direct(object):
    def __init__(self, fileinfo, fields=None):
        
        self._fileinfo = fileinfo
        self._fields = fields
    
    def __len__(self):
        return self._fileinfo['nrecords']
    
    def __getitem__(self,key):
        
        #process indexing key
        mode = 'block'
        
        if isinstance(key,slice):
            start,stop,step = key.indices(self._fileinfo['nrecords'])
            if step!=1:
                mode = 'random'
                indices = np.arange(start,stop,step,dtype=np.int64)
                nrows = len(indices)
            else:
                blockstart = start
                blocksize = stop-start
                nrows = blocksize
        else:
            mode = 'random'
            indices = np.array(key,dtype=np.int64).ravel()
            nrows = len(indices)
            if np.any( np.logical_or( indices<0, indices>=self._fileinfo['nrecords'] ) ):
                raise IndexError("Record index out of range")
        
        #create arrays for read data
        if self._fileinfo['dtype'].fields is None or self._fields is None:
            in_data = np.empty( (nrows,), dtype=self._fileinfo['dtype'] )
            nofields = True
        else:
            #determine which data fields to read
            fields_to_read = set()
            for alias, (flds, func) in self._fields.iteritems():
                if isinstance(flds,str):
                    fields_to_read.add( flds )
                else:
                    fields_to_read.update( flds )
                
            in_data = { fld:np.empty( (nrows,), dtype=self._fileinfo['dtype'].fields[fld][0] ) for fld in fields_to_read }
            nofields = False
        
        #read data from file
        if mode=='random':
            with open(self._fileinfo['filename']) as f:
                for idx,recidx in enumerate(indices):
                    f.seek( self._fileinfo['offset'] + recidx*self._fileinfo['recordsize'], os.SEEK_SET )
                    rawdata = np.fromfile(f, dtype=self._fileinfo['dtype'], count=1)
                    if nofields:
                        in_data[idx] = rawdata
                    else:
                        for fld in fields_to_read:
                            in_data[fld][idx] = rawdata[ fld ]
        else:
            nrec = 10000000 // self._fileinfo['dtype'].itemsize #maximum number of records to read at one time
            
            with open(self._fileinfo['filename']) as f:
                
                for b,n in blocks(blocksize,nrec):
                    f.seek(self._fileinfo['offset'] + (b+blockstart)*self._fileinfo['recordsize'], os.SEEK_SET)
                    rawdata = np.fromfile(f, dtype=self._fileinfo['dtype'], count=n)
                    if nofields:
                        in_data[b:(b+n)] = rawdata
                    else:
                        for fld in fields_to_read:
                            in_data[fld][b:(b+n)] = rawdata[ fld ]
        
        #apply post-processing functions
        if not nofields:
            out_data = dict()
            
            for alias,(flds,func) in self._fields.iteritems():
                if isinstance(flds,str):
                    flds = [flds]
                
                selected_data = { k:in_data[k] for k in flds }
                
                if func is not None:
                    out_data[alias] = func( **selected_data )
                elif len(flds)==1:
                    out_data[alias] = in_data[flds[0]]
                else:
                    out_data[alias] = selected_data
            
            #do not return dictionary when only one field is requested
            if len(out_data)==1:
                out_data = out_data.values()[0]
            
        else:
            out_data = in_data

        return out_data
