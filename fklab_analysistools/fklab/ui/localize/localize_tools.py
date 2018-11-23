# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:07:22 2017

@author: fklab
"""
import numpy as np
import os
import scipy.interpolate
import collections

import fklab.utilities.yaml as yaml

def _get_default_options():
    
    d = collections.OrderedDict()
    
    d['source'] = collections.OrderedDict( [['path',''],['epoch',None]] )
    d['tracking'] = collections.OrderedDict( [['colors',[]], ['orientation',0]] )
    d['regions'] = collections.OrderedDict([['enabled',False],['include',[]],['exclude',[]]])
    d['jumps'] = collections.OrderedDict( [['enabled',True], ['size',20], ['duration',0.5]])
    d['small_gaps'] = collections.OrderedDict([['enabled',True], ['gap_size',0.1]])
    d['diode_distance'] = collections.OrderedDict([['enabled',True], ['threshold',1.5]])
    d['missing_diode'] = collections.OrderedDict([['enabled',True], ['gap_size',0.25]])
    d['large_gaps'] = collections.OrderedDict([['enabled',True], ['gap_size',0.5]])
    d['behavior'] = collections.OrderedDict([['robust',True], ['velocity_smooth',0.25], ['direction_smooth',0.5]])
    
    return d

def makedirs(path, exist_ok=True):
    try:
        os.makedirs(path)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise

def get_nlx_video_time_lut( path, target='VT1' ):
    
    import bs4
    
    lut_file = os.path.join( path, target + '_time_lut.npy' )
    
    if os.path.isfile( lut_file ):
        #load lut
        ts = np.load( lut_file, mmap_mode='r' )
    else:
        filename = os.path.join( path, target + '.smi' )
        
        if not os.path.isfile( filename ):
            print filename
            raise ValueError('No video caption file found')
    
        ts = []
        
        with open(filename) as fid:
            soup = bs4.BeautifulSoup(fid, 'lxml')
            for k in soup.find_all('sync'):
                ts.append( [int(k['start']), int(k.p.text)] )
        
        if len(ts)==0:
            ts = np.zeros( (0,2) )
        else:
            ts = np.array(ts)
            ts = ts / [[1000., 1000000.]] # convert to seconds
    
        #save to target_time_lut.npy
        np.save( lut_file, ts )
        
    return ts

def get_nlx_video_time( path, t ):
    
    import glob
    
    # do we have summary file?
    summary_file = os.path.join( path, 'video_summary.yaml' )
    
    if not os.path.isfile( summary_file ):
        summary = []
        # construct summary file and LUTs
        # list all mpg and smi files
        video_files = glob.glob( os.path.join( path, 'VT1*.mpg' ) )
        video_files = [ os.path.splitext(x)[0] for x in video_files ]
        smi_files = glob.glob( os.path.join( path, 'VT1*.smi' ) )
        smi_files = [ os.path.splitext(x)[0] for x in smi_files ]
        
        # only work with cases where we have both mpg and smi files
        video_files = sorted( list( set(video_files).intersection(smi_files) ) )
        
        # for each video, create LUT
        for k in video_files:
            ts = get_nlx_video_time_lut( path, target=k)
            
            start_time = np.inf if ts.shape[0]==0 else ts[0,1]
            stop_time = -np.inf if ts.shape[0]==0 else ts[-1,1]
            
            basepath, base = os.path.split(k)
            
            summary.append( dict( path=basepath,
                                  base=base,
                                  video_file=base+'.mpg',
                                  caption_file=base+'.smi', 
                                  time_lut_file=base+'_time_lut.npy',
                                  start_time=float(start_time),
                                  stop_time=float(stop_time) ) )
        
        with open( summary_file, 'w' ) as f:
            yaml.dump( summary, stream=f )
    
    else:
        
        with open(summary_file, 'r') as f:
            summary = yaml.load(f)
        
    for k in summary:
        ts = get_nlx_video_time_lut( k['path'], target=k['base'] )
        if t>k['start_time'] and t<k['stop_time']:
             video_t = scipy.interpolate.interp1d( ts[:,1], ts[:,0], kind='linear' )(t)
             return video_t, k
        
    return None, None


def nlx_extract_video_image( path, t, outputfile, overwrite=False ):
    
    if os.path.isfile( outputfile ) and not overwrite:
        return
    
    video_t, video = get_nlx_video_time( path, t )
    
    if video_t is None:
        return
    
    extract_video_image( os.path.join( video['path'], video['video_file'] ), video_t, outputfile )


def extract_video_image( inputfile, t, outputfile ):
    
    import subprocess
    
    command = ['avconv', '-ss', str(t), '-i', inputfile, '-frames:v', '1', outputfile ]
    with open(os.devnull,'w') as f:
        if subprocess.call( ['which', 'avconv'], stdout=f, stderr=f ) == 0:
            subprocess.call( command, stdout=f, stderr=f ) 
        else:
            print("Cannot extract images from video file. Please install avconv using `sudo apt-get install libav-tools`")
