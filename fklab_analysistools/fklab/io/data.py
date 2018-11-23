"""
==================================
Data Import (:mod:`fklab.io.data`)
==================================

.. currentmodule:: fklab.io.data

Function for importing experimental data in FKLab.

.. autosummary::
    :toctree: generated/
    
    import_session_info
    import_position
    import_epoch_info
    import_environment
    
"""

__all__ = ['import_session_info', 'import_position', 'import_epoch_info',
           'import_environment']

import os
import collections

import h5py

import fklab.utilities.yaml as yaml
import fklab.geometry.shapes

def import_session_info( session ):
    """Import session information.
    
    Parameters
    ----------
    session : str
        path to recording session
    
    Returns
    -------
    OrderedDict
    
    """
    
    if not isinstance(session,str) or not os.path.isdir( session ):
        raise IOError('Invalid session path.')
    
    info_file = os.path.join( session, 'info.yaml' )
    if not os.path.isfile( info_file ):
        raise IOError('No session info file found.')
    
    with open(info_file) as f:
        info = yaml.load(f)
    
    return info

def import_epoch_info( session ):
    """Import epoch information.
    
    Parameters
    ----------
    session : str
        path to recording session
    
    Returns
    -------
    OrderedDict
    
    """
    
    info = import_session_info( session )
    
    epochs = collections.OrderedDict()
    
    if 'epochs' in info:
        for e in info['epochs']:
            epochs[e['id']] = e
            epochs[e['id']]['path'] = os.path.abspath( os.path.join( session, 'epochs', e['id'] ) )
        
    return epochs
    
def import_position( session, epoch=None, fields=None ):
    """Import position data.
    
    Parameters
    ----------
    session : str
        path to recording session
    epoch : str, list of str
        epochs for which to load position data
    
    Returns
    -------
    OrderedDict
    
    """
       
    epoch_info = import_epoch_info( session )
    
    if isinstance( epoch, str):
        epoch = [epoch,]
    elif epoch is None:
        epoch = epoch_info.keys()
    
    result = collections.OrderedDict()
    
    if fields is None:
        fields = ['time','position','velocity','head_direction','diodes']
    elif isinstance(fields,str):
        fields = [fields,]
    
    for k in epoch:
        
        if not k in epoch_info.keys():
            raise ValueError('Non-existing epoch.')
        
        pos_file = os.path.join( epoch_info[k]['path'], 'position.hdf5' )
        
        if not os.path.isfile(pos_file):
            #raise IOError('No position data found for epoch {epoch}.'.format(epoch=k))
            continue
        
        fid = h5py.File(pos_file,'r')
        
        pos = dict()
        
        for field in fields:
            if field=='diodes':
                pos[field] = dict()
                for diode in fid['diodes'].keys():
                    pos[field][diode] = fid[field][diode][:]
            else:
                pos[field] = fid[field][:]
        
        pos_info_file = os.path.join( epoch_info[k]['path'], 'position.yaml')
        
        if os.path.isfile(pos_info_file):
            with open(pos_info_file) as f:
                pos['preprocessing'] = yaml.load( f )
        
        result[k] = pos
    
    return result

def import_video_image( session, epoch=None, load=True ):
    """Import video image.
    
    Parameters
    ----------
    session : str
        path to recording session
    epoch : str, list of str
        epochs for which to import video image
    load : bool
        load video image, or return path to file
    
    Returns
    -------
    OrderedDict
    
    """
    
    epoch_info = import_epoch_info( session )
    
    if isinstance( epoch, str):
        epoch = [epoch,]
    elif epoch is None:
        epoch = epoch_info.keys()
    
    result = collections.OrderedDict()
    
    for k in epoch:
        
        if not k in epoch_info.keys():
            raise ValueError('Non-existing epoch.')
        
        video_file = os.path.join( epoch_info[k]['path'], 'video_image.png' )
        
        if not os.path.isfile(video_file):
            img = None
        else:
            if load:
                img = skimage.io.imread( video_file )
                img = skimage.color.rgb2gray( img )
            else:
                img = video_file
        
        result[k] = img
    
    return result
    

def import_environment( session, epoch=None ):
    """Import evenironment data."""
    
    epoch_info = import_epoch_info( session )
    
    if isinstance( epoch, str):
        epoch = [epoch,]
    elif epoch is None:
        epoch = epoch_info.keys()
    
    result = collections.OrderedDict()
    
    for k in epoch:
        
        if not k in epoch_info.keys():
            raise ValueError('Non-existing epoch.')
        
        env_file = os.path.join( epoch_info[k]['path'], 'environment.yaml' )
        
        if not os.path.isfile(env_file):
            continue
        
        with open(env_file) as f:
            result[k] = yaml.load( f )
        
    return result

def export_environment( session, epoch, env):
    """Export environment data."""
    
    epoch_info = import_epoch_info( session )
    
    if not epoch in epoch_info.keys():
        raise ValueError('Non-existing epoch.')
    
    env_file = os.path.join( epoch_info[epoch]['path'], 'environment.yaml' )
    
    if os.path.isfile(env_file):
        # create backup
        import shutil
        import datetime
        shutil.move( env_file, env_file + '.' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') )
    
    with open(env_file, 'w') as f:
        yaml.dump( env, stream=f )

    


