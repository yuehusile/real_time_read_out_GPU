#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:01:39 2017

@author: davide
"""

import numpy as np
import os
import fklab.io.neuralynx as nlx
import fklab.behavior.preprocessing
import networkx as nx
import fklab.utilities.yaml as yaml
from functools import reduce

import localize_tools as tools


class ComputationDependencyTree:
    def __init__(self):
        self._graph = nx.DiGraph()
    
    def add_computation( self, computation, dependencies=[] ):
        
        if not computation in self._graph:
            self._graph.add_node( computation, dirty=True )
        
        for k in dependencies:
            if not k in self._graph:
                self._graph.add_node( k, dirty=True)
            
            self._graph.add_path( [computation, k] )
    
    def add_computation_chain( self, computations ):
        
        for k in computations:
            if not k in self._graph:
                self._graph.add_node( k, dirty=True)
        
        self._graph.add_path( computations )
    
    def set_dirty( self, computation ):
        
        if not computation in self._graph:
            raise KeyError('No such computation in graph.')
        
        self._graph.node[computation]['dirty']=True
        
        successors = nx.algorithms.traversal.dfs_successors( self._graph, source=computation)
        successors = reduce( lambda x,y: set(x).union(y), list(successors.values()), set() )
        
        for k in successors:
            self._graph.node[k]['dirty'] = True
    
    def set_clean( self, computation ):
        
        if not computation in self._graph:
            raise KeyError('No such computation in graph.')
        
        self._graph.node[computation]['dirty']=False
    
    def is_dirty( self, computation ):
        
        if not computation in self._graph:
            raise KeyError('No such computation in graph.')
        
        return self._graph.node[computation]['dirty']

class NlxSingleTargetTracker:
    
    def __init__(self):
        
        self._fid = None
        
        self._options = tools._get_default_options()
        
        self._color_validity = None
        self._file_info = dict()
        
        self._dependency_graph = ComputationDependencyTree()
        self._dependency_graph.add_computation_chain( ['load', 'correct', 'behavior'] )
        self._dependency_graph.add_computation_chain( ['load', 'validity'] )
    
    @property
    def info(self):
        return self._file_info
    
    def save_options(self, path, overwrite=True):
        
        if overwrite or not os.path.isfile( path ):
            with open( path, 'w' ) as f:
                #ruamel.yaml.dump( self._options, stream=f, Dumper=yaml.dumper.RoundTripDumper )
                yaml.dump( self._options, stream=f )
    
    def load_options(self, path):
        with open( path ) as f:
            #self._options = ruamel.yaml.load(f, Loader=ruamel.yaml.loader.RoundTripLoader )
            self._options = yaml.load(f)
    
    def get_source(self):
        return self._options['source'].copy()
    
    def set_source(self, filename, epoch=None):
        
        self._fid = nlx.NlxOpen( filename )
        
        self._file_info = dict( filename=filename,
                                fs=self._fid.header['SamplingFrequency'], 
                                intensity=self._fid.header['IntensityThreshold'],
                                red=self._fid.header['RedThreshold'],
                                green=self._fid.header['GreenThreshold'],
                                blue=self._fid.header['BlueThreshold'],
                                resolution=self._fid.header['Resolution'] )
        
        self._tracked_colors = []
        for k in ('red','green','blue','intensity'):
            if self._file_info[k][0]:
                self._tracked_colors.append(k)
        
        self._options['source']['path'] = self._fid.fullpath
        self._options['source']['epoch'] = epoch
        
        # remove any no-tracked colors
        self._options['tracking']['colors'] = [x for x in self._options['tracking']['colors'] if x in self._tracked_colors]
        if len(self._options['tracking']['colors'])==0:
            self._options['tracking']['colors'] = self._tracked_colors[0:min(2,len(self._tracked_colors))]
        
        self._dependency_graph.set_dirty('load')
    
    def get_regions_option(self,option=None):
        if option is None:
            return self._options['regions'].copy()
        else:
            return self._options['regions'][option]
      
    def get_tracked_colors(self):
        return self._tracked_colors
        
    def get_target_colors(self):
        return self._options['tracking']['colors']
    
    def get_orientation(self):
        return self._options['tracking']['orientation']
    
    def set_regions_options( self, **kwargs ):
        
        for k,v in kwargs.items():
            if k in self._options['regions']:
                self._options['regions'][k] = v
        
        self._dependency_graph.set_dirty( 'load' )
    
    def set_target_colors( self, colors ):
        
        if isinstance(colors,str):
            colors = [colors,]
        
        if not isinstance(colors, (list,tuple)):
            raise ValueError('Invalid sequence of colors')
        
        if not all( [k in self._tracked_colors for k in colors] ):
            raise ValueError('Invalid target colors.')
        
        if len(colors)<0 or len(colors)>2:
            raise ValueError('Invalid number of target colors')
        
        self._options['tracking']['colors'] = colors
        
        self._dependency_graph.set_dirty( 'correct' )
    
    def set_orientation(self, val):
        self._options['tracking']['orientation'] = float(val)
        
        self._dependency_graph.set_dirty('behavior')
    
    @property
    def tracking_mode(self):
        
        n = len(self._options['tracking']['colors'])
        
        if n==0:
            raise ValueError('No target colors set.')
            
        return n
    
    @property
    def color_validity(self):
        if self._dependency_graph.is_dirty( 'validity' ):
            self.compute_validity()
        return self._color_validity
    
    def get_correction_option(self, step, option):
        return self._options[step][option]
    
    def set_correction_options(self, step, **kwargs ):
        
        if not step in self._options or step in ('source','tracking','regions'):
            raise KeyError('Unknown correction step.')
        
        for k,v in kwargs.items():
            if k in self._options[step]:
                self._options[step][k] = v

        self._dependency_graph.set_dirty( 'correct' )
    
    def set_jump_correction_options(self, **kwargs):
        self.set_correction_options('jumps',**kwargs)
    
    def set_small_gap_correction_options(self, **kwargs):
        self.set_correction_options('small_gaps',**kwargs)
    
    def set_target_distance_correction_options(self, **kwargs):
        self.set_correction_options('diode_distance',**kwargs)
    
    def set_missing_target_correction_options(self, **kwargs):
        self.set_correction_options('missing_diode',**kwargs)
    
    def set_large_gap_correction_options(self, **kwargs):
        self.set_correction_options('large_gaps',**kwargs)
    
    def get_behavior_option( self, option ):
        return self._options['behavior'][option]
    
    def set_behavior_options( self, **kwargs):
        
        for k,v in kwargs.items():
            if k in self._options['behavior']:
                self._options['behavior'][k] = v
        
        self._dependency_graph.set_dirty( 'behavior' )
    
    
    def reload( self ):
        
        if self._fid is None:
            raise ValueError('No source set.')
        
        epoch = self._options['source']['epoch']
        
        if not epoch is None:
            s = self._fid.timeslice( start=epoch[0], stop=epoch[1] )
        else:
            s = slice(None)
        
        self._time = self._fid.data.time[s]
        
        targets = self._fid.data.targets[s]
        
        if self._options['regions']['enabled']:
            include = self._options['regions']['include']
            exclude = self._options['regions']['exclude']
        else:
            include = []
            exclude = []
        
        self._color_coordinates = fklab.behavior.preprocessing.diode_coordinates(
            targets, 
            colors=self._tracked_colors,
            include_regions=include,
            exclude_regions=exclude )
        
        # default target colors
        #if self._target_colors is None:
        #    self.set_target_colors( self._tracked_colors[0:min(len(self._tracked_colors),2)] )
        
        self._dependency_graph.set_clean( 'load' )
    
    def compute_validity(self):
        coordinates = self.color_target_coordinates
        self._color_validity = { k: 100-np.sum(np.isnan(v[:,0]))*100./v.shape[0] for k,v in coordinates.items() }
        
        self._dependency_graph.set_clean('validity' )
    
    def apply_correction(self):
        
        data = self.color_target_coordinates
        
        data = [ data[k] for k in self._options['tracking']['colors'] ]
        
        self._corrected_coordinates = fklab.behavior.preprocessing.correct_diode_coordinates( 
            self._time, data,
            regions = False,
            jumps = self._options['jumps'],
            small_gaps = self._options['small_gaps'],
            diode_distance = self._options['diode_distance'],
            missing_diode = self._options['missing_diode'],
            large_gaps = self._options['large_gaps'] )
        
        self._dependency_graph.set_clean( 'correct' )
    
    def compute_behavior(self):
        
        coordinates = self.corrected_target_coordinates
        
        dx = 1./self._file_info['fs']
        
        if len(coordinates)==0:
            self._position = np.full( (len(self._time),2), np.nan )
        else:
            self._position = fklab.behavior.preprocessing.compute_position(
                coordinates, robust=self._options['behavior']['robust'] )
            
        self._velocity = fklab.behavior.preprocessing.compute_velocity(
            self._position, dx=dx, smooth=self._options['behavior']['velocity_smooth'] )
        
        if len(coordinates)==2:
            self._headdirection = fklab.behavior.preprocessing.compute_head_direction(
                coordinates[0], coordinates[1], dx=dx, orientation=self._options['tracking']['orientation'],
                smooth=self._options['behavior']['direction_smooth'] )
        else:   
            self._headdirection = np.full( self._velocity.shape, np.nan )
        
        self._dependency_graph.set_clean( 'behavior' )
    
    @property
    def corrected_target_coordinates(self):
        
        if self._dependency_graph.is_dirty( 'correct' ):
            self.apply_correction()
        
        return self._corrected_coordinates
        
    @property
    def behavior(self):
        
        if self._dependency_graph.is_dirty( 'behavior' ):
            self.compute_behavior()
        
        return dict( time=self._time, position=self._position, velocity=self._velocity, head_direction=self._headdirection )
    
    @property
    def color_target_coordinates(self):
        
        if self._dependency_graph.is_dirty( 'load' ):
            self.reload()
        
        return self._color_coordinates