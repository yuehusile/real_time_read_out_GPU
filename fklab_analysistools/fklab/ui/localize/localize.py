import sys
import os
import argparse

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication

import numpy as np

import collections

from localize_main import Ui_MainWindow

#import fklab.plot.interaction
#import fklab.statistics.circular

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import seaborn

#import ruamel.yaml
# make sure ruamel.yaml works properly with OrderedDicts
#def dict_representer(dumper,data):
#    return dumper.represent_ordereddict( data )

#ruamel.yaml.add_representer(collections.OrderedDict, dict_representer, Dumper=ruamel.yaml.dumper.RoundTripDumper )

import fklab.utilities.yaml as yaml
import fklab.behavior.preprocessing
import localize_tools as tools


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

from tracker import NlxSingleTargetTracker

# start up
# prepare_session( session ) -> get epoch names and times, create epoch directories

# select epoch
# prepare_epoch_data( session, epoch ) -> get options, update widgets, get video image (create if not exist), update_diode_data(session, epoch)
# update_epoch_data( session, epoch ) -> load time and targets from nvt file for epoch, extract diode coordinates for all colors, update_diode_correction()
# update_diode_correction() -> apply correction, save to disk, update_behavior()
# update_behavior()
# update_plot()


class LocalizeWindow(QtWidgets.QMainWindow):
    def __init__(self, subject='.', session='', epoch=''):
        QtWidgets.QMainWindow.__init__(self)
        
        self._enable_recalculation = True
        
        self._tracker = NlxSingleTargetTracker()
        self._regions = []
        
        #1. set up the user interface from Designer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #2. set up matplotlib figures
        self.setup_plots()
        
        self._subject_path = os.path.realpath( subject )
        self.prepare_subject(session, epoch)
    
    def setup_plots(self):
        import fklab.plot.interaction
        
        self._mpl_figure_top = Figure()
        self._mpl_figure_bottom = Figure()
        self._mpl_canvas_top = FigureCanvas( self._mpl_figure_top )
        self._mpl_canvas_bottom = FigureCanvas( self._mpl_figure_bottom )
        
        self.ui.figureContainer.addWidget( self._mpl_canvas_top )
        self.ui.figureContainer.addWidget( self._mpl_canvas_bottom )
        
        self._mpl_axes_position2d = self._mpl_figure_top.add_subplot(121)
        self._mpl_axes_position2d.set_xlabel('x position [pixels]')
        self._mpl_axes_position2d.set_ylabel('y position [pixels]')
        
        self._mpl_axes_direction = self._mpl_figure_top.add_subplot(122)
        
        self._mpl_axes_position1d = self._mpl_figure_bottom.add_subplot(211)
        self._mpl_axes_position1d.set_ylabel('position [pixels]')
        self._mpl_axes_speed = self._mpl_figure_bottom.add_subplot(212, sharex=self._mpl_axes_position1d)
        self._mpl_axes_speed.set_xlabel('time [s]')
        self._mpl_axes_speed.set_ylabel('speed [pixels/s]')
        self._mpl_axes_speed.grid(False)
        
        self._mpl_axes_headdirection = self._mpl_axes_speed.twinx()
        self._mpl_axes_headdirection.set_ylabel('head direction [deg]')
        self._mpl_axes_headdirection.grid(False)
        
        self._mpl_canvas_top.setFocusPolicy( QtCore.Qt.ClickFocus )
        self._mpl_canvas_bottom.setFocusPolicy( QtCore.Qt.ClickFocus )
        self._mpl_canvas_top.setFocus()
        
        #import fklab.plot
        self._mpl_zoom_axes_position1d = fklab.plot.interaction.ScrollPanZoom( self._mpl_axes_position1d)
        self._mpl_zoom_axes_speed = fklab.plot.interaction.ScrollPanZoom( self._mpl_axes_speed)
        #self._mpl_zoom_axes_position2d = fklab.plot.interaction.ScrollPanZoom( self._mpl_axes_position2d)
        
        self._mpl_plot_position1d_x = self._mpl_axes_position1d.plot( [], color='black' )[0]
        self._mpl_plot_position1d_y = self._mpl_axes_position1d.plot( [], color='gray' )[0]
        
        self._mpl_plot_speed = self._mpl_axes_speed.plot( [], color='black' )[0]
        self._mpl_plot_headdirection = self._mpl_axes_headdirection.plot( [], color='skyblue')[0]
        
        self._mpl_plot_position2d = self._mpl_axes_position2d.plot( [], '.', color='black', alpha=0.1 )[0]
        
        #self._mpl_plot_direction = self._mpl_axes_direction.plot( [], color='black' )
        
        self._mpl_canvas_top.draw()
        self._mpl_canvas_bottom.draw()
        
        fklab.plot.interaction.interactive_figure(self._mpl_figure_top)
    
    def refresh_options(self):
        
        self._enable_recalculation = False
        
        self.ui.diode1_selection.blockSignals(True)
        self.ui.diode2_selection.blockSignals(True)
        
        self.ui.diode1_selection.clear()
        self.ui.diode1_selection.addItems( ['none',] + self._tracker.get_tracked_colors() )
        
        self.ui.diode2_selection.clear()
        self.ui.diode2_selection.addItems( ['none',] + self._tracker.get_tracked_colors() )
        
        targets = self._tracker.get_target_colors()
        
        if len(targets)>0:
            idx = self.ui.diode1_selection.findText( targets[0] )
            if idx<0:
                raise ValueError('Unknown value for diode1')
        else:
            idx = 0
            
        self.ui.diode1_selection.setCurrentIndex( idx )
        
        if len(targets)>1:
            idx = self.ui.diode2_selection.findText( targets[1] )
            if idx<0:
                raise ValueError('Unknown value for diode1')
        else:
            idx = 0
        
        self.ui.diode2_selection.setCurrentIndex( idx )
        
        self.ui.diode1_selection.blockSignals(False)
        self.ui.diode2_selection.blockSignals(False)
        
        self.ui.diode_orientation.setValue( self._tracker.get_orientation() )
        
        self.ui.diode_plot.setOrientation( self._tracker.get_orientation() )
        self.ui.diode_plot.setDiodeColors( targets )
        
        self.ui.regionsTools.setChecked( self._tracker.get_regions_option('enabled') )
        
        for k in self._tracker.get_regions_option('include'):
            self.add_region( 'include', k.vertices )
        
        for k in self._tracker.get_regions_option('exclude'):
            self.add_region( 'exclude', k.vertices )
        
        options = self._tracker.get_correction_option
        
        self.ui.removeJumpsTools.setChecked( options('jumps','enabled') )
        self.ui.jumpsize.setValue( options('jumps','size') )
        self.ui.jumpduration.setValue( options('jumps','duration') )
        
        self.ui.smallGapInterpolationTools.setChecked( options('small_gaps','enabled') )
        self.ui.small_gapsize_interpolation.setValue( options('small_gaps','gap_size') )
        
        self.ui.correctDiodeDistanceTools.setChecked( options('diode_distance','enabled') )
        self.ui.diode_distance_threshold.setValue( options('diode_distance','threshold') )
        
        self.ui.missingDiodeInterpolationTools.setChecked( options('missing_diode','enabled') )
        self.ui.missingdiode_gapsize_interpolation.setValue( options('missing_diode','gap_size') )
        
        self.ui.finalGapInterpolationTools.setChecked( options('large_gaps','enabled') )
        self.ui.final_gapsize_interpolation.setValue( options('large_gaps','gap_size') )
        
        options = self._tracker.get_behavior_option
        
        self.ui.robust_position.setChecked( options('robust') )
        self.ui.vel_smooth.setValue( options('velocity_smooth') )
        self.ui.hd_smooth.setValue( options('direction_smooth') )
        
        self.refresh_diode_validity()
        
        self._enable_recalculation = True
    
    def subject_change_request(self):
        # show directory selection dialog
        d = QtWidgets.QFileDialog.getExistingDirectory( caption="Select subject folder", directory=self._subject_path )
        d = str(d)
        
        if len(d)>0 and d != self._subject_path:
            self._subject_path = d
            self.prepare_subject()
    
    def prepare_subject(self, session=None, epoch=None):
        
        self.ui.subjectSelectorButton.setText( self._subject_path )
        
        # list all sub folders
        _, sessions, _ = next(os.walk(self._subject_path))
        
        self.ui.sessionSelector.blockSignals(True)
        self.ui.sessionSelector.clear()
        
        if len(sessions)==0:
            self._current_session = None
        else:
            
            sessions = sorted( [ os.path.split(x)[1] for x in sessions ] )
            
            self.ui.sessionSelector.addItems( sessions )
            
            self._current_session = sessions[0]
            
            if not session is None:
                if session in sessions:
                    self._current_session = session
            
            idx = sessions.index(self._current_session)
            
            self.ui.sessionSelector.setCurrentIndex( idx )
        
        self.ui.sessionSelector.blockSignals(False)
        
        self.prepare_session(epoch)
    
    def session_changed(self,val):
        if self._current_session!=str(val):
            self._current_session = str(val)
            self.prepare_session()
    
    def prepare_session(self, epoch=None):
        
        self.ui.epochSelector.blockSignals(True)
        
        self.ui.epochSelector.clear()
        
        self._current_epoch = None
        self._epochs = collections.OrderedDict()
        
        if not self._current_session is None:
            info_path = os.path.join( self._subject_path, self._current_session, 'info.yaml' )
            
            if os.path.isfile( info_path ):
                with open( info_path ) as f:
                    #info = ruamel.yaml.load(f, Loader=ruamel.yaml.loader.RoundTripLoader )
                    info = yaml.load(f)
                
                if 'epochs' in info and len(info['epochs'])>0:
    
                    self._epochs = collections.OrderedDict( [ [x['id'],x['time']] for x in info['epochs']] )
                    
                    #create epochs/<epoch> directories
                    for ep in self._epochs:
                        tools.makedirs( os.path.join( self._subject_path, self._current_session, 'epochs', ep ) )
                    
                    if not epoch is None and epoch in self._epochs:
                        self._current_epoch = epoch
                        self.ui.epochSelector.addItems(list(self._epochs.keys()))
                        self.ui.epochSelector.setCurrentIndex( list(self._epochs.keys()).index(self._current_epoch) )
                    else:
                        self.ui.epochSelector.addItem('Please select epoch')
                        self.ui.epochSelector.addItems(list(self._epochs.keys()))
                        self.ui.epochSelector.setCurrentIndex(0)
        
            #extract video image
            for epoch, t in self._epochs.items():
                    video_image_file = os.path.join( self._subject_path, self._current_session, 'epochs', epoch, 'video_image.png' )
                    tools.nlx_extract_video_image( os.path.join( self._subject_path, self._current_session), 
                            np.mean(t), video_image_file)
        
        self.ui.epochSelector.blockSignals(False)
        
        self.prepare_epoch()
    
    def epoch_selection_changed(self, val):
        
        idx =  self.ui.epochSelector.findText('Please select epoch')
        if idx>-1:
            self.ui.epochSelector.removeItem(idx)
        
        self._current_epoch = str(val)
        
        self.prepare_epoch()
    
    def prepare_epoch(self):
        
        if self._current_epoch is None:
            self.ui.toolContainer.setEnabled(False)
            self.ui.diode1_validity.setText( '' )
            self.ui.diode2_validity.setText( '' )
            self.clear_plots()
            return
        
        source_path = os.path.join( self._subject_path, self._current_session, 'VT1.nvt' )
        epoch = self._epochs[self._current_epoch]
        self._tracker.set_source( source_path, epoch )
        
        #check if session/epochs/<epoch>/position.yaml exists
        options_file = os.path.join( self._subject_path, self._current_session, 'epochs', self._current_epoch, 'position.yaml' )
        if os.path.isfile( options_file ):
            self._tracker.load_options( options_file )
        else:
            self._tracker.save_options( options_file )
        
        self.ui.toolContainer.setEnabled(True)
        
        self.refresh_options()
        
        self.prepare_plots()
        self.refresh_plot()
    
    def refresh_diode_validity(self):
        
        self.ui.diode1_validity.setText( '' )
        self.ui.diode2_validity.setText( '' )
        
        if self._tracker.color_validity is None:
            return
        
        colors = self._tracker.get_target_colors()
        
        if len(colors)>0:
            val = self._tracker.color_validity[colors[0]]
            val_string = '{p:.1f}%'.format(p=val)
            color_string = 'red' if val<50 else 'black'
            self.ui.diode1_validity.setStyleSheet('QLabel {{color: {c};}}'.format(c=color_string))
        else:
            val_string = ''
        
        self.ui.diode1_validity.setText( val_string )
        
        if len(colors)>1:
            val = self._tracker.color_validity[colors[1]]
            val_string = '{p:.1f}%'.format(p=val)
            color_string = 'red' if val<50 else 'black'
            self.ui.diode2_validity.setStyleSheet('QLabel {{color: {c};}}'.format(c=color_string))
        else:
            val_string = ''
        
        self.ui.diode2_validity.setText( val_string )
    
    def prepare_plots(self):
        
        x,y = self._tracker.info['resolution']
        
        self._mpl_axes_position1d.set_ylim([0,y])
        self._mpl_axes_position1d.set_xlim(self._epochs[self._current_epoch])
        
        self._mpl_axes_speed.set_ylim([0,200])
        self._mpl_axes_speed.set_xlim(self._epochs[self._current_epoch])
        
        self._mpl_axes_headdirection.set_ylim([0,360])
        
        self._mpl_axes_position2d.set_xlim([0,x])
        self._mpl_axes_position2d.set_ylim([0,y])
        self._mpl_axes_position2d.invert_yaxis()
        
        self._mpl_axes_direction.set_xlim([-180, 180])
        
        self.clear_plots()
    
    def clear_plots(self):
        
        #plot x,y vs time
        self._mpl_plot_position1d_x.set_xdata( [] )
        self._mpl_plot_position1d_y.set_xdata( [] )
        self._mpl_plot_position1d_x.set_ydata( [] )
        self._mpl_plot_position1d_y.set_ydata( [] )

        #plot speed, head direction vs time
        self._mpl_plot_speed.set_xdata( [] )
        self._mpl_plot_speed.set_ydata( [] )
        
        self._mpl_plot_headdirection.set_xdata( [] )
        self._mpl_plot_headdirection.set_ydata( [] )

        self._mpl_canvas_bottom.draw()
        
        #plot y vs x
        self._mpl_plot_position2d.set_xdata( [] )
        self._mpl_plot_position2d.set_ydata( [] )
        
        # motion direction - head direction
        self._mpl_axes_direction.clear()
        
        self._mpl_canvas_top.draw()

    def refresh_plot(self):
        
        if not self._enable_recalculation:
            return
        
        behav = self._tracker.behavior
        
        valid = 100-np.sum(np.isnan(np.sum(behav['position'],axis=1)))*100./behav['position'].shape[0]
        self.ui.valid_position.setText( 'valid: {p:.2f}%'.format(p=valid) )
        
        #plot x,y vs time
        self._mpl_plot_position1d_x.set_xdata( behav['time'] )
        self._mpl_plot_position1d_y.set_xdata( behav['time'] )
        self._mpl_plot_position1d_x.set_ydata( behav['position'][:,0] )
        self._mpl_plot_position1d_y.set_ydata( behav['position'][:,1] )

        #plot speed, head direction vs time
        speed = np.abs(behav['velocity'])
        self._mpl_plot_speed.set_xdata( behav['time'] )
        self._mpl_plot_speed.set_ydata( speed )
        
        self._mpl_plot_headdirection.set_xdata( behav['time'] )
        self._mpl_plot_headdirection.set_ydata( 180.*behav['head_direction']/np.pi )
        
        self._mpl_canvas_bottom.draw()
        
        #plot y vs x
        self._mpl_plot_position2d.set_xdata( behav['position'][:,0] )
        self._mpl_plot_position2d.set_ydata( behav['position'][:,1] )
        
        #plot move direction - head direction distribution
        speed_threshold = np.nanpercentile( speed, 70 )
        direction = fklab.statistics.circular.diff( np.angle( behav['velocity'] ), behav['head_direction'], directed=True )
        direction = direction[np.logical_and( np.logical_not(np.isnan(direction)), speed>speed_threshold ) ]
        
        self._mpl_axes_direction.clear()
        if len(direction)>0:
            seaborn.distplot( 180.*direction/np.pi, bins=100, ax=self._mpl_axes_direction, kde=False )
        
        #(plot diode distance distribution?)
        self._mpl_canvas_top.draw()
            
    def on_key_press(self, event):
        print('Key press!')

    def enable_correct_diode_distance(self, val):
        self._tracker.set_correction_options('diode_distance', enabled=bool(val))
        self.refresh_plot()
    
    def enable_finalgapinterpolation(self, val):
        self._tracker.set_correction_options('large_gaps', enabled=bool(val))
        self.refresh_plot()
    
    def enable_missingdiode_interpolation(self, val):
        self._tracker.set_correction_options('missing_diode', enabled=bool(val))
        self.refresh_plot()
    
    def enable_regions(self, val):
        self._tracker.set_regions_options(enabled=bool(val))
        if len(self._regions)>0:
            self.refresh_plot()
    
    def enable_removejumps(self, val):
        self._tracker.set_correction_options('jumps', enabled=bool(val))
        self.refresh_plot()
    
    def enable_smallgapinterpolation(self, val):
        self._tracker.set_correction_options('small_gaps', enabled=bool(val))
        self.refresh_plot()
    
    def robust_position_changed(self, val):
        self._tracker.set_behavior_options(robust=bool(val))
        self.refresh_plot()
    
    def remove_region(self):
        row = int(self.ui.regions.currentRow())
        
        if row>-1:
            self._regions[row][2].remove()
            del self._regions[row]
            self.ui.regions.takeItem(row)
            self._mpl_canvas_top.draw()
            
            self.refresh_regions()
    
    def add_region(self, kind, vertices ):
        
        if kind=='include':
            col, highlight = 'green', 'limegreen'
        else:
            col, highlight = 'red', 'salmon'
        
        artist = fklab.plot.interaction.iPolygon( vertices,
            color=col, facecolor='none', zorder=100,
            highlight_color=highlight, linewidth=2)
        artist.interactive=False
        
        self._regions.append( (vertices, kind, artist) )
        self.ui.regions.addItem( kind )
        
        self._mpl_axes_position2d.add_patch( artist )
        self._mpl_canvas_top.draw()
    
    def add_include_region(self):
        pdata = fklab.plot.interaction.create_polygon( self._mpl_axes_position2d )
        if not pdata is None:
            self.add_region( 'include', pdata )
            self.refresh_regions()
            
    def add_exclude_region(self):
        pdata = fklab.plot.interaction.create_polygon( self._mpl_axes_position2d )
        if not pdata is None:
            self.add_region( 'exclude', pdata )
            self.refresh_regions()
    
    def refresh_regions(self):
        
        include = []
        exclude = []
        
        for k in self._regions:
            if k[1]=='include':
                include.append( fklab.geometry.shapes.polygon( k[0] ) )
            else:
                exclude.append( fklab.geometry.shapes.polygon( k[0] ) )
        
        self._tracker.set_regions_options( include=include, exclude=exclude )
        self.refresh_plot()
    
    def region_selection_changed(self,row):
        for k,v in enumerate( self._regions ):
            if k==row:
                v[2].set_highlight(True)
            else:
                v[2].set_highlight(False)
        self._mpl_canvas_top.draw()
            
    def diode_orientation_changed(self,val):
        val = float(val)
        rad = 2*np.pi*val / 360
        self._tracker.set_orientation(rad)
        self.ui.diode_plot.setOrientation(rad)
        self.refresh_plot()
    
    def diode_selection_changed(self,val):
        cols = []
        
        val = str(self.ui.diode1_selection.currentText())
        if val!='none' and val!='':
            cols.append( val )
        
        val = str(self.ui.diode2_selection.currentText())
        if val!='none' and val!='':
            cols.append( val )
            
        self._tracker.set_target_colors( cols )
        self.ui.diode_plot.setDiodeColors( cols )
        
        self.refresh_diode_validity()
        self.refresh_plot()
    
    def jumpsize_changed(self,val):
        self._tracker.set_correction_options('jumps',size=int(val) )
        self.refresh_plot()
    
    def jumpduration_changed(self,val):
        self._tracker.set_correction_options('jumps',duration=float(val) )
        self.refresh_plot()
    
    def vel_smooth_changed(self,val):
        self._tracker.set_behavior_options(velocity_smooth=float(val) )
        self.refresh_plot()
    
    def hd_smooth_changed(self,val):
        self._tracker.set_behavior_options(direction_smooth=float(val) )
        self.refresh_plot()
    
    def small_gapsize_interpolation_changed(self,val):
        self._tracker.set_correction_options('small_gaps',gap_size=float(val) )
        self.refresh_plot()
    
    def final_gapsize_interpolation_changed(self,val):
        self._tracker.set_correction_options('large_gaps',gap_size=float(val) )
        self.refresh_plot()
    
    def missingdiode_gapsize_interpolation_changed(self,val):
        self._tracker.set_correction_options('missing_diode',gap_size=float(val) )
        self.refresh_plot()
    
    def diode_distance_threshold_changed(self,val):
        self._tracker.set_correction_options('diode_distance',threshold=float(val) )
        self.refresh_plot()
    
    def close_button_clicked(self):
        pass
    
    def reset_button_clicked(self):
        pass
    
    def save_button_clicked(self):

        import h5py
        
        options_file = os.path.join( self._subject_path, self._current_session, 'epochs', self._current_epoch, 'position.yaml' )
        self._tracker.save_options( options_file )
        
        behav = self._tracker.behavior
        targets = self._tracker.corrected_target_coordinates
        
        filename = os.path.join( self._subject_path, self._current_session, 'epochs', self._current_epoch, 'position.hdf5' )
        fid = h5py.File(filename,'w')
        fid.create_dataset("time", data=behav['time'])
        grp = fid.create_group("diodes")
        
        if len(targets)>0:
            grp.create_dataset("diode1", data=targets[0])
        
        if len(targets)>1:
            grp.create_dataset("diode2", data=targets[1])
            
        fid.create_dataset("position", data=behav['position'])
        fid.create_dataset("velocity", data=behav['velocity'])
        fid.create_dataset("head_direction", data=behav['head_direction'])
        
        fid.close()
    

def main(app):
    
    parser = argparse.ArgumentParser(description='Localize your rodent!')
    parser.add_argument('subject', nargs='?', default='.', help='path to subject data folder')
    parser.add_argument('session', nargs='?', default='', help='name of the session')
    parser.add_argument('epoch', nargs='?', default='', help='name of an epoch')
    
    args = parser.parse_args()
        
    window = LocalizeWindow(args.subject, args.session, args.epoch)
    window.show()
    return app.exec_()
    

app = QApplication(sys.argv)
sys.exit(main(app))

