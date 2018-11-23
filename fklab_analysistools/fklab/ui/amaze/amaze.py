import sys
import os
import datetime
import argparse
import random
from collections import OrderedDict, namedtuple

import numpy as np
import skimage
import skimage.io
import h5py

from PyQt4 import QtGui
from PyQt4.QtGui import QApplication, QMainWindow, QTableWidgetItem, QPushButton, QLabel, QDialog, QIcon
from PyQt4 import QtCore

from amaze_main import Ui_MainWindow
from amaze_about import Ui_Dialog

from environment_properties import Ui_EnvironmentDialog as env_dlg
from shape_properties import Ui_ShapeDialog as shape_dlg

import fklab.io.data
import fklab.geometry.shapes as shapes

# start up: check arguments to see if it is a directory with info.yaml
# if so, load info.yaml, extract epochs, load epoch data, update tree, select epoch/env, update position plot, update shapes/shapes list

# user adds new shape: addShape(kind, ...) -> shape_added signal -> add to shapes list, add to data model
# 

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class ShapeClass:
    def __init__(self, name, tags=[], comments='', shape=None):
        self.name = name
        self.tags = tags
        self.comments = comments
        self.shape = shape
        self.ID=-1
    
    def todict(self):
        d = OrderedDict()
        d['tags'] = self.tags
        d['comments'] = self.comments
        d['shape'] = self.shape
        
        return self.name, d
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, val):
        if not isinstance(val, str) or val=='':
            raise ValueError('Name should be a valid string.')
        self._name = val
    
    @property
    def comments(self):
        return self._comments
    
    @comments.setter
    def comments(self,val):
        self._comments = str(val)
    
    @property
    def tags(self):
        return self._tags
    
    @tags.setter
    def tags(self,val):
        if isinstance( val, str ):
            val = [ x.strip() for x in val.split() ]
        elif not isinstance(val, (tuple, list)):
            raise ValueError('Invalid list of tags')
        else:
            val = [ str(x) for x in val ]
        
        # get unique tags without losing order
        val = [ x for k,x in enumerate(val) if val.index(x)==k ]
        
        self._tags = val 
    
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self,val):
        if not val is None and not isinstance(val, fklab.geometry.shapes.shape ):
            raise ValueError('Invalid shape.')
        
        self._shape = val
    
    @property
    def ID(self):
        return self._id
    
    @ID.setter
    def ID(self,val):
        self._id = int(val)
    
class EnvironmentClass:
    def __init__(self, name, comments='', shapes=[]):
        self.name = name
        self.comments = comments
        self.shapes = shapes
    
    def todict(self):
        d = OrderedDict()
        d['comments'] = self.comments
        d['shapes'] = OrderedDict( [ x.todict() for x in self.shapes ] )
        
        return self.name, d
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,val):
        if not isinstance(val, str) or val=='':
            raise ValueError('Name should be a valid string.')
            
        self._name = val
    
    @property
    def comments(self):
        return self._comments
    
    @comments.setter
    def comments(self,val):
        self._comments = str(val)
    
    @property
    def shapes(self):
        return self._shapes
    
    @shapes.setter
    def shapes(self,val):
        if not isinstance(val,list) or not all( [isinstance(x,ShapeClass) for x in val] ):
            raise ValueError('Invalid list of shapes.')
        
        self._shapes = val
    

class AboutDialog(QDialog):
    def __init__(self,**kwargs):
        QDialog.__init__(self,**kwargs)
        
        #1. set up the user interface from Designer
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

class EnvironmentPropertiesDialog(QDialog):
    def __init__(self, name, comments, invalid_names=[], **kwargs):
        QDialog.__init__(self,**kwargs)
        
        self._invalid_names = invalid_names
        
        #1. set up user interface
        self.ui = env_dlg()
        self.ui.setupUi(self)
        
        #2. populate widgets
        self.ui.Name.setText(name)
        self.ui.Comments.setPlainText(comments)
        
        self.enable_ok( self.is_valid_name(name) )
    
    def is_valid_name(self, val):
        import re
        return not re.match("[_A-Za-z][_\-a-zA-Z0-9]*$", val) is None
    
    def enable_ok(self,val):
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).setEnabled(val)
    
    def name_changed(self, val):
        val = str(val)
        valid = self.is_valid_name( val ) and not val in self._invalid_names
        self.enable_ok(valid)
  
class ShapePropertiesDialog(QDialog):
    def __init__(self, name, tags, comments, invalid_names=[], **kwargs):
        QDialog.__init__(self,**kwargs)
        
        self._invalid_names = invalid_names
        
        #1. set up user interface
        self.ui = shape_dlg()
        self.ui.setupUi(self)
        
        #2. populate widgets
        self.ui.Name.setText(name)
        self.set_tags(tags)
        self.ui.Comments.setPlainText(comments)
        
        self.enable_ok( self.is_valid_name(name) )
    
    def update_tag_buttons(self, tags):
        ntags = 8
        for k in xrange(ntags):
            tag_button = getattr(self.ui, 'tag' + str(k+1))
            button_text = str(tag_button.text())
            if button_text!='':
                tag_button.setChecked( button_text in tags )
        
    def is_valid_name(self, val):
        import re
        return not re.match("[_A-Za-z][_\-a-zA-Z0-9]*$", val) is None
    
    def enable_ok(self,val):
        self.ui.buttonBox.button(QtGui.QDialogButtonBox.Ok).setEnabled(val)
    
    def name_changed(self, val):
        val = str(val)
        valid = self.is_valid_name( val ) and not val in self._invalid_names
        self.enable_ok(valid)
    
    def get_tags(self):
        tags = str(self.ui.Tags.toPlainText()).strip().replace('\n','').replace('\r','').split(',')
        tags = [ x.strip() for x in tags ]
        tags = [ x for x in tags if x!='' ]
        return tags
    
    def set_tags(self,tags):
        tags_text = ', '.join(tags)
        self.ui.Tags.setPlainText(tags_text)
    
    def tags_changed(self):
        tags = self.get_tags()
        cursor = self.ui.Tags.textCursor()
        self.update_tag_buttons(tags)
        self.ui.Tags.setTextCursor(cursor)
    
    def set_tag(self, tag, add):
        tags = self.get_tags()
        
        if add:
            if not tag in tags:
                tags.append(tag)
        else:
            tags = [ x for x in tags if x!=tag ]
        
        self.set_tags( tags )
    
    def tag1_clicked(self,val):
        self.set_tag( str(self.ui.tag1.text()), val )
    
    def tag2_clicked(self,val):
        self.set_tag( str(self.ui.tag2.text()), val )
        
    def tag3_clicked(self,val):
        self.set_tag( str(self.ui.tag3.text()), val )
    
    def tag4_clicked(self,val):
        self.set_tag( str(self.ui.tag4.text()), val )
    
    def tag5_clicked(self,val):
        self.set_tag( str(self.ui.tag5.text()), val )
        
    def tag6_clicked(self,val):
        self.set_tag( str(self.ui.tag6.text()), val )
    
    def tag7_clicked(self,val):
        self.set_tag( str(self.ui.tag7.text()), val )
    
    def tag8_clicked(self,val):
        self.set_tag( str(self.ui.tag8.text()), val )

    
class EnvironmentModel(QtCore.QObject):
    
    haschanged = QtCore.pyqtSignal(bool)
    
    def __init__(self, *args, **kwargs):
        QtCore.QObject.__init__(self,*args,**kwargs)
        self.clear()
    
    def clear(self):
        self._environments = []
        self.changed = False
        self._current_environment = None
    
    @property
    def changed(self):
        return self._changed
    
    @changed.setter
    def changed(self, value):
        self._changed = bool(value)
        self.haschanged.emit(self._changed)
    
    def import_file(self, filename):
        with open(filename,'r') as fid:
            env = fklab.utilities.yaml.load(fid)
        if not isinstance(env, dict):
            raise IOError('Invalid environment definition file.')
        return self.import_dict( env )
    
    def import_dict(self, data):
        
        new_names = []
        
        for name, d in data.iteritems():
            
            env = EnvironmentClass(name=name, comments=d.get('comments',''), shapes=[])
            
            if not 'shapes' in d or not isinstance(d['shapes'],dict):
                raise IOError('Invalid shape definition: no shapes')
            
            for shape_name, shape in d['shapes'].iteritems():
                
                if not 'shape' in shape or not isinstance(shape['shape'], shapes.shape):
                    raise IOError('Invalid shape definition: unrecognized shape' )
                
                env.shapes.append( ShapeClass(name=shape_name, tags=shape.get('tags',[]), comments=shape.get('comments',''), shape=shape['shape']) )
            
            new_names.append( self.add_environment( env ) )
        
        return new_names
    
    def add_environment(self, env):
        if not isinstance(env, EnvironmentClass):
            raise TypeError()
            
        if env.name in self.environment_names():
            env.name = env.name + str(random.randint(1000,9999))
            
        self._environments.append(env)
        self.changed = True
        
        if self._current_environment is None:
            self._current_environment = env.name
        
        return env.name
    
    def new_environment(self, name, comments=''):
        if name in self.environment_names():
            raise ValueError('Environment with same name already exists.')

        self._environments.append( EnvironmentClass(name=name, comments=comments, shapes=[]) )
        self.changed = True
        
        if self._current_environment is None:
            self._current_environment = name
        
    def export(self, destination):
        
        if len(self._environments)==0:
            return
        
        env = OrderedDict( [ x.todict() for x in self._environments ] )
        
        with open(destination, 'w') as f:
            fklab.utilities.yaml.dump( env, stream=f )
    
    def import_(self, source):
        if isinstance(source,str):
            n = self.import_file(source)
        elif isinstance(source, dict):
            n = self.import_dict(source)
        self.changed=True
        return n
    
    def open(self, source):
        self.clear()
        n = self.import_(source)
        self.changed=False
        return n
    
    def save(self, filename):
        self.export(filename)
        self.changed=False
    
    @property
    def nenvironments(self):
        return len(self._environments)
    
    @property
    def currentEnvironment(self):
        return self._current_environment
    
    @property
    def currentEnvironmentIndex(self):
        if self.currentEnvironment is None:
            return None
        
        return self.environment_names().index( self.currentEnvironment )
    
    @currentEnvironment.setter
    def currentEnvironment(self, val):
        if not val in self.environment_names():
            raise ValueError('Invalid environment.')
        
        self._current_environment = val
    
    def environment_names(self):
        return [ x.name for x in self._environments]
    
    def remove_environment(self, name):
        
        idx = self.environment_names().index( name )
        
        del self._environments[idx]
        self.changed = True
        
        if self._current_environment==name:
            n = self.nenvironments
            idx = idx if idx<n else n-1
            self._current_environment = None if idx<0 else self._environments[idx].name
    
    def rename_environment(self, oldname, newname):
        
        for k in self._environments:
            if k.name==oldname:
                k.name = newname
                self.changed = True
        
        if self._current_environment == oldname:
            self._current_environment = newname
    
    def shapes(self, environment=None):
        env = self.environment(environment)
        return env.shapes
    
    def shape_names(self, environment=None):
        return [ x.name for x in self.shapes(environment)]
    
    def create_shape(self, name, tags=[], comments=[], shape=None, environment=None):
        if name in self.shape_names(environment):
            raise ValueError('Shape with this name alraedy exists.')
        
        self.shapes(environment).append( ShapeClass(name=name, tags=tags, comments=comments, shape=shape) )
        self.changed = True
    
    def remove_shape(self, name, environment=None):
        
        idx = self.shape_names(environment).index( name )
        
        del self.shapes(environment)[idx]
        self.changed = True
    
    def rename_shape(self, oldname, newname, environment=None):
        
        for k in self.shapes(environment):
            if k.name==oldname:
                k.name = newname
                self.changed = True
       
    def environment(self, environment=None):
        
        if environment is None:
            if self._current_environment is None:
                raise ValueError('No such environment')
            environment = self._current_environment
        
        for x in self._environments:
            if x.name == environment:
                return x
        
        raise ValueError('No such environment.')
    
    def shape(self, shape, environment=None):
        
        env = self.environment(environment)
            
        for x in env.shapes:
            if x.name == shape:
                return x
        
        raise ValueError('No such shape.')
    
    def shapeByID(self, ID, environment=None):
        
        env = self.environment(environment)
        
        for x in env.shapes:
            if x.ID == ID:
                return x
        
        raise ValueError('No such shape.')


class AmazeWindow(QMainWindow):
    def __init__(self, env=None, pos=None, img=None, target=None):
        QMainWindow.__init__(self)
        
        #1. set up the user interface from Designer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #2. set up the shape canvas
        self.ui.Canvas.scene().setSceneRect( QtCore.QRectF(0, 0, 720, 576) )
        self.ui.Canvas.setMouseTracking(True)
        
        #3. add coordinate label to statusbar
        self.ui.coordlabel = QLabel()
        self.ui.statusbar.addPermanentWidget(self.ui.coordlabel,0)
        
        #4. connect signals from scene to methods
        self.ui.Canvas.coordinates_changed.connect( self.showCoordinates )
        self.ui.Canvas.scene().instructions.connect( self.showMessage )
        self.ui.Canvas.scene().setInstructions()
        self.ui.Canvas.scene().selection_changed.connect( self.shape_selected )
        self.ui.Canvas.scene().mode_changed.connect( self.canvas_mode_changed )
        self.ui.Canvas.scene().shape_edited.connect( self.shape_edited )
        
        
        self._model = EnvironmentModel(parent=self)
        self._model.haschanged.connect( self._model_changed )
        
        self.action_clear()
        
        if env is not None:
            # check if env is existing file
            if not isinstance(env,str) or os.path.isfile(env):
                try:
                    n = self._model.open( env )
                    for k in n:
                        self.ui.EnvironmentList.addItem( k )
                    
                    if n>0:
                        self.ui.EnvironmentList.setCurrentRow(0)
                except Exception as e:
                    raise IOError("Could not load environment: " + e.message )
                    
            if isinstance(env, str):
                self.save_target = env
        
        if target is not None:
            target = os.path.abspath(str(target))
            self.save_target = target
        
        if pos is not None:
            if isinstance(pos,str):
                s = pos
                pos = self.load_position( pos )
                self.ui.position_data_source.setText(s)
            self.plot_position( pos )
        
        if img is not None:
            #if isinstance(img,str):
            #    img = self.load_image( img )
            self.plot_image( img )
            self.ui.video_image_source.setText( img )
    
    def closeEvent(self, event):
        self.return_value = OrderedDict( [ x.todict() for x in self._model._environments ] )
        
        if not self.action_clear():
            event.ignore()
            return
        
        event.accept()

    
    @QtCore.pyqtSlot(bool)
    def _model_changed(self, value):
        self.ui.actionSave.setEnabled( value )
    
    @property
    def save_target(self):
        return self._save_target
    
    @save_target.setter
    def save_target(self, value):
        if value is None:
            self._save_target = None
            self.ui.env_description_file.setText( "" )
        else:
            self._save_target = str(value)
            self.ui.env_description_file.setText( self._save_target )
    
    def action_save(self):
        # environment file specified?
        if self.save_target is None:
            self.action_save_as()
        else:
            try:
                self._model.save( self.save_target )
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error saving environment definition: " + e.message )
                return
    
    def action_save_as(self):        
        d = str(QtGui.QFileDialog.getSaveFileName( parent=self, caption="Save as", filter="Environment (*.yaml)" ))
        if len(d)>0:
            try:
                self._model.save( d )
                self.save_target = d
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error saving environment definition: " + e.message )
                return
    
    def action_import(self):
        d = str(QtGui.QFileDialog.getOpenFileName( parent=self, caption="Import", filter="Environment (*.yaml)" ))
        if len(d)>0:
            try:
                new_env_names = self._model.import_( d )
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error importing environment definition: " + e.message )
                return
            
            for k in new_env_names:
                self.ui.EnvironmentList.addItem( k )
            
            if self._model.nenvironments==len(new_env_names):
                self.ui.EnvironmentList.setCurrentRow(0)
    
    def action_export(self):
        d = str(QtGui.QFileDialog.getSaveFileName( parent=self, caption="Export", filter="Environment (*.yaml)" ))
        if len(d)>0:
            try:
                self._model.export( d )
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error exporting environment definition: " + e.message )
                return
                
    
    def action_open(self):
        
        d = str(QtGui.QFileDialog.getOpenFileName( parent=self, caption="Open", filter="Environment (*.yaml)" ))
        
        if len(d)>0:
            
            if not self.action_clear():
                return
            
            try:
                new_env_names = self._model.open( d )
                self.save_target = d
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error opening environment definition: " + e.message )
                return
            
            for k in new_env_names:
                self.ui.EnvironmentList.addItem( k )
            
            if self._model.nenvironments==len(new_env_names):
                self.ui.EnvironmentList.setCurrentRow(0)
            
    
    def action_clear(self):
        if self._model.changed:
            reply = QtGui.QMessageBox.question(self, "Quit", "Would you like to save the changes for the current environment definition?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel, QtGui.QMessageBox.Cancel )
            
            if reply == QtGui.QMessageBox.Cancel:
                return False
            
            if reply == QtGui.QMessageBox.Yes:
                self.action_save()
        
        self._model.clear()
        self.save_target = None
        
        self.plot_position()
        self.plot_image()
        
        self.ui.EnvironmentList.setCurrentItem(None)
        self.ui.EnvironmentList.clear()
        
        self.ui.position_data_source.setText("")
        self.ui.video_image_source.setText("")
        
        return True
    
    def action_clear_position(self):
        self.plot_position()
        self.ui.position_data_source.setText("")
    
    def action_clear_image(self):
        self.plot_image()
        self.ui.video_image_source.setText("")
    
    def action_set_position(self):
        d = str(QtGui.QFileDialog.getOpenFileName( parent=self, caption="Import position data", filter="Processed position (*.hdf5)" ))
        if len(d)>0:
            try:
                pos = self.load_position( d )
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error importing position data: " + e.message )
                return
            
            self.plot_position( pos )
            self.ui.position_data_source.setText(d)
            
    def action_set_image(self):
        d = str(QtGui.QFileDialog.getOpenFileName( parent=self, caption="Import image", filter="Image (*.png *.jpg *.tif)" ))
        if len(d)>0:
            try:
                self.plot_image( d )
            except Exception as e:
                QtGui.QMessageBox.warning(self, "Error", "Error importing image: " + e.message )
                return
            
            self.ui.video_image_source.setText(d)
    
    def load_position(self, x):
            
        if isinstance(x, str):
            # load image from file
            fid = h5py.File(x,'r')
            data = fid["position"][:]
        else:
            data = np.array(x)
        
        return data
    
    def plot_position(self, xy=None):
        self.ui.Canvas.scene().setTrackerData( xy )
            
    def plot_image(self, img=None):
        self.ui.Canvas.scene().setTrackerImage( img )
    
    
    def show_about_dialog(self):
        about = AboutDialog(parent=self)
        about.exec_()
    
      
    def environment_selection_changed(self):
        
        #1. get currently selected item
        selection = self.ui.EnvironmentList.currentItem()
         
        #2. clear shape table
        while self.ui.ShapeTable.rowCount()>0:
            self.ui.ShapeTable.removeRow(0)
        
        #3. clear shapes
        self.ui.Canvas.scene().removeAllShapes()
        
        if selection is None:
            self.ui.ShapeFrame.setEnabled(False)
            return
        
        self._model.currentEnvironment = str( selection.text() )
        
        #4. add shapes to table
        for shape in self._model.shapes():
            
            row = self.ui.ShapeTable.rowCount()
            self.ui.ShapeTable.insertRow( row )
            
            #populate cells in row
            self.ui.ShapeTable.setItem( row, 0, QTableWidgetItem(shape.name))
            self.ui.ShapeTable.setItem( row, 1, QTableWidgetItem(', '.join(shape.tags)))
            
            shape.ID = self.ui.Canvas.scene().addShape( shape.shape )
        
        self.ui.ShapeFrame.setEnabled(True)
    
    def environment_edit_request(self,item):
        
        name = str(item.text())
        env = self._model.environment(environment=name)
        
        new_name, comments = self.edit_environment_data(name, env.comments)
        
        env.comments = comments
        
        if new_name != name:
            #rename
            self._model.rename_environment( name, new_name )
            item.setText( new_name )
            
    def edit_environment_data(self, name='', comments=''):
        
        invalid_names = [k for k in self._model.environment_names() if k!=name]
        
        dialog = EnvironmentPropertiesDialog(
            name, comments, 
            invalid_names,
            parent=self
            )
            
        if dialog.exec_():
            name = str(dialog.ui.Name.text())
            comments = str(dialog.ui.Comments.toPlainText())
        
        return name, comments
        
    def add_environment(self):
        
        name, comments = self.edit_environment_data()
        
        if name!='':
            self._model.new_environment( name, comments )
            self.ui.EnvironmentList.addItem( name )
            
            if self._model.nenvironments==1:
                self.ui.EnvironmentList.setCurrentRow(0)

    def remove_environment(self):
        
        #1. get selection
        selection = self.ui.EnvironmentList.currentRow()
        env_name = str( self.ui.EnvironmentList.currentItem().text() )
        
        self.ui.EnvironmentList.takeItem(selection)
        self._model.remove_environment( env_name )
    
   
    def shape_edit_request(self, index):
        
        row = index.row()
        name = str( self.ui.ShapeTable.item(row,0).text() )
        
        shape = self._model.shape(name)
        
        new_name, tags, comments = self.edit_shape_data(name=name,tags=shape.tags,
            comments=shape.comments)
        
        shape.comments = comments
        shape.tags = tags
        
        if new_name!=name:
            #rename
            self._model.rename_shape(name, new_name)
            self.ui.ShapeTable.item(row,0).setText(new_name)
        
        self.ui.ShapeTable.setItem( row, 1, QTableWidgetItem(', '.join(tags)))
    
    def edit_shape_data(self, name='', tags=[], comments=''):
        
        invalid_names = [k for k in self._model.shape_names() if k!=name]
        
        dialog = ShapePropertiesDialog(
            name=name, tags=tags, comments=comments, parent=self, invalid_names=invalid_names
            )
        
        if dialog.exec_():
            name = str(dialog.ui.Name.text())
            tags = str(dialog.ui.Tags.toPlainText()).replace('\n','').replace('\r','').split(',')
            comments = str(dialog.ui.Comments.toPlainText())
        
        return name, tags, comments
    
    def create_new_shape(self, kind, tags=[], **kwargs):
        self.ui.Canvas.setFocus(0)
        shape = self.ui.Canvas.scene().createShape( kind, **kwargs )
        
        if not shape is None:
            name, tags, comments = self.edit_shape_data(name='',tags=tags,comments='')
            
            if name!='':
                self._model.create_shape(name=name, tags=tags, comments=comments, shape=shape)
                
                row = self.ui.ShapeTable.rowCount()
                self.ui.ShapeTable.insertRow( row )
                
                #populate cells in row
                self.ui.ShapeTable.setItem( row, 0, QTableWidgetItem(name))
                self.ui.ShapeTable.setItem( row, 1, QTableWidgetItem(', '.join(tags)))
                
                self._model.shape(name).ID = self.ui.Canvas.scene().addShape( shape )
    
    def remove_shapes(self):
        
        #1. get selected shapes (reverse ordered)
        selection = self.ui.ShapeTable.selectionModel().selectedRows()
        rows = [ k.row() for k in selection]
        rows.sort(reverse=True)
        
        table = self.ui.ShapeTable
        QtCore.QObject.disconnect(table, QtCore.SIGNAL(_fromUtf8("itemSelectionChanged()")), self.shape_selection_changed)
        
        #2. instruct scene to remove shapes
        for k in rows:
            shape_name = str(self.ui.ShapeTable.item(k,0).text())
            shape = self._model.shape( shape_name )
            self.ui.Canvas.scene().removeShapeByID( shape.ID )
            self._model.remove_shape( shape_name )
            self.ui.ShapeTable.removeRow(k)
            
        QtCore.QObject.connect(table, QtCore.SIGNAL(_fromUtf8("itemSelectionChanged()")), self.shape_selection_changed)


    def canvas_mode_changed(self, mode):
        
        if mode=='default':
            self.ui.ToolboxFrame.setEnabled(True)
        else:
            self.ui.ToolboxFrame.setEnabled(False)
    
    def shape_edited(self, ID):
        
        shape = self._model.shapeByID( ID )
        shape.shape = self.ui.Canvas.scene().getShape(ID).toshape()
        
        self._model.changed = True
    
    def create_linear_track(self):
        self.create_new_shape( 'polyline', closed=False, spline=False, tags=['track',] )
    
    def create_circular_track1(self):
        self.create_new_shape( 'ellipse', method='3-points', tags=['track',] )
    
    def create_circular_track2(self):
        self.create_new_shape( 'ellipse', method='2-points', tags=['track',] )
    
    def create_circular_track3(self):
        self.create_new_shape( 'ellipse', method='center square', tags=['track',] )
            
    def create_rectangular_track1(self):
        self.create_new_shape( 'rectangle', method='3-points', tags=['track',] )
    
    def create_rectangular_track2(self):
        self.create_new_shape( 'rectangle', method='2-points', tags=['track',] )
    
    def create_rectangular_track3(self):
        self.create_new_shape( 'rectangle', method='center square', tags=['track',] )
        
    def create_graph_track(self):
        self.create_new_shape( 'graph', tags=['track',] )
    
    def create_circular_field1(self):
        self.create_new_shape( 'ellipse', method='3-points', tags=['field',] )
    
    def create_circular_field2(self):
        self.create_new_shape( 'ellipse', method='2-points', tags=['field',])
    
    def create_circular_field3(self):
         self.create_new_shape( 'ellipse', method='center square', tags=['field',] )
    
    def create_rectangular_field1(self):
        self.create_new_shape( 'rectangle', method='3-points', tags=['field',] )
    
    def create_rectangular_field2(self):
        self.create_new_shape( 'rectangle', method='2-points', tags=['field',])
    
    def create_rectangular_field3(self):
        self.create_new_shape( 'rectangle', method='center square', tags=['field',] )
    
    def create_polygon_field(self):
        self.create_new_shape( 'polygon', closed=True, spline=False, tags=['field',] )
    
    def create_circular_object1(self):
        self.create_new_shape( 'ellipse', method='3-points', tags=['object',] )
    
    def create_circular_object2(self):
        self.create_new_shape( 'ellipse', method='2-points', tags=['object',] )
    
    def create_circular_object3(self):
        self.create_new_shape( 'ellipse', method='center square', tags=['object',] )
    
    def create_rectangular_object1(self):
        self.create_new_shape( 'rectangle', method='3-points', tags=['object',] )
    
    def create_rectangular_object2(self):
        self.create_new_shape( 'rectangle', method='2-points', tags=['object',] )
    
    def create_rectangular_object3(self):
        self.create_new_shape( 'rectangle', method='center square', tags=['object',] )
    
    def create_polygon_object(self):
        self.create_new_shape( 'polygon', closed=True, spline=False, tags=['object',] )
    
    
    def showCoordinates(self, x, y ):
        if x is None or y is None:
            s = ""
        else:
            s = "(" + str(x) +"," + str(y) +")"
        
        self.ui.coordlabel.setText( s )
    
    def showMessage(self, message):
        self.ui.statusbar.showMessage( message )
    
    
    def shape_selection_changed(self):
        #called when Table emits itemSelectionChanged signal (when user selects rows)
        #we need to respond by instructing scene to select corresponding shapes
        
        #1. retrieve selected rows
        selection = self.ui.ShapeTable.selectionModel().selectedRows()
        rows = [ k.row() for k in selection]
        
        #2. select shapes in canvas
        self.ui.Canvas.scene().selectByIndex( rows, value=True, add=False )
    
    def shape_selected(self):
        #called when scene emits selection_changed signal
        #we need to respond by selecting the appropriate rows in the table
        
        #1. get selected shapes
        rows = self.ui.Canvas.scene().getSelectionIndex()
        
        #2. temporarily disconnect signal
        #Table will emit itemSelectionChanged signal if we select rows
        #to avoid getting stuck in loop, disconnect signal
        
        table = self.ui.ShapeTable
        QtCore.QObject.disconnect(table, QtCore.SIGNAL(_fromUtf8("itemSelectionChanged()")), self.shape_selection_changed)
        
        #3. clear selection, temporarily set selection mode to multi-select
        table.clearSelection()
        table.setSelectionMode( table.MultiSelection )
        
        #4. select rows
        for k in rows:
            table.selectRow( k )
            
        #5. undo temporary changes
        table.setSelectionMode( table.ExtendedSelection )
        QtCore.QObject.connect(self.ui.ShapeTable, QtCore.SIGNAL(_fromUtf8("itemSelectionChanged()")), self.shape_selection_changed)
    

def queued_return_value( f ):
    
    def g(*args, **kwargs):
        
        queue = kwargs.pop( '__queue__', None )
        
        ret = f( *args, **kwargs)
        
        if queue is not None:
            queue.put( ret )
        
        return ret
    
    return g

@queued_return_value
def start_ui( mainwindow, *args, **kwargs):
    
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle('cleanlooks')
    
    window = mainwindow( *args, **kwargs )
    window.show()
    
    app.exec_()
    
    return window.return_value


def ui_starter(window):
    
    def f(*args,**kwargs):
        
        if kwargs.pop('__multiprocessing__',True):
                   
            import multiprocessing
            q = multiprocessing.Queue()
            kwargs['__queue__'] = q
            p = multiprocessing.Process(target=start_ui, args=(window,)+args, kwargs=kwargs)
            p.start()
            return q.get()
        
        else:
            
            return start_ui( window, *args, **kwargs)
    
    f.__doc__ = """Help me!"""
    
    return f

amaze = ui_starter(AmazeWindow)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Amaze me!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('env', nargs='?', default=None, help='Path to the enviroment definition file (*.yaml)')
    parser.add_argument('--pos', default=None, help='Position data file (*.hdf5)')
    parser.add_argument('--img', default=None, help='Image file (*.png *.jpg *.tif)')
    parser.add_argument('--target', default=None, help='Output file name for environment definition')
    
    args = parser.parse_args()
    
    amaze(__multiprocessing__=False, env=args.env, pos=args.pos, img=args.img, target=args.target)
    sys.exit( 0 )
