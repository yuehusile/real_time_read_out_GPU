"""
===============================================
PyQt core utilities (:mod:`fklab.ui.pyqt.core`)
===============================================

.. currentmodule:: fklab.ui.pyqt.core

Core module for PyQt applications.

.. autosummary::
    :toctree: generated/
    
    UiExecutor
    
"""

import multiprocessing

import PyQt5 as pyqt
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication

__all__ = ['pyqt', 'QApplication', 'QtCore', 'QtWidgets', 'QtGui', 'UiExecutor', 'SignalWait']

def _run_ui( mainwindow, *args, **kwargs):
    
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle('cleanlooks')
    
    window = mainwindow( *args, **kwargs )
    window.show()
    
    ret = app.exec_()
    
    try:
        ret = window.__retrieve_result__()
    except AttributeError:
        pass
    
    return ret

class UiExecutor:
    """Helper class to start a pyqt application.
    
    The application can either be started in the same process as the caller,
    or in a separate process (i.e. asynchronously).
    
    app = UiExecutor(main_window)
    result = app( args, kwds ) # blocks until app is closed
    async_result = app.async( args, kwds ) # result can be retrieved once app is closed
    
    Parameters
    ----------
    ui : Qt main window class
    
    """
    def __init__(self, ui):
        self._ui = ui
    
    def __call__(self, args=tuple(), kwds=dict()):
        return _run_ui(self._ui, *args, **kwds)
    
    def async(self, args=tuple(), kwds=dict(), callback=None):
        pool = multiprocessing.Pool(processes=1)
        return pool.apply_async( _run_ui, args=(self._ui,)+args, kwds=kwds, callback=callback )
    
class SignalWait(QtCore.QObject):
    """Helper class to block until signal is emitted.
    
    Qt events continue to be process while waiting for signal is emitted.
    Once the signal is emitted, its arguments are returned.
    If the signal is not emitted before the (optional) time out,
    a TimeoutError exception is raised.
    
    Parameters
    ----------
    signal : Qt Signal
    
    """
    
    def __init__(self,signal):
        
        QtCore.QObject.__init__(self)
        
        signal.connect( self.finish )
        self._done = False
        self.result = tuple()
    
    def finish(self, *args):
        self.result = args
        self._done = True
        
    def wait(self, timeout=0):
        """Wait for signal.
        
        Parameters
        ----------
        timeout : scalar
            Maximum time (in milliseconds) to wait for signal. A TimeoutError
            exception is raised if signal is not emitted in time. When the value
            is zero, then no time out is applied.
        
        Returns
        -------
        tuple : signal arguments
        
        """
        app = QtCore.QCoreApplication.instance()
        
        if timeout>0:
            timer = QtCore.QTime(0,0,0,timeout)
            timer.start();
        
        while( not self._done and (timeout==0 or timer.elapsed()<timeout) ):
            app.processEvents()
        
        if not self._done:
            raise TimeoutError('Timed out while waiting for signal.')
        
        return self.result
