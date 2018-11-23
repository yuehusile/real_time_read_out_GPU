# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'environment_properties.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_EnvironmentDialog(object):
    def setupUi(self, EnvironmentDialog):
        EnvironmentDialog.setObjectName(_fromUtf8("EnvironmentDialog"))
        EnvironmentDialog.resize(400, 300)
        self.verticalLayout = QtGui.QVBoxLayout(EnvironmentDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.frame = QtGui.QFrame(EnvironmentDialog)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.formLayout = QtGui.QFormLayout(self.frame)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(self.frame)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.Name = QtGui.QLineEdit(self.frame)
        self.Name.setObjectName(_fromUtf8("Name"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.Name)
        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.Comments = QtGui.QPlainTextEdit(self.frame)
        self.Comments.setObjectName(_fromUtf8("Comments"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.Comments)
        self.verticalLayout.addWidget(self.frame)
        self.buttonBox = QtGui.QDialogButtonBox(EnvironmentDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(EnvironmentDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), EnvironmentDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), EnvironmentDialog.reject)
        QtCore.QObject.connect(self.Name, QtCore.SIGNAL(_fromUtf8("textChanged(QString)")), EnvironmentDialog.name_changed)
        QtCore.QMetaObject.connectSlotsByName(EnvironmentDialog)

    def retranslateUi(self, EnvironmentDialog):
        EnvironmentDialog.setWindowTitle(_translate("EnvironmentDialog", "Environment Properties", None))
        self.label.setText(_translate("EnvironmentDialog", "name", None))
        self.label_2.setText(_translate("EnvironmentDialog", "comments", None))

