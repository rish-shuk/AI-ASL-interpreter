# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_browseModel_dialogBox.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(632, 273)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SelectModelLabel = QtWidgets.QLabel(Dialog)
        self.SelectModelLabel.setObjectName("SelectModelLabel")
        self.verticalLayout.addWidget(self.SelectModelLabel)
        self.ListModelExampleLabel = QtWidgets.QLabel(Dialog)
        self.ListModelExampleLabel.setObjectName("ListModelExampleLabel")
        self.verticalLayout.addWidget(self.ListModelExampleLabel)
        self.SelectModelList = QtWidgets.QListWidget(Dialog)
        self.SelectModelList.setObjectName("SelectModelList")
        item = QtWidgets.QListWidgetItem()
        self.SelectModelList.addItem(item)
        self.verticalLayout.addWidget(self.SelectModelList)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.SelectModelLabel.setText(_translate("Dialog", "Select a model: "))
        self.ListModelExampleLabel.setText(_translate("Dialog", "Model Name | Accuracy | Train/Validation Ratio | Model Type | Batch Size"))
        __sortingEnabled = self.SelectModelList.isSortingEnabled()
        self.SelectModelList.setSortingEnabled(False)
        item = self.SelectModelList.item(0)
        item.setText(_translate("Dialog", "Test"))
        self.SelectModelList.setSortingEnabled(__sortingEnabled)
