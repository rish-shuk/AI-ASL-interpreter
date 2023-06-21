################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PySide2
## V: 1.0.0
##
################################################################################

## ==> GUI FILE
from main import *
import pandas as pd
# from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import *
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import time
import os
import sys
import copy
import shutil

from torch.utils.data import DataLoader
import torchvision

import numpy as np
from PIL import Image

from View.ui_main_josefAdditions import Ui_MainWindow

# IMPORTING MODULE FILE FROM DIFFERENT FOLDER
sys.path.append(os.getcwd() + "/backend")

from backend.models.model_functions_classTest import ModelTrainer
from backend.models.csvconvert import CSVConvert, LenetTransform, CNNTransform, show_img, AlexNetTransform, Img_LenetTransform, Img_AlexNetTransform, Img_CNNTransform

import backend.models.lenet5 
import backend.models.alexnet_model
import backend.models.cnn_model 

# for general functions that are used throughout the GUI
class GeneralFunctions():
    selectedDatasetPaths = None
    selectedModelInfo = None
    selectedTestingImages = None

    def __init__(self, ui_main: Ui_MainWindow):
        self.consoleLineCounter = 0
        self.ui = ui_main
    
    def sendToConsole(self, info: str):
        consoleText = "{" + str(datetime.now()) + "}: " + f'{info}'
        self.displayOnConsole(consoleText)

    def displayOnConsole(self, text: str): 
        widgetText = QLabel()
        widgetText.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        widgetText.setObjectName(f'Console Line {self.consoleLineCounter}')
        widgetText.setText(text)
        self.consoleLineCounter = self.consoleLineCounter + 1
        self.ui.consoleVLayout.addWidget(widgetText)
        self.ui.DebugConsoleArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.ui.DebugConsoleArea.verticalScrollBar().rangeChanged.connect(self.scrollToBottom)
    
    def scrollToBottom(self, min, max):
        self.ui.DebugConsoleArea.verticalScrollBar().setValue(max)
    
    def retrieveSelectedTrainCSVPath(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return
        
        self.sendToConsole(f"Current data set path: {GeneralFunctions.selectedDatasetPaths} ")
        trainCSVPath = GeneralFunctions.selectedDatasetPaths[0]
        return trainCSVPath
    
    def retrieveSelectedTrainImagePath(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return
        
        self.sendToConsole(f"Current data set path: {GeneralFunctions.selectedDatasetPaths} ")
        trainImagePath = GeneralFunctions.selectedDatasetPaths[1]
        return trainImagePath

    def retrieveSelectedTestCSVPath(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return

        self.sendToConsole(f"Current data set selected: {GeneralFunctions.selectedDatasetPaths} ")
        testCSVPath = GeneralFunctions.selectedDatasetPaths[2]
        return testCSVPath

    def retrieveSelectedTestImagePath(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return
        
        self.sendToConsole(f"Current data set path: {GeneralFunctions.selectedDatasetPaths} ")
        testImagePath = GeneralFunctions.selectedDatasetPaths[3]
        return testImagePath

    def retrieveSelectedDatasetImagesAmount(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return
        return GeneralFunctions.selectedDatasetPaths[4]
    
    def retrieveSelectedDatasetName(self):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.sendToConsole(f"Error Training: No data set selected!")
            return
        return GeneralFunctions.selectedDatasetPaths[5]
    
    def retrieveSelectedModelName(self):
        if (GeneralFunctions.selectedModelInfo == None):
            self.sendToConsole(f"Error getting model info: No model selected!")
            return
        return GeneralFunctions.selectedModelInfo[1]

    def retrieveSelectedModelType(self):
        if (GeneralFunctions.selectedModelInfo == None):
            self.sendToConsole(f"Error getting model info: No model selected!")
            return
        return GeneralFunctions.selectedModelInfo[4]

class SaveLoadListsFunctions():
    settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope ,'ISM', 'SL_AI_Trainer')

    def __init__(self, ui_main: Ui_MainWindow):
        self.ui = ui_main
        self.generalFunctions = GeneralFunctions(self.ui)
        self.settingsFilePath = os.getcwd()
        SaveLoadListsFunctions.settings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, self.settingsFilePath)

    def saveLists(self):
        self.saveDataSetList()
        self.saveModelList()

    def restoreLists(self):
        self.restoreDataSetList()
        self.restoreModelList()

    def saveModelList(self):
        # save list of items of dataset
        savedSettings = QSettings(f"{self.settingsFilePath}/ISM/SL_AI_Trainer.ini", QSettings.Format.IniFormat)
        savedSettings.beginGroup("ModelList")
        listOfItems = []
        for item_index in range(self.ui.SelectModelList.count()):
            item = []
            item.append(self.ui.SelectModelList.item(item_index).text())
            item.append(self.ui.SelectModelList.item(item_index).data(Qt.ItemDataRole.UserRole))
            listOfItems.append(copy.copy(item))

        savedSettings.setValue("listOfModels", listOfItems)
        savedSettings.endGroup()
    
    def restoreModelList(self):
        savedSettings = QSettings(f"{self.settingsFilePath}/ISM/SL_AI_Trainer.ini", QSettings.Format.IniFormat)
        savedSettings.beginGroup("ModelList")
        listOfItems = savedSettings.value("listOfModels")
        if (listOfItems == None):
            self.generalFunctions.sendToConsole("No saved model list found!")
            return

        self.ui.SelectModelList.clear()

        for item in listOfItems:
            if (item == None):
                continue

            item_name = item[0]
            item_data = item[1]

            if (item_data == None):
                continue

            # check if widget file directories still exist otherwise skip
            if (not os.path.exists(item_data[0])):
                self.generalFunctions.sendToConsole(f"Error: PTH File missing! Removing saved model ({item_name}).")
                listOfItems.remove(item)
                continue

            newListWidget = QListWidgetItem(self.ui.SelectModelList)
            newListWidget.setText(item_name)

            if (not item_data == None):
                newListWidget.setData(Qt.ItemDataRole.UserRole, item_data)

            self.ui.SelectModelList.addItem(newListWidget)

        savedSettings.endGroup()

    def saveDataSetList(self):
        # save list of items of dataset
        savedSettings = QSettings(f"{self.settingsFilePath}/ISM/SL_AI_Trainer.ini", QSettings.Format.IniFormat)
        savedSettings.beginGroup("DataSetList")
        listOfItems = []
        for item_index in range(self.ui.listWidget.count()):
            item = []
            item.append(self.ui.listWidget.item(item_index).text())
            item.append(self.ui.listWidget.item(item_index).data(Qt.ItemDataRole.UserRole))
            
            listOfItems.append(copy.copy(item))

        # listOfItems = self.ui.listWidget.items()
        # itemNumber = 1
        # for item in listOfItems:
        #     SaveLoadListsFunctions.settings.setValue(f"datasetItem_{itemNumber}", item)

        # list of items contain a list of items
        # where each item consists of 
        #   item[0] = item_name
        #   item[1] = item_data
        savedSettings.setValue("listOfItems", listOfItems)

        savedSettings.endGroup()

    def restoreDataSetList(self):
        savedSettings = QSettings(f"{self.settingsFilePath}/ISM/SL_AI_Trainer.ini", QSettings.Format.IniFormat)
        savedSettings.beginGroup("DataSetList")
        listOfItems = savedSettings.value("listOfItems")
        if (listOfItems == None):
            self.generalFunctions.sendToConsole("No saved data sets found!")
            print("wow there is nothing here")
            print(listOfItems)
            return

        self.ui.listWidget.clear()

        for item in listOfItems:
            if (item == None):
                continue

            item_name = item[0]
            item_data = item[1]

            if (item_data == None):
                continue
            # check if widget file directories still exist otherwise skip

            if ((not os.path.exists(item_data[0])) or (not os.path.exists(item_data[2]))):
                self.generalFunctions.sendToConsole(f"Error: CSV File missing! Removing saved dataset ({item_name}).")
                listOfItems.remove(item)
                continue

            newListWidget = QListWidgetItem(self.ui.listWidget)
            newListWidget.setText(item_name)

            if (not item_data == None):
                newListWidget.setData(Qt.ItemDataRole.UserRole, item_data)
            #print(f"{item_name}: {item_data}")
            self.ui.listWidget.addItem(newListWidget)

        savedSettings.endGroup()

# Importing and converting functions of the dataset
class ImportFunctions():
    def __init__(self, ui_main: Ui_MainWindow):
        self.ui = ui_main
        self.thread = None
        self.trainRatio = 50
        self.testRatio = 50
        self.datasetNumLabels = [self.ui.aLabel,self.ui.bLabel,self.ui.cLabel,self.ui.dLabel,self.ui.eLabel,self.ui.fLabel,self.ui.gLabel,self.ui.hLabel,self.ui.iLabel,self.ui.kLabel,self.ui.lLabel,self.ui.mLabel,self.ui.nLabel,self.ui.oLabel,self.ui.pLabel,self.ui.qLabel,self.ui.rLabel,self.ui.sLabel,self.ui.tLabel,self.ui.uLabel,self.ui.vLabel,self.ui.wLabel,self.ui.xLabel,self.ui.yLabel]
        self.generalFunctions = GeneralFunctions(self.ui)
        self.connectItemWidgetSignal()
        self.settings = SaveLoadListsFunctions(self.ui)

    def importData(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        trainFile = QFileDialog.getOpenFileName(None, "Choose Train File", ".","CSV Files (*.csv)", options=options)[0]
        testFile = QFileDialog.getOpenFileName(None, "Choose Test File", ".","CSV Files (*.csv)", options=options)[0]

        trainFileName = os.path.basename(trainFile).split('.')[0] # name of csv
        testFileName = os.path.basename(testFile).split('.')[0]

        self.generalFunctions.sendToConsole(f"Selected train file: {trainFile}")
        self.generalFunctions.sendToConsole(f"Selected test file: {testFile}")

        self.thread = QThread()
        self.worker = ImportWorker(trainFile, testFile, trainFileName, testFileName)
        self.thread.started.connect(self.worker.run)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)
        self.worker.progress.connect(self.updateProgressBar)
        self.worker.finished.connect(self.ui.TimeLeftText.setText)
        self.worker.progress.connect(self.updateProgressBar)
        self.worker.arrays.connect(self.getFilterDict)
        self.worker.saveDataset.connect(self.saveImportedDataset)
        self.worker.sendToConsole.connect(self.generalFunctions.sendToConsole)
        self.thread.start()
        """
        trainFileName = os.path.basename(trainFile).split('.')[0] # name of csv
        testFileName = os.path.basename(testFile).split('.')[0]
        self.thread = ImportThread(trainFile, testFile, trainFileName, testFileName)
        self.thread.progress.connect(self.updateProgressBar)
        self.thread.arrays.connect(self.getFilterDict)
        self.thread.finished.connect(self.ui.TimeLeftText.setText)
        self.thread.saveDataset.connect(self.saveImportedDataset)
        self.thread.sendToConsole.connect(self.generalFunctions.sendToConsole)
        self.thread.start()
        """

    def getFilterDict(self, dict):
        labelToLetter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
        currentIndex = 0
        labelNum = 0
        totalImages = 0
        while currentIndex < 24:
            if currentIndex == 9 :
                labelNum = labelNum + 1
            else:
                pass
            letter = labelToLetter[labelNum]
            lenArray = len(dict[letter])
            currentLabel = self.datasetNumLabels[currentIndex]
            currentLabel.setText(f"{letter} :{lenArray}")
            currentIndex = currentIndex + 1
            labelNum = labelNum + 1
            totalImages = totalImages + lenArray
        self.ui.NumberOfImagesText.setText(f"Number of Images :{totalImages}")

    def stopImport(self):
        if self.thread is not None:
            self.thread.quit()

    def updateProgressBar(self, value):
        self.ui.progressBar.setValue(value)
        QApplication.processEvents()

    def saveImportedDataset(self, data: list):
        newDatasetItemWidget = QListWidgetItem(self.ui.listWidget)
        # order of list:

        # train csv path
        # train images path
        # test csv path
        # test images path    
        
        trainPath = data[0]
        trainImagesPath = data[1]
        testPath = data[2]
        testImagesPath = data[3]

        trainFileName = os.path.basename(trainPath).split('.')[0] # name of csv
        testFileName = os.path.basename(testPath).split('.')[0]

        datasetName = f"{trainFileName} | {testFileName}"

        data.append(datasetName)

        # set name of item
        newDatasetItemWidget.setText(datasetName)

        # add data to item 
        newDatasetItemWidget.setData(Qt.ItemDataRole.UserRole, data)
        self.ui.listWidget.addItem(newDatasetItemWidget)

        self.settings.saveDataSetList()
        
    def connectItemWidgetSignal(self):
        self.ui.listWidget.itemClicked.connect(self.onItemClicked)
    
    def onItemClicked(self, item: QListWidgetItem):
        
        if (item == None or item.data(Qt.ItemDataRole.UserRole) == None):
            print("No data found!")
            self.generalFunctions.sendToConsole("Error: selected dataset contains no info about the dataSet")
            return
        
        print(f"Current data set selected: {item.text()} ")

        GeneralFunctions.selectedDatasetPaths = copy.copy(item.data(Qt.ItemDataRole.UserRole))
        
        self.generalFunctions.sendToConsole(f"Current data set selected: {item.text()} ")
        self.ui.selectedDatasetLabel.setText(f"Selected Dataset: {item.text()}")
        self.generalFunctions.sendToConsole(f"Data Set Info: ")
        for data in GeneralFunctions.selectedDatasetPaths:
            self.generalFunctions.sendToConsole(data)
        # Getting the data embedded in each item from the listWidget
        # item_data = item.data(Qt.ItemDataRole.UserRole)  

        # Getting the datatext of each item from the listWidget
        # item_name = item.text() 

        # print(item_name)
        # print(item_data)
        # self.generalFunctions.sendToConsole(item_name)
        # self.generalFunctions.sendToConsole(item_data)

class ImportWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    arrays = pyqtSignal(object)
    sendToConsole = pyqtSignal(str)
    saveDataset = pyqtSignal(list)
    done = pyqtSignal()
    
    def __init__(self, train_file, test_file, trainName, testName):
        super().__init__()
        self.train_file = train_file # csv file path
        self.test_file = test_file
        self.filterDict = None
        self.cancelled = False

        self.trainFileName = trainName
        self.testFileName = testName

    def run(self):
        # CHECK IF FILE HAS ACTUALLY BEEN SELECTED
        if self.train_file == '' or self.test_file == '':
            print("No file selected")
            self.sendToConsole.emit("Importing failed! One or more files weren't found!")
            self.done.emit()
            return

        self.sendToConsole.emit(f"Beginning Import!")
        labelToImage = {}
        labelToLetter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
        for i in range(25):
            if i == 9 :
                pass
            else:
                labelToImage[labelToLetter[i]] = []
        cwd = os.getcwd()

        # creation of file path to images if path doesnt exist
        if (not os.path.exists(os.getcwd() + f"/dataset/{self.testFileName}/test_images") or (not os.getcwd() + f"/dataset/{self.trainFileName}/train_images")):
            os.makedirs(os.getcwd() + f"/dataset/{self.testFileName}/test_images")
            os.makedirs(os.getcwd() + f"/dataset/{self.trainFileName}/train_images")
        
        self.testImageDirectiory = os.getcwd() + f"/dataset/{self.testFileName}/test_images"
        self.trainImageDirectory = os.getcwd() + f"/dataset/{self.trainFileName}/train_images"
        
        test_data = np.genfromtxt(self.test_file, delimiter=',')
        train_data = np.genfromtxt(self.train_file, delimiter=',')
        test_images = test_data.shape[0]
        train_images = train_data.shape[0]
        total = train_images + test_images

        print(total)
        print(f"Test Images Amount = {test_images}")

        self.totalImages = int(total)
        print(self.totalImages)

        completed = 0
        start_time = time.time()
        for i in range(test_images - 1):
            if self.cancelled:
                break
            labelData = test_data[i + 1, 0].reshape((-1, 1))
            labelData = labelData.astype(int)
            labelData = labelToLetter[labelData[0][0]]
            data = test_data[i + 1, 1:].reshape((28, 28))
            data = np.interp(data, (data.min(), data.max()), (0, 255)).astype(np.uint8)
            image = Image.fromarray(data)
            image.save(f"{self.testImageDirectiory}/image_{i + 1}_test.jpg")
            labelToImage[labelData].append(f"image_{i + 1}_test.jpg")
            completed += 1
            progress = int((completed / total) * 100)
            
                #timeLeft = self.calcProcessTime(start_time,currentIteration,total)
            telapsed = time.time() - start_time
            testimated = (telapsed/completed)*(total)

            timeLeft = testimated-telapsed
            string = f"{int(timeLeft)} seconds left"
            self.finished.emit(string)
            self.progress.emit(progress)

        for i in range(train_images - 1):
            if self.cancelled:
                break
            labelData = train_data[i + 1, 0].reshape((-1, 1))
            labelData = labelData.astype(int)
            labelData = labelToLetter[labelData[0][0]]
            data = train_data[i + 1, 1:].reshape((28, 28))
            data = np.interp(data, (data.min(), data.max()), (0, 255)).astype(np.uint8)
            image = Image.fromarray(data)
            image.save(f"{self.trainImageDirectory}/image_{i + 1}_train.jpg")
            labelToImage[labelData].append(f"image_{i + 1}_train.jpg")
            completed += 1
            progress = int((completed / total) * 100)
            telapsed = time.time() - start_time
            testimated = (telapsed/completed)*(total)

            timeLeft = testimated-telapsed
            string = f"{int(timeLeft)} seconds left"
            self.finished.emit(string)
            self.progress.emit(progress)

        if self.cancelled:
            for file_name in os.listdir(cwd):
                if file_name.endswith('.jpg'):
                    os.remove(os.path.join(cwd, file_name))
            self.progress.emit(0)
            self.finished.emit("Upload Cancelled")
            self.emit_to_console("Upload Cancelled!")
            self.done.emit()
        else:
            self.filterDict = labelToImage
            self.arrays.emit(labelToImage)
            self.progress.emit(100)
            self.finished.emit("Upload Complete")
            self.emit_to_console("Upload Complete!")

            datasetInfo = []
            datasetInfo.append(self.train_file)
            datasetInfo.append(self.trainImageDirectory)
            datasetInfo.append(self.test_file)
            datasetInfo.append(self.testImageDirectiory)
            datasetInfo.append(int(total))
            self.emit_saveDataset(datasetInfo)
            self.done.emit()

    def cancel(self):
        self.cancelled = True

    def emit_to_console(self, text):
        self.sendToConsole.emit(text)
    
    def emit_saveDataset(self, data):
        self.saveDataset.emit(data)

# region TESTING REMOVED CODE
# class ImportThread(QThread):
#     progress = pyqtSignal(int)
#     finished = pyqtSignal(str)
#     arrays = pyqtSignal(object)
#     sendToConsole = pyqtSignal(str)
#     saveDataset = pyqtSignal(list)

#     def __init__(self, trainFile, testFile, trainName, testName):
#         super().__init__()
#         self.testFile = testFile
#         self.trainFile = trainFile
#         self.filterDict = None
#         self.trainName = trainName
#         self.testName = testName
        
#     def run(self):
#         if self.trainFile == '' or self.testFile == '':
#             print("No file selected")
#             self.sendToConsole.emit("Importing failed! One or more files weren't found!")

#         else:
#             self.worker = ImportWorker(self.trainFile, self.testFile, self.trainName, self.testName)
#             self.worker.progress.connect(self.emit_progress)
#             self.worker.finished.connect(self.emit_finished)
#             self.worker.arrays.connect(self.emit_arrays)
#             self.worker.sendToConsole.connect(self.emit_to_console)
#             self.worker.saveDataset.connect(self.emit_saveDataset)
#             self.worker.run()

#     def cancel(self):
#         self.worker.cancel()
#         self.emit_to_console("Import cancelled!")

#     def emit_progress(self, value):
#         self.progress.emit(value)

#     def emit_finished(self, message):
#         self.finished.emit(message)
#         self.emit_to_console("Import has finished!")

#     def emit_arrays(self,array):
#         self.arrays.emit(array)

#     def emit_saveDataset(self, data):
#         self.saveDataset.emit(data)
    
#     def emit_to_console(self, text):
#         self.sendToConsole.emit(text)
# endregion TESTING REMOVED CODE

class TrainFunctions():
    def __init__(self, mainUI: Ui_MainWindow):
        self.ui = mainUI
        self.thread = None
        self.generalFunctions = GeneralFunctions(self.ui)
        self.modelTrainer = ModelTrainer(mainUI)
        self.settings = SaveLoadListsFunctions(self.ui)
        self.trainRatio = 50
        self.testRatio = 50
        self.connectWidgetSignals()

    def connectWidgetSignals(self):
        self.ui.TrainModelBtn.clicked.connect(self.trainModel)

        #batchSize = self.ui.BatchSizeSpinBox.valueFromText()
        self.ui.BatchSizeSpinBox.valueChanged.connect(self.setBatchSize)

        #epochAmount = self.ui.EpochNumberSpinBox.valueFromText()
        self.ui.EpochNumberSpinBox.valueChanged.connect(self.setEpochAmount)

        self.ui.modelTypeCBox.textActivated.connect(self.modelTypeSelected)

        # Models tab list widget
        self.ui.SelectModelList.itemClicked.connect(self.onItemClicked)

        # set name on change
        self.ui.modelNameLineEdit.textChanged.connect(self.setName)

        # Save model
        self.ui.SaveModelBtn.clicked.connect(self.saveTrainedModel)

        # stop training model and skip to testing
        self.ui.StopTrainingBtn.clicked.connect(self.modelTrainer.cancelTraining)
    
    def setBatchSize(self, batchSize):
        self.modelTrainer.setBatchSize(batchSize)
    
    def setEpochAmount(self, epochAmount):
        self.modelTrainer.setEpoch(epochAmount)
    
    def setName(self, name):
        self.modelTrainer.setName(name)

    def setRatio(self, testRatio, trainRatio):
        self.modelTrainer.setRatio(testRatio, trainRatio)

    def modelTypeSelected(self, selectedModelType):
        #selectedModelType = self.ui.modelTypeCBox.currentText()
        if (selectedModelType == "AlexNet"):
            modelType = backend.models.alexnet_model.AlexNet()
            
        elif (selectedModelType == "LeNet5"):
            modelType = backend.models.lenet5.LeNet5()

        elif (selectedModelType == "CNN"):
            modelType = backend.models.cnn_model.CNN_Model()
        elif (selectedModelType == "Select a model"):
            self.generalFunctions.sendToConsole(f"Select a model!")
            return
        else:
            modelType = backend.models.cnn_model.CNN_Model()
            self.generalFunctions.sendToConsole("Error: Model does not exist, defaulting to CNN")
        
        self.generalFunctions.sendToConsole(f"Selected Model: {selectedModelType}")
        self.modelTrainer.setModel(modelType)

    def convertDataSet(self):
        selectedModelType = self.ui.modelTypeCBox.currentText()
        transformType = None
        #modelType = None

        if (selectedModelType == "AlexNet"):
            transformType = AlexNetTransform()
            self.modelTrainer.transformType = transformType
            #modelType = backend.models.alexnet_model.AlexNet()
            
        elif (selectedModelType == "LeNet5"):
            transformType = LenetTransform()
            self.modelTrainer.transformType = transformType
            #modelType = backend.models.lenet5.LeNet5()

        elif (selectedModelType == "CNN"):
            transformType = CNNTransform()
            self.modelTrainer.transformType = transformType
            #modelType = backend.models.cnn_model.CNN_Model()
        else:
            #modelType = backend.models.cnn_model.CNN_Model()
            self.generalFunctions.sendToConsole("Error: Model does not exist, select another model")
            return


        self.generalFunctions.sendToConsole(f"Transforming dataset using {selectedModelType} transformation.")
        self.modelTrainer.transformDataset(self.generalFunctions.retrieveSelectedTrainCSVPath(), self.generalFunctions.retrieveSelectedTestCSVPath(), transformType)

    def trainModel(self):
        if (self.modelTrainer.model == None):
            self.generalFunctions.sendToConsole(f"Error: Train failed! - Select a model!")
            return
        
        self.convertDataSet()
        # epoch = 3
        # model = AlexNet()
        # trainAndTest(model, train_loader, test_loader, epoch)
        
        # assuming modeltrainer.setModel
        #              ""     .transformDatset (happens at convertDataset)
        #              ""     .setEpoch (default is 1: changes value whenever spinbox is changed)
        #              ""     .setBatchSize (default is 2: changes value whenever spinbox is changed)

        # have all been executed before hand

        # execute via thread
        self.thread = TrainThread(self.modelTrainer, self.ui)
        self.thread.sendToConsole.connect(self.generalFunctions.sendToConsole)
        self.thread.progress.connect(self.updateProgressBar)
        self.thread.finished.connect(self.ui.T_EstimatedTimeText.setText)
        self.thread.saveModelData.connect(self.saveTrainedModel)
        self.thread.start()
    
    def updateProgressBar(self, value):
        self.ui.T_ProgressBar.setValue(value)

    def saveTrainedModel(self):
    #   When save model is pressed
    #  -> save current model
    #  -> create item list widget and append to select model list widget
    #  -> save model information as a list in itemlistwidget.data
    # - data list contains
    # - filepath to model file
    # - name
    # - accuracy
    # - train/validation ratio
    # - model type
    # - batch size
    # - epoch
    #  -> when selected in listwidget, display data in related labels
        if (not self.modelTrainer.hasFinishedTraining):
            self.generalFunctions.sendToConsole("Cannot save! Model has not been trained yet!")
            return
        

        modelFilePath = self.modelTrainer.saveCurrentModel()
        doesModelAlreadyExist = False

        for item_index in range(self.ui.SelectModelList.count()):

            item_data = self.ui.SelectModelList.item(item_index).data(Qt.ItemDataRole.UserRole)
            item_text = self.ui.SelectModelList.item(item_index).text()

            if ((item_data == None) or (item_text == None)):
                continue

            if (item_data[0] == modelFilePath):
                doesModelAlreadyExist = True
        
        if (doesModelAlreadyExist):
            self.generalFunctions.sendToConsole("Error saving model: Model already exists!")
            return
        
        newModelItemWidget = QListWidgetItem(self.ui.SelectModelList)
        dataInfo = []

        modelName = self.modelTrainer.name
        newModelItemWidget.setText(modelName)
        accuracy = "debug incorrect / debug correct"
        trainingRatio = f"Train: {self.trainRatio} / Test: {self.testRatio}"
        modelType = self.modelTrainer.model.__class__.__name__
        batchSize = self.modelTrainer.batchSize
        epoch = self.modelTrainer.epoch

        dataInfo.append(modelFilePath) # 0
        dataInfo.append(modelName) # 1
        dataInfo.append(accuracy) # 2
        dataInfo.append(trainingRatio) # 3
        dataInfo.append(modelType) # 4
        dataInfo.append(batchSize) # 5
        dataInfo.append(epoch) # 6

        # add data to item 
        newModelItemWidget.setData(Qt.ItemDataRole.UserRole, dataInfo)
        
        self.ui.SelectModelList.addItem(newModelItemWidget)
        self.generalFunctions.sendToConsole("Saved PyTorch Model State to " + modelFilePath)

        self.settings.saveModelList()
        
    def onItemClicked(self, item: QListWidgetItem):
        if (item == None or item.data(Qt.ItemDataRole.UserRole) == None):
            print("No data found!")
            self.generalFunctions.sendToConsole("Error: selected model contains no info about the dataSet")
            return
        
        GeneralFunctions.selectedModelInfo = copy.copy(item.data(Qt.ItemDataRole.UserRole))
        modelInfo = GeneralFunctions.selectedModelInfo
    
        print(f"Current Model selected: {modelInfo[1]} ")

        # update labels
        self.ui.SelectModelLabel.setText(f"Selected Model: {modelInfo[1]}")

        # model information
        self.ui.CurrentModelName.setText(f"Current Model: {modelInfo[1]}")
        self.ui.TrainingLossAccuracyText.setText(f"Training Loss/Validation Accuracy: {modelInfo[2]}")
        self.ui.TrainingValidationRatioText.setText(f"Training/Validation Ratio: {modelInfo[3]}")
        self.ui.DeepNeuralNetworkText.setText(f"Model Type: {modelInfo[4].__class__.__name__}")
        self.ui.BatchSizeText.setText(f"Batch Size: {modelInfo[5]}")
        self.ui.EpochLabel.setText(f"Epoch: {modelInfo[6]}")

    def updateBoxes(self,value):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.generalFunctions.sendToConsole("Error changing slider: No dataset selected!")
            return
        
        diff = 100 - value
        self.ui.TrainSpinBox.setValue(value)
        self.ui.ValidateSpinBox.setValue(diff)

    def updateTrainBoxValue(self, value):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.generalFunctions.sendToConsole("Error changing slider: No dataset selected!")
            return
        floatRatio = float(value / 100.0)
        diff = 100 - value
        self.trainRatio = value
        self.ui.TrainValidateSlider.setValue(value)
        self.ui.ValidateSpinBox.setValue(diff)

        self.setRatio(self.testRatio, self.trainRatio)
        imagesAmount = self.generalFunctions.retrieveSelectedDatasetImagesAmount()
        self.ui.TrainImagesAmountText.setText(f'Images: {int(imagesAmount * floatRatio)}')

    def updateValidateBoxValue(self, value):
        if (GeneralFunctions.selectedDatasetPaths == None):
            self.generalFunctions.sendToConsole("Error changing slider: No dataset selected!")
            return
        floatRatio = float((value) / 100.0)
        
        diff = 100 - value
        self.testRatio = value
        self.ui.TrainValidateSlider.setValue(diff)
        self.ui.TrainSpinBox.setValue(diff)

        self.setRatio(self.testRatio, self.trainRatio)
        imagesAmount = self.generalFunctions.retrieveSelectedDatasetImagesAmount()
        self.ui.ValidateAmountText.setText(f'Images: {int(imagesAmount * floatRatio)}')

class TrainThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    sendToConsole = pyqtSignal(str)
    saveModelData = pyqtSignal(list)

    def __init__(self, modelTrainer: ModelTrainer, ui: Ui_MainWindow):
        super().__init__()
        self.modelTrainer = modelTrainer
        self.ui = ui
        
    def run(self):
        #self.total = testData.shape[0] + trainData.shape[0]
        self.worker = TrainWorker(self.modelTrainer)
        # self.worker.progress.connect(self.emit_progress)
        # self.worker.finished.connect(self.emit_finished)
        self.worker.sendToConsole.connect(self.emit_to_console)
        self.worker.saveModelData.connect(self.emit_save_model_data)

        self.worker.progress.connect(self.emit_progress)
        self.worker.finished.connect(self.finished)

        self.worker.run()

    def cancel(self):
        self.worker.cancel()
        self.emit_to_console("Train cancelled! Starting test!")

    def emit_progress(self, value):
        self.progress.emit(value)

    def emit_finished(self, message):
        self.finished.emit(message)
        self.emit_to_console("Training has finished!")
    
    def emit_to_console(self, text):
        self.sendToConsole.emit(text)
    
    def emit_save_model_data(self, dataList):
        self.saveModelData.emit(dataList)

class TrainWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    sendToConsole = pyqtSignal(str)
    saveModelData = pyqtSignal(list)
    
    def __init__(self, modelTrainer:ModelTrainer):
        super().__init__()
        #self.total = total
        self.completed = 0
        self.cancelled = False
        self.modelTrainer = modelTrainer

    def run(self):
        self.sendToConsole.emit(f"Beginning Model Training!")
        #start_time = time.time()
        # create emit functions in modelTrainier
        # connect modelTrainer emit functions to TrainWorker functions
        self.modelTrainer.sendToConsole.connect(self.emit_to_console)
        self.modelTrainer.saveModelData.connect(self.emit_save_model_data)

        self.modelTrainer.finished.connect(self.emit_finished)
        self.modelTrainer.progress.connect(self.emit_progress)

        # run modelTrainer.executeModelTrainer
        self.modelTrainer.executeModelTrainer()
    
        # if self.cancelled:
        #     # cancel trainiing -> go to testing
        #     # break

        # if self.cancelled:
        #     # self.emit_progress(0)
        #     # self.emit_finished("Training stopped!")
        # else:
        #     self.emit_progress(100)
        #     self.emit_finished("Training Complete")

    def cancel(self):
        self.cancelled = True

    def emit_progress(self, value):
        self.progress.emit(value)

    def emit_finished(self, message):
        self.finished.emit(message)
    
    def emit_to_console(self, text):
        self.sendToConsole.emit(text)
    
    def emit_save_model_data(self, dataList):
        self.saveModelData.emit(dataList)

# dialog boxes code
class LoadModelsFunctions():
    def __init__(self, ui_main: Ui_MainWindow, ui_dialog: ui_browseModel_dialogBox.Ui_Dialog):
        self.ui_main = ui_main
        self.ui_dialog = ui_dialog
        self.generalFunctions = GeneralFunctions(self.ui_main)
        self.connectSignalSlots()
    
    #https://stackoverflow.com/questions/62408781/copying-all-items-from-qlistwidget-to-another
    def initialiseModelList(self):
        # copy items
        for i in range(self.ui_main.SelectModelList.count()):
            clone_it = self.ui_main.SelectModelList.item(i).clone()
            self.ui_dialog.SelectModelList.addItem(clone_it)

    def connectSignalSlots(self):
        self.ui_dialog.SelectModelList.itemClicked.connect(self.onItemClicked)
    
    def onItemClicked(self, item: QListWidgetItem):

        if (item == None or item.data(Qt.ItemDataRole.UserRole) == None):
            print("No data found!")
            self.generalFunctions.sendToConsole("Error: selected model contains no information")
            return
        
        print(f"Current data set selected: {item.text()} ")

        GeneralFunctions.selectedModelInfo = copy.copy(item.data(Qt.ItemDataRole.UserRole))

        testingImagesPath = self.generalFunctions.retrieveSelectedTestImagePath()
        currentDataset = self.generalFunctions.retrieveSelectedDatasetName()
                                                               
        self.generalFunctions.sendToConsole(f"Current data set selected: {currentDataset} ")
        self.generalFunctions.sendToConsole(f"DEBUG-Selected Dataset Testing Images Path: {testingImagesPath}")
        self.generalFunctions.sendToConsole(f"Selected Model: {testingImagesPath}")

class TestingImagesFunctions():
    def __init__(self, ui_main: Ui_MainWindow, ui_dialog: ui_testingImages_dialogBox.Ui_Dialog):
        self.ui_main = ui_main
        self.ui_dialog = ui_dialog
        self.generalFunctions = GeneralFunctions(self.ui_main)

class PredictViewerFunctions():
    def __init__(self, ui_main: Ui_MainWindow):
        self.ui = ui_main
        self.generalFunctions = GeneralFunctions(self.ui)
        self.selectedImages = []
        self.current_image = 0
        self.modelTrainer = ModelTrainer(self.ui)
        self.dataLoader = None

        self.generalFunctions.selectedTestingImages = self.selectedImages

    def connectSignalSlots(self):
        self.ui.NextImageButton.clicked.connect(self.nextImage)
        self.ui.PrevImageBtn.clicked.connect(self.prevImage)

    def startPredictions(self):
        # get model transform
        # get model from load(modelname)
        # get selectedDataset from filepath
        # get dataloader usning DataLoader(selectedDataset,batch_size=self.selectedImages.count),num_workers=2,shuffle=True)
        print("ARE WE HERE YET")

        filePath = os.getcwd() + "/dataset/selectedImages"

        modelType = self.generalFunctions.retrieveSelectedModelType()

        print(f"TRANSFORM TYPE: {modelType}")

        if (modelType == "AlexNet"):
            transformType = Img_AlexNetTransform()
            print("YUP")
            print(transformType)
            self.modelTrainer.transformType = transformType
            #modelType = backend.models.alexnet_model.AlexNet()
            
        elif (modelType == "LeNet5"):
            transformType = Img_LenetTransform()
            print("YUP")
            print(transformType)
            #modelType = backend.models.lenet5.LeNet5()

        elif (modelType == "CNN_Model"):
            transformType = Img_CNNTransform()
            print("YUP")
            print(transformType)
        else:
            print("nope")
            self.generalFunctions.sendToConsole("Error: Model does not exist")
            return

        selectedDataset = torchvision.datasets.ImageFolder(filePath, transformType)

        self.dataLoader = DataLoader(selectedDataset,len(self.selectedImages),num_workers=2,shuffle=True)

        self.loadedModel = self.modelTrainer.load(self.generalFunctions.retrieveSelectedModelName())

        #self.modelTrainer.showPrediction(self.dataLoader, self.current_image, self.loadedModel)

        
    def initialiseImageViewer(self, selectedImagesFromUser):

        self.selectedImages = selectedImagesFromUser

        if (self.selectedImages == None or len(self.selectedImages) <= 0):
            self.generalFunctions.sendToConsole("Error predicting!: No testing images selected!")
            return

        if (not os.path.exists(os.getcwd() + f"/dataset/selectedImages/images")):
            os.makedirs(os.getcwd() + f"/dataset/selectedImages/images")

        for filePath in self.selectedImages:
            original = rf"{filePath}.jpg"
            target = rf"{os.getcwd()}/dataset/selectedImages/images"
            shutil.copy(original, target)

        self.startPredictions()
            
        self.generalFunctions.sendToConsole("Initialise!")
        self.current_image = 0
        self.setImageText()
        self.showImage()
    
    def nextImage(self):
        self.current_image = (self.current_image + 1) % len(self.selectedImages)
        self.generalFunctions.sendToConsole("Next!")
        self.setImageText()
        self.showImage()

        current_img, current_label, current_pred, current_acc = self.modelTrainer.showPrediction(self.dataLoader, self.current_image, self.loadedModel)

        labelToLetter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
        
        predletter = labelToLetter[current_pred.item()]
        actualLetter = labelToLetter[current_label.item()]

        self.ui.predictionAccuracy.setText(f"Actual: {actualLetter}, Prediction: {predletter}, Accuracy: {current_acc}")
        self.ui.GuessText.setText(f"Top Guess: {predletter}")

        
    def prevImage(self):
        self.current_image = (self.current_image - 1) % len(self.selectedImages)
        self.generalFunctions.sendToConsole("Back!")
        self.setImageText()
        self.showImage()

        current_img, current_label, current_pred, current_acc = self.modelTrainer.showPrediction(self.dataLoader, self.current_image, self.loadedModel)

        labelToLetter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}

        
        predletter = labelToLetter[current_pred.item()]
        actualLetter = labelToLetter[current_label.item()]

        self.ui.predictionAccuracy.setText(f"Actual: {actualLetter}, Prediction: {predletter}, Accuracy: {current_acc}")
        self.ui.GuessText.setText(f"Top Guess: {predletter}")

        
    def setImageText(self):
        self.generalFunctions.sendToConsole("Set!")
        self.ui.CurrentImageText.setText(f'Image {self.current_image + 1} of {len(self.selectedImages)} - {os.path.basename(self.selectedImages[self.current_image])}')

    def showImage(self):
        self.generalFunctions.sendToConsole("Show!")
        pixmap = QPixmap(self.selectedImages[self.current_image])

        current_img, current_label, current_pred, current_acc = self.modelTrainer.showPrediction(self.dataLoader, self.current_image, self.loadedModel)

        labelToLetter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}

        
        predletter = labelToLetter[current_pred.item()]
        actualLetter = labelToLetter[current_label.item()]

        self.ui.predictionAccuracy.setText(f"Actual: {actualLetter}, Prediction: {predletter}, Accuracy: {current_acc.item()}")
        self.ui.GuessText.setText(f"Top Guess: {predletter}")

        self.ui.ImageToPredict.setPixmap(pixmap)
        self.ui.ImageToPredict.setAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
