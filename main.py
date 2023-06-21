import sys
import View.ui_filter_dialogBox as ui_filter_dialogBox
import View.ui_testingImages_dialogBox as ui_testingImages_dialogBox
import View.ui_browseModel_dialogBox as ui_browseModel_dialogBox
import View.ui_cameraContainer_test as ui_cameraContainer_dialogBox
import ui.camera as cameraWidget

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import View.ui_functions as ui_functions

from View.ui_main_josefAdditions import Ui_MainWindow

from View.widget_imagegrid import ImageGrid

from datetime import datetime

import os
import copy

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()   
        self.ui.setupUi(self)
        self.ui_generalFunctions = ui_functions.GeneralFunctions(self.ui)
        self.settings = ui_functions.SaveLoadListsFunctions(self.ui)
        self.predictViewer = ui_functions.PredictViewerFunctions(self.ui)

        # restore any saved lists
        self.settings.restoreLists()

        self.connectSignalsSlots()
        self.show()
        
    def connectSignalsSlots(self):
        self.sideMenuButtons()
        self.dataSetButtons()
        self.predictButtons()
        self.TrainPage = ui_functions.TrainFunctions(self.ui) # IMPORTANT
        self.jeromesStuff()
       
    def sideMenuButtons(self):
         # Toggle menu and Page buttons
        self.ui.Btn_Toggle.clicked.connect(lambda: self.toggleMenu(250, True))
        self.ui.dataset_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.dataset_page))
        self.ui.train_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.train_page))
        self.ui.predict_btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.PredictPage))

    # need to refactor later
    def jeromesStuff(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.tabWidget.setCurrentIndex(0)

        self.importFunctions = ui_functions.ImportFunctions(self.ui)

        self.ui.D_ImportBtn.clicked.connect(lambda: self.importFunctions.importData())
        self.ui.cancel_btn.clicked.connect(lambda: self.importFunctions.stopImport())

        self.ui.TrainValidateSlider.valueChanged.connect(lambda value: self.TrainPage.updateBoxes(value))
        self.ui.TrainSpinBox.valueChanged.connect(lambda value: self.TrainPage.updateTrainBoxValue(value))
        self.ui.ValidateSpinBox.valueChanged.connect(lambda value: self.TrainPage.updateValidateBoxValue(value))
    
    def toggleMenu(self, maxWidth, enable):
        if enable:

            # GET WIDTH
            width = self.ui.frame_left_menu.width()
            maxExtend = maxWidth
            standard = 70

            # SET MAX WIDTH
            if width == 70:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # ANIMATION
            self.animation = QPropertyAnimation(self.ui.frame_left_menu, b"minimumWidth")
            self.animation.setDuration(400)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QEasingCurve.InOutQuart)
            self.animation.start()

    # UI FUNCTION TESTING - WILL REFACTOR LATER AND PUT INTO UI_FUNCTIONS
    # Button functionailities
    def dataSetButtons(self):
        # filter tool box - opens dialog box 
        self.ui.FilterToolBtn.clicked.connect(self.onFilterToolBoxClicked)

    def predictButtons(self):
        # testing images btn - opens dialog box
        self.ui.TestingImagesBtn.clicked.connect(self.onTestingImagesClicked)
        self.ui.BrowseModelsBtn.clicked.connect(self.onBrowseModelsCLicked)
        self.ui.OpenWebcamBtn.clicked.connect(self.onOpenWebCamClicked)
        
        self.ui.NextImageButton.clicked.connect(self.predictViewer.nextImage)
        self.ui.PrevImageBtn.clicked.connect(self.predictViewer.prevImage)


    def onFilterToolBoxClicked(self):
        dlg = FilterToolBoxDlg(self.ui)
        self.ui_generalFunctions.sendToConsole("Filter Tool Dialog Box Opened!")
        dlg.exec()
        self.ui_generalFunctions.sendToConsole("Filter Tool Dialog box closed.")

    def onTestingImagesClicked(self):
        dlg = TestingImagesBoxDlg(self.ui, self.predictViewer)
        self.ui_generalFunctions.sendToConsole("Testing Images Dialog Box Opened! Please wait.")
        dlg.exec()
        self.ui_generalFunctions.sendToConsole("Testing Images Dialog box closed.")
    
    def onBrowseModelsCLicked(self):
        dlg = BrowseModelsBoxDlg(self.ui)
        self.ui_generalFunctions.sendToConsole("Load Models Dialog Box Opened!")
        dlg.loadModelFunctions.initialiseModelList()
        dlg.exec()
        self.ui_generalFunctions.sendToConsole("Load Models Dialog box closed.")
    
    def onOpenWebCamClicked(self):
        dlg = CameraBoxDlg(self.ui)
        self.ui_generalFunctions.sendToConsole("Webcam Dialog Box Opened!")
        dlg.exec()
        self.ui_generalFunctions.sendToConsole("Webcam Dialog box closed.")



class FilterToolBoxDlg(QDialog):
    def __init__(self, mainUi, parent=None):
        super().__init__(parent)
        self.ui = ui_filter_dialogBox.Ui_Dialog()
        self.ui.setupUi(self)

class TestingImagesBoxDlg(QDialog):
    def __init__(self, mainUi: Ui_MainWindow, predictViewer, parent=None):
        super().__init__(parent)
        self.ui_dialog = ui_testingImages_dialogBox.Ui_Dialog()
        self.ui_dialog.setupUi(self)
        self.ui_main = mainUi
        self.generalFunctions = ui_functions.GeneralFunctions(self.ui_main)
        self.testingImagesFunctions = ui_functions.TestingImagesFunctions(self.ui_main, self.ui_dialog)
        self.predictViewer = predictViewer

        self.cameraImageGrid = None
        self.testingImageGrid = None

        self.initialiseAllImageGrids()
        self.connectSignalSlots()

    def connectSignalSlots(self):
        self.ui_dialog.clearSelectedBtn.clicked.connect(self.clearAll)
        self.ui_dialog.ButtonBox.accepted.connect(self.addSelectedImages)
        #self.ui_dialog.ButtonBox.rejected.connect(self.removeSelectedImages)
    
    def addSelectedImages(self):
        allSelectedPaths = []

        for imagePath in self.testingImageGrid.selected_image_paths:
            allSelectedPaths.append(imagePath)

        for imagePath in self.cameraImageGrid.selected_image_paths:
            allSelectedPaths.append(imagePath)

        self.generalFunctions.selectedTestingImages = copy.copy(allSelectedPaths)
        self.predictViewer.initialiseImageViewer(allSelectedPaths)

        print(f"gone thru!")

    def clearAll(self):
        self.cameraImageGrid.clear_selected_images()
        self.testingImageGrid.clear_selected_images()
    
    def initialiseAllImageGrids(self):
        self.initialiseTestingImagesGrid()
        self.initialiseCameraImagesGrid()

    def initialiseCameraImagesGrid(self):
        image_paths = self.createImagePathsListFromCamera()

        if image_paths == None:
            ui_functions.GeneralFunctions(self.ui_main).sendToConsole("No images found!")
            return

        self.cameraImageGrid = ImageGrid(image_paths)
        self.ui_dialog.cameraImagesVBox.addWidget(self.cameraImageGrid)
        self.cameraImageGrid.show()

    def initialiseTestingImagesGrid(self):
        testImagesPath = self.generalFunctions.retrieveSelectedTestImagePath()

        if testImagesPath == None:
            ui_functions.GeneralFunctions(self.ui_main).sendToConsole("No images found!")
            return
        
        image_paths = self.createImagePathsListFromDataset(testImagesPath)

        # self.images = ['image1.png', 'image2.jpg', 'image3.png']
        self.testingImageGrid = ImageGrid(image_paths)
        self.ui_dialog.datasetSAVbox.addWidget(self.testingImageGrid)
        self.testingImageGrid.show()
    
    def createImagePathsListFromDataset(self, pathToImageFolder):
        listOfImagePaths = []

        imgNumber = 1

        while (os.path.exists(pathToImageFolder + f"/image_{imgNumber}_test.jpg") or os.path.exists(pathToImageFolder + f"/image_{imgNumber}_test.png")):
            row = []
            for j in range(10):
                imagePath = pathToImageFolder + f"/image_{imgNumber}_test"
                row.append(imagePath)
                imgNumber = imgNumber + 1
            listOfImagePaths.append(row)

        return listOfImagePaths
    
    # no parameter because fixed path
    def createImagePathsListFromCamera(self):
        listOfImagePaths = []
        fixedCameraImagesFolderPath = os.getcwd() + "/ui"

        imgNumber = 1
        while (os.path.exists(fixedCameraImagesFolderPath + f"/image_{imgNumber}_test.jpg")):
            row = []
            for j in range(10):
                imagePath = fixedCameraImagesFolderPath + f"/image_{imgNumber}_test"
                row.append(imagePath)
                imgNumber = imgNumber + 1
            listOfImagePaths.append(row)

        return listOfImagePaths   

class BrowseModelsBoxDlg(QDialog):
    def __init__(self, mainUi, parent=None):
        super().__init__(parent)
        self.ui = ui_browseModel_dialogBox.Ui_Dialog()
        self.ui.setupUi(self)
        self.loadModelFunctions = ui_functions.LoadModelsFunctions(mainUi, self.ui)
    
class CameraBoxDlg(QDialog):
    def __init__(self, mainUI, parent=None):
        super().__init__(parent)
        self.ui = ui_cameraContainer_dialogBox.Ui_Dialog()
        self.ui.setupUi(self)

        cWidget = (cameraWidget.Camera())
        self.ui.verticalLayout.addWidget(cWidget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())