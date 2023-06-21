# importing required libraries
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import time
import platform

# Main window class

#Code is inspired and sourced from:
# https://www.geeksforgeeks.org/creating-a-camera-application-using-pyqt5/
# https://www.codepile.net/pile/ey9KAnxn
class Camera(QMainWindow):
    def __init__(self):
        super(Camera, self).__init__()

        # setting geometry
        self.setGeometry(100, 100,800, 600)
        self.setStyleSheet("background : lightgrey;")
        self.VBox = QVBoxLayout()

        # getting available cameras
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            # exit the code
            sys.exit()

        # creating a status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background : white;")
        self.setStatusBar(self.status)

        # path to save
        self.save_path = "/images"

        if (platform.system() == "Windows"):
            savePath = "/images"
            if (not os.path.exists(os.getcwd() + savePath)):
                os.makedirs(os.getcwd() + savePath)

            self.save_path = os.getcwd() + savePath

        # creating a QCameraViewfinder object
        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.VBox.addWidget(self.viewfinder)
        self.select_camera(0)
        
        toolbar = QToolBar("Camera Tool Bar")
        self.addToolBar(toolbar)

        # creating a photo action to take photo
        self.captureButton = QPushButton("Capture Photo")
        self.captureButton.clicked.connect(self.click_photo)
        self.VBox.addWidget(self.captureButton)
        self.captureButton.setStatusTip("This will capture picture")

        # creating action for viewing images
        self.viewImages = QPushButton("View Images")
        self.viewImages.clicked.connect(self.imageExplorer)
        self.VBox.addWidget(self.viewImages)
        self.viewImages.setStatusTip("Can view images taken in file and make changes on deleting/keeping.")

        self.closeButton = QPushButton("Save and close")
        self.closeButton.clicked.connect(self.closeWindow)
        self.VBox.addWidget(self.closeButton)

         # creating a combo box for selecting camera
        camera_selector = QComboBox()
        camera_selector.setStatusTip("Choose camera to take pictures")
        camera_selector.setToolTip("Select Camera")
        camera_selector.setToolTipDuration(2500)
        camera_selector.addItems([camera.description() for camera in self.available_cameras])
        camera_selector.currentIndexChanged.connect(self.select_camera)
        toolbar.addWidget(camera_selector)
        toolbar.setStyleSheet("background : white;")

        # setting window title
        self.setWindowTitle("Camera")
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(self.VBox)

        # method to select camera
    def select_camera(self, i):

        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

        # start the camera
        self.camera.start()
        self.capture = QCameraImageCapture(self.camera)

        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg))
        self.capture.imageCaptured.connect(lambda d,i: self.status.showMessage("Image captured : "+ str(self.save_seq)))
        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0

    # method to take photo
    def click_photo(self):

        # time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        self.capture.capture(os.path.join(self.save_path,"%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        )))
        self.save_seq += 1
        self.status.showMessage("Picture taken")

    # change folder method
    def imageExplorer(self):
        # open the dialog to select path
        QFileDialog.getOpenFileName(self, "Picture Location", filter="JPEG(*.jpg *.jpeg)")

    
    def closeWindow(self):
        self.camera.stop()
        self.close()

    def alert(self, msg):
        error = QErrorMessage(self)
        error.showMessage(msg)

# Driver code
if __name__ == "__main__" :
    App = QApplication(sys.argv)
    window = Camera()
    window.show()
    sys.exit(App.exec())
