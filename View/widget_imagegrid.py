# chatgpt

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageGrid(QWidget):
    def __init__(self, image_paths):
        super().__init__()
        self.selected_widgets = []
        self.selected_image_paths = []
        self.Imagelayout = QGridLayout()
        self.image_paths = image_paths
        imgNumber = 1
        for row in range(len(image_paths)):
            for col in range(len(image_paths[row])):
                pixmap = QPixmap(image_paths[row][col])
                label = QLabel()
                label.setPixmap(pixmap) # pixmap contains filepath
                label.setObjectName(f"image_train_{imgNumber}")
                #label.setFixedSize(50,35)
                label.setAlignment(Qt.AlignCenter)
                label.mousePressEvent = lambda event, r=row, c=col: self.select_image(r, c)
                self.Imagelayout.addWidget(label, row, col)
                imgNumber = imgNumber + 1

        self.setLayout(self.Imagelayout)
        self.setWindowTitle('Image Grid')

    def select_image(self, row, col):
        label = self.layout().itemAtPosition(row, col).widget()
        self.selected_image_paths.append(self.image_paths[row][col])

        if label in self.selected_widgets:
            self.selected_widgets.remove(label)
            label.setStyleSheet('')
        else:
            self.selected_widgets.append(label)
            label.setStyleSheet('border: 2px solid red;')
    
    def clear_selected_images(self):
        for row in range(len(self.image_paths)):
            for col in range(len(self.image_paths[row])):
                self.selected_image_paths.remove(self.image_paths[row][col])

                label = self.layout().itemAtPosition(row, col).widget()
                if label in self.selected_widgets:
                    self.selected_widgets.remove(label)
                    label.setStyleSheet('')
                

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     image_paths = [['image1.png', 'image2.jpg', 'image3.png'],
#                    ['image4.jpg', 'image5.jpg', 'image6.jpg'],
#                    ['image7.jpg', 'image8.jpg', 'image9.jpg']]
#     window = ImageGrid(image_paths)
#     window.show()
#     sys.exit(app.exec_())