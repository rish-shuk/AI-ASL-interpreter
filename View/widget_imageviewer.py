# chatgpt

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

class ImageViewer(QWidget):
    def __init__(self, images):
        super().__init__()

        self.images = images
        self.current_image = 0

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
    
        next_button = QPushButton('Next', self)
        next_button.clicked.connect(self.next_image)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(next_button)

        self.show_image()

    def show_image(self):
        pixmap = QPixmap(self.images[self.current_image])
        self.label.setPixmap(pixmap)
        self.setWindowTitle(f'Image Viewer - {self.images[self.current_image]}')

    def next_image(self):
        self.current_image = (self.current_image + 1) % len(self.images)
        self.show_image()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     images = ['image1.png', 'image2.jpg', 'image3.png']
#     viewer = ImageViewer(images)
#     viewer.show()
#     sys.exit(app.exec_())