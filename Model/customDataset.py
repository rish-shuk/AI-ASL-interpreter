import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the label and pixel values for the image
        label = self.data.iloc[idx, 0]
        pixels = self.data.iloc[idx, 1:]

        # Reshape the pixel values into a 28x28 image
        img = torch.tensor(pixels.values.reshape(28, 28))

        return img, label


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a layout for the image widgets
        self.layout = QVBoxLayout()

        # Load the dataset
        dataset = MNISTDataset('sign_mnist_train.csv')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create a label widget for each image in the dataset and add it to the layout
        for i, (img, label) in enumerate(dataloader):
            # Convert the tensor to a numpy array
            numpy_array = img.numpy()

            # Create a QPixmap from the numpy array
            pixmap = QPixmap.fromImage(QImage(numpy_array, numpy_array.shape[1], numpy_array.shape[0], QImage.Format_Grayscale8))

            # Create a QLabel with the QPixmap and add it to the layout
            label_widget = QLabel()
            label_widget.setPixmap(pixmap)
            self.layout.addWidget(label_widget)

        # Create a central widget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(self.layout)

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)


if __name__ == '__main__':
    # Create the application and main window
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    # Run the application
    sys.exit(app.exec_())