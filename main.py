import os
import sys

from skimage import io, color
from skimage.transform import resize
from skimage.util import img_as_ubyte

from torchvision.transforms import transforms

from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QWidget,
    QFileDialog,
)
from PySide2.QtGui import QImage, QPixmap, QColor

import numpy as np

from net import ResImageNet

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = ResImageNet(device="cpu", try_to_train=True, image_dir="Images")
        self.image_in_arr = None
        self.image_out_arr = None

        self.setWindowTitle("ResImage")
        self.setMinimumSize(QSize(800, 500))

        menu_bar = self.menuBar()

        menu = menu_bar.addMenu("Menu")
        self.open_action = menu.addAction("Open file")
        self.open_action.triggered.connect(self.openFile)
        self.start_action = menu.addAction("Process the image")
        self.start_action.triggered.connect(self.processImage)
        self.save_action = menu.addAction("Save image to file")
        self.save_action.triggered.connect(self.saveImage)

        layout = QHBoxLayout()

        self.image_in = QLabel()
        self.image_in.setScaledContents(True)
        self.image_out = QLabel()
        self.image_out.setScaledContents(True)

        self.button = QPushButton()

        layout.addWidget(self.image_in)
        layout.addWidget(self.image_out)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
    
    def openFile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select image", os.getcwd(),
                                                "Image files (*.jpeg *.jpg *.jpe *.jfif *.png *.bmp *.gif)")
        if file_name != None and file_name != "":

            self.image_in.clear()
            self.image_out.clear()

            tmp_array = io.imread(file_name)
            w = tmp_array.shape[0]
            w = (w // 4) * 4
            h = tmp_array.shape[1]
            h = (h // 4) * 4
            tmp_array = resize(tmp_array, (w, h))

            # Gray to RGB conversion
            if len(tmp_array.shape) < 3:
                tmp_array = color.gray2rgb(tmp_array)
            # RGBA to RGB conversion
            if tmp_array.shape[2] > 3:
                tmp_array = color.rgba2rgb(tmp_array)

            image_in = img_as_ubyte(tmp_array.copy())
            tensor_transform = transforms.Compose([transforms.ToTensor()])
            self.image_in_arr = tensor_transform(tmp_array)

            height, width, _ = image_in.shape
            bytes_per_line = 3 * width
            qimage_in = QImage(image_in, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_in.setPixmap(QPixmap(qimage_in))

            tmp_pix = QPixmap(width, height)
            tmp_pix.fill(QColor(200, 200, 200))
            self.image_out.setPixmap(tmp_pix)

    def processImage(self):
        if self.image_in_arr != None:
            image_out = self.model(self.image_in_arr)

            image_out = image_out.squeeze().numpy()
            image_out = np.transpose(image_out, (1, 2, 0)).copy()
            self.image_out_arr = img_as_ubyte(image_out)

            height, width, _ = self.image_out_arr.shape
            bytes_per_line = 3 * width
            qimage_out = QImage(self.image_out_arr, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_out.setPixmap(QPixmap(qimage_out))

    def saveImage(self):
        dialog_filters = "JPEG (*.jpg *.jpeg *.jpe *.jfif);;PNG (*.png);;GIF (*.gif);;BMP (*.bmp)"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save at...", os.getcwd(),
                                                   dialog_filters)
        if file_name != None and file_name != "":
            io.imsave(file_name, self.image_out_arr)

def main():
    #gui = GUI()
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()