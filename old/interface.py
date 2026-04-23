from PyQt5.QtWidgets import QWidget, QInputDialog, QPushButton, QLineEdit, QMessageBox, QVBoxLayout, QApplication, QProgressBar
from search_image import search_image
from PyQt5 import QtWidgets
import sys
from search_image import load_clip_model, load_and_encode_images
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess, self.tokenizer = load_clip_model()
        self.images, self.original_images = \
            load_and_encode_images(self.preprocess)

        self.setWindowTitle("Seismic Image Search")
        self.setGeometry(200, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText(
            "Enter the seismic image description...")
        self.layout.addWidget(self.input_box)

        self.button = QPushButton("Search", self)
        self.button.clicked.connect(self.run_image_search)
        self.layout.addWidget(self.button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)


    def run_image_search(self):
        caption = self.input_box.text().strip()

        if caption:
            # QApplication.processEvents()

            best_image = search_image(
                self.model, caption, self.images, self.original_images,
                self.tokenizer
            )
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)

            self.ax.imshow(best_image, cmap='gray')
            self.canvas.draw()
        else:
            QMessageBox.warning(
                self, "Input Error", "Please enter a search query.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
