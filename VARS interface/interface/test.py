"""
A class where 90 percent of the screen starting from the left are used to show a video and the 10 percent on the right shows a text about the video by using pyqt5
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Player')
        self.setGeometry(300, 300, 600, 400)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.red)
        painter.setBrush(Qt.red)
        painter.drawRect(0, 0, 540, 400)

class TextWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text')
        self.setGeometry(300, 300, 600, 400)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.red)
        painter.setBrush(Qt.red)
        painter.drawRect(540, 0, 60, 400)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video = VideoWindow()
    text = TextWindow()
    sys.exit(app.exec_())


