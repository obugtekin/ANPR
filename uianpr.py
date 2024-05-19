import os
import re
import sys
from datetime import datetime
import cv2
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
    QPushButton, QListWidget, QSizePolicy, QHBoxLayout, QGridLayout, QSpacerItem,
    QTabWidget, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt, QTime, QDate
from databaseanpr import DatabaseManager
from anprappvideoprocesing import VideoThread


class ANPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread()
        self.initUI()
        self.populatePlatesList()  # Populate plates list with database data
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateDatabaseList)
        self.timer.start(1000)  # Update every 5 seconds

    def initUI(self):
        self.showFullScreen()

        dark_theme = """
        QMainWindow {
            background: #f2f2f2;
            color: #000000;
            border: 2px solid #d9d9d9;
            border-radius: 10px;
        }

        QPushButton {
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            width: 100px;  /* Adjust width */
            height: 100px;  /* Adjust height */
            margin: 5px;
        }

        QPushButton#in_button,
        QPushButton#out_button {
            width: 80px;  /* Adjust width */
            height: 80px;  /* Adjust height */
        }

        QPushButton#in_button {
            background: #4CAF50; /* Green */
        }


        QPushButton#out_button {
            background: #F44336; /* Red */
        }


        QPushButton:hover {
            opacity: 0.8;
        }

                QTabWidget::pane {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 white, stop:1 lightgreen);
                    border: 2px solid #d9d9d9;
                    border-radius: 10px;
                }

                QTabBar::tab {
                    background-color: #f9f9f9;
                    border: 1px solid #d9d9d9;
                    border-bottom-color: none;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                    min-width: 100px;
                    padding: 10px;
                    margin-top: 10px;
                    margin-bottom: -1px;
                }

                QTabBar::tab:selected {
                    background-color: lightgreen;
                    border: 2px solid #d9d9d9;
                    border-bottom-color: none;
                    border-top-left-radius: 10px;
                    border-top-right-radius: 10px;
                }

        QListWidget {
            background: #f9f9f9; /* Lighter gray background */
            border: 1px solid #d9d9d9;
            color: #000000;
            border-radius: 10px;
        }
        QPushButton#quit_button { /* Added styling for the Quit button */
            background: #607D8B; /* Grey-blue */
            width: 80px;  /* Adjust width */
            height: 80px;  /* Adjust height */
        }

        QLabel {
            color: #000000;
            border-radius: 10px; /* Add border-radius for rounding edges */
            border: 2px solid #d9d9d9; /* Add border for better visibility */
        }
        """
        self.setStyleSheet(dark_theme)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QGridLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        self.label = QLabel()
        self.label.setMinimumSize(960, 720)
        layout.addWidget(self.label, 0, 0)

        side_layout = QVBoxLayout()

        self.label_title = QLabel("PARS")
        self.label_title.setAlignment(Qt.AlignCenter)
        font_title = QFont("Arial", 24, QFont.Bold)
        self.label_title.setFont(font_title)
        side_layout.addWidget(self.label_title)

        # Create a tab widget
        tab_widget = QTabWidget()
        side_layout.addWidget(tab_widget)

        # Create a list widget for CARSIN
        self.list_carsin = QListWidget()
        self.list_carsin.setStyleSheet(
            "QListWidget { background: #f2f2f2; border: none; color: #000000; }"
        )
        self.list_carsin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        font_list = QFont("Arial", 16)
        self.list_carsin.setFont(font_list)

        # Add CARSIN list to the first tab
        tab_widget.addTab(self.list_carsin, "CARSIN")

        # Create a list widget for CARS INSIDE
        self.list_carsinside = QListWidget()
        self.list_carsinside.setStyleSheet(
            "QListWidget { background: #f2f2f2; border: none; color: #000000; }"
        )
        self.list_carsinside.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        font_list = QFont("Arial", 16)
        self.list_carsinside.setFont(font_list)

        # Add CARS INSIDE list to the second tab
        tab_widget.addTab(self.list_carsinside, "CARS INSIDE")

        # Create a list widget for CARSOUT
        self.list_carsout = QListWidget()
        self.list_carsout.setStyleSheet(
            "QListWidget { background: #f2f2f2; border: none; color: #000000; }"
        )
        self.list_carsout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        font_list = QFont("Arial", 16)
        self.list_carsout.setFont(font_list)

        # Add CARSOUT list to the third tab
        tab_widget.addTab(self.list_carsout, "CARSOUT")

        hbox = QHBoxLayout()
        spacer_left = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hbox.addItem(spacer_left)
        self.label_screenshot = QLabel()
        hbox.addWidget(self.label_screenshot)
        spacer_right = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hbox.addItem(spacer_right)
        side_layout.addLayout(hbox)

        self.label_plate_image = QLabel()
        self.label_plate_image.setAlignment(Qt.AlignCenter)  # Align the plate image to the center
        side_layout.addWidget(self.label_plate_image)

        self.label_most_confident = QLabel()
        self.label_most_confident.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        side_layout.addWidget(self.label_most_confident)

        # Add "IN" and "OUT" buttons side by side
        buttons_layout = QHBoxLayout()
        in_button = QPushButton("IN")
        in_button.setObjectName("in_button")
        in_button.clicked.connect(self.handleInButton)
        buttons_layout.addWidget(in_button)

        out_button = QPushButton("OUT")
        out_button.setObjectName("out_button")
        out_button.clicked.connect(self.handleOutButton)
        buttons_layout.addWidget(out_button)

        side_layout.addLayout(buttons_layout)

        self.list_carsin.itemClicked.connect(self.showImage)
        self.list_carsinside.itemClicked.connect(self.showImage)
        self.list_carsout.itemClicked.connect(self.showImage)

        layout.addLayout(side_layout, 0, 1)

        quit_button = QPushButton("Quit")
        quit_button.setObjectName("quit_button")  # Set object name for the Quit button
        exit_icon_path = "C:/Users/Onur/Desktop/ANPR/exit_icon.png"
        quit_button.setIcon(QIcon(exit_icon_path))
        quit_button.clicked.connect(self.quitApp)
        layout.addWidget(quit_button, 1, 1, Qt.AlignRight | Qt.AlignBottom)

        central_widget.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

        self.video_thread.most_confident_plate_signal.connect(self.updateMostConfidentPlate)

        # Start video processing thread
        self.video_thread.start()

    def handleInButton(self):
        self.video_thread.plate_in_out_signal.emit("IN")
        # Change background color temporarily
        self.highlightButton("in_button")

    def handleOutButton(self):
        self.video_thread.plate_in_out_signal.emit("OUT")
        # Change background color temporarily
        self.highlightButton("out_button")

    def highlightButton(self, button_name):
        button = self.findChild(QPushButton, button_name)
        original_style_sheet = button.styleSheet()
        highlighted_style_sheet = original_style_sheet + "border: 2px solid yellow;"
        button.setStyleSheet(highlighted_style_sheet)
        QTimer.singleShot(100, lambda: button.setStyleSheet(original_style_sheet))

    def updateFrame(self):
        ret, frame = self.video_thread.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (960, 720))
        height, width, channel = resized_frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(resized_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)



    def updateMostConfidentPlate(self, plate, screenshot, plate_img):
        font = QFont("Arial", 18)
        self.label_most_confident.setFont(font)

        plate_text = "".join(c for c in plate if c.isalnum())

        if any(c.islower() for c in plate_text):
            plate_text = plate_text.upper()

        current_date_time = datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        plate_with_datetime = f"{current_date_time} - PLAKA: {plate_text}"

        self.label_most_confident.setText(plate_with_datetime)
        self.list_carsin.addItem(f" {plate_with_datetime}")

        q_image = QImage(screenshot.data, screenshot.shape[1], screenshot.shape[0], screenshot.strides[0],
                         QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.label_screenshot.setPixmap(pixmap.scaled(480, 360, Qt.KeepAspectRatio))

        plate_image = QImage(plate_img.data, plate_img.shape[1], plate_img.shape[0], plate_img.strides[0],
                             QImage.Format_BGR888)
        plate_pixmap = QPixmap.fromImage(plate_image)
        self.label_plate_image.setPixmap(plate_pixmap.scaled(480, 360, Qt.KeepAspectRatio))

        # Save plate image and screenshot image together
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plate_text_clean = "".join(c for c in plate_text if c.isalnum())
        folder_name = f"{timestamp}_{plate_text_clean}"
        folder_path = os.path.join("images", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        plate_image_path = os.path.join(folder_path, "plate_image.png")
        screenshot_image_path = os.path.join(folder_path, "screenshot.png")

        # Save images to the folder
        plate_pixmap.save(plate_image_path)
        pixmap.save(screenshot_image_path)

    def showImage(self, item):
        # Retrieve the text of the clicked item
        plate_text = item.text()

        # Extract the plate text using regular expressions
        plate_match = re.search(r'PLAKA: (\w+)', plate_text)
        if plate_match:
            plate_text = plate_match.group(1).strip()
        else:
            QMessageBox.critical(self, "Error", "Plate text not found.")
            return

        # Iterate through the folders in the images directory
        images_dir = "images"
        for folder_name in os.listdir(images_dir):
            # Check if the folder name contains the plate text
            if plate_text in folder_name:
                folder_path = os.path.join(images_dir, folder_name)
                plate_image_path = os.path.join(folder_path, "plate_image.png")
                plate_pixmap = QPixmap(plate_image_path)
                self.label_plate_image.setPixmap(plate_pixmap)
                screenshot_image_path = os.path.join(folder_path, "screenshot.png")
                screenshot_pixmap = QPixmap(screenshot_image_path)
                self.label_screenshot.setPixmap(screenshot_pixmap)
                break
        else:
            QMessageBox.critical(self, "Error", "Images not found for the selected plate.")

    def populatePlatesList(self):
        tables = ["CARSIN", "CARSINSIDE", "CARSOUT"]
        for table_name in tables:
            plates_data = self.video_thread.db_manager.get_latest_plates(table_name)
            list_widget = getattr(self, f"list_{table_name.lower().replace(' ', '_')}")
            for plate_data in plates_data:
                plate_text, timestamp = plate_data
                item_text = f"{timestamp} - PLAKA: {plate_text}"
                list_widget.addItem(item_text)

    def updateDatabaseList(self):
        tables = ["CARSIN", "CARSINSIDE", "CARSOUT"]
        for table_name in tables:
            list_widget = getattr(self, f"list_{table_name.lower().replace(' ', '_')}")
            list_widget.clear()
            plates_data = self.video_thread.db_manager.get_latest_plates(table_name)
            for plate_data in plates_data:
                plate_text, timestamp = plate_data
                item_text = f"{timestamp} - PLAKA: {plate_text}"
                list_widget.addItem(item_text)

    def quitApp(self):
        self.video_thread.stop()
        self.video_thread.wait()  # Wait for the thread to finish before exiting
        self.timer.stop()
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    mainWindow = ANPRApp()
    # Show the main window
    mainWindow.show()

    sys.exit(app.exec_())
