import sys
import cv2
import numpy as np
import sqlite3
import concurrent.futures
from datetime import datetime
from anpr import ObjectDetectionProcessor
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
    QPushButton, QListWidget, QSizePolicy, QHBoxLayout, QGridLayout, QSpacerItem,
    QTabWidget
)
from PyQt5.QtCore import QTimer, pyqtSignal, Qt, QTime, QDate, QThread, QObject, QMutexLocker, QMutex

class DatabaseManager(QObject):
    save_plate_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect('plates.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS plates (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                plate_text TEXT,
                                timestamp TEXT)''')
        self.save_plate_signal.connect(self.save_plate_to_database)

    def save_plate_to_database(self, plate_text):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO plates (plate_text, timestamp) VALUES (?, ?)", (plate_text, current_datetime))
        self.conn.commit()

    def get_latest_plates(self, limit=50):
        try:
            self.cursor.execute(f"SELECT plate_text, timestamp FROM plates ORDER BY timestamp DESC LIMIT {limit}")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print("Error fetching plates from database:", e)
            return []

class VideoThread(QThread):
    most_confident_plate_signal = pyqtSignal(str, np.ndarray, np.ndarray)  # Added np.ndarray for plate image
    plate_screenshot_signal = pyqtSignal(np.ndarray)
    plate_detected_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor = ObjectDetectionProcessor()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
        self.target_size = (600, 400)
        self.skip_frames = 0
        self.detected_plates, self.current_plate = [], None
        self.no_detection_count, self.consecutive_no_detection_count = 0, 0
        self.max_consecutive_no_detection = 2
        self.stop_thread = False
        self.last_detected_plate = None
        self.is_plate_detected = False
        self.skip_frames_after_detection = 0
        self.mutex = QMutex()
        self.db_manager = DatabaseManager()
        self.plate_detected_signal.connect(self.db_manager.save_plate_signal)

    def run(self):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                while not self.stop_thread and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    if self.is_plate_detected:
                        self.skip_frames += 1
                        if self.skip_frames >= self.skip_frames_after_detection:
                            self.is_plate_detected = False
                            self.skip_frames = 0
                        continue

                    resized_frame = cv2.resize(frame, self.target_size)

                    # Perform object detection and OCR in parallel
                    future_detection = executor.submit(self.processor.detect_objects, resized_frame)
                    future_ocr = executor.submit(self.processor.apply_ocr_to_detection, resized_frame, future_detection.result())

                    detections = future_detection.result()
                    text, _, confidence, ocr_confidence, plate_image = future_ocr.result()  # Added plate_image

                    if text is not None:
                        if not self.is_plate_detected:
                            self.current_plate = text
                            if self.current_plate != self.last_detected_plate:
                                with QMutexLocker(self.mutex):
                                    self.detected_plates.append((self.current_plate, confidence, ocr_confidence, plate_image))  # Added plate_image
                                    self.no_detection_count, self.consecutive_no_detection_count = 0, 0
                                    self.current_screenshot = frame
                                    self.plate_screenshot_signal.emit(frame)
                                    self.is_plate_detected = True
                                    self.last_detected_plate = self.current_plate
                            else:
                                self.is_plate_detected = False
                    else:
                        self.no_detection_count += 1
                        self.consecutive_no_detection_count += 1

                        if self.consecutive_no_detection_count >= self.max_consecutive_no_detection:
                            if self.detected_plates:
                                with QMutexLocker(self.mutex):
                                    plate_confidences = [plate[1] for plate in self.detected_plates]
                                    ocr_confidences = [plate[2] for plate in self.detected_plates]
                                    new_confidences = [0.25 * conf + 0.75 * ocr_conf for conf, ocr_conf in
                                                       zip(plate_confidences, ocr_confidences)]

                                    above_median_plates = [(plate, new_conf) for plate, new_conf in
                                                           zip(self.detected_plates, new_confidences)
                                                           if new_conf > np.median(new_confidences)]

                                    if above_median_plates:
                                        most_confident_plate, _ = max(above_median_plates, key=lambda x: x[1])
                                        most_confident_plate_str = str(most_confident_plate[0]) if isinstance(
                                            most_confident_plate, tuple) else str(most_confident_plate)
                                        most_confident_plate_img = most_confident_plate[3]  # Added plate image
                                        self.most_confident_plate_signal.emit(most_confident_plate_str, self.current_screenshot, most_confident_plate_img)  # Added plate image
                                        self.plate_detected_signal.emit(most_confident_plate_str)
                                        self.detected_plates.clear()
                                        self.consecutive_no_detection_count = 0
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cap.release()

    def stop(self):
        self.stop_thread = True


class ANPRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread()
        self.initUI()
        self.populatePlatesList()  # Populate plates list with database data
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateDatabaseList)
        self.timer.start(5000)  # Update every 5 seconds

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
    background: #4d94ff;
    color: #FFFFFF;
    border: none;
    border-radius: 20px;
    padding: 10px 20px;
    margin: 5px;
}

QPushButton:hover {
    background: #1a75ff;
}

QListWidget {
    background: #f2f2f2;
    border: 1px solid #d9d9d9;
    color: #000000;
    border-radius: 10px;
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

        self.label_title = QLabel("PLAKALAR")
        self.label_title.setAlignment(Qt.AlignCenter)
        font_title = QFont("Arial", 24, QFont.Bold)
        self.label_title.setFont(font_title)
        side_layout.addWidget(self.label_title)

        # Create a tab widget
        tab_widget = QTabWidget()
        side_layout.addWidget(tab_widget)

        # Create two list widgets for real-time and database plates
        self.list_confident_output = QListWidget()
        self.list_confident_output.setStyleSheet(
            "QListWidget { background: #2E2E2E; border: 1px solid #404040; color: #FFFFFF; }"
        )
        self.list_confident_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        font_list = QFont("Arial", 16)
        self.list_confident_output.setFont(font_list)

        # Add real-time plates list to the first tab
        tab_widget.addTab(self.list_confident_output, "Real-time Plates")

        # Create a list widget for database plates
        self.list_database_plates = QListWidget()
        self.list_database_plates.setStyleSheet(
            "QListWidget { background: #2E2E2E; border: 1px solid #404040; color: #FFFFFF; }"
        )
        self.list_database_plates.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        font_list = QFont("Arial", 16)
        self.list_database_plates.setFont(font_list)

        # Add database plates list to the second tab
        tab_widget.addTab(self.list_database_plates, "Database Plates")

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

        layout.addLayout(side_layout, 0, 1)

        quit_button = QPushButton("Quit")
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

        current_time = QTime.currentTime().toString("hh:mm:ss")
        current_date = QDate.currentDate().toString(Qt.DefaultLocaleLongDate)

        plate_with_datetime = f"{current_date} {current_time} - PLAKA: {plate_text}"

        self.label_most_confident.setText(plate_with_datetime)
        self.list_confident_output.addItem(f" {plate_with_datetime}")

        q_image = QImage(screenshot.data, screenshot.shape[1], screenshot.shape[0], screenshot.strides[0],
                         QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.label_screenshot.setPixmap(pixmap.scaled(480, 360, Qt.KeepAspectRatio))

        plate_image = QImage(plate_img.data, plate_img.shape[1], plate_img.shape[0], plate_img.strides[0], QImage.Format_BGR888)
        plate_pixmap = QPixmap.fromImage(plate_image)
        self.label_plate_image.setPixmap(plate_pixmap.scaled(480, 360, Qt.KeepAspectRatio))

    def populatePlatesList(self):
        plates_data = self.video_thread.db_manager.get_latest_plates()
        for plate_data in plates_data:
            plate_text, timestamp = plate_data
            item_text = f"{timestamp} - PLAKA: {plate_text}"
            self.list_database_plates.addItem(item_text)

    def updateDatabaseList(self):
        self.list_database_plates.clear()
        self.populatePlatesList()

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