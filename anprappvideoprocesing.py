import cv2
import numpy as np
import concurrent.futures
from anpr import ObjectDetectionProcessor
from databaseanpr import DatabaseManager
from PyQt5.QtCore import QThread, pyqtSignal, QMutexLocker, QMutex


class VideoThread(QThread):
    most_confident_plate_signal = pyqtSignal(str, np.ndarray, np.ndarray)  # Added np.ndarray for plate image
    plate_screenshot_signal = pyqtSignal(np.ndarray)
    plate_detected_signal = pyqtSignal(str)
    plate_in_out_signal = pyqtSignal(str)  # Signal to indicate IN or OUT button click

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
        self.plate_in_out_signal.connect(self.handle_in_out_button)

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
                    future_ocr = executor.submit(self.processor.apply_ocr_to_detection, resized_frame,
                                                 future_detection.result())

                    detections = future_detection.result()
                    text, _, confidence, ocr_confidence, plate_image = future_ocr.result()  # Added plate_image

                    if text is not None:
                        if not self.is_plate_detected:
                            self.current_plate = text
                            if self.current_plate != self.last_detected_plate:
                                with QMutexLocker(self.mutex):
                                    self.detected_plates.append((self.current_plate, confidence, ocr_confidence,
                                                                 plate_image))  # Added plate_image
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
                                        self.most_confident_plate_signal.emit(most_confident_plate_str,
                                                                              self.current_screenshot,
                                                                              most_confident_plate_img)  # Added plate image
                                        self.plate_detected_signal.emit(most_confident_plate_str)
                                        self.detected_plates.clear()
                                        self.consecutive_no_detection_count = 0
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cap.release()

    def stop(self):
        self.stop_thread = True

    def handle_in_out_button(self, button):
        # Update cars_inside flag based on button clicked
        self.db_manager.cars_inside = button == "IN"
