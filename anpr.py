import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow as tf
import easyocr
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Model ve etiket haritası konfigürasyonları
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'
DETECTION_THRESHOLD = 0.2
REGION_THRESHOLD = 0.1
OCR_CONFIDENCE_THRESHOLD = 0.2

# Dosya yolları ve konfigürasyonlar
ANNOTATION_PATH = os.path.join('Tensorflow', 'workspace', 'annotations')
CHECKPOINT_PATH = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME)
PIPELINE_CONFIG = os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config')
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)

# Nesne Tespiti İşleme sınıfı
class ObjectDetectionProcessor:

    def __init__(self):
        # TensorFlow konfigürasyonlarını ayarla
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # Paralel işleme için ThreadPoolExecutor'ı başlat
        self.executor = ThreadPoolExecutor(max_workers=8)
        configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        ckpt = tf.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-28')).expect_partial()
        self.reader = easyocr.Reader(['en', 'tr'], detector='dbnet18')

    def detect_objects(self, image_np):
        # Giriş resmi üzerinde nesne tespiti yap
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return detections

    def detect_fn(self, image):
        # Model kullanarak nesne tespiti yap
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def apply_ocr_to_detection(self, image_np_with_detections, detections):
        # Resimdeki tespit edilen nesnelere OCR uygula
        scores = list(filter(lambda x: x > DETECTION_THRESHOLD, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]

        if len(boxes) == 0:
            print("NO DETECTION")
            return None, None, None, None, None  # Tespit yoksa OCR güvenliğini None olarak döndür

        box = boxes[0]
        width, height = image_np_with_detections.shape[1], image_np_with_detections.shape[0]
        roi = box * [height, width, height, width]
        car_region = image_np_with_detections[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])].copy()
        cv2.rectangle(image_np_with_detections, (int(roi[1]), int(roi[0])), (int(roi[3]), int(roi[2])), (0, 255, 0), 2)
        confidence = scores[0]

        # Önceden yüklenmiş EasyOCR modelini kullanarak OCR gerçekleştir
        ocr_future = self.executor.submit(self.reader.readtext, car_region, batch_size=5)
        ocr_result = ocr_future.result()
        confident_ocr_results = [result for result in ocr_result if result[2] > OCR_CONFIDENCE_THRESHOLD]
        text = self.filter_text(car_region, confident_ocr_results, REGION_THRESHOLD)

        if confident_ocr_results:
            ocr_confidence = confident_ocr_results[0][2]
            if ocr_confidence > 0.1:
                display_text = f"OCR Sonucu: {text}, Confidence: {confidence:.2f}, OCR Confidence: {ocr_confidence:.2f}"
                cv2.putText(image_np_with_detections, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                print(display_text)
                return text, car_region, confidence, ocr_confidence, car_region  # Return the plate image as well

        return None, None, None, None, None

    def filter_text(self, region, ocr_result, region_threshold):
        # Bölge boyutu eşiği baz alınarak OCR sonuçlarını filtrele
        rectangle_size = region.shape[0] * region.shape[1]
        plate = []

        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))

            if length * height / rectangle_size > region_threshold:
                plate.append(result[1])

        return plate

""""# OCR sonuçlarını Excel'e kaydetme fonksiyonu
def save_to_excel(plate):
    # Dosya varsa kontrol et, DataFrame oluştur ve Excel'e kaydet
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Plate'])
    new_row = pd.DataFrame({'Plate': [plate]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_file_path, index=False)
    print(f"Sonuçlar {excel_file_path} adresine kaydedildi.")"""





"""def format_license_plate(self, text, mapping=None):
        # Apply character mapping if provided
        if mapping:
            text = ''.join([mapping[char] if char in mapping else char for char in text])

        first_part = ''
        second_part = ''

        # Find the index of the first letter
        letter_index = next((i for i, char in enumerate(text) if char.isalpha()), None)

        if letter_index is not None:
            first_part = str(int(text[:letter_index]))  # Convert the numeric part to a string
            remaining_text = text[letter_index:]  # Extract the remaining part with letters

            # Split the remaining text into letters and numeric parts
            letter_parts = ''.join(filter(str.isalpha, remaining_text))
            numeric_parts = ''.join(filter(str.isdigit, remaining_text))

            # Apply character mapping to letter parts if provided
            if mapping:
                letter_parts = ''.join([mapping[char] if char in mapping else char for char in letter_parts])

            # Combine the letter and numeric parts
            second_part = f"{letter_parts} {numeric_parts}" if letter_parts else int(numeric_parts)

        # Add formatting logic based on the specified patterns
        if len(first_part) == 2 and len(str(second_part)) == 4:
            return f"{first_part} X {second_part}"
        elif len(first_part) == 2 and len(str(second_part)) == 5:
            return f"{first_part} XX {second_part}"
        elif len(first_part) == 2 and len(str(second_part)) == 3:
            return f"{first_part} XXX {second_part}"
        else:
            return text"""







"""    def apply_tesseract_ocr(self, image_np_with_detections, detections):
        with self.lock:
            scores = list(filter(lambda x: x > DETECTION_THRESHOLD, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]

            if len(boxes) == 0:
                print("No detections found.")
                return None, None

            box = boxes[0]
            width, height = image_np_with_detections.shape[1], image_np_with_detections.shape[0]
            roi = box * [height, width, height, width]
            car_region = image_np_with_detections[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])].copy()
            cv2.rectangle(image_np_with_detections, (int(roi[1]), int(roi[0])), (int(roi[3]), int(roi[2])), (0, 255, 0),
                          2)
            confidence = scores[0]

            # Apply Tesseract OCR with additional configurations and processing steps
            text = self.apply_tesseract_with_config(car_region)

            if text:
                display_text = f"Tesseract OCR Result: {text}, Confidence: {confidence:.2f}"
                cv2.putText(image_np_with_detections, display_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                print(display_text)
                return text, car_region

            return None, None

    def apply_tesseract_with_config(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply additional processing or filtering steps if needed
        # Example: You can apply image morphology, denoising, etc., before using pytesseract

        # pytesseract configuration with additional options
        config_options = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        text = pytesseract.image_to_string(thresh, config=config_options)
        return text """