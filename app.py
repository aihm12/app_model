from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageStat
import pytesseract
import base64
import io
import cv2
import numpy as np
import gdown

app = Flask(__name__)

# تحميل النموذج من Google Drive
MODEL_FILE = "yolov8_license_plate.pt"
GOOGLE_DRIVE_FILE_ID = "1ewShjHYyro3adOU5IATMd1KQF8HdW_oD"

def download_model():
    """تحميل النموذج من Google Drive"""
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# تحميل النموذج عند بدء التشغيل
download_model()
model = YOLO(MODEL_FILE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # تشغيل الموديل والحصول على النتائج
        results = model.predict(image)
        
        # استخراج أول صندوق للوحة السيارة
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return jsonify({'error': 'No license plate detected'}), 400

        x_min, y_min, x_max, y_max = map(int, boxes[0])
        cropped_plate = image.crop((x_min, y_min, x_max, y_max))

        # فحص اللون الغالب
        dominant_color = ImageStat.Stat(cropped_plate).mean
        plate_type = "غير معروف"
        if dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
            plate_type = "نقل"
        elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
            plate_type = "أجرة"
        elif dominant_color[2] > dominant_color[0] and dominant_color[2] > dominant_color[1]:
            plate_type = "خصوصي"

        # استخراج النصوص باستخدام OCR
        plate_number = pytesseract.image_to_string(cropped_plate, lang="eng", config="--psm 7").strip()

        response = {
            'coordinates': [x_min, y_min, x_max, y_max],
            'plate_number': plate_number,
            'plate_type': plate_type
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
