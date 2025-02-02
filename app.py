import time
import json
import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageStat
import pytesseract
import os

app = Flask(__name__)

# 🟢 **تحديد مسار النموذج داخل المستودع**
MODEL_FILE = "yolov8_license_plate.pt"

# **تأكد من أن النموذج موجود**
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ الملف {MODEL_FILE} غير موجود. تأكد من رفع النموذج إلى GitHub.")

# تحميل النموذج
print("✅ تحميل النموذج...")
model = YOLO(MODEL_FILE)

def process_image(image):
    """تحليل الصورة واستخراج أرقام اللوحة"""
    try:
        results = model.predict(image)
        
        # استخراج أول صندوق (إن وجد)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None, "❌ لم يتم العثور على لوحة رقمية"

        x_min, y_min, x_max, y_max = map(int, boxes[0])
        cropped_plate = image.crop((x_min, y_min, x_max, y_max))

        # تحليل اللون الغالب
        dominant_color = ImageStat.Stat(cropped_plate).mean
        plate_type = "غير معروف"
        if dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
            plate_type = "نقل"
        elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
            plate_type = "أجرة"
        elif dominant_color[2] > dominant_color[0] and dominant_color[2] > dominant_color[1]:
            plate_type = "خصوصي"

        # تنفيذ OCR لاستخراج النص
        plate_number = pytesseract.image_to_string(
            cropped_plate, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()

        return {
            "coordinates": [x_min, y_min, x_max, y_max],
            "plate_number": plate_number,
            "plate_type": plate_type
        }, None
    except Exception as e:
        return None, str(e)

@app.route('/predict', methods=['POST'])
def predict():
    """API لاستقبال الصور وتحليلها"""
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({"error": "❌ لم يتم إرسال صورة"}), 400

        # فك تشفير الصورة من Base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # معالجة الصورة والحصول على النتائج
        response, error = process_image(image)
        if error:
            return jsonify({"error": error}), 400

        # حفظ النتائج في ملف `result.json`
        with open("result.json", "w") as f:
            json.dump(response, f)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🛑 تشغيل التطبيق مؤقتًا ثم الإغلاق تلقائيًا بعد 10 ثوانٍ
if __name__ == '__main__':
    print("🚀 تشغيل النموذج...")
    app.run(debug=False, host='0.0.0.0', port=8000)
    time.sleep(10)  # تشغيل السيرفر لمدة قصيرة
    print("🛑 إيقاف التشغيل التلقائي")
    exit(0)
