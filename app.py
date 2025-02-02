from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageStat
import pytesseract
import base64
import io
import gdown
import json
import time

app = Flask(__name__)

# 🟢 تحميل النموذج من Google Drive
MODEL_FILE = "yolov8_license_plate.pt"
GOOGLE_DRIVE_FILE_ID = "1ewShjHYyro3adOU5IATMd1KQF8HdW_oD"

def download_model():
    """تحميل النموذج من Google Drive إذا لم يكن موجودًا"""
    try:
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
        print("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")

# تحميل النموذج عند بدء التشغيل
download_model()
model = YOLO(MODEL_FILE)


def process_image(image):
    """تحليل الصورة لاستخراج أرقام اللوحة"""
    try:
        results = model.predict(image)
        
        # استخراج أول صندوق للوحة السيارة
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None, "❌ لم يتم العثور على لوحة رقمية"
        
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
        plate_number = pytesseract.image_to_string(
            cropped_plate, lang="eng", config="--psm 7"
        ).strip()

        return {
            'coordinates': [x_min, y_min, x_max, y_max],
            'plate_number': plate_number,
            'plate_type': plate_type
        }, None
    except Exception as e:
        return None, str(e)


@app.route('/predict', methods=['POST'])
def predict():
    """API لتحليل الصور المرسلة"""
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({"error": "❌ لم يتم إرسال صورة"}), 400

        # فك تشفير الصورة من Base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # تحليل الصورة
        response, error = process_image(image)
        if error:
            return jsonify({'error': error}), 400

        # حفظ النتائج في ملف `result.json`
        with open("result.json", "w") as f:
            json.dump(response, f)

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ تشغيل Flask لمدة محددة ثم إغلاق التطبيق تلقائيًا
if __name__ == '__main__':
    print("🚀 تشغيل النموذج وتحليل الصورة...")
    app.run(debug=False, host='0.0.0.0', port=8000)
    time.sleep(10)  # تشغيل السيرفر لمدة 10 ثوانٍ فقط
    print("🛑 إيقاف التشغيل التلقائي")
    exit(0)
