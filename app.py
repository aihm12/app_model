import json
import base64
import io
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageStat
import pytesseract

# 🟢 تحديد مسار النموذج
MODEL_FILE = "yolov8_license_plate.pt"

# تأكد من أن النموذج موجود
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ الملف {MODEL_FILE} غير موجود. تأكد من رفع النموذج إلى GitHub.")

# تحميل النموذج
print("✅ تحميل النموذج...")
model = YOLO(MODEL_FILE)

# مسار حفظ النتائج
RESULTS_DIR = "outputs"
RESULTS_FILE = os.path.join(RESULTS_DIR, "result.json")

# تأكد من أن مجلد outputs موجود
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_image(image):
    """تحليل الصورة واستخراج أرقام اللوحة"""
    try:
        results = model.predict(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None, "❌ لم يتم العثور على لوحة رقمية"

        x_min, y_min, x_max, y_max = map(int, boxes[0])
        cropped_plate = image.crop((x_min, y_min, x_max, y_max))

        dominant_color = ImageStat.Stat(cropped_plate).mean
        plate_type = "غير معروف"
        if dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
            plate_type = "نقل"
        elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
            plate_type = "أجرة"
        elif dominant_color[2] > dominant_color[0] and dominant_color[2] > dominant_color[1]:
            plate_type = "خصوصي"

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

def main():
    """تشغيل المعالجة تلقائيًا عند تشغيل السكريبت"""
    try:
        image_path = "test_image.jpg"  # 🟢 تأكد من رفع صورة اختبار إلى المستودع
        if not os.path.exists(image_path):
            raise FileNotFoundError("❌ لم يتم العثور على الصورة! تأكد من رفع الصورة إلى المستودع.")

        image = Image.open(image_path).convert('RGB')

        response, error = process_image(image)
        if error:
            print(f"❌ خطأ: {error}")
            return

        # 🔹 حفظ النتائج في `outputs/result.json`
        with open(RESULTS_FILE, "w") as f:
            json.dump(response, f)

        print(f"✅ تمت المعالجة بنجاح! تم حفظ النتائج في {RESULTS_FILE}")

    except Exception as e:
        print(f"❌ خطأ أثناء المعالجة: {e}")

if __name__ == '__main__':
    main()
