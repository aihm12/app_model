import json
import base64
import io
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageStat
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # تأكد من أن هذا هو المسار الصحيح على Ubuntu

MODEL_FILE = "yolov8_license_plate.pt"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ النموذج غير موجود! تأكد من رفعه إلى المستودع.")

print("✅ تحميل النموذج...")
model = YOLO(MODEL_FILE)

def process_image(image):
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
    try:
        image_path = "test_image.jpg"
        if not os.path.exists(image_path):
            raise FileNotFoundError("❌ لم يتم العثور على الصورة! تأكد من رفع `test_image.jpg` إلى المستودع.")

        image = Image.open(image_path).convert('RGB')

        response, error = process_image(image)
        if error:
            print(f"❌ خطأ: {error}")
            return

        with open("result.json", "w") as f:
            json.dump(response, f)

        print("✅ تمت المعالجة بنجاح! ✅")

    except Exception as e:
        print(f"❌ خطأ أثناء المعالجة: {e}")

if __name__ == '__main__':
    main()
