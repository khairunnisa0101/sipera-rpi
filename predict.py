from ultralytics import YOLO
import os
import json

# Load model
model = YOLO("best.pt")

# Folder gambar
folder = "/home/pi/images/"

results_list = []

# Loop semua gambar
for filename in os.listdir(folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder, filename)
        
        # Run inference
        results = model(img_path)
        
        # Ambil hasil deteksi (misal nama class + confidence)
        predictions = []
        for r in results:
            boxes = r.boxes  # semua bounding box
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                predictions.append({"class": cls_name, "confidence": conf})

        results_list.append({"filename": filename, "predictions": predictions})

# Simpan ke file JSON (opsional)
with open("results.json", "w") as f:
    json.dump(results_list, f, indent=2)

print("Selesai, hasil disimpan di results.json")