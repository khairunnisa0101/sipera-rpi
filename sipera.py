import cv2
import os
import datetime
import requests
from deepface import DeepFace
from picamera2 import Picamera2
from time import sleep, time
from ultralytics import YOLO

# --- (aktif) GPIO untuk LED indikator ---
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
LED_HIJAU = 17
LED_MERAH = 27
GPIO.setup(LED_HIJAU, GPIO.OUT)
GPIO.setup(LED_MERAH, GPIO.OUT)

# --- Konfigurasi umum ---
database_path = "face_database"
base_dir = "Catatan_Presensi"
os.makedirs(base_dir, exist_ok=True)

API_URL = "https://casipera.com/api/presensi"  # Ganti dengan URL endpoint API kamu
API_KEY = "2|vScnE6aCLbeMYYYymUx8GlKf4VDL3aaYyhUtO9rOc9c3eb99"            # Ganti dengan API key kamu

MAX_RUNTIME_MINUTES = 10
presensi_harian = set()
last_date = datetime.date.today()

# --- Inisialisasi kamera ---
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (480, 640)})
picam2.configure(config)
picam2.start()
sleep(2)

# HaarCascade untuk trigger deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load model YOLO ---
model = YOLO("/home/pi/capstone1/best.pt")

print("[INFO] Sistem presensi dan kelengkapan siap.")
start_time = time()

try:
    while True:
        if (time() - start_time) > MAX_RUNTIME_MINUTES * 60:
            print("[INFO] Waktu maksimal tercapai, sistem dihentikan otomatis.")
            break

        # Reset harian jika berganti hari
        today_date = datetime.date.today()
        if today_date != last_date:
            presensi_harian.clear()
            last_date = today_date
            print("[INFO] Hari baru, reset presensi harian.")

        # Ambil frame kamera
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            temp_file = "snapshot.jpg"
            cv2.imwrite(temp_file, frame)

            try:
                result = DeepFace.find(
                    img_path=temp_file,
                    db_path=database_path,
                    model_name="Facenet",
                    distance_metric="cosine",
                    enforce_detection=False,
                    silent=True,
                    detector_backend="opencv"
                )

                if len(result) > 0 and len(result[0]) > 0:
                    best_match = result[0].iloc[0]
                    id_siswa = os.path.basename(os.path.dirname(best_match['identity']))
                    presensi_key = (id_siswa, today_date)

                    if presensi_key not in presensi_harian:
                        presensi_harian.add(presensi_key)

                        foto_path = os.path.join(base_dir, f"{id_siswa}_presensi.jpg")
                        cv2.imwrite(foto_path, frame)

                        print(f"[PRESENSI] {id_siswa} dikenali, menjalankan deteksi atribut...")

                        # --- Jalankan YOLO untuk deteksi atribut ---
                        results = model(foto_path)
                        atribut = {"dasi": 0.0, "badge": 0.0, "sabuk": 0.0}

                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                cls_name = model.names[cls_id].lower()
                                conf = float(box.conf[0])
                                if cls_name in atribut and conf > atribut[cls_name]:
                                    atribut[cls_name] = conf

                        # --- Logika kelengkapan (pakai confidence threshold 0.4) ---
                        lengkap = 1
                        for conf in atribut.values():
                            if conf < 0.4:
                                lengkap = 0
                                break

                        # --- Kirim hasil ke API ---
                        payload = {
                            "id_siswa": id_siswa,
                            "kehadiran": "Hadir",
                            "kelengkapan": lengkap
                        }

                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {API_KEY}"
                        }

                        try:
                            response = requests.post(API_URL, json=payload, headers=headers)
                            if response.status_code == 200:
                                print(f"[API] Data {id_siswa} terkirim ke server ✅")
                            else:
                                print(f"[API] Gagal kirim data ({response.status_code}): {response.text}")
                        except Exception as e:
                            print("[ERROR API]", e)

                        # --- LED hijau berkedip ---
                        for _ in range(3):
                            GPIO.output(LED_HIJAU, GPIO.HIGH)
                            sleep(0.3)
                            GPIO.output(LED_HIJAU, GPIO.LOW)
                            sleep(0.3)

                    else:
                        print(f"[INFO] {id_siswa} sudah presensi hari ini.")

                else:
                    print("[PRESENSI] Wajah tidak dikenali.")
                    GPIO.output(LED_MERAH, GPIO.HIGH)
                    sleep(1)
                    GPIO.output(LED_MERAH, GPIO.LOW)

            except Exception as e:
                print("[ERROR]", e)

        sleep(1)

except KeyboardInterrupt:
    print("[INFO] Sistem dihentikan manual.")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("[INFO] Kamera dimatikan.")
