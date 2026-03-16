import cv2
import os
import datetime
import requests
import traceback
from deepface import DeepFace
from picamera2 import Picamera2
from time import sleep, time
from ultralytics import YOLO
import RPi.GPIO as GPIO
import numpy as np
import hmac
import hashlib
import json
from dotenv import load_dotenv

# --- ROOT PROJECT PATH ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load environment variable ---
load_dotenv("/home/pi/capstone1/.env")

API_URL = os.getenv("API_URL", "https://casipera.com/api/presensi")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Fungsi buat signature HMAC ---
def generate_signature(payload, secret):
    payload_str = json.dumps(payload, sort_keys=True)
    return hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
    
# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
LED_HIJAU = 22
LED_MERAH = 17
LED_KUNING = 27
GPIO.setup(LED_HIJAU, GPIO.OUT)
GPIO.setup(LED_MERAH, GPIO.OUT)
GPIO.setup(LED_KUNING, GPIO.OUT)

# --- Konfigurasi umum ---
model_name = "Facenet512"
model_dir = "/home/pi/capstone1/models"

database_path = os.path.join(BASE_PATH, "face_database")
base_dir = os.path.join(BASE_PATH, "Catatan_Presensi")
os.makedirs(base_dir, exist_ok=True)

presensi_harian = set()
last_date = datetime.date.today()


# --- Inisialisasi kamera ---
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (360, 480)})
picam2.configure(config)
picam2.start()
sleep(2)

# --- HaarCascade untuk deteksi wajah ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load model YOLO ---
model = YOLO("/home/pi/capstone1/best3.pt")

print("[INFO] Sistem presensi dan kelengkapan siap. Tekan Ctrl+C untuk menghentikan secara manual.")

try:
    while True:
        today_date = datetime.date.today()
        if today_date != last_date:
            presensi_harian.clear()
            last_date = today_date
            print("[INFO] Hari baru, reset presensi harian.")

        # --- Buat folder log harian ---
        log_dir = os.path.join(base_dir, str(today_date))
        os.makedirs(log_dir, exist_ok=True)

        # Ambil frame kamera
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            temp_file = "snapshot.jpg"
            cv2.imwrite(temp_file, frame)

            try:
                # --- Ukur waktu deteksi & pengenalan wajah ---
                start_face = time()
                result = DeepFace.find(
                    img_path=temp_file,
                    db_path=database_path,
                    model_name=model_name,
                    distance_metric="cosine",
                    enforce_detection=False,
                    silent=True,
                    detector_backend="opencv"
                )

                end_face = time()
                durasi_face = end_face - start_face

                if len(result) > 0 and len(result[0]) > 0:
                    best_match = result[0].iloc[0]
                    id_siswa = os.path.basename(os.path.dirname(best_match['identity']))


                    # --- Ambil jarak pengenalan (distance dari DeepFace) ---
                    cosine_distance = best_match.get("distance", None)


                    threshold = 0.4
                    if cosine_distance is not None:
                        print(f"[WAKTU] Deteksi wajah: {durasi_face:.2f}s | Cosine Distance: {cosine_distance:.3f} (Threshold: {threshold})")
                    else:
                        print(f"[WAKTU] Deteksi wajah: {durasi_face:.2f}s | Cosine Distance: N/A (Threshold: {threshold})")


                    # --- Evaluasi hasil ---
                    if cosine_distance is None or cosine_distance <= threshold:
                        presensi_key = (id_siswa, today_date)
                        if presensi_key not in presensi_harian:
                            presensi_harian.add(presensi_key)

                            foto_path = os.path.join(log_dir, f"{id_siswa}_presensi.jpg")
                            cv2.imwrite(foto_path, frame)

                            print(f"[PRESENSI] {id_siswa} dikenali, menjalankan deteksi atribut...")

                            # --- Ukur waktu YOLO ---
                            start_yolo = time()
                            results = model(foto_path)
                            end_yolo = time()
                            durasi_yolo = end_yolo - start_yolo
                                
                            # --- Analisis atribut hasil YOLO ---
                            atribut = {"dasi": 0.0, "badge": 0.0, "sabuk": 0.0}
                            for r in results:
                                for box in r.boxes:
                                    cls_id = int(box.cls[0])
                                    cls_name = model.names[cls_id].lower()
                                    conf = float(box.conf[0])
                                    if cls_name in atribut and conf > atribut[cls_name]:
                                        atribut[cls_name] = conf

                            lengkap = "1"
                            for nama, conf in atribut.items():
                                if nama == "sabuk":
                                    if conf < 0.3:
                                        lengkap = "0"
                                        break
                                else:
                                    if conf < 0.4:
                                        lengkap = "0"
                                        break

                            payload = {
                                "id_siswa": id_siswa,
                                "status": "Hadir",
                                "kelengkapan": lengkap
                            }
                            signature = generate_signature(payload, SECRET_KEY)
                            payload["signature"] = signature

                            headers = {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {API_KEY}"
                            }

                            try:
                                response = requests.post(API_URL, json=payload, headers=headers)
                                if response.ok:
                                    print(f"[API] Data {id_siswa} terkirim ? | Response: {response.text}")
                                else:
                                    with open(os.path.join(BASE_PATH, "error_log.txt"), "a") as log:
                                        log.write(f"[{datetime.datetime.now()}] API gagal ({response.status_code}): {response.text}\n")
                                    print(f"[API ERROR] {response.status_code}: lihat error_log.txt")
                            except Exception as e:
                                with open(os.path.join(BASE_PATH, "error_log.txt"), "a") as log:
                                    log.write(f"\n[{datetime.datetime.now()}] Exception API:\n{traceback.format_exc()}")
                                print(f"[API ERROR] {type(e).__name__}: {str(e)[:100]} ...")

                            # --- Hitung total waktu ---
                            end_total = time()
                            durasi_total = end_total - start_face  # mulai dari face

                            # --- Simpan log ---
                            with open("waktu_pengujian.txt", "a") as f:
                                f.write(f"[{datetime.datetime.now()}] {id_siswa} | Wajah: {durasi_face:.2f}s | YOLO: {durasi_yolo:.2f}s | Total: {durasi_total:.2f}s\n")

                            # --- Simpan log waktu ---
                            with open("waktu_pengujian.txt", "a") as f:
                                f.write(
                                    f"[{datetime.datetime.now()}] {id_siswa} | "
                                    f"Wajah: {durasi_face:.2f}s | "
                                    f"YOLO: {durasi_yolo:.2f}s | "
                                    f"Total: {durasi_total:.2f}s\n"
                                )

                            # --- LED hijau berkedip 3 kali ---
                            for _ in range(3):
                                GPIO.output(LED_HIJAU, GPIO.HIGH)
                                sleep(0.3)
                                GPIO.output(LED_HIJAU, GPIO.LOW)
                                sleep(0.3)
                        else:
                            print(f"[INFO] {id_siswa} sudah presensi hari ini.")
                    else:
                        print(f"[PRESENSI] Wajah tidak cocok (cosine={cosine_distance:.3f} > {threshold}).")

                # --- LED merah berkedip 3 kali jika tidak ada hasil cocok ---
                else:
                    print("[INFO] Tidak ada hasil dari DeepFace.find(). Wajah tidak cocok dengan database.")
                    for _ in range(3):
                        GPIO.output(LED_MERAH, GPIO.HIGH)
                        sleep(0.3)
                        GPIO.output(LED_MERAH, GPIO.LOW)
                        sleep(0.3)

            except Exception as e:
                print("[ERROR]", e)

            # --- Mode standby 2 detik ---
            print("[STANDBY] Menunggu 2 detik sebelum deteksi berikutnya...")
            GPIO.output(LED_KUNING, GPIO.HIGH)
            sleep(2)
            GPIO.output(LED_KUNING, GPIO.LOW)

        else:
            GPIO.output(LED_KUNING, GPIO.HIGH)
            sleep(0.2)
            GPIO.output(LED_KUNING, GPIO.LOW)
            sleep(0.2)

except KeyboardInterrupt:
    print("\n[INFO] Sistem dihentikan manual.")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("[INFO] Kamera dimatikan dan GPIO dibersihkan.")
