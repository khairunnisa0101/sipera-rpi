import cv2
import os
import datetime
import requests
import traceback
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

API_URL = "https://casipera.com/api/presensi"
API_KEY = "4|APLnCwDE68dy7MHU1N83cpNy2LzKRsTdAKv1SVFJ201ad667"

MAX_RUNTIME_MINUTES = 10
presensi_harian = set()
last_date = datetime.date.today()

# --- Inisialisasi kamera ---
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (480, 640)})
picam2.configure(config)
picam2.start()
sleep(2)

# --- HaarCascade untuk deteksi wajah ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load model YOLO ---
model = YOLO("/home/pi/capstone1/best1.pt")

print("[INFO] Sistem presensi dan kelengkapan siap. Auto-stop setelah", MAX_RUNTIME_MINUTES, "menit.")
start_time = time()

try:
    while True:
        # Hentikan otomatis jika waktu maksimal tercapai
        if (time() - start_time) > MAX_RUNTIME_MINUTES * 60:
            print("[INFO] Waktu maksimal tercapai, sistem dihentikan otomatis.")
            break

        # Reset harian jika berganti hari
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

        # Deteksi wajah menggunakan HaarCascade
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
                    model_name="Facenet",
                    distance_metric="cosine",
                    enforce_detection=False,
                    silent=True,
                    detector_backend="opencv"
                )
                end_face = time()
                durasi_face = end_face - start_face
                print(f"[WAKTU] Deteksi & pengenalan wajah: {durasi_face:.2f} detik")

                if len(result) > 0 and len(result[0]) > 0:
                    best_match = result[0].iloc[0]
                    id_siswa = os.path.basename(os.path.dirname(best_match['identity']))
                    presensi_key = (id_siswa, today_date)

                    if presensi_key not in presensi_harian:
                        presensi_harian.add(presensi_key)

                        # --- Simpan foto hasil presensi ke folder log harian ---
                        foto_path = os.path.join(log_dir, f"{id_siswa}_presensi.jpg")
                        cv2.imwrite(foto_path, frame)

                        print(f"[PRESENSI] {id_siswa} dikenali, menjalankan deteksi atribut...")

                        # --- Ukur waktu deteksi atribut (YOLO) ---
                        start_yolo = time()
                        results = model(foto_path)
                        end_yolo = time()
                        durasi_yolo = end_yolo - start_yolo
                        print(f"[WAKTU] Deteksi atribut (YOLO): {durasi_yolo:.2f} detik")

                        # --- Simpan hasil pengujian waktu ke file ---
                        with open("waktu_pengujian.txt", "a") as f:
                            f.write(f"[{datetime.datetime.now()}] {id_siswa} | Wajah: {durasi_face:.2f}s | YOLO: {durasi_yolo:.2f}s\n")

                        # --- Analisis atribut hasil YOLO ---
                        atribut = {"dasi": 0.0, "badge": 0.0, "sabuk": 0.0}
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                cls_name = model.names[cls_id].lower()
                                conf = float(box.conf[0])
                                if cls_name in atribut and conf > atribut[cls_name]:
                                    atribut[cls_name] = conf

                        print("[ATRIBUT] Nilai deteksi:")
                        for nama, conf in atribut.items():
                            print(f"  - {nama.capitalize()}: {conf:.2f}")

                        # --- Logika kelengkapan atribut ---
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

                        # --- Kirim hasil ke API ---
                        payload = {
                            "id_siswa": id_siswa,
                            "status": "Hadir",
                            "kelengkapan": lengkap
                        }
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {API_KEY}"
                        }

                        try:
                            response = requests.post(API_URL, json=payload, headers=headers)
                            if response.ok:
                                print(f"[API] Data {id_siswa} terkirim ke server ✅ | Response: {response.text}")
                            else:
                                with open("error_log.txt", "a") as log:
                                    log.write(f"[{datetime.datetime.now()}] API gagal ({response.status_code}): {response.text}\n")
                                print(f"[API ERROR] {response.status_code}: lihat detail di error_log.txt")
                        except Exception as e:
                            with open("error_log.txt", "a") as log:
                                log.write(f"\n[{datetime.datetime.now()}] Exception saat kirim data API:\n")
                                log.write(traceback.format_exc())
                            print(f"[API ERROR] {type(e).__name__}: {str(e)[:120]} ... (lihat error_log.txt)")

                        # --- LED hijau berkedip tiga kali ---
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

except KeyboardInterrupt:
    print("[INFO] Sistem dihentikan manual.")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("[INFO] Kamera dimatikan.")
