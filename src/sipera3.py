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

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv("/home/pi/capstone1/.env")

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

def generate_signature(payload, secret):
    payload_str = json.dumps(payload, sort_keys=True)
    return hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()

GPIO.setmode(GPIO.BCM)

LED_HIJAU = 22
LED_MERAH = 17
LED_KUNING = 27

GPIO.setup(LED_HIJAU, GPIO.OUT)
GPIO.setup(LED_MERAH, GPIO.OUT)
GPIO.setup(LED_KUNING, GPIO.OUT)

model_name = "Facenet512"

database_path = os.path.join(BASE_PATH, "face_database")
base_dir = os.path.join(BASE_PATH, "Catatan_Presensi")

os.makedirs(base_dir, exist_ok=True)

presensi_harian = set()
last_date = datetime.date.today()

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (360,480)})
picam2.configure(config)
picam2.start()

sleep(2)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model = YOLO("/home/pi/capstone1/best3.pt")

print("[INFO] Sistem presensi siap")

try:

    while True:

        today_date = datetime.date.today()

        if today_date != last_date:
            presensi_harian.clear()
            last_date = today_date

        log_dir = os.path.join(base_dir, str(today_date))
        os.makedirs(log_dir, exist_ok=True)

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces) == 1:

            temp_file = "snapshot.jpg"
            cv2.imwrite(temp_file, frame)

            try:

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

                    id_siswa = os.path.basename(
                        os.path.dirname(best_match['identity'])
                    )

                    cosine_distance = best_match.get("distance",None)

                    threshold = 0.4

                    if cosine_distance is None or cosine_distance <= threshold:

                        presensi_key = (id_siswa,today_date)

                        if presensi_key not in presensi_harian:

                            presensi_harian.add(presensi_key)

                            foto_path = os.path.join(
                                log_dir,
                                f"{id_siswa}_presensi.jpg"
                            )

                            cv2.imwrite(foto_path, frame)

                            print("[INFO] Menjalankan YOLO...")

                            start_yolo = time()

                            results = model(foto_path)

                            end_yolo = time()
                            durasi_yolo = end_yolo - start_yolo

                            atribut = {
                                "dasi":0,
                                "badge":0,
                                "sabuk":0
                            }

                            for r in results:
                                for box in r.boxes:

                                    cls_id = int(box.cls[0])
                                    cls_name = model.names[cls_id].lower()
                                    conf = float(box.conf[0])

                                    if cls_name in atribut and conf > atribut[cls_name]:
                                        atribut[cls_name] = conf

                            lengkap = "1"

                            for nama,conf in atribut.items():

                                if nama == "sabuk":

                                    if conf < 0.3:
                                        lengkap = "0"

                                else:

                                    if conf < 0.4:
                                        lengkap = "0"

                            payload = {
                                "id_siswa":id_siswa,
                                "status":"Hadir",
                                "kelengkapan":lengkap
                            }

                            signature = generate_signature(payload,SECRET_KEY)
                            payload["signature"] = signature

                            headers = {
                                "Content-Type":"application/json",
                                "Authorization":f"Bearer {API_KEY}"
                            }

                            response = requests.post(
                                API_URL,
                                json=payload,
                                headers=headers
                            )

                            end_total = time()
                            durasi_total = end_total - start_face

                            print(
                                f"Wajah:{durasi_face:.2f}s "
                                f"YOLO:{durasi_yolo:.2f}s "
                                f"TOTAL:{durasi_total:.2f}s"
                            )

                            with open("waktu_pengujian.txt","a") as f:

                                f.write(
                                    f"{datetime.datetime.now()} "
                                    f"{id_siswa} "
                                    f"Face:{durasi_face:.2f} "
                                    f"YOLO:{durasi_yolo:.2f} "
                                    f"Total:{durasi_total:.2f}\n"
                                )

                            for _ in range(3):

                                GPIO.output(LED_HIJAU,GPIO.HIGH)
                                sleep(0.3)

                                GPIO.output(LED_HIJAU,GPIO.LOW)
                                sleep(0.3)

            except Exception as e:

                print("[ERROR]",e)

        else:

            GPIO.output(LED_KUNING,GPIO.HIGH)
            sleep(0.2)
            GPIO.output(LED_KUNING,GPIO.LOW)
            sleep(0.2)

except KeyboardInterrupt:

    print("Sistem dihentikan")

finally:

    picam2.stop()
    GPIO.cleanup()