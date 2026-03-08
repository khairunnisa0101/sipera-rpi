import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
PIN = 22

GPIO.setup(PIN, GPIO.OUT)

print("LED HARUS NYALA TERUS (cek manual)")
GPIO.output(PIN, GPIO.HIGH)

input("Tekan ENTER untuk matikan...")

GPIO.output(PIN, GPIO.LOW)
GPIO.cleanup()
