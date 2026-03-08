from deepface import DeepFace
import pandas as pd
import os

# Path ke folder database wajah kamu
database_path = "face_database"

# Path ke salah satu gambar wajah untuk pengujian (ubah sesuai gambar yang ada)
test_image_path = "test_image.jpg"

# Daftar model yang ingin dicek
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

print("[INFO] Mengecek model-model DeepFace dan kolom hasilnya...\n")

# Loop semua model untuk melihat hasil struktur kolom
for model in models:
    print(f"=== Model: {model} ===")
    try:
        df = DeepFace.find(
            img_path=test_image_path,
            db_path=database_path,
            model_name=model,
            enforce_detection=False
        )
        if isinstance(df, list):
            df = df[0]
        print("Kolom hasil:", list(df.columns))
        if any("cosine" in col.lower() for col in df.columns):
            cosine_cols = [col for col in df.columns if "cosine" in col.lower()]
            print("? Kolom cosine terdeteksi:", cosine_cols)
        else:
            print("? Tidak ada kolom cosine.")
    except Exception as e:
        print("Gagal memproses model:", e)
    print()

print("\n[SELESAI] Pengecekan selesai.")
