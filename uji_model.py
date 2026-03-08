import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Konfigurasi umum ---
database_path = "face_database"
csv_label_path = "labels.csv"  # file CSV berisi file_name dan true_label
models_to_test = ["Facenet512"]
distance_metric = "cosine"
THRESHOLD = 0.4  # batas maksimum cosine distance agar dianggap cocok
detector_backend = "mtcnn"

print("=" * 90)
print("[INFO] Pengujian Model DeepFace dengan Evaluasi & Threshold")
print(f"[INFO] Database : {database_path}")
print(f"[INFO] Metric   : {distance_metric}")
print(f"[INFO] Threshold : {THRESHOLD}")
print(f"[INFO] Detector Backend : {detector_backend}")
print("=" * 90)

# --- Baca CSV label ground truth ---
labels_df = pd.read_csv(csv_label_path)
labels_df["true_label"] = labels_df["true_label"].astype(str)
labels_df["file_name"] = labels_df["file_name"].astype(str)

# --- Buat daftar gambar uji ---
test_images = [
    os.path.join(database_path, row["true_label"], row["file_name"])
    for _, row in labels_df.iterrows()
]

total_images = len(test_images)
print(f"[INFO] Total gambar untuk diuji: {total_images}\n")

# --- Simpan hasil keseluruhan ---
hasil_final = []

for model_name in models_to_test:
    print(f"\n{'=' * 70}")
    print(f"[MODEL] {model_name}")
    print(f"{'=' * 70}")

    y_true, y_pred, hasil_pengujian = [], [], []

    for idx, row in labels_df.iterrows():
        true_label = str(row["true_label"])
        file_name = str(row["file_name"])
        img_path = os.path.join(database_path, true_label, file_name)

        print(f"[{idx+1}/{total_images}] Menguji {file_name} ...")
        start_time = time.time()

        try:
            result = DeepFace.find(
                img_path=img_path,
                db_path=database_path,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
                detector_backend="mtcnn"
            )
            durasi = time.time() - start_time

            if len(result) > 0 and len(result[0]) > 0:
                best_match = result[0].iloc[0]
                detected_id = os.path.basename(os.path.dirname(best_match["identity"]))
                cosine_distance = best_match.get("distance", None)

                # --- Cek apakah match berdasarkan threshold ---
                if cosine_distance is not None and cosine_distance <= THRESHOLD:
                    final_id = detected_id
                else:
                    final_id = "None"

                hasil_pengujian.append({
                    "model": model_name,
                    "gambar_uji": file_name,
                    "true_label": true_label,
                    "id_terdeteksi": final_id,
                    "cosine_distance": cosine_distance,
                    "durasi_deteksi": durasi
                })

                y_true.append(true_label)
                y_pred.append(final_id)

                print(f"    -> ID: {final_id} | Cosine: {cosine_distance:.3f} | Waktu: {durasi:.2f}s")
            else:
                print("    -> Tidak ditemukan kecocokan.")
                hasil_pengujian.append({
                    "model": model_name,
                    "gambar_uji": file_name,
                    "true_label": true_label,
                    "id_terdeteksi": "None",
                    "cosine_distance": None,
                    "durasi_deteksi": durasi
                })
                y_true.append(true_label)
                y_pred.append("None")

        except Exception as e:
            print(f"    [ERROR] {e}")
            hasil_pengujian.append({
                "model": model_name,
                "gambar_uji": file_name,
                "true_label": true_label,
                "id_terdeteksi": "ERROR",
                "cosine_distance": None,
                "durasi_deteksi": None
            })
            y_true.append(true_label)
            y_pred.append("ERROR")

    # --- Evaluasi performa ---
    df_model = pd.DataFrame(hasil_pengujian)
    hasil_final.append(df_model)

    df_valid = df_model.dropna(subset=["cosine_distance"])
    if not df_valid.empty:
        rata_cosine = df_valid["cosine_distance"].mean()
        rata_waktu = df_valid["durasi_deteksi"].mean()
    else:
        rata_cosine = 0
        rata_waktu = 0

    # --- Hitung metrik evaluasi ---
    labels_unik = sorted(list(set(y_true + y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels_unik)

    print(f"\n[HASIL RATA-RATA UNTUK {model_name}]")
    print(f"  - Rata-rata Cosine Distance : {rata_cosine:.3f}")
    print(f"  - Rata-rata Waktu Deteksi   : {rata_waktu:.2f}s")
    print(f"  - Accuracy  : {acc:.3f}")
    print(f"  - Precision : {prec:.3f}")
    print(f"  - Recall    : {rec:.3f}")
    print(f"  - F1-Score  : {f1:.3f}")

    # --- Tampilkan Confusion Matrix ---
    print("\nConfusion Matrix:")
    conf_df = pd.DataFrame(conf_mat, index=labels_unik, columns=labels_unik)
    print(conf_df)

    # --- Visualisasi Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    timestamp_img = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"confusion_matrix_{model_name}_{timestamp_img}.png")
    plt.close()

# --- Gabungkan semua hasil ---
df_all = pd.concat(hasil_final, ignore_index=True)

# --- Simpan hasil ke CSV ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"hasil_pengujian_deepface_{timestamp}.csv"
df_all.to_csv(output_file, index=False)

print("\n" + "=" * 90)
print(f"[INFO] Semua pengujian selesai.")
print(f"[INFO] Hasil lengkap disimpan ke: {output_file}")
print(f"[INFO] Gambar confusion matrix juga disimpan di folder saat ini.")
print("=" * 90)
