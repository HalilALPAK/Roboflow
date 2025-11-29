"""from roboflow import Roboflow
rf = Roboflow(api_key="7rVVO91C1xm0t3qc520o")
project = rf.workspace("peter-7x3gt").project("edge-cases")
version = project.version(2)
dataset = version.download("yolov8")"""""
import os
from ultralytics import YOLO

# -----------------------------
# Dataset ve model yolları
# -----------------------------
dataset_dir = r"C:\Users\User\Desktop\nudity\edge-cases-2"
data_yaml = os.path.join(dataset_dir, "data.yaml")

# Yol kontrolü
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")
else:
    print("✅ Kullanılacak dataset:", data_yaml)

# YOLO modelini yükle (önceden eğitilmiş)
model = YOLO("yolov8n.pt")  # Veya kendi modeliniz yolunu yazın

# -----------------------------
# Eğitim başlat
# -----------------------------
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=2,
    verbose=True,
)

print("✅ Eğitim tamamlandı.")


model.save('nude.pt')

print("Model 'nude.pt' olarak kaydedildi.")
