# Gerekli kütüphaneleri içe aktar
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os # Dosya yollarıyla çalışmak için

# Masaüstünüzdeki model dosyasının yolu.
# Lütfen bu yolu kendi dosya konumunuza göre güncelleyin.
model_path = 'yolov8n_fire.pt' # Örneğin: '/Users/kullaniciadi/Desktop/yolov8n_fire.pt'

# Test etmek istediğiniz resim dosyasının yolu.
# Lütfen bu yolu kendi dosya konumunuza göre güncelleyin.
image_path = 'yangin.jpeg' # Örneğin: '/Users/kullaniciadi/Desktop/yangin.jpeg'


# Kaydedilmiş modeli yükle
try:
    loaded_model = YOLO(model_path)
    print(f"Model şu adresten yüklendi: {model_path}")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    print("Lütfen model dosyasının yolunu kontrol edin.")
    exit() # Hata durumunda programı sonlandır

# Resmi yükle
try:
    img = Image.open(image_path)
    print(f"Resim şu adresten yüklendi: {image_path}")
except FileNotFoundError:
    print(f"Hata: Resim dosyası bulunamadı: {image_path}")
    print("Lütfen resim dosyasının yolunu kontrol edin.")
    exit() # Hata durumunda programı sonlandır


# Yüklenen modeli kullanarak tahmin yap
print("Tahminler yapılıyor...")
results = loaded_model(image_path)

# İlk sonucu al (tek bir resim işlendiğini varsayarak)
result = results[0]

# Tahminleri içeren görüntüyü matplotlib ile görüntülenebilecek formata dönüştür
# YOLOv8 sonuç nesnesi, numpy dizisi döndüren bir plot() metoduna sahiptir
img_with_predictions = result.plot()

# Tahminleri içeren görüntüyü matplotlib kullanarak göster
plt.figure(figsize=(10, 10))
plt.imshow(img_with_predictions)
plt.axis('off') # Eksenleri gizle
plt.title('Model Tahminleri') # Başlık ekle
plt.show()

print("Tahminler tamamlandı ve görüntü gösterildi.")