# Derin Öğrenme / Görüntü İşleme Çalışmaları

Roboflow ile etiketlenmiş veri setleri üzerinde eğitilen, çeşitli görüntü sınıflandırma ve nesne tespiti denemelerini içeren notebook koleksiyonu.

## İçindekiler

| Klasör / Dosya | Konu |
|---|---|
| [`ANN_cifar.ipynb`](ANN_cifar.ipynb) | CIFAR veri seti üzerinde yapay sinir ağı (ANN) ile görüntü sınıflandırma. |
| [`cifar.ipynb`](cifar.ipynb) | CIFAR veri seti üzerinde ek sınıflandırma denemeleri. |
| [`basket_topu_tespiti/`](basket_topu_tespiti) | Basketbol topu nesne tespiti. |
| [`duygu/`](duygu) | Yüz ifadesinden duygu (emotion) tanıma. |
| [`futbolcu_tespit/`](futbolcu_tespit) | Futbolcu tespiti. |
| [`helmet/`](helmet) | Baret/kask takılı mı tespiti (iş güvenliği). |
| [`mimik/`](mimik) | Yüz mimikleri analizi. |
| [`nudity/`](nudity) | Uygunsuz içerik (nudity) tespiti. |
| [`tenis_topu_tespiti/`](tenis_topu_tespiti) | Tenis topu nesne tespiti. |
| [`yanginn/`](yanginn) | Yangın/duman tespiti. |
| [`yuz/`](yuz) | Yüz tespiti. |

## Kullanılan Teknolojiler

- Roboflow (veri seti etiketleme/versiyonlama)
- YOLO / CNN tabanlı modeller
- Python, Jupyter Notebook

## Kurulum

```bash
pip install roboflow ultralytics opencv-python numpy matplotlib
```

Her klasördeki notebook kendi veri setini Roboflow API üzerinden indirir; çalıştırmadan önce ilgili notebook içindeki Roboflow API anahtarını kendi hesabınızla değiştirin.
