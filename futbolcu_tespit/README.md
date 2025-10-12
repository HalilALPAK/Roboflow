# Football Object Detection with YOLOv8

Bu proje, YOLOv8 kullanarak futbol objelerini tespit etmek için eğitilmiş bir makine öğrenmesi modelidir.

## Dosyalar

- `f.ipynb` - Model test notebook'u
- `futbol.ipynb` - Model eğitim notebook'u
- `best_model.pt` - Eğitilmiş YOLOv8 modeli
- `real.jpeg` - Test görseli
- `yolo11n.pt`, `yolov8s.pt` - YOLOv8 model dosyaları

## Kullanım

1. Model test etmek için `f.ipynb` notebook'unu çalıştırın
2. Yeni model eğitmek için `futbol.ipynb` notebook'unu kullanın
3. Test görseli olarak `real.jpeg` dosyası kullanılır

## Özellikler

- ✅ YOLOv8 ile futbol objesi tespiti
- ✅ Early stopping ile model eğitimi
- ✅ Gerçek zamanlı görüntü testi
- ✅ Görselleştirme ve sonuç analizi

## Gereksinimler

```bash
pip install ultralytics opencv-python matplotlib roboflow
```

## Sonuçlar

Eğitilmiş model futbol sahası, top, oyuncular ve diğer futbol objelerini yüksek doğrulukla tespit edebilir.
