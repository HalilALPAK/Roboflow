import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from scipy.spatial.distance import cosine

# =====================
# CONFIG
# =====================
ME_DIR = "me"
THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# MODEL (EfficientNet)
# =====================
model = models.efficientnet_b0(pretrained=True)
model.classifier = nn.Identity()
model = model.to(DEVICE)
model.eval()

# =====================
# FACE DETECTOR
# =====================
mtcnn = MTCNN(image_size=224, margin=20, device=DEVICE)

# =====================
# AUGMENTATION
# =====================
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
])

# =====================
# BUILD "ME" EMBEDDING
# =====================
def build_me_embedding():
    embeddings = []

    for img_name in os.listdir(ME_DIR):
        img = Image.open(os.path.join(ME_DIR, img_name)).convert("RGB")
        face = mtcnn(img)

        if face is None:
            continue

        # Orijinal
        with torch.no_grad():
            emb = model(face.unsqueeze(0).to(DEVICE))
            embeddings.append(emb.cpu().numpy())

        # Augmented versiyonlar
        for _ in range(3):
            aug_img = augment(img)
            face_aug = mtcnn(aug_img)
            if face_aug is None:
                continue

            with torch.no_grad():
                emb = model(face_aug.unsqueeze(0).to(DEVICE))
                embeddings.append(emb.cpu().numpy())

    return np.mean(embeddings, axis=0)

ME_EMBEDDING = build_me_embedding()
print("✅ ME embedding hazır")

# =====================
# RECOGNITION
# =====================
def recognize(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return "NO FACE"

    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

    dist = cosine(emb, ME_EMBEDDING)

    if dist < THRESHOLD:
        return "ME"
    else:
        return "UNKNOWN"

# =====================
# TEST
# =====================
if __name__ == "__main__":
    test_image = "test.jpg"
    result = recognize(test_image)
    print(f"🧠 Sonuç: {result}")
