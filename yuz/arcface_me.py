import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# =====================
# CONFIG
# =====================
ME_DIR = "me"
THRESHOLD = 0.8  # ArcFace için ideal
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# FACE DETECTOR
# =====================
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=DEVICE
)

# =====================
# ARC FACE MODEL
# =====================
model = InceptionResnetV1(
    pretrained="vggface2"
).to(DEVICE).eval()

# =====================
# AUGMENTATION
# =====================
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
])

# =====================
# BUILD ME EMBEDDING
# =====================
def build_me_embedding():
    embeddings = []

    for img_name in os.listdir(ME_DIR):
        img = Image.open(os.path.join(ME_DIR, img_name)).convert("RGB")

        # Orijinal
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = model(face.unsqueeze(0).to(DEVICE))
                embeddings.append(emb.cpu().numpy())

        # Augmented
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

    similarity = 1 - cosine(emb, ME_EMBEDDING)

    if similarity > THRESHOLD:
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
