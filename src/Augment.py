"""
STEP 3: Data Augmentation
==========================
بيعمل augmentation على الصور النظيفة علشان:
  - يزود عدد الصور لـ 1000+ لكل class
  - يعوّض أي class imbalance
  - يخلي الموديل أكثر robustness

التحويلات المستخدمة:
  - Horizontal/Vertical Flip
  - Rotation (±25°)
  - Brightness & Contrast variation
  - Zoom & Crop
  - Color Jitter (مهم لأن الصور في أضواء مختلفة)
  - Gaussian Blur
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# ===================== CONFIG =====================
CLEAN_DIR = "dataset/clean"
AUG_DIR   = "dataset/augmented"
TARGET_PER_CLASS = 1000   # عدد الصور المطلوب لكل class
SEED = 42
# ==================================================

random.seed(SEED)
np.random.seed(SEED)


def augment_image(img: Image.Image) -> Image.Image:
    """يطبق مجموعة عشوائية من التحويلات على صورة واحدة"""
    img = img.convert("RGB")

    # Horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip (أحياناً مفيد)
    if random.random() > 0.8:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

    # Brightness
    factor = random.uniform(0.6, 1.4)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation (Color Jitter)
    factor = random.uniform(0.7, 1.4)
    img = ImageEnhance.Color(img).enhance(factor)

    # Sharpness
    factor = random.uniform(0.5, 1.5)
    img = ImageEnhance.Sharpness(img).enhance(factor)

    # Zoom (Random Crop then resize back)
    if random.random() > 0.5:
        w, h = img.size
        zoom = random.uniform(0.8, 1.0)
        new_w, new_h = int(w * zoom), int(h * zoom)
        left  = random.randint(0, w - new_w)
        top   = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    # Gaussian Blur (أحياناً)
    if random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


def augment_class(label: str) -> None:
    src_dir  = Path(CLEAN_DIR) / label
    dest_dir = Path(AUG_DIR) / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    orig_files = list(src_dir.glob("*"))
    orig_files = [f for f in orig_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    if not orig_files:
        print(f"   ⚠️  No images found in {src_dir}")
        return

    print(f"   Found {len(orig_files)} original images → target {TARGET_PER_CLASS}")

    # انسخ الأصليين الأول
    for f in orig_files:
        dest = dest_dir / f.name
        dest.write_bytes(f.read_bytes())

    # ابدأ الـ augmentation
    aug_count = 0
    needed = TARGET_PER_CLASS - len(orig_files)

    if needed <= 0:
        print(f"   ✅ Already have {len(orig_files)} images, no augmentation needed.")
        return

    while aug_count < needed:
        src_file = random.choice(orig_files)
        try:
            with Image.open(src_file) as img:
                aug_img = augment_image(img)
                save_path = dest_dir / f"aug_{aug_count:05d}.jpg"
                aug_img.save(save_path, "JPEG", quality=90)
                aug_count += 1
        except Exception as e:
            print(f"   Warning: skipped {src_file.name} — {e}")
            continue

    print(f"   ✅ Generated {aug_count} augmented images → total: {len(orig_files) + aug_count}")


def main():
    print("=" * 60)
    print("  STEP 3: Data Augmentation")
    print("=" * 60)

    labels = [d for d in os.listdir(CLEAN_DIR) if os.path.isdir(f"{CLEAN_DIR}/{d}")]

    for label in sorted(labels):
        print(f"\n▶ Augmenting: {label}")
        augment_class(label)

    # Summary
    print("\n📊 Final Dataset Summary:")
    for label in sorted(labels):
        count = len(list((Path(AUG_DIR) / label).glob("*")))
        bar = "█" * (count // 50)
        print(f"   {label:<15}: {count:>5} images  {bar}")

    print("\n   Now run: python step4_train.py")


if __name__ == "__main__":
    main()