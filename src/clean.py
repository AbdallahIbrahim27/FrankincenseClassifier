"""
STEP 2: Data Cleaning
======================
بيعمل:
  1. يشيل الصور التالفة أو اللي مش هتفتح
  2. يشيل الصور المكررة (duplicate hashing)
  3. يشيل الصور الصغيرة جداً (< 100x100)
  4. يطبع تقرير بعدد الصور لكل class
"""

import os
import hashlib
from PIL import Image
from pathlib import Path
from collections import defaultdict

# ===================== CONFIG =====================
RAW_DIR = "dataset/raw"
CLEAN_DIR = "dataset/clean"
MIN_SIZE = (100, 100)       # أقل حجم مقبول
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ==================================================


def get_image_hash(filepath: str) -> str:
    """Hash بسيط للصورة للكشف عن المكررات"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def is_valid_image(filepath: str) -> tuple[bool, str]:
    """تتحقق إن الصورة صالحة وكبيرة بما يكفي"""
    try:
        with Image.open(filepath) as img:
            img.verify()
    except Exception:
        return False, "corrupted"

    try:
        with Image.open(filepath) as img:
            w, h = img.size
            if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                return False, f"too_small ({w}x{h})"
    except Exception:
        return False, "unreadable"

    return True, "ok"


def clean_class(label: str) -> dict:
    raw_path = Path(RAW_DIR) / label
    clean_path = Path(CLEAN_DIR) / label
    clean_path.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    seen_hashes = set()

    files = [f for f in raw_path.iterdir() if f.suffix.lower() in VALID_EXTS]
    stats["total_raw"] = len(files)

    for filepath in files:
        # تحقق من صلاحية الصورة
        valid, reason = is_valid_image(str(filepath))
        if not valid:
            stats[f"removed_{reason}"] += 1
            continue

        # تحقق من التكرار
        img_hash = get_image_hash(str(filepath))
        if img_hash in seen_hashes:
            stats["removed_duplicate"] += 1
            continue
        seen_hashes.add(img_hash)

        # انسخ للـ clean folder
        dest = clean_path / filepath.name
        dest.write_bytes(filepath.read_bytes())
        stats["kept"] += 1

    return dict(stats)


def main():
    print("=" * 60)
    print("  STEP 2: Cleaning Data")
    print("=" * 60)

    labels = [d for d in os.listdir(RAW_DIR) if os.path.isdir(f"{RAW_DIR}/{d}")]
    total_kept = 0

    for label in sorted(labels):
        print(f"\n▶ Cleaning: {label}")
        stats = clean_class(label)

        for k, v in stats.items():
            print(f"   {k:<30}: {v}")
        total_kept += stats.get("kept", 0)

    print(f"\n✅ Total clean images: {total_kept}")

    # تحذير لو class عندها صور قليلة
    for label in labels:
        count = len(list((Path(CLEAN_DIR) / label).glob("*")))
        if count < 200:
            print(f"⚠️  WARNING: '{label}' has only {count} images — consider adding more!")

    print("\n   Now run: python step3_augment.py")


if __name__ == "__main__":
    main()