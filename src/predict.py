"""
STEP 6: Predict on New Images
==============================
استخدم الموديل المدرب على صور جديدة.

Usage:
  python step6_predict.py --image path/to/image.jpg
  python step6_predict.py --folder path/to/folder/
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import tensorflow as tf
from PIL import Image

# ===================== CONFIG =====================
MODEL_PATH = "models/frankincense_classifier_final.keras"
CLASS_JSON = "models/class_indices.json"
IMG_SIZE   = (224, 224)

CLASS_INFO = {
    "no_resin": {
        "arabic": "بدون راتنج",
        "color":  "#e74c3c",
        "emoji":  "🌳",
        "advice": "الشجرة ما زالت تحتاج وقتاً قبل النقر."
    },
    "not_ready": {
        "arabic": "راتنج غير ناضج",
        "color":  "#f39c12",
        "emoji":  "⏳",
        "advice": "الراتنج موجود لكن يحتاج مزيداً من الوقت ليجف."
    },
    "ready": {
        "arabic": "جاهز للحصاد",
        "color":  "#27ae60",
        "emoji":  "✅",
        "advice": "الراتنج ناضج وجاهز للجمع الآن!"
    },
}
# ==================================================


def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    return model, classes


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(model, classes: list, image_path: str, show: bool = True) -> dict:
    arr = preprocess_image(image_path)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "image": str(image_path),
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }

    if show:
        _show_prediction(image_path, result, classes, probs)

    return result


def _show_prediction(image_path, result, classes, probs):
    info = CLASS_INFO[result["prediction"]]
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # Image
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_title(
        f"{info['emoji']} {result['prediction']}  ({result['confidence']:.1%})\n{info['arabic']}",
        color="white", fontsize=13, pad=10
    )
    ax_img.axis("off")
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        linewidth=3,
        edgecolor=info["color"],
        facecolor="none",
        transform=ax_img.transAxes,
        clip_on=False
    )
    ax_img.add_patch(rect)

    # Bar chart
    colors = [CLASS_INFO[c]["color"] for c in classes]
    bars = ax_bar.barh(classes, probs * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)", color="white")
    ax_bar.set_title("Class Probabilities", color="white", pad=10)
    ax_bar.tick_params(colors="white")
    ax_bar.set_facecolor("#16213e")
    for bar, prob in zip(bars, probs):
        ax_bar.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", color="white", fontsize=10
        )

    # Advice
    fig.text(0.5, 0.02, f"💡 {info['advice']}", ha="center",
             color="#aaaaaa", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = Path(image_path).stem + "_prediction.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"   📊 Saved: {out_path}")


def predict_folder(model, classes, folder_path: str):
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))

    if not image_files:
        print("No images found in folder.")
        return

    print(f"Found {len(image_files)} images\n")
    results = []
    for img_path in image_files:
        r = predict_single(model, classes, str(img_path), show=False)
        results.append(r)
        print(f"  {img_path.name:<40} → {r['prediction']:<12} ({r['confidence']:.1%})")

    # Summary
    from collections import Counter
    counter = Counter(r["prediction"] for r in results)
    print(f"\n📊 Summary:")
    for cls, cnt in counter.items():
        print(f"   {CLASS_INFO[cls]['emoji']} {cls}: {cnt} images")


def main():
    parser = argparse.ArgumentParser(description="Frankincense Resin Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    model, classes = load_model_and_classes()
    print(f"✅ Model loaded | Classes: {classes}\n")

    if args.image:
        result = predict_single(model, classes, args.image, show=True)
        print(f"\n🎯 Prediction : {result['prediction']}")
        print(f"   Confidence : {result['confidence']:.1%}")
        print(f"   Advice     : {CLASS_INFO[result['prediction']]['advice']}")
    else:
        predict_folder(model, classes, args.folder)


if __name__ == "__main__":
    main()