"""
STEP 5: Model Evaluation (FIXED VERSION)
========================================
✔ نفس preprocessing زي التدريب
✔ Confusion Matrix
✔ Classification Report
✔ Wrong Predictions
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===================== CONFIG =====================
AUG_DIR    = "dataset/augmented"
MODEL_PATH = "models/frankincense_classifier_final.keras"
CLASS_JSON = "models/class_indices.json"
EVAL_DIR   = "evaluation"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
# ==================================================

os.makedirs(EVAL_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  STEP 5: Evaluating Model (FIXED)")
    print("=" * 60)

    # ── Load model ───────────────────────────────
    print("\n📦 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # ── Load class mapping ───────────────────────
    with open(CLASS_JSON) as f:
        class_indices = json.load(f)

    idx_to_class = {v: k for k, v in class_indices.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"   Classes: {classes}")

    # ── FIXED: نفس preprocessing زي التدريب ──────
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.15,
    )

    test_gen = datagen.flow_from_directory(
        AUG_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        subset="validation",
        seed=SEED,
        shuffle=False,  # مهم جدًا
    )

    # ── Predictions ───────────────────────────────
    print("\n🔍 Running predictions...")
    y_prob = model.predict(test_gen, verbose=1)

    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # ── Classification Report ─────────────────────
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)

    print("\n📋 Classification Report:\n")
    print(report)

    with open(f"{EVAL_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # ── Confusion Matrix ──────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)

    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/confusion_matrix.png", dpi=150)
    plt.show()

    print(f"   📊 Saved: {EVAL_DIR}/confusion_matrix.png")

    # ── Normalized Heatmap ────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        xticklabels=classes,
        yticklabels=classes,
        cmap="YlOrRd",
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/confusion_matrix_normalized.png", dpi=150)
    plt.show()

    # ── Wrong Predictions ─────────────────────────
    print("\n🔎 Generating wrong predictions gallery...")

    filepaths = [Path(p) for p in test_gen.filepaths]
    wrong_idx = np.where(y_pred != y_true)[0]

    sample = wrong_idx[:16]

    if len(sample) > 0:
        cols = 4
        rows = (len(sample) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
        axes = axes.flatten()

        for i, idx in enumerate(sample):
            img = plt.imread(str(filepaths[idx]))
            axes[i].imshow(img)

            axes[i].set_title(
                f"True: {classes[y_true[idx]]}\n"
                f"Pred: {classes[y_pred[idx]]} ({y_prob[idx][y_pred[idx]]:.1%})",
                fontsize=8,
                color="red",
            )

            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Wrong Predictions", fontsize=14)
        plt.tight_layout()

        plt.savefig(f"{EVAL_DIR}/wrong_predictions.png", dpi=150)
        plt.show()

        print(f"   📊 Saved: {EVAL_DIR}/wrong_predictions.png")

    else:
        print("   🎉 No wrong predictions!")

    # ── Summary ───────────────────────────────────
    acc = np.mean(y_pred == y_true)

    print("\n" + "=" * 40)
    print(f"  Overall Accuracy: {acc:.2%}")
    print(f"  Wrong predictions: {len(wrong_idx)} / {len(y_true)}")
    print(f"  Results saved in: {EVAL_DIR}/")
    print("=" * 40)

    print("\n✅ Done — Evaluation Correct Now!")


if __name__ == "__main__":
    main()