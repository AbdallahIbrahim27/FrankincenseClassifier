'''"""
STEP 4: Model Training
=======================
Transfer Learning باستخدام EfficientNetB0
  - Pretrained على ImageNet
  - Fine-tuning على dataset اللبان
  - يحفظ أحسن موديل تلقائياً
  - يرسم learning curves

Architecture:
  EfficientNetB0 (frozen) → GlobalAvgPool → Dense(256) → Dropout → Dense(3, softmax)
  ثم Fine-tuning: unfreeze آخر 30 layer
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===================== CONFIG =====================
AUG_DIR    = "dataset/augmented"
MODEL_DIR  = "models"
LOGS_DIR   = "logs"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN    = 10    # Phase 1: train head only
EPOCHS_FINETUNE  = 20    # Phase 2: fine-tune top layers
SEED = 42
CLASSES = ["no_resin", "not_ready", "ready"]
# ==================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────
# 1. Data Generators
# ─────────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.15,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
)

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    train_gen = train_datagen.flow_from_directory(
        AUG_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="training",
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        AUG_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


# ─────────────────────────────────────────────────
# 2. Build Model
# ─────────────────────────────────────────────────
def build_model(num_classes: int = 3) -> tf.keras.Model:
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False  # freeze في البداية

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


# ─────────────────────────────────────────────────
# 3. Callbacks
# ─────────────────────────────────────────────────
def get_callbacks(phase: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return [
        callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/best_model_{phase}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(f"{LOGS_DIR}/training_{phase}_{timestamp}.csv"),
    ]


# ─────────────────────────────────────────────────
# 4. Plot Learning Curves
# ─────────────────────────────────────────────────
def plot_history(history, phase: str, offset: int = 0):
    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss    = history.history["loss"]
    val_loss= history.history["val_loss"]
    epochs  = range(offset + 1, offset + len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training — {phase}", fontsize=14, fontweight="bold")

    ax1.plot(epochs, acc, "b-o", label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-o", label="Val Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, "b-o", label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{LOGS_DIR}/curves_{phase}.png", dpi=150)
    plt.show()
    print(f"   📊 Saved: {LOGS_DIR}/curves_{phase}.png")


# ─────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STEP 4: Training EfficientNetB0")
    print("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n🖥️  GPUs available: {len(gpus)}")

    # Data
    print("\n📂 Loading data generators...")
    train_gen, val_gen = build_generators()
    print(f"   Train batches : {len(train_gen)}")
    print(f"   Val batches   : {len(val_gen)}")
    print(f"   Class indices : {train_gen.class_indices}")

    # Save class mapping
    with open(f"{MODEL_DIR}/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Build
    print("\n🏗️  Building model...")
    model, base_model = build_model(num_classes=len(CLASSES))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Phase 1: Train head only ──────────────────
    print(f"\n🔵 Phase 1: Training head ({EPOCHS_FROZEN} epochs, base frozen)")
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )
    plot_history(h1, "Phase1_HeadOnly", offset=0)

    # ── Phase 2: Fine-tune top layers ────────────
    print(f"\n🟡 Phase 2: Fine-tuning top 30 layers ({EPOCHS_FINETUNE} epochs)")
    base_model.trainable = True
    # Freeze all except last 30 layers
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )
    plot_history(h2, "Phase2_FineTune", offset=EPOCHS_FROZEN)

    # ── Save final model ──────────────────────────
    model.save(f"{MODEL_DIR}/frankincense_classifier_final.keras")
    print(f"\n✅ Final model saved: {MODEL_DIR}/frankincense_classifier_final.keras")
    print("   Now run: python step5_evaluate.py")


if __name__ == "__main__":
    main()'''
    
    
"""
STEP 4: Model Training (FIXED VERSION)
======================================
EfficientNetB0 + Proper Preprocessing + Strong Fine-Tuning
"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===================== CONFIG =====================
AUG_DIR    = "dataset/augmented"
MODEL_DIR  = "models"
LOGS_DIR   = "logs"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 30

SEED = 42
CLASSES = ["no_resin", "not_ready", "ready"]
# ==================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ───────────────────────────────────────────────
# 1. Data Generators (FIXED)
# ───────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.15,

        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],

        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ✅ FIXED
        validation_split=0.15,
    )

    train_gen = train_datagen.flow_from_directory(
        AUG_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="training",
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        AUG_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


# ───────────────────────────────────────────────
# 2. Build Model (Improved Head)
# ───────────────────────────────────────────────
def build_model(num_classes=3):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


# ───────────────────────────────────────────────
# 3. Callbacks
# ───────────────────────────────────────────────
def get_callbacks(phase):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    return [
        callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/best_model_{phase}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(f"{LOGS_DIR}/training_{phase}_{timestamp}.csv"),
    ]


# ───────────────────────────────────────────────
# 4. Plot
# ───────────────────────────────────────────────
def plot_history(history, phase, offset=0):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(offset + 1, offset + len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{LOGS_DIR}/curves_{phase}.png")
    plt.show()


# ───────────────────────────────────────────────
# 5. Main
# ───────────────────────────────────────────────
def main():
    print("\n🚀 Training EfficientNetB0 (Fixed Version)\n")

    train_gen, val_gen = build_generators()

    print("Classes:", train_gen.class_indices)

    # Save class mapping
    with open(f"{MODEL_DIR}/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model, base_model = build_model(len(CLASSES))

    # ── Phase 1 ──
    print("\n🔵 Phase 1: Train Head")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=get_callbacks("phase1"),
    )

    plot_history(h1, "phase1")

    # ── Phase 2 ──
    print("\n🟡 Phase 2: Fine-Tuning")

    base_model.trainable = True

    # Freeze early layers
    for layer in base_model.layers[:-120]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=get_callbacks("phase2"),
    )

    plot_history(h2, "phase2", offset=EPOCHS_FROZEN)

    model.save(f"{MODEL_DIR}/frankincense_classifier_final.keras")

    print("\n✅ DONE — Model Saved Successfully!")


if __name__ == "__main__":
    main()