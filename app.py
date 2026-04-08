import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

from tensorflow.keras.applications.efficientnet import preprocess_input

# ================= CONFIG =================
MODEL_PATH = "models/frankincense_classifier_final.keras"
CLASS_JSON = "models/class_indices.json"
IMG_SIZE = (224, 224)
# ==========================================

# ── Load Model ─────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_classes():
    with open(CLASS_JSON) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    return classes

model = load_model()
classes = load_classes()

# ── UI ─────────────────────────────────────
st.title("🌳 Frankincense Classifier")
st.write("Upload an image to classify resin status")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict
    preds = model.predict(img)[0]
    pred_idx = np.argmax(preds)
    pred_class = classes[pred_idx]
    confidence = preds[pred_idx]

    # Result
    st.subheader("📊 Prediction")

    st.success(f"🟢 Class: {pred_class}")
    st.info(f"Confidence: {confidence:.2%}")

    # Show all probabilities
    st.subheader("🔎 All Class Probabilities")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {preds[i]:.2%}")