import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\Users\\komtr\\Downloads\\Brain_MRI_PREDICTION\\data\\brain_tumor_model.h5")

model = load_model()

# Class labels (IMPORTANT: order must match training)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Title
st.markdown(
    "<h1 style='text-align:center;'>🧠 Brain Tumor Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Upload MRI scan to detect tumor type</p>",
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Detect Tumor"):
        input_img = preprocess_image(image)
        predictions = model.predict(input_img)[0]

        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(
            f"🧠 Prediction: **{predicted_class.upper()}** ({confidence:.2f}%)"
        )

        # -----------------------------
        # 📊 Confidence Bar Chart
        # -----------------------------
        st.subheader("📊 Prediction Confidence")

        fig, ax = plt.subplots()
        ax.bar(class_names, predictions, alpha=0.8)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# -----------------------------
# 📈 Model Metrics (Static Example)
# -----------------------------
st.subheader("📈 Model Performance")

accuracy = [0.65, 0.78, 0.86, 0.90]
val_accuracy = [0.62, 0.75, 0.83, 0.88]

fig2, ax2 = plt.subplots()
ax2.plot(accuracy, label="Train Accuracy")
ax2.plot(val_accuracy, label="Validation Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
st.pyplot(fig2)

st.markdown(
    "<p style='text-align:center; color:gray;'>Built using Deep Learning & Streamlit</p>",
    unsafe_allow_html=True
)



