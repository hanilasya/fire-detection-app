import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(page_title="Fire Detection App", page_icon="🔥", layout="centered")

# Load model (cached)
@st.cache_resource
def load_fire_model():
    return load_model('model/fire_smoke_model.h5', compile=False)

model = load_fire_model()

# UI
st.title("🔥 Fire Detection System")
st.write("Upload an image to detect if it contains **Fire** or **No Fire**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        pred = model.predict(img)[0][0]

    # Result
    if pred < 0.5:
        label = "🔥 Fire Detected!"
        confidence = (1 - pred) * 100
        st.error(f"**Prediction: {label}**")
    else:
        label = "✅ No Fire"
        confidence = pred * 100
        st.success(f"**Prediction: {label}**")

    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    st.progress(int(confidence))

else:
    st.info("👆 Please upload an image to get started.")