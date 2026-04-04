import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/fire_smoke_model.h5')

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    # CORRECT
if pred < 0.5:
    label = "🔥 Fire"
    confidence = (1 - pred) * 100
else:
    label = "✅ No Fire"
    confidence = pred * 100
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")

# Test with one image — change path to any image you have
predict_image('dataset/test/fire/fire.601.png')