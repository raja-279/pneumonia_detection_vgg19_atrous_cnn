import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ================================
# LOAD LOCKED MODEL
# ================================
MODEL_PATH = "best_vgg19_atrous.h5"
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")

# ================================
# SETTINGS
# ================================
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # decision threshold

DEMO_DIR = "demo_images"

# ================================
# PREDICTION FUNCTION
# ================================
def predict_xray(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]

    if prob >= THRESHOLD:
        label = "PNEUMONIA"
        confidence = prob * 100
    else:
        label = "NORMAL"
        confidence = (1 - prob) * 100

    return label, confidence, prob


# ================================
# RUN DEMO PREDICTIONS
# ================================
print("\n🔍 Running predictions on demo images:\n")

for file in os.listdir(DEMO_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(DEMO_DIR, file)

        label, confidence, raw_prob = predict_xray(img_path)

        print(f"Image: {file}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw model probability: {raw_prob:.4f}")
        print("-" * 40)
