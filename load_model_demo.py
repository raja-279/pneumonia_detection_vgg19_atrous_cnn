import tensorflow as tf

# Load the locked model
model = tf.keras.models.load_model("best_vgg19_atrous.h5")

# Display architecture
model.summary()
from tensorflow.keras.preprocessing import image
import numpy as np

# Load a sample image
img_path = "sample_xray.jpg"  # keep one test image ready
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

if prediction[0][0] >= 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")
