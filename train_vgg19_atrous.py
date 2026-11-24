# ================================
"""
MODEL LOCKED
Best model: best_vgg19_atrous.h5
Validation Accuracy: ~97.7%
Test Accuracy: ~98%
Do NOT retrain or overwrite.
"""

# ================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
    Dense, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import classification_report, confusion_matrix

# ================================
# PATHS (CHANGE ONLY IF NEEDED)
# ================================
BASE_DIR = r"C:\Users\Admin\Downloads\minorproj"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# ================================
# PARAMETERS
# ================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4

# ================================
# DATA GENERATORS
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class mapping:", train_gen.class_indices)

# ================================
# MODEL: VGG19 + ATROUS CNN
# ================================
input_layer = Input(shape=(224, 224, 3))

vgg19 = VGG19(
    weights="imagenet",
    include_top=False,
    input_tensor=input_layer
)

# Freeze VGG19
for layer in vgg19.layers:
    layer.trainable = False

x = vgg19.output

# Extra Conv block
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)
x = Dropout(0.4)(x)

# Atrous Convolutions
x = Conv2D(64, (3, 3), dilation_rate=1, activation="relu", padding="same")(x)
x = Conv2D(64, (3, 3), dilation_rate=2, activation="relu", padding="same")(x)
x = Conv2D(64, (3, 3), dilation_rate=4, activation="relu", padding="same")(x)

x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# CALLBACKS (VERY IMPORTANT)
# ================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        "best_vgg19_atrous.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ================================
# TRAINING
# ================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ================================
# TESTING (FINAL EVALUATION)
# ================================
test_gen.reset()
y_prob = model.predict(test_gen)
y_pred = (y_prob >= 0.5).astype(int).ravel()
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ================================
# SAVE RESULTS
# ================================
results = pd.DataFrame({
    "True_Label": y_true,
    "Predicted_Prob": y_prob.ravel(),
    "Predicted_Label": y_pred
})

results.to_csv("pneumonia_results.csv", index=False)
print("\nResults saved as pneumonia_results.csv")
