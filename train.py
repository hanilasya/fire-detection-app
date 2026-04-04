import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
TRAIN_DIR = 'dataset/train'
TEST_DIR  = 'dataset/test'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 10

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary: fire vs no_fire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# Save model
model.save(os.path.join(MODEL_DIR, 'fire_smoke_model.h5'))
print("\n✅ Model saved to model/fire_smoke_model.h5")

# Final accuracy
loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc*100:.2f}%")