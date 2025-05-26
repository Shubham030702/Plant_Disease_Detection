import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU device(s) found: {gpus}")
else:
    print("No GPU found, training will run on CPU.")

# Config
DATASET_DIR = "/home/smurfy/Desktop/Plant_Disease_Detection/DATASET/Bean_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50

# Load dataset
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

# Apply data augmentation only to training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Depthwise Separable Conv Block
def depthwise_separable_conv(x, pointwise_filters, strides):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Build MobileNet from scratch with Dropout added before final Dense layer
def build_mobilenet(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = depthwise_separable_conv(x, 64, 1)
    x = depthwise_separable_conv(x, 128, 2)
    x = depthwise_separable_conv(x, 128, 1)
    x = depthwise_separable_conv(x, 256, 2)
    x = depthwise_separable_conv(x, 256, 1)
    x = depthwise_separable_conv(x, 512, 2)

    for _ in range(5):
        x = depthwise_separable_conv(x, 512, 1)

    x = depthwise_separable_conv(x, 1024, 2)
    x = depthwise_separable_conv(x, 1024, 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # Dropout to reduce overfitting
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model

model = build_mobilenet(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define callbacks
checkpoint_path = "best_mobilenet_model.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Save the final model
model.save("mobilenet_model_final.h5")
print("✅ Final model saved.")

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()
    