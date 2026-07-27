import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# =====================================
# PROJECT SETTINGS
# =====================================

DATASET_PATH = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

# =====================================
# LOAD DATASET
# =====================================

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("\n==============================")
print("Classes Found")
print("==============================")

for name in class_names:
    print(name)

print("\nTotal Classes :", len(class_names))

# =====================================
# OPTIMIZE DATASET
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================
# DATA AUGMENTATION
# =====================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# =====================================
# BUILD CNN MODEL
# =====================================

model = models.Sequential([

    layers.Input(shape=(224,224,3)),

    data_augmentation,

    layers.Rescaling(1./255),

    # Block 1
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(128,3,activation="relu"),
    layers.MaxPooling2D(),

    # Block 4
    layers.Conv2D(256,3,activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256,activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(len(class_names),activation="softmax")

])

# =====================================
# SHOW MODEL
# =====================================

print("\n==============================")
print("CNN MODEL")
print("==============================")

model.summary()

# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Compiled Successfully!")

# =====================================
# CREATE MODEL FOLDER
# =====================================

os.makedirs("models", exist_ok=True)

# =====================================
# CALLBACKS
# =====================================

checkpoint = ModelCheckpoint(
    "models/soybean_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# =====================================
# SAVE FINAL MODEL
# =====================================

model.save("models/final_soybean_model.keras")

print("\nModel Saved Successfully!")

# =====================================
# PLOT ACCURACY
# =====================================

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# =====================================
# PLOT LOSS
# =====================================

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")