import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ----------------------------
# Configuration
# ----------------------------

DATASET_PATH = "dataset"

IMAGE_SIZE = (224,224)

BATCH_SIZE = 16

EPOCHS = 10

# ----------------------------
# Dataset
# ----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

# ----------------------------
# Base Model
# ----------------------------

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ----------------------------
# Model
# ----------------------------

model = models.Sequential([

    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )

])

# ----------------------------
# Compile
# ----------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ----------------------------
# Train
# ----------------------------

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS

)

# ----------------------------
# Save
# ----------------------------

model.save("models/final_soybean_model.keras")

print("Model Saved Successfully!")