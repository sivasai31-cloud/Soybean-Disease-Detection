import os
import numpy as np
import tensorflow as tf

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_PATH = os.path.join("models", "final_soybean_model.keras")
IMAGE_SIZE = (224, 224)

# Update this list if your training order is different.
CLASS_NAMES = [
    "bacterial_blight",
    "brown_spot",
    "crestamento",
    "ferrugen",
    "healthy",
    "Mosaic Virus",
    "powdery_mildew",
    "septoria",
    "Southern blight",
    "Sudden Death Syndrome",
    "Yellow Mosaic"
]

# -----------------------------
# Load Model Once
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_path):
    """
    Load and preprocess image for prediction.
    """

    image = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.preprocessing.image.img_to_array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):
    """
    Predict disease from image.
    Returns:
        disease_name
        confidence_percentage
    """

    if model is None:
        raise Exception("Model is not loaded.")

    image = preprocess_image(image_path)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[0][predicted_index]) * 100

    disease = CLASS_NAMES[predicted_index]

    return disease, round(confidence, 2)


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    sample = "test.jpg"

    if os.path.exists(sample):

        disease, confidence = predict_image(sample)

        print("Prediction :", disease)
        print("Confidence :", confidence, "%")

    else:

        print("Place a test image named test.jpg in the project folder.")