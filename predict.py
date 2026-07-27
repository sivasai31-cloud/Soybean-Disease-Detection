import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =====================================
# LOAD MODEL
# =====================================

model = tf.keras.models.load_model("models/final_soybean_model.keras")

# Class names (must be in the same order as training)
class_names = [
    "Mossaic Virus",
    "Southern blight",
    "Sudden Death Syndrone",
    "Yellow Mosaic",
    "bacterial_blight",
    "brown_spot",
    "crestamento",
    "ferrugen",
    "healthy",
    "powdery_mildew",
    "septoria"
]

# =====================================
# IMAGE PATH
# =====================================

image_path = "test.jpg"   # Change this to your test image

# =====================================
# LOAD IMAGE
# =====================================

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# =====================================
# PREDICT
# =====================================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

confidence = np.max(prediction) * 100

print("\n=============================")
print("Prediction Result")
print("=============================")

print("Disease :", class_names[predicted_index])
print(f"Confidence : {confidence:.2f}%")