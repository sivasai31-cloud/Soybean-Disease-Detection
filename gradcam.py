import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------
# Load Model
# ----------------------------

model = tf.keras.models.load_model("models/final_soybean_model.keras")

# ----------------------------
# Class Names
# ----------------------------

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

# ----------------------------
# Image Path
# ----------------------------

IMAGE_PATH = "test.jpg"

# ----------------------------
# Load Image
# ----------------------------

img = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ----------------------------
# MobileNetV2 Base Model
# ----------------------------

base_model = model.layers[1]

# Last convolution layer
last_conv_layer = base_model.get_layer("Conv_1")

# ----------------------------
# Build Grad-CAM Model
# ----------------------------

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

# ----------------------------
# Forward Pass
# ----------------------------

with tf.GradientTape() as tape:

    conv_outputs, features = grad_model(img_array)

    predictions = model(img_array)

    class_index = tf.argmax(predictions[0])

    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

heatmap = tf.maximum(heatmap, 0)

heatmap /= tf.reduce_max(heatmap)

heatmap = heatmap.numpy()

# ----------------------------
# Read Original Image
# ----------------------------

original = cv2.imread(IMAGE_PATH)

original = cv2.resize(original, (224, 224))

# ----------------------------
# Resize Heatmap
# ----------------------------

heatmap = cv2.resize(heatmap, (224, 224))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

superimposed = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)

# ----------------------------
# Save
# ----------------------------

cv2.imwrite(
    "gradcam_result.jpg",
    superimposed
)

print("Grad-CAM image saved as gradcam_result.jpg")

# ----------------------------
# Display
# ----------------------------

predicted_class = int(class_index.numpy())

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Prediction: {CLASS_NAMES[predicted_class]}")
plt.show()