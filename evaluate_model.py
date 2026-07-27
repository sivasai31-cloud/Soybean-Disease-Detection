import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------
# Load Model
# ----------------------------

model = tf.keras.models.load_model(
    "models/final_soybean_model.keras"
)

# ----------------------------
# Dataset
# ----------------------------

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224,224),
    batch_size=16,
    shuffle=False
)

class_names = dataset.class_names

# ----------------------------
# Predictions
# ----------------------------

y_true = []
y_pred = []

for images, labels in dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted)

# ----------------------------
# Accuracy
# ----------------------------

accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# Classification Report
# ----------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ----------------------------
# Confusion Matrix
# ----------------------------

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(12,12))

disp.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()