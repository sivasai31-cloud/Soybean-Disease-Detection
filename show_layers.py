import tensorflow as tf

model = tf.keras.models.load_model("models/final_soybean_model.keras")

print("="*60)
print("MODEL SUMMARY")
print("="*60)

model.summary(expand_nested=True)

print("\n")
print("="*60)
print("TOP LEVEL LAYERS")
print("="*60)

for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))