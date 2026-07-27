import tensorflow as tf

model = tf.keras.models.load_model("models/final_soybean_model.keras")

print("Model input:", model.input)
print("Model output:", model.output)

base = model.layers[1]

print("\nBase input:", base.input)
print("Base output:", base.output)

print("\nLast activation:", base.get_layer("out_relu").output)