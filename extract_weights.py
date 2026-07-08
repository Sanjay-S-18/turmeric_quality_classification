import os
import pickle
import numpy as np
import tensorflow as tf

CNN_PATH = "turmeric_cnn_feature_extractor.keras"

print("Loading model...")
model = tf.keras.models.load_model(CNN_PATH)
model.summary()

# Extract weights of each layer
weights_dict = {}
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for idx, w in enumerate(weights):
            print(f"  weight {idx} shape: {w.shape}")
        weights_dict[layer.name] = weights

# Save the weights to a pickle file
with open("turmeric_cnn_weights.pkl", "wb") as f:
    pickle.dump(weights_dict, f)
print("Saved weights to turmeric_cnn_weights.pkl")
