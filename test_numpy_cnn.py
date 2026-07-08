import os
import pickle
import numpy as np
import tensorflow as tf

def cnn_predict_numpy(img, weights):
    # img shape: (H, W, C) or (1, H, W, C)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    
    # Layer 1: conv2d
    w_conv1, b_conv1 = weights['conv2d']
    # w_conv1 shape: (kh, kw, in_c, out_c) = (3, 3, 3, 32)
    # b_conv1 shape: (out_c,) = (32,)
    # Output shape: (1, 126, 126, 32)
    
    # Efficient 2D Conv using NumPy strides or simple loops
    # Since batch size is 1, let's write a clear loop-based or vectorised conv
    b, h, w, c = img.shape
    kh, kw, in_c, out_c = w_conv1.shape
    
    # Conv 1
    out_h1 = h - kh + 1
    out_w1 = w - kw + 1
    # We can use np.lib.stride_tricks.as_strided to extract patches
    # Shape of patches: (1, 126, 126, 3, 3, 3)
    shape = (b, out_h1, out_w1, kh, kw, in_c)
    strides = (img.strides[0], img.strides[1], img.strides[2], img.strides[1], img.strides[2], img.strides[3])
    patches1 = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    
    # Perform convolution: multiply and sum over (kh, kw, in_c)
    # patches1 shape: (1, 126, 126, 3, 3, 3)
    # w_conv1 shape: (3, 3, 3, 32)
    # We want to sum over axes 3, 4, 5
    conv1 = np.tensordot(patches1, w_conv1, axes=((3, 4, 5), (0, 1, 2))) + b_conv1
    # conv1 shape: (1, 126, 126, 32)
    
    # ReLU
    relu1 = np.maximum(conv1, 0)
    
    # MaxPooling 1 (2, 2)
    # Input shape: (1, 126, 126, 32)
    # Output shape: (1, 63, 63, 32)
    pool1 = relu1[:, :126:2, :126:2, :]
    for offset_h in [0, 1]:
        for offset_w in [0, 1]:
            pool1 = np.maximum(pool1, relu1[:, offset_h:126:2, offset_w:126:2, :])
            
    # Layer 2: conv2d_1
    w_conv2, b_conv2 = weights['conv2d_1']
    b, h, w, c = pool1.shape
    kh, kw, in_c, out_c = w_conv2.shape
    out_h2 = h - kh + 1
    out_w2 = w - kw + 1
    
    shape = (b, out_h2, out_w2, kh, kw, in_c)
    strides = (pool1.strides[0], pool1.strides[1], pool1.strides[2], pool1.strides[1], pool1.strides[2], pool1.strides[3])
    patches2 = np.lib.stride_tricks.as_strided(pool1, shape=shape, strides=strides)
    conv2 = np.tensordot(patches2, w_conv2, axes=((3, 4, 5), (0, 1, 2))) + b_conv2
    
    # ReLU
    relu2 = np.maximum(conv2, 0)
    
    # MaxPooling 2 (2, 2)
    # Input: (1, 61, 61, 64) -> Output: (1, 30, 30, 64)
    # Note: drop last row/col because 61 is odd
    pool2 = relu2[:, :60:2, :60:2, :]
    for offset_h in [0, 1]:
        for offset_w in [0, 1]:
            pool2 = np.maximum(pool2, relu2[:, offset_h:60:2, offset_w:60:2, :])
            
    # Layer 3: conv2d_2
    w_conv3, b_conv3 = weights['conv2d_2']
    b, h, w, c = pool2.shape
    kh, kw, in_c, out_c = w_conv3.shape
    out_h3 = h - kh + 1
    out_w3 = w - kw + 1
    
    shape = (b, out_h3, out_w3, kh, kw, in_c)
    strides = (pool2.strides[0], pool2.strides[1], pool2.strides[2], pool2.strides[1], pool2.strides[2], pool2.strides[3])
    patches3 = np.lib.stride_tricks.as_strided(pool2, shape=shape, strides=strides)
    conv3 = np.tensordot(patches3, w_conv3, axes=((3, 4, 5), (0, 1, 2))) + b_conv3
    
    # ReLU
    relu3 = np.maximum(conv3, 0)
    
    # MaxPooling 3 (2, 2)
    # Input: (1, 28, 28, 128) -> Output: (1, 14, 14, 128)
    pool3 = relu3[:, :28:2, :28:2, :]
    for offset_h in [0, 1]:
        for offset_w in [0, 1]:
            pool3 = np.maximum(pool3, relu3[:, offset_h:28:2, offset_w:28:2, :])
            
    # Flatten
    flat = pool3.reshape(b, -1)
    
    # Dense: dense_features
    w_dense, b_dense = weights['dense_features']
    # w_dense shape: (25088, 64)
    # b_dense shape: (64,)
    dense_out = np.dot(flat, w_dense) + b_dense
    
    # ReLU
    dense_relu = np.maximum(dense_out, 0)
    
    return dense_relu

# Test equivalence
np.random.seed(42)
img = np.random.rand(1, 128, 128, 3).astype(np.float32)

print("Loading Keras model...")
keras_model = tf.keras.models.load_model("turmeric_cnn_feature_extractor.keras")
keras_pred = keras_model.predict(img)

print("Loading weights...")
with open("turmeric_cnn_weights.pkl", "rb") as f:
    weights = pickle.load(f)

print("Running NumPy prediction...")
numpy_pred = cnn_predict_numpy(img, weights)

print(f"Keras prediction shape: {keras_pred.shape}")
print(f"NumPy prediction shape: {numpy_pred.shape}")

diff = np.abs(keras_pred - numpy_pred)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")

assert max_diff < 1e-4, "Predictions differ too much!"
print("SUCCESS: NumPy implementation is 100% equivalent!")
