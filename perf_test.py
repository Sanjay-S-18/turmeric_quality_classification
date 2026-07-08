import os
import time
import pickle
import numpy as np
import tensorflow as tf

def cnn_predict_numpy(img, weights):
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    
    b, h, w, c = img.shape
    
    # Layer 1: conv2d
    w_conv1, b_conv1 = weights['conv2d']
    kh, kw, in_c, out_c = w_conv1.shape
    out_h1 = h - kh + 1
    out_w1 = w - kw + 1
    
    shape = (b, out_h1, out_w1, kh, kw, in_c)
    strides = (img.strides[0], img.strides[1], img.strides[2], img.strides[1], img.strides[2], img.strides[3])
    patches1 = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    conv1 = np.tensordot(patches1, w_conv1, axes=((3, 4, 5), (0, 1, 2))) + b_conv1
    relu1 = np.maximum(conv1, 0)
    
    # MaxPooling 1
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
    relu2 = np.maximum(conv2, 0)
    
    # MaxPooling 2
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
    relu3 = np.maximum(conv3, 0)
    
    # MaxPooling 3
    pool3 = relu3[:, :28:2, :28:2, :]
    for offset_h in [0, 1]:
        for offset_w in [0, 1]:
            pool3 = np.maximum(pool3, relu3[:, offset_h:28:2, offset_w:28:2, :])
            
    # Flatten & Dense
    flat = pool3.reshape(b, -1)
    w_dense, b_dense = weights['dense_features']
    dense_out = np.dot(flat, w_dense) + b_dense
    dense_relu = np.maximum(dense_out, 0)
    
    return dense_relu

# Measure time
img = np.random.rand(1, 128, 128, 3).astype(np.float32)

print("Loading Keras model...")
t0 = time.time()
keras_model = tf.keras.models.load_model("turmeric_cnn_feature_extractor.keras")
print(f"Model load time: {time.time() - t0:.4f}s")

# Warm up Keras
keras_model.predict(img, verbose=0)
t0 = time.time()
for _ in range(10):
    keras_model.predict(img, verbose=0)
print(f"Keras average prediction time: {(time.time() - t0)/10:.4f}s")

print("\nLoading weights...")
t0 = time.time()
with open("turmeric_cnn_weights.pkl", "rb") as f:
    weights = pickle.load(f)
print(f"Weights load time: {time.time() - t0:.4f}s")

# Warm up NumPy
cnn_predict_numpy(img, weights)
t0 = time.time()
for _ in range(10):
    cnn_predict_numpy(img, weights)
print(f"NumPy average prediction time: {(time.time() - t0)/10:.4f}s")
