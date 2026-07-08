import sys
import numpy as np

# Ensure tensorflow is not imported yet
assert 'tensorflow' not in sys.modules, "TensorFlow already loaded!"

print("Importing app modules and resources...")
import app

# Force load resources
pipeline, feat_extractor = app.load_resources()

print(f"Loaded pipeline: {type(pipeline)}")
print(f"Loaded feat_extractor: {type(feat_extractor)}")
assert isinstance(feat_extractor, app.NumPyFeatureExtractor), "Feature extractor is not NumPyFeatureExtractor!"

# Run mock prediction
img = np.random.rand(1, 128, 128, 3).astype(np.float32)
out = feat_extractor.predict(img)
print(f"Prediction output shape: {out.shape}")
assert out.shape == (1, 64), "Prediction output shape is incorrect!"

# Verify tensorflow is still not imported
is_tf_imported = 'tensorflow' in sys.modules
print(f"Is TensorFlow imported? {is_tf_imported}")
assert not is_tf_imported, "TensorFlow was imported during NumPy-based load or inference!"

print("SUCCESS: App initialized and ran mock prediction with ZERO TensorFlow imports!")
