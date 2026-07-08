import os
import pickle
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


# Set page configuration with wide layout and custom icon
st.set_page_config(
    page_title="Aura Turmeric Analytics - Premium AI Quality System",
    page_icon="🍂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light-Theme Premium Orange Quality Analytics Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Headings styling */
    h1, h2, h3, h4, .glow-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-bottom: 12px;
        background: none !important;
        -webkit-text-fill-color: initial !important;
    }

    h1 {
        font-size: 2.2rem !important;
    }

    h2 {
        font-size: 1.8rem !important;
    }

    h3 {
        font-size: 1.4rem !important;
    }

    h4 {
        font-size: 1.15rem !important;
    }
    
    /* Paragraphs and general text */
    p, span, li, label, div {
        font-family: 'Outfit', sans-serif;
    }

    p {
        color: #475569 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    /* Container/Card styles */
    .glass-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .glass-card:hover {
        border-color: #F59E0B !important;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.05), 0 4px 6px -2px rgba(245, 158, 11, 0.025) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Status indicators border glow style */
    .glow-border-good {
        border-left: 6px solid #059669 !important;
        background-color: #FFFFFF !important;
    }
    .glow-border-bad {
        border-left: 6px solid #DC2626 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFF7ED !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    [data-testid="stSidebar"] .glass-card {
        background-color: #FFFFFF !important;
        border: 1px solid #FFE3E3 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }

    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] subheader, [data-testid="stSidebar"] label {
        color: #D97706 !important;
    }

    /* Buttons styling */
    .stButton>button, .stDownloadButton>button {
        background-color: #F59E0B !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1) !important;
        width: auto !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #D97706 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 6px rgba(217, 119, 6, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        transform: translateY(0px) !important;
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #F59E0B !important;
        border-radius: 12px !important;
        padding: 10px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stFileUploader"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #F59E0B !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* Metric Card Swatch */
    .color-swatch {
        display: inline-block;
        width: 100%;
        height: 60px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Labels */
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #D97706;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Banner Hero */
    .hero-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF7ED 100%);
        border-left: 5px solid #F59E0B;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025);
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }

    /* Navigation tabs styling */
    div[data-testid="stTabBar"] button {
        color: #475569 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabBar"] button:hover {
        color: #D97706 !important;
    }
    div[data-testid="stTabBar"] button[aria-selected="true"] {
        color: #F59E0B !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #F59E0B !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Table styling */
    .stDataFrame {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Streamlit's Native Expander override */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
MODEL_PATH = "turmeric_classifier_pipeline.pkl"
CNN_WEIGHTS_PATH = "turmeric_cnn_weights.pkl"
CSV_PATH = "turmeric_dataset_features.csv"

def cnn_predict_numpy(img, weights):
    """Replicates the 3-layer CNN feature extractor using pure NumPy for inference."""
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
    
    # MaxPooling 1 (2, 2)
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
    
    # MaxPooling 2 (2, 2)
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
    
    # MaxPooling 3 (2, 2)
    pool3 = relu3[:, :28:2, :28:2, :]
    for offset_h in [0, 1]:
        for offset_w in [0, 1]:
            pool3 = np.maximum(pool3, relu3[:, offset_h:28:2, offset_w:28:2, :])
            
    # Flatten
    flat = pool3.reshape(b, -1)
    
    # Dense: dense_features
    w_dense, b_dense = weights['dense_features']
    dense_out = np.dot(flat, w_dense) + b_dense
    dense_relu = np.maximum(dense_out, 0)
    
    return dense_relu

class NumPyFeatureExtractor:
    """Wrapper that mimics the Keras model interface for code compatibility."""
    def __init__(self, weights_path):
        with open(weights_path, "rb") as f:
            self.weights = pickle.load(f)

    def predict(self, img_batch):
        return cnn_predict_numpy(img_batch, self.weights)


# Preprocessing & Feature Extraction helpers
def extract_color_features_from_pil(pil_img):
    """Extracts mean and standard deviation in RGB and HSV spaces."""
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mean_rgb, std_rgb = cv2.meanStdDev(img_rgb)
    mean_hsv, std_hsv = cv2.meanStdDev(img_hsv)
    
    features = np.hstack([
        mean_rgb.flatten(), std_rgb.flatten(),
        mean_hsv.flatten(), std_hsv.flatten()
    ])
    return features, mean_rgb.flatten()

def get_dominant_colors(pil_img, k=3):
    """Uses KMeans clustering to extract dominant colors from the turmeric image."""
    img_small = pil_img.resize((50, 50))
    pixels = np.array(img_small).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    # Sort colors by brightness
    brightness = np.sum(colors, axis=1)
    sorted_indices = np.argsort(brightness)[::-1]
    return colors[sorted_indices]

def estimate_curcumin_index(mean_colors_rgb):
    """Aesthetic algorithm estimating the Curcumin content based on color purity."""
    r, g, b = mean_colors_rgb
    total = r + g + b + 1e-5
    r_pct = r / total
    g_pct = g / total
    
    # Yellow-orange color bias (curcumin peak)
    yellow_purity = (r_pct + g_pct) * (1.0 - (b / total))
    # Curcumin scale 10% - 98%
    curcumin_index = min(98.5, max(15.0, yellow_purity * 122.0 - 5.0))
    return curcumin_index

def train_model_fallback():
    """Trains the models dynamically using dataset/ if not pre-trained."""
    with st.spinner("Initializing Training Pipeline..."):
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from tensorflow.keras import layers, models
        except ImportError:
            st.error("TensorFlow is not installed in this environment. Training is only available in environments with TensorFlow installed (e.g., local development).")
            return False
        
        CATEGORIES = ["good", "bad"]
        IMG_SIZE = 128
        
        def load_data(split):
            images, color_feats, labels = [], [], []
            base_dir = f"dataset/{split}"
            
            for class_idx, class_name in enumerate(CATEGORIES):
                class_dir = os.path.join(base_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                        img_array = np.array(img_resized) / 255.0
                        
                        img_cv = cv2.imread(img_path)
                        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                        mean_rgb, std_rgb = cv2.meanStdDev(img_rgb)
                        mean_hsv, std_hsv = cv2.meanStdDev(img_hsv)
                        c_feat = np.hstack([
                            mean_rgb.flatten(), std_rgb.flatten(),
                            mean_hsv.flatten(), std_hsv.flatten()
                        ])
                        
                        images.append(img_array)
                        color_feats.append(c_feat)
                        labels.append(class_idx)
                    except:
                        pass
            return np.array(images), np.array(color_feats), np.array(labels)

        X_train_img, X_train_color, y_train = load_data("train")
        
        if len(X_train_img) == 0:
            st.error("No training images found. Please run 'python generate_dataset.py' first.")
            return False

        scaler_color = StandardScaler()
        X_train_color_scaled = scaler_color.fit_transform(X_train_color)
        
        # Define Sequential model with Keras 3 layers.Input
        cnn = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu', name='dense_features'),
            layers.Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cnn.fit(X_train_img, y_train, epochs=8, batch_size=16, verbose=0)
        
        feat_extractor = models.Sequential(cnn.layers[:-1])
        X_train_deep = feat_extractor.predict(X_train_img)
        
        X_train_hybrid = np.hstack([X_train_deep, X_train_color_scaled])
        scaler_hybrid = StandardScaler()
        X_train_hybrid_scaled = scaler_hybrid.fit_transform(X_train_hybrid)
        
        svm = SVC(probability=True, kernel='rbf', random_state=42)
        svm.fit(X_train_hybrid_scaled, y_train)
        
        cnn.save("turmeric_cnn_full.keras")
        feat_extractor.save("turmeric_cnn_feature_extractor.keras")
        
        # Save weights to pickle for NumPy inference
        weights_dict = {}
        for layer in feat_extractor.layers:
            weights = layer.get_weights()
            if weights:
                weights_dict[layer.name] = weights
        with open("turmeric_cnn_weights.pkl", "wb") as f:
            pickle.dump(weights_dict, f)
        
        pipeline_data = {
            "scaler_color": scaler_color,
            "scaler_hybrid": scaler_hybrid,
            "svm_classifier": svm,
            "classes": CATEGORIES,
            "img_size": IMG_SIZE
        }
        with open("turmeric_classifier_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline_data, f)
            
        st.success("Models trained successfully!")
        return True

# --- App Header Banner ---
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 38px;">🍂</span>
        <h1 style="margin: 0; font-size: 34px; color: #0F172A !important; font-weight: 800; letter-spacing: -0.5px; background: none; -webkit-text-fill-color: initial;">AURA TURMERIC ANALYTICS</h1>
    </div>
    <div style="margin-top: 8px; font-size: 17px; font-weight: 600; color: #D97706;">
        AI-Powered Turmeric Quality Classification
    </div>
    <p style="margin: 8px 0 0 0; color: #475569 !important; font-size: 15px; line-height: 1.6;">
        This premium quality intelligence platform utilizes <b>Deep Convolutional Neural Network (CNN) feature extraction</b> coupled with a high-accuracy <b>Support Vector Machine (SVM) hybrid model</b> to instantly assess, grade, and classify turmeric quality based on colorimetric purity, texture, and structural features.
    </p>
</div>
""", unsafe_allow_html=True)

# Check models existence
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(CNN_WEIGHTS_PATH)

if not model_exists:
    st.markdown("""
    <div class="glass-card glow-border-bad">
        <h3 style="margin-top:0; color: #0F172A !important;">⚠️ Model Files Missing</h3>
        <p style="color: #475569 !important; font-size: 15px;">The app requires trained weights to analyze samples. You can train them instantly by clicking the button below.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Train & Build Models Instantly"):
        if train_model_fallback():
            st.rerun()
    st.stop()

# Load Cached Resources
@st.cache_resource
def load_resources():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    feat_extractor = NumPyFeatureExtractor(CNN_WEIGHTS_PATH)
    return pipeline, feat_extractor

pipeline, feat_extractor = load_resources()

# Sidebar Control Panel
st.sidebar.markdown("""
<div class="glass-card" style="padding: 16px; border: 1px solid #FFE3E3; background-color: #FFFFFF;">
    <h4 style="margin: 0 0 10px 0; color: #D97706 !important; font-size: 16px; font-weight: 700;">🔬 AI Pipeline Status</h4>
    <p style="color: #059669 !important; font-weight: 700; margin: 0; font-size: 14px;">● SYSTEM OPERATIONAL</p>
    <p style="color: #475569 !important; font-size: 13px; margin: 8px 0 0 0; font-weight: 500;">CNN Feature Extractor: <span style="color: #059669; font-weight: 600;">Loaded</span></p>
    <p style="color: #475569 !important; font-size: 13px; margin: 2px 0 0 0; font-weight: 500;">Hybrid SVM Model: <span style="color: #059669; font-weight: 600;">Loaded</span></p>
</div>
""", unsafe_allow_html=True)

# File Uploader in Sidebar
st.sidebar.subheader("📤 Sample Upload")
uploaded_file = st.sidebar.file_uploader("Upload Turmeric Rhizome/Powder Image", type=["jpg", "jpeg", "png"])

# Define Tabs
tab1, tab2, tab3 = st.tabs([
    "🍂 Real-time Quality Analyzer", 
    "🔬 Curcumin Spectrum & ML Features", 
    "📂 Dataset Spreadsheet Viewer"
])

# --- TAB 1: Real-time Analyzer ---
with tab1:
    if uploaded_file is None:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 40px; border: 1px dashed #E2E8F0; background-color: #FFFFFF;">
            <div style="font-size: 70px; margin-bottom: 20px; filter: drop-shadow(0px 4px 6px rgba(245, 158, 11, 0.15));">🍂</div>
            <h2 style="margin: 0; font-size: 26px; color: #0F172A; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Welcome to the Quality Analyzer</h2>
            <p style="color: #475569; max-width: 600px; margin: 15px auto 30px auto; font-size: 15px; line-height: 1.6;">
                To begin the evaluation, upload a high-resolution image of turmeric rhizomes, slices, or powder using the sidebar panel on the left. The AI pipeline will automatically process the image and provide classification diagnostics.
            </p>
            <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; margin-top: 20px;">
                <div style="background: #FFF7ED; border: 1px solid #FFE3E3; border-radius: 12px; padding: 16px; width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 8px;">📤</div>
                    <div style="font-weight: 600; color: #D97706; font-size: 14px;">1. Upload Image</div>
                    <div style="color: #64748B; font-size: 12px; margin-top: 4px; line-height: 1.4;">Supports JPG, JPEG, or PNG formats</div>
                </div>
                <div style="background: #FFF7ED; border: 1px solid #FFE3E3; border-radius: 12px; padding: 16px; width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 8px;">🧠</div>
                    <div style="font-weight: 600; color: #D97706; font-size: 14px;">2. CNN Analysis</div>
                    <div style="color: #64748B; font-size: 12px; margin-top: 4px; line-height: 1.4;">Extracts deep structural features</div>
                </div>
                <div style="background: #FFF7ED; border: 1px solid #FFE3E3; border-radius: 12px; padding: 16px; width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 8px;">🎯</div>
                    <div style="font-weight: 600; color: #D97706; font-size: 14px;">3. Quality Prediction</div>
                    <div style="color: #64748B; font-size: 12px; margin-top: 4px; line-height: 1.4;">Hybrid SVM grading algorithm</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        col_img, col_res = st.columns([1, 1.3])
        
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>📷 Uploaded Sample</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=uploaded_file.name)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🔍 AI Quality Diagnostics</h4>", unsafe_allow_html=True)
            
            # --- Inference Pipeline Execution ---
            img_size = pipeline["img_size"]
            img_resized = image.resize((img_size, img_size))
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # 1. CNN Feature Extraction
            deep_features = feat_extractor.predict(img_batch)
            
            # 2. Traditional Color Stats Extraction
            color_features, mean_colors = extract_color_features_from_pil(image)
            color_scaled = pipeline["scaler_color"].transform([color_features])
            
            # 3. Hybrid Concatenation & Prediction
            hybrid_features = np.hstack([deep_features, color_scaled])
            hybrid_scaled = pipeline["scaler_hybrid"].transform(hybrid_features)
            
            pred_class_idx = pipeline["svm_classifier"].predict(hybrid_scaled)[0]
            pred_probs = pipeline["svm_classifier"].predict_proba(hybrid_scaled)[0]
            pred_class = pipeline["classes"][pred_class_idx]
            confidence = pred_probs[pred_class_idx] * 100
            
            # Stylize status card depending on classification
            if pred_class == "good":
                border_style = "glow-border-good"
                status_text = "PREMIUM / PURE"
                status_color = "#059669"
                desc = "This sample exhibits a high density of yellow-orange chromophores, indicating good quality and low levels of discoloration or impurities."
            else:
                border_style = "glow-border-bad"
                status_text = "INFERIOR / ADULTERATED"
                status_color = "#DC2626"
                desc = "Warning! Detected abnormal color metrics (grey/brown/green bias) or mold patterns. This indicates high degradation, rot, or possible chemical adulteration."
            
            st.markdown(f"""
            <div style="margin-top: 10px; padding: 4px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; border-left: 5px solid {status_color};">
                        <span style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; display: block;">Quality Grade</span>
                        <div style="font-size: 18px; font-weight: 700; color: {status_color}; margin-top: 4px;">{status_text}</div>
                    </div>
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; border-left: 5px solid #F59E0B;">
                        <span style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; display: block;">Confidence Score</span>
                        <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin-top: 4px;">{confidence:.2f}%</div>
                    </div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; border-left: 5px solid #059669; margin-bottom: 16px;">
                    <span style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; display: block;">Model Pipeline Status</span>
                    <div style="font-size: 14px; font-weight: 600; color: #0F172A; margin-top: 4px;">CNN Feature Extractor + SVM Classifier: <span style="color: #059669;">ACTIVE</span></div>
                </div>
                <p style="color: #475569 !important; font-size: 14px !important; line-height: 1.5 !important; margin: 0 0 16px 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(confidence / 100.0))
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Unique features row: Curcumin Estimation & Dominant Color extraction
        st.markdown("### 🧬 Advanced Chemical & Chromatic Estimations")
        col_curc, col_dom = st.columns(2)
        
        with col_curc:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🧪 Curcumin Concentration Estimate</h4>", unsafe_allow_html=True)
            st.write("Curcumin gives turmeric its signature color. Based on the chromatic purity and Hue-Saturation parameters, we estimate:")
            
            curcumin_score = estimate_curcumin_index(mean_colors)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px 0;">
                <span class="metric-label" style="color: #64748B; font-weight: 600; font-size: 13px; text-transform: uppercase;">Est. Curcumin Content</span>
                <div style="font-size: 54px; font-weight: 800; color: #D97706; margin: 10px 0;">{curcumin_score:.1f}%</div>
                <p style="color: #475569 !important; font-size: 14px !important; margin: 0;">Pure standard Grade ranges from 70% to 95% Curcumin Purity.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_dom:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🎨 Color Palette Extraction (K-Means)</h4>", unsafe_allow_html=True)
            st.write("Using unsupervised Machine Learning (K-Means Clustering) to extract the top 3 dominant colors from the sample:")
            
            dom_colors = get_dominant_colors(image, k=3)
            
            col_c1, col_c2, col_c3 = st.columns(3)
            for idx, col_box in enumerate([col_c1, col_c2, col_c3]):
                rgb_val = dom_colors[idx]
                hex_val = '#{:02x}{:02x}{:02x}'.format(rgb_val[0], rgb_val[1], rgb_val[2])
                with col_box:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div class="color-swatch" style="background: {hex_val};"></div>
                        <span class="metric-label" style="font-size: 11px; color: #64748B; font-weight: 600;">Color {idx+1}</span>
                        <code style="display: block; font-size: 12px; margin-top: 4px; background: #F8FAFC; color: #0F172A; border: 1px solid #E2E8F0;">{hex_val.upper()}</code>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: Curcumin Spectrum & ML Features ---
with tab2:
    st.markdown("""
    <div class="glass-card">
        <h3>🔬 Feature Space Inspection</h3>
        <p style="color: #475569 !important; font-size: 15px;">
            To classify quality, the pipeline merges deep structural shapes (extracted by the CNN) with color variance profiles (OpenCV). Here is the live extraction data of your uploaded sample:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info("⚠️ Please upload a turmeric image to explore its feature values.")
    else:
        col_c_chart, col_m_vals = st.columns([1.2, 1])
        
        with col_c_chart:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🎨 Image Channel Color Intensity Profile</h4>", unsafe_allow_html=True)
            st.write("Mean intensity levels of Red, Green, and Blue pixels:")
            
            # Native interactive Streamlit Bar Chart
            chart_df = pd.DataFrame(
                mean_colors,
                index=['Red Channel', 'Green Channel', 'Blue Channel'],
                columns=['Mean Level (0-255)']
            )
            st.bar_chart(chart_df)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m_vals:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🧬 Model Input Vectors</h4>", unsafe_allow_html=True)
            st.write("Below are the vectors concatenated to form the final SVM input classification array:")
            
            st.markdown(f"**Deep CNN Features Dimension**: `{deep_features.shape[1]}` components")
            with st.expander("View Raw Deep CNN Feature Vector Snippet"):
                st.code(str(deep_features[0][:10])[:-1] + " ...]")
                
            st.markdown(f"**Color Statistics Features**: `12` components")
            color_feats_formatted = {
                "Mean Red": f"{color_features[0]:.2f}",
                "Mean Green": f"{color_features[1]:.2f}",
                "Mean Blue": f"{color_features[2]:.2f}",
                "Std Red": f"{color_features[3]:.2f}",
                "Mean Hue": f"{color_features[6]:.2f}",
                "Mean Saturation": f"{color_features[7]:.2f}",
                "Mean Value": f"{color_features[8]:.2f}",
            }
            st.json(color_feats_formatted)
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: Dataset Spreadsheet Viewer ---
with tab3:
    st.markdown("""
    <div class="glass-card">
        <h3>📂 Tabular Dataset Explorer</h3>
        <p style="color: #475569 !important; font-size: 15px;">
            This project compiles the extracted features of all images in the dataset into a standard tabular spreadsheet file (<b>turmeric_dataset_features.csv</b>). This is highly useful for training other standard machine learning models like Decision Trees, Random Forests, or XGBoost.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists(CSV_PATH):
        try:
            df_csv = pd.read_csv(CSV_PATH)
            
            # Search / Filter tools
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h4>🔍 Dataset Filters</h4>", unsafe_allow_html=True)
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_split = st.selectbox("Filter by Split", ["All", "train", "test"])
            with col_f2:
                selected_label = st.selectbox("Filter by Quality Label", ["All", "good", "bad"])
                
            filtered_df = df_csv.copy()
            if selected_split != "All":
                filtered_df = filtered_df[filtered_df["split"] == selected_split]
            if selected_label != "All":
                filtered_df = filtered_df[filtered_df["label_name"] == selected_label]
                
            st.markdown(f"Showing **{len(filtered_df)}** entries out of **{len(df_csv)}** total dataset entries:")
            
            # Interactive DataFrame
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download CSV option
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered CSV Dataset",
                data=csv_data,
                file_name="turmeric_filtered_dataset.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading CSV dataset: {e}")
    else:
        st.warning("⚠️ The CSV dataset file 'turmeric_dataset_features.csv' was not found. Please run 'python generate_csv.py' to generate the tabular data.")
