# Turmeric Quality Classification using CNN and Machine Learning

This is an end-to-end Machine Learning and Deep Learning project designed for ML students to classify turmeric quality. It distinguishes between **Good Quality** (pure, bright orange-yellow turmeric) and **Bad Quality** (degraded, moldy, greyish, or adulterated turmeric) using a hybrid approach combining **Convolutional Neural Networks (CNN)** and **traditional color features**.

---

## 🏗️ Architecture Overview

To achieve robust and lightweight quality classification, this project uses a **Hybrid Classifier**:
1. **Deep CNN Features**: A Convolutional Neural Network extracts spatial patterns, texture, and structural characteristics (shape of rhizomes or powder mounds) from the images.
2. **Traditional ML Features**: Average intensity and variance statistics from the **RGB** and **HSV** color spaces are extracted to capture exact color tones (essential for turmeric purity).
3. **SVM Classifier**: The CNN deep features (64 dimensions) and traditional color features (12 dimensions) are concatenated and classified using a **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel.
4. **Connectivity**: The pipeline (scalers and SVM) is saved in a **Pickle** (`.pkl`) file, while the CNN weights are serialized in Keras (`.keras`) format.

---

## 📂 Project Structure

```text
turmeric_quality_classification/
│
├── dataset/                     # Generated image dataset (train & test splits)
│   ├── train/
│   │   ├── good/
│   │   └── bad/
│   └── test/
│       ├── good/
│       └── bad/
│
├── app.py                       # Premium Streamlit web application
├── generate_dataset.py          # Synthetic dataset generator script
├── generate_csv.py              # Script to extract features and build the CSV dataset
├── turmeric_dataset_features.csv# Tabular CSV dataset containing image paths, labels, and color features
├── turmeric_classification.ipynb# Jupyter Notebook for preprocessing, training & serialization
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

---

## 🚀 Getting Started

Follow these steps to run the project on your local machine:

### 1. Set Up Environment
It is recommended to use a virtual environment (conda or venv):
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset & CSV File
If you don't have a dataset of real turmeric images yet, run the synthetic dataset generator script. This creates a realistic dummy dataset of 150 images representing good and bad turmeric characteristics. Afterward, compile these images and their extracted color features into a tabular CSV file:
```bash
# Generate image files
python generate_dataset.py

# Compile them and extract features into a CSV
python generate_csv.py
```
*Note: To use a real dataset, simply replace the images in the `dataset/train/good`, `dataset/train/bad`, `dataset/test/good`, and `dataset/test/bad` folders with your own images and re-run `python generate_csv.py` to rebuild your CSV dataset.*

### 3. Open and Run Jupyter Notebook
Open the notebook to explore the preprocessing, feature extraction, CNN model architecture, hybrid classifier, and serialization steps:
```bash
jupyter notebook turmeric_classification.ipynb
```
Run all cells in the notebook. This will output:
- `turmeric_cnn_full.keras` (full trained CNN model)
- `turmeric_cnn_feature_extractor.keras` (CNN feature extractor model)
- `turmeric_classifier_pipeline.pkl` (pickled scaler and SVM classifier pipeline)

### 4. Launch the Streamlit User Interface
Once the model files are generated, launch the Streamlit app to analyze images in real-time through a premium web dashboard:
```bash
streamlit run app.py
```
*(If the model files are missing when you open the app, it will display a warning and offer an **"Instantly Train Models"** button to train and load them automatically on the spot!)*

---

## 🎨 Features in Streamlit App
- **Interactive File Upload**: Upload images of turmeric powder or roots.
- **Real-Time Inference**: Displays classification results ("Good Quality" vs. "Bad Quality") with a confidence progress bar.
- **Interactive Visualizations**: Shows an RGB Color Profile chart of the uploaded image to highlight color analysis.
- **Model Pipeline Metrics**: Explains the structural data dimensions flowing from the CNN to the SVM.
