import os
import cv2
import numpy as np
import pandas as pd

def extract_color_features(img_path):
    """Extracts average color and variance features in RGB and HSV color spaces."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mean_rgb, std_rgb = cv2.meanStdDev(img_rgb)
    mean_hsv, std_hsv = cv2.meanStdDev(img_hsv)
    
    return {
        "mean_R": mean_rgb[0][0],
        "mean_G": mean_rgb[1][0],
        "mean_B": mean_rgb[2][0],
        "std_R": std_rgb[0][0],
        "std_G": std_rgb[1][0],
        "std_B": std_rgb[2][0],
        "mean_H": mean_hsv[0][0],
        "mean_S": mean_hsv[1][0],
        "mean_V": mean_hsv[2][0],
        "std_H": std_hsv[0][0],
        "std_S": std_hsv[1][0],
        "std_V": std_hsv[2][0]
    }

def create_dataset_csv(base_dir="dataset", output_csv="turmeric_dataset_features.csv"):
    """Scans the image folder structure and extracts features into a single CSV file."""
    data = []
    
    splits = ["train", "test"]
    categories = ["good", "bad"]
    
    print("Scanning dataset and extracting features...")
    
    for split in splits:
        for class_idx, cat in enumerate(categories):
            class_dir = os.path.join(base_dir, split, cat)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                # Extract features
                features = extract_color_features(img_path)
                if features is not None:
                    # Append metadata
                    row = {
                        "image_name": img_name,
                        "image_path": img_path.replace("\\", "/"),
                        "split": split,
                        "label_idx": class_idx,
                        "label_name": cat
                    }
                    row.update(features)
                    data.append(row)
                    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV dataset created successfully with {len(df)} rows: '{output_csv}'")
    print(df.head())

if __name__ == "__main__":
    create_dataset_csv()
