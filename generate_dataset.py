import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_directory_structure(base_dir="dataset"):
    """Creates the dataset folders for train and test splits."""
    paths = [
        os.path.join(base_dir, "train", "good"),
        os.path.join(base_dir, "train", "bad"),
        os.path.join(base_dir, "test", "good"),
        os.path.join(base_dir, "test", "bad")
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    print("Dataset directories created successfully.")

def generate_synthetic_turmeric_image(quality="good", size=(128, 128)):
    """Generates a synthetic image of turmeric (rhizome or powder)."""
    # Create background (light grey/beige table texture)
    bg_color = (random.randint(220, 240), random.randint(215, 230), random.randint(205, 220))
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add subtle background noise (wood/table texture)
    for _ in range(200):
        x = random.randint(0, size[0]-1)
        y = random.randint(0, size[1]-1)
        draw.point((x, y), fill=(random.randint(180, 210), random.randint(180, 210), random.randint(180, 210)))
        
    # Draw turmeric shape (rhizome: a series of overlapping ellipses, powder: a mound/polygon)
    is_rhizome = random.choice([True, False])
    
    if quality == "good":
        # Rich deep orange-yellow tones
        primary_color = (random.randint(220, 255), random.randint(140, 180), random.randint(0, 30))
        secondary_color = (random.randint(200, 230), random.randint(110, 140), random.randint(0, 20))
    else:
        # Dull yellow, greyish, brown, or moldy green spots
        primary_color = (random.randint(120, 170), random.randint(110, 140), random.randint(60, 100))
        secondary_color = (random.randint(80, 120), random.randint(80, 110), random.randint(50, 80))

    if is_rhizome:
        # Generate irregular rhizome shape
        center_x = size[0] // 2 + random.randint(-10, 10)
        center_y = size[1] // 2 + random.randint(-10, 10)
        
        # Main body
        width = random.randint(40, 60)
        height = random.randint(20, 35)
        angle = random.randint(0, 180)
        
        # Draw overlapping segments to look like a ginger/turmeric root
        segments = random.randint(2, 4)
        for i in range(segments):
            seg_x = center_x + random.randint(-20, 20)
            seg_y = center_y + random.randint(-15, 15)
            seg_w = random.randint(20, 35)
            seg_h = random.randint(15, 25)
            
            # Draw segment
            draw.ellipse(
                [seg_x - seg_w, seg_y - seg_h, seg_x + seg_w, seg_y + seg_h],
                fill=primary_color,
                outline=secondary_color
            )
            
            # Texture lines on rhizome
            for line_offset in range(-10, 11, 5):
                draw.arc(
                    [seg_x - seg_w + 3, seg_y - seg_h + line_offset, seg_x + seg_w - 3, seg_y + seg_h + line_offset],
                    start=30, end=150,
                    fill=secondary_color
                )
    else:
        # Generate a powder pile (polygon mound)
        center_x = size[0] // 2 + random.randint(-10, 10)
        center_y = size[1] // 2 + random.randint(-10, 10)
        r = random.randint(30, 45)
        
        # Build list of points for an irregular mound
        num_points = 8
        points = []
        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            dist = r + random.randint(-8, 8)
            x = center_x + int(dist * np.cos(angle))
            y = center_y + int(dist * np.sin(angle))
            points.append((x, y))
            
        draw.polygon(points, fill=primary_color, outline=secondary_color)
        
        # Add granular texture to powder
        for _ in range(300):
            gx = center_x + random.randint(-r+5, r-5)
            gy = center_y + random.randint(-r+5, r-5)
            # check if point is inside approximate circle
            if (gx - center_x)**2 + (gy - center_y)**2 < (r-5)**2:
                draw.point((gx, gy), fill=secondary_color if random.random() > 0.5 else (255, 220, 100))

    # Add quality-specific characteristics
    if quality == "bad":
        # Add moldy patches (grey/green) or dark decay spots
        for _ in range(random.randint(3, 8)):
            spot_x = size[0] // 2 + random.randint(-25, 25)
            spot_y = size[1] // 2 + random.randint(-25, 25)
            spot_r = random.randint(4, 10)
            spot_color = random.choice([
                (70, 90, 70),   # mold green
                (40, 40, 40),   # black rot
                (140, 110, 90)  # dry decay
            ])
            draw.ellipse(
                [spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r],
                fill=spot_color
            )
            
    # Apply a light blur to make features look natural and continuous
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img

def build_dataset(base_dir="dataset", num_train_per_class=60, num_test_per_class=15):
    """Generates and saves the synthetic dataset."""
    create_directory_structure(base_dir)
    
    categories = ["good", "bad"]
    
    for cat in categories:
        # Generate Train images
        print(f"Generating {num_train_per_class} training images for category: {cat}...")
        for i in range(num_train_per_class):
            img = generate_synthetic_turmeric_image(quality=cat)
            filename = os.path.join(base_dir, "train", cat, f"turmeric_{cat}_{i:03d}.jpg")
            img.save(filename, "JPEG")
            
        # Generate Test images
        print(f"Generating {num_test_per_class} test/validation images for category: {cat}...")
        for i in range(num_test_per_class):
            img = generate_synthetic_turmeric_image(quality=cat)
            filename = os.path.join(base_dir, "test", cat, f"turmeric_{cat}_{i:03d}.jpg")
            img.save(filename, "JPEG")

    print("\nDataset generation completed successfully!")
    print(f"Total training images: {num_train_per_class * 2}")
    print(f"Total test images: {num_test_per_class * 2}")

if __name__ == "__main__":
    # Use random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    build_dataset()
