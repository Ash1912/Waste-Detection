import cv2
import numpy as np
import os
import shutil
from Data_Classification.config import Config

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"Cleared and prepared directory: {dir_path}")

def detect_waste(image):
    return np.mean(image) < 100  # Example logic for waste detection

def preprocess_image(image, size=Config.IMAGE_SIZE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 150, 250)
    contrast_enhanced = cv2.equalizeHist(blurred)
    enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
    resized_image = cv2.resize(enhanced_image, size)
    normalized_image = resized_image / 255.0
    is_waste = detect_waste(normalized_image)
    return normalized_image.flatten(), is_waste

def preprocess_and_classify_images(input_dir, output_dirs):
    for dir_type in [Config.TRAIN_DIR, Config.TEST_DIR]:
        for class_name in Config.CLASSES:
            clear_directory(os.path.join(output_dirs[dir_type], class_name))

    processed_images = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.jpg'):
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    processed_image, is_waste = preprocess_image(image)
                    class_subdir = 'waste' if is_waste else 'no_waste'
                    train_test_dir = Config.TRAIN_DIR if 'train' in root else Config.TEST_DIR
                    save_dir = os.path.join(output_dirs[train_test_dir], class_subdir)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, processed_image * 255)
                    processed_images.append(processed_image)
                    print(f"Processed and saved: {filename} to {class_subdir} in {train_test_dir}")
                else:
                    print(f"Failed to load image {filename}")
    print(f"Total preprocessed images: {len(processed_images)}")

if __name__ == '__main__':
    input_dir = os.path.join(Config.BASE_DIR, 'images')
    output_dirs = {
        Config.TRAIN_DIR: os.path.join(Config.DESTINATION_DIR, Config.TRAIN_DIR),
        Config.TEST_DIR: os.path.join(Config.DESTINATION_DIR, Config.TEST_DIR)
    }
    preprocess_and_classify_images(input_dir, output_dirs)
