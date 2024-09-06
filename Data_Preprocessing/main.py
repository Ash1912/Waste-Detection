import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from Data_Preprocessing.config import Config

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"Cleared and prepared directory: {dir_path}")

def preprocess_image(image, size=Config.IMAGE_SIZE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 150, 250)
    contrast_enhanced = cv2.equalizeHist(blurred)
    enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
    resized_image = cv2.resize(enhanced_image, size)
    normalized_image = resized_image / 255.0
    return normalized_image.flatten()

def preprocess_and_save_images(input_dir=Config.INPUT_DIR, output_dir=Config.OUTPUT_DIR):
    clear_directory(output_dir)
    processed_images = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if any(filename.endswith(ext) for ext in Config.VALID_IMAGE_EXTENSIONS):
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    processed_image = preprocess_image(image)
                    processed_images.append(processed_image)
                    relative_path = os.path.relpath(root, input_dir)
                    save_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, filename)
                    cv2.imwrite(output_path, processed_image.reshape(Config.IMAGE_SIZE) * 255)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Failed to load image {file_path}")
    print(f"Total processed images: {len(processed_images)}")
    return processed_images

def process_and_display_image(image_path=Config.IMAGE_PATH):
    original, enhanced = process_image(image_path)
    if original is not None and enhanced is not None:
        display_images(original, enhanced)
        print('Image dimensions:', original.shape)
    else:
        print("Failed to process the image due to an error.")

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or unable to load: {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 150, 250)
    contrast_enhanced = cv2.equalizeHist(blurred)
    enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
    return image, enhanced_image

def display_images(original, enhanced):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Grayscale Image')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    preprocess_and_save_images()
    process_and_display_image()
