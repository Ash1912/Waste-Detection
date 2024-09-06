import os
import cv2
import numpy as np
from Data_Shuffling.config import Config

def load_images_from_directory(directory):
    """Load and preprocess images from a specified directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is not None:
                processed_image = preprocess_image(image)
                images.append(processed_image)
            else:
                print(f"Failed to load image: {filename}")
    return images

def preprocess_image(image, size=Config.IMAGE_SIZE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 150, 250)
    contrast_enhanced = cv2.equalizeHist(blurred)
    enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
    resized_image = cv2.resize(enhanced_image, size)
    normalized_image = resized_image / 255.0
    return normalized_image.flatten()

def shuffle_data(images):
    """Shuffle the provided list of images."""
    np.random.seed(Config.SHUFFLE_SEED)
    np.random.shuffle(images)
    return images

if __name__ == '__main__':
    image_directory = Config.IMAGE_DIRECTORY  # Make sure to set this in your config module
    preprocessed_images = load_images_from_directory(image_directory)
    if preprocessed_images:
        shuffled_images = shuffle_data(preprocessed_images)
        print(f"Data shuffled. Total images: {len(shuffled_images)}")
    else:
        print("No preprocessed images available for shuffling.")
