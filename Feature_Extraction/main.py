import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from Feature_Extraction.config import Config

def load_and_preprocess_images(directory):
    """Load images from a directory and preprocess them."""
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is not None:
                preprocessed_image = preprocess_image(image)
                images.append(preprocessed_image)
            else:
                print(f"Failed to load image: {filename}")
    return images

def preprocess_image(image):
    """Convert image to grayscale and resize it according to configuration."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, Config.IMAGE_SIZ)
    return resized_image

def extract_features(images):
    """Extract combined HOG and LBP features from a list of images."""
    hog_features = []
    lbp_features = []
    for image in images:
        # Assuming image is already reshaped into its original 2D shape (32x32)
        image_reshaped = image.reshape(32, 32)
        
        # Convert image to 8-bit unsigned integers for LBP
        image_uint = np.uint8(image_reshaped * 255)
        
        # Extract HOG features: captures edge direction and intensity information
        fd_hog = hog(image_reshaped, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_features.append(fd_hog)
        
        # Extract LBP features: provides a texture descriptor
        lbp = local_binary_pattern(image_uint, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 256), density=True)
        lbp_features.append(lbp_hist)
    
    # Convert lists to numpy arrays for easier handling
    hog_features = np.array(hog_features)
    lbp_features = np.array(lbp_features)
    
    # Concatenate HOG and LBP features into a single feature vector per image
    combined_features = np.hstack((hog_features, lbp_features))
    return combined_features

if __name__ == '__main__':
    image_directory = Config.IMAGE_DIR  # Ensure this is set in your config.py
    images = load_and_preprocess_images(image_directory)
    if images:
        features = extract_features(images)
        print(f"Extracted features for {len(features)} images.")
    else:
        print("No images to process.")

