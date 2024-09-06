import os
import cv2
import numpy as np
from Data_Processing.config import Config
from Data_Saving.main import InMemoryImageStore

def load_super_resolution_model():
    """ Load and return the super-resolution model. """
    print("Loading super-resolution model...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(Config.MODEL_PATH)
    sr.setModel('edsr', 4)  # Set model to EDSR and upscale factor to 4
    return sr

def apply_enhancements(image):
    """ Apply various image enhancements and return the processed image. """
    print("Applying image enhancements...")
    # Super-resolution
    sr = load_super_resolution_model()  # Load model
    if np.var(image) > 100:  # Apply super-resolution based on variance
        image = sr.upsample(image)

    # Convert to grayscale, apply Gaussian Blur and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 150, 250)
    contrast_enhanced = cv2.equalizeHist(blurred)
    enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
    highlighted = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    # Gamma correction for low luminance
    mean_luminance = np.mean(gray)
    if mean_luminance < Config.LUMINANCE_THRESHOLD * 255:
        inv_gamma = 1.0 / Config.GAMMA_VALUE
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        highlighted = cv2.LUT(highlighted, table)

    # Resize and normalize the image
    resized_image = cv2.resize(highlighted, Config.IMAGE_SIZE)
    normalized_image = resized_image / 255.0
    return normalized_image.flatten()

def process_images(directory):
    """ Process a list of images and return the processed images. """
    print("Processing images...")
    image_store = InMemoryImageStore()
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(tuple(Config.VALID_IMAGE_EXTENSIONS))]

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = apply_enhancements(image)
            image_store.save_image(processed_image)
        else:
            print(f"Failed to load image from {image_path}")

    print(f"Total images saved in memory: {len(image_store.get_images())}")

if __name__ == '__main__':
    process_images(Config.TRAIN_IMAGE_DIR)  # Process training images

