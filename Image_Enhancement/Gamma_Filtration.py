# Import functions from the super-resolution script
from Image_Enhancement.Image_Super_Resolution import load_super_resolution_model, apply_super_resolution, enhance_image_with_edges_and_contours

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil

# Function to adjust gamma of the image
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Integrating gamma correction within the processing pipeline
def apply_gamma_correction(image, gamma_value=2.0, luminance_threshold=0.5):
    # Convert to grayscale to compute the mean luminance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_luminance = np.mean(gray_image)
    
    # Check if the luminance is below a certain threshold to decide whether to apply gamma correction
    if mean_luminance < luminance_threshold * 255:  # Normalize threshold to pixel range
        return adjust_gamma(image, gamma=gamma_value)
    return image

# Function to apply gamma correction for visualization
def display_images_with_gamma_correction(folder_path, num_images=2, gamma_value=2.0, luminance_threshold=0.5):
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(images_paths, min(len(images_paths), num_images))
    sr_model = load_super_resolution_model()  # Load the super-resolution model once

    # Calculate appropriate figure size: 15 inches wide by 10 inches per image set
    plt.figure(figsize=(15, 10 * num_images))

    for index, img_path in enumerate(selected_images):
        image = cv2.imread(img_path)
        if image is not None:
            super_res_image = apply_super_resolution(image, sr_model, variance_threshold=10)
            enhanced_image = enhance_image_with_edges_and_contours(super_res_image)
            gamma_corrected_image = apply_gamma_correction(enhanced_image, gamma_value, luminance_threshold)

            # Plot original image
            plt.subplot(num_images, 3, 3*index + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image {index + 1}')
            plt.axis('off')

            # Plot enhanced super-resolved image
            plt.subplot(num_images, 3, 3*index + 2)
            plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Enhanced Super-Resolved Image {index + 1}')
            plt.axis('off')

            # Plot gamma corrected image
            plt.subplot(num_images, 3, 3*index + 3)
            plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Gamma Corrected Image {index + 1}', pad=10)
            plt.axis('off')

    # Adjust layout to avoid overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust horizontal and vertical spacing
    plt.tight_layout()
    plt.show()

# Function to clear a directory
def clear_directory(dir_path, force_clear=False):
    """Clears the directory if it exists and force_clear is True, otherwise just ensures it exists."""
    if force_clear and os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # Remove directory and all contents
        print(f"Cleared directory: {dir_path}")
        os.makedirs(dir_path)  # Create the directory if it was cleared
        print(f"Re-prepared directory after clearing: {dir_path}")

def user_decision_to_clear():
    """Ask user whether to clear the existing directory or not."""
    response = input("Clear the existing processed images directory? (y/n): ").strip().lower()
    return response == 'y'

def directory_is_empty(dir_path):
    """Check if the directory is empty or does not exist."""
    return not os.listdir(dir_path) if os.path.exists(dir_path) else True

    
# Modify the existing processing function to include gamma correction
def preprocess_and_save_images(input_dir, output_dir, gamma_value=2.0, luminance_threshold=0.5, force_clear=False):
    """Preprocess and save images with an option to not clear the directory."""
    if force_clear or directory_is_empty(output_dir):
        clear_directory(output_dir, force_clear=force_clear)  # Prepare the output directory based on user decision
        sr_model = load_super_resolution_model()  # Assume this function is defined elsewhere
        processed_images = []
        
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.jpg'):
                    file_path = os.path.join(root, filename)
                    image = cv2.imread(file_path)
                    if image is not None:
                        # Apply super-resolution
                        super_res_image = apply_super_resolution(image, sr_model, variance_threshold=10)  # Assume defined elsewhere
                        # Enhance image
                        enhanced_image = enhance_image_with_edges_and_contours(super_res_image)  # Assume defined elsewhere
                        # Apply gamma correction
                        gamma_corrected_image = apply_gamma_correction(enhanced_image, gamma_value, luminance_threshold)  # Assume defined elsewhere
                        
                        # Save the processed image
                        relative_path = os.path.relpath(root, input_dir)
                        save_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(save_dir, exist_ok=True)
                        output_path = os.path.join(save_dir, filename)
                        cv2.imwrite(output_path, cv2.cvtColor(gamma_corrected_image, cv2.COLOR_RGB2BGR))
                        print(f"Processed and saved: {output_path}")
                        processed_images.append(gamma_corrected_image)
                    else:
                        print(f"Failed to load image {file_path}")
        print(f"Total processed images: {len(processed_images)}")
        return processed_images
    else:
        print(f"All images are already processed and saved in this directory: {output_dir}")

if __name__ == '__main__':
    
    # Example usage
    image_folder = './Dataset/Water_Trash_Dataset/images/train'
    display_images_with_gamma_correction(image_folder, num_images=2)