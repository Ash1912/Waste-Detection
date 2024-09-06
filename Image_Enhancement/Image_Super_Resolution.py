import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
from Image_Enhancement.config import Config

# Function to load the super-resolution model
def load_super_resolution_model(model_path=Config.MODEL_PATH):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('edsr', 4)  # Set model and upscale factor
    return sr

# Apply super-resolution to an image based on a threshold criterion
def apply_super_resolution(image, sr, variance_threshold=100):
    if np.var(image) > variance_threshold:  # Apply SR only if variance is above the threshold
        return sr.upsample(image)
    else:
        return image  # Return original if below threshold

# Enhance image with edge highlights and contours
def enhance_image_with_edges_and_contours(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur
        edges = cv2.Canny(blurred, 150, 250)  # Enhance edges using Canny
        contrast_enhanced = cv2.equalizeHist(blurred)  # Adjust contrast using histogram equalization
        enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)  # Merge edges with the base image
        
        highlighted_img_colored = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for plotting
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted_img_colored, contours, -1, (0, 255, 0), 2)  # Draw green contours
        
        return highlighted_img_colored
    except Exception as e:
        print(f"Failed to enhance image: {str(e)}")
        return image  # Return the original if enhancement fails


# Function to preprocess and display multiple images
def process_and_display_images(folder_path, num_images=2):
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(images_paths, min(len(images_paths), num_images))

    sr_model = load_super_resolution_model()  # Load the super-resolution model once

    plt.figure(figsize=(20, num_images * 15))  # Adjust the figure size accordingly
    
    for index, img_path in enumerate(selected_images):
        image = cv2.imread(img_path)
        if image is not None:
            super_res_image = apply_super_resolution(image, sr_model, variance_threshold=10)
            enhanced_image = enhance_image_with_edges_and_contours(super_res_image)
            
            # Plot original image
            plt.subplot(num_images, 2, 2*index+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image {index+1}',pad=20)
            plt.axis('off')
            
            # Plot enhanced super-resolved image
            plt.subplot(num_images, 2, 2*index+2)
            plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Enhanced Super-Resolved Image {index+1}',pad=20)
            plt.axis('off')

    plt.subplots_adjust(hspace=0.6, wspace=0.2)  # Adjust space between rows and columns
    plt.tight_layout()
    plt.show()

# Define the function to clear the directory
def clear_directory(dir_path, force_clear=False):
    """Clears the directory if it exists and force_clear is True, otherwise just ensures it exists."""
    if force_clear and os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # Remove directory and all contents
        print(f"Cleared directory: {dir_path}")
        os.makedirs(dir_path)  # Create the directory if it was cleared
        print(f"Re-prepared directory after clearing: {dir_path}")

def user_decision_to_clear():
    response = input("Clear the existing processed images directory? (y/n): ").strip().lower()
    return response == 'y'

def directory_is_empty(dir_path):
    """Check if the directory is empty or does not exist."""
    return not os.listdir(dir_path) if os.path.exists(dir_path) else True

# Define the main processing function
def preprocess_and_save_images(input_dir, output_dir, sr, force_clear=False):
    """Preprocess and save images with an option to not clear the directory."""
    clear_directory(output_dir, force_clear=force_clear)  # Prepare the output directory based on user decision

    processed_images = []  # Store processed images
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.jpg'):
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    # Assuming apply_super_resolution and enhance_image_with_edges_and_contours are defined
                    super_res_image = apply_super_resolution(image, sr)  # Placeholder function
                    enhanced_image = enhance_image_with_edges_and_contours(super_res_image)  # Placeholder function

                    # Save the processed image
                    relative_path = os.path.relpath(root, input_dir)
                    save_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, filename)
                    cv2.imwrite(output_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for correct color saving
                    print(f"Processed and saved: {output_path}")
                    processed_images.append(enhanced_image)
                else:
                    print(f"Failed to load image {file_path}")

    print(f"Total processed images: {len(processed_images)}")
    return processed_images

if __name__ == '__main__':
    # Directory containing the images
    image_folder = './Dataset/Water_Trash_Dataset/images/train'
    process_and_display_images(image_folder, num_images=2)
