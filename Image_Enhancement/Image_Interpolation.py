from Image_Enhancement.Image_Super_Resolution import load_super_resolution_model, apply_super_resolution, enhance_image_with_edges_and_contours
from Image_Enhancement.Gamma_Filtration import apply_gamma_correction

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Function to apply interpolation to an image
def apply_interpolation(image, method='bilinear', scale_factor=1.5):
    height, width = image.shape[:2]
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        return image  # Return the original if no valid method specified

    # Calculate new size
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return resized_image

# Example usage within the visualization function
def display_interpolation_effects(folder_path, num_images=5, scale_factor=1.5):
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(images_paths, min(len(images_paths), num_images))
    sr_model = load_super_resolution_model()  # Load the super-resolution model once

    plt.figure(figsize=(20, num_images * 6))  # Adjust the figure size accordingly

    for index, img_path in enumerate(selected_images):
        image = cv2.imread(img_path)
        if image is not None:
            super_res_image = apply_super_resolution(image, sr_model, variance_threshold=10)
            enhanced_image = enhance_image_with_edges_and_contours(super_res_image)
            gamma_corrected_image = apply_gamma_correction(enhanced_image, 2.0, 0.5)

            # Interpolation effects
            nearest_image = apply_interpolation(gamma_corrected_image, method='nearest', scale_factor=scale_factor)
            bilinear_image = apply_interpolation(gamma_corrected_image, method='bilinear', scale_factor=scale_factor)
            bicubic_image = apply_interpolation(gamma_corrected_image, method='bicubic', scale_factor=scale_factor)

            # Plot original and interpolated images
            plt.subplot(num_images, 4, 4*index + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image {index + 1}')
            plt.axis('off')

            plt.subplot(num_images, 4, 4*index + 2)
            plt.imshow(cv2.cvtColor(nearest_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Nearest Neighbor Interpolation')
            plt.axis('off')

            plt.subplot(num_images, 4, 4*index + 3)
            plt.imshow(cv2.cvtColor(bilinear_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Bilinear Interpolation')
            plt.axis('off')

            plt.subplot(num_images, 4, 4*index + 4)
            plt.imshow(cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Bicubic Interpolation')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image_folder = './Dataset/Water_Trash_Dataset/images/train'
    display_interpolation_effects(image_folder, num_images=5, scale_factor=1.5)