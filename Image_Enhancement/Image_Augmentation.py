import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Function to rotate an image
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# Function to shift an image
def translate_image(image, x, y):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    shifted_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return shifted_image

# Function to flip an image horizontally
def flip_image(image, flipCode=1):
    return cv2.flip(image, flipCode)

# Function to adjust image brightness
def adjust_brightness(image, brightness_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], brightness_factor)
    bright_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return bright_image

# Function to apply various augmentations and display them
def display_augmentations(folder_path, num_images=5):
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(images_paths, min(len(images_paths), num_images))

    plt.figure(figsize=(20, num_images * 6))  # Adjust the figure size accordingly
    
    for index, img_path in enumerate(selected_images):
        image = cv2.imread(img_path)
        if image is not None:
            rotated_image = rotate_image(image, 45)  # Rotate by 45 degrees
            shifted_image = translate_image(image, 50, -50)  # Shift right by 50 and up by 50
            flipped_image = flip_image(image)  # Horizontal flip
            bright_image = adjust_brightness(image, 40)  # Increase brightness by 40 units
            
            # Plot original and augmented images
            plt.subplot(num_images, 5, 5*index + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.axis('off')

            plt.subplot(num_images, 5, 5*index + 2)
            plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
            plt.title('Rotated')
            plt.axis('off')

            plt.subplot(num_images, 5, 5*index + 3)
            plt.imshow(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB))
            plt.title('Shifted')
            plt.axis('off')

            plt.subplot(num_images, 5, 5*index + 4)
            plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
            plt.title('Flipped')
            plt.axis('off')

            plt.subplot(num_images, 5, 5*index + 5)
            plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
            plt.title('Brightness Adjusted')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Directory containing the images
    image_folder = './Dataset/Water_Trash_Dataset/images/train'
    display_augmentations(image_folder, num_images=5)
