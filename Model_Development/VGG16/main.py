# Import necessary libraries
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from Model_Development.VGG16.config import Config
# Ensure true randomness in selecting images
random.seed()

# Function to load and preprocess images for VGG16
def load_and_preprocess_image_vgg(img_path):
    """
    Load an image file, resizing it to 224x224 pixels and preprocessing it for VGG16.
    """
    img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function to predict using VGG16
def predict_with_vgg16(img_path, model):
    """
    Predict the content of an image using VGG16 and return the top-3 predictions.
    """
    processed_image = load_and_preprocess_image_vgg(img_path)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=3)[0]

# Load the VGG16 model
vgg_model = VGG16(weights='imagenet')

# Path to a sample image
image_path = './Dataset/Water_Trash_Dataset/images/train/800wm.jpg'

# Predict and display results
predictions = predict_with_vgg16(image_path, vgg_model)
print("VGG16 Predictions:")
for p in predictions:
    print(p)

# Function to add edge highlights to an image
def add_edge_highlights(img_path):
    """
    Add edge highlights to an image using Canny edge detection.
    This helps to emphasize the structure within the image by displaying it in grayscale.
    """
    try:
        original_img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance edges using Canny Edge Detection
        edges = cv2.Canny(blurred, 150, 250)
        
        # Adjust contrast using histogram equalization
        contrast_enhanced = cv2.equalizeHist(gray)
        
        # Merge edges back into the original image to highlight them
        enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
        
        # Convert the enhanced image to RGB for plotting
        highlighted_img_colored = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        
        # Draw contours around the edges for better visibility
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted_img_colored, contours, -1, (0, 255, 0), 2)  # Draw green contours

        return highlighted_img_colored
    except Exception as e:
        print(f"Failed to process image {img_path}: {str(e)}")
        return None

# Function to display images and predictions
def display_images_and_predictions_vgg():
    """
    Display original and edge-highlighted images along with VGG16 predictions.
    Selects a random subset of images from the directory.
    """
    
    model = VGG16(weights='imagenet')
    images = [os.path.join(Config.IMAGES_DIRECTORY, img) for img in os.listdir(Config.IMAGES_DIRECTORY) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(images, min(len(images), Config.NUM_IMAGES))

    cols = 3  # Adding one extra column for predictions
    rows = len(selected_images)

    plt.figure(figsize=(15, 10 * rows))

    for i, img_path in enumerate(selected_images):
        original_img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
        highlighted_img = add_edge_highlights(img_path)
        predictions = predict_with_vgg16(img_path, model)

        ax1 = plt.subplot(rows, cols, 3 * i + 1)
        plt.imshow(original_img)
        plt.axis('off')
        ax1.set_title("Original Image")

        ax2 = plt.subplot(rows, cols, 3 * i + 2)
        if highlighted_img is not None:
            plt.imshow(highlighted_img, cmap='gray')  # Ensure the highlighted image is shown in grayscale
        plt.axis('off')
        ax2.set_title("Edge Highlighted")

        ax3 = plt.subplot(rows, cols, 3 * i + 3)
        ax3.axis('off')
        prediction_texts = "\n".join([f'{p[1]}: {p[2]*100:.2f}%' for p in predictions])
        plt.text(0.5, 0.5, "Predictions:\n" + prediction_texts, ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Predictions")

    # Adjust subplot parameters to modify spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust spacing to prevent any overlap
    plt.tight_layout()
    plt.show()

    # Save the model in h5 format
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'vgg16_model.h5'))
    print("VGG16 Model saved successfully.")

if __name__ == '__main__':
    # Display images and predictions using VGG16
    display_images_and_predictions_vgg()
