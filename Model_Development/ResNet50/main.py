# Import Libraries
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from Model_Development.ResNet50.config import Config

# Ensure true randomness in selecting images
random.seed()

# Preparing the Image Data
# The images need to be preprocessed to match the input requirements of ResNet50. This involves resizing the images to 
# 224x224 pixels (the input size expected by ResNet50) and applying specific preprocessing like normalization

#Load and Preprocess an Image
def load_and_preprocess_image(img_path):
    """
    Load an image file, resizing it to 224x224 pixels and preprocessing it for ResNet50.
    """
    img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function to load and preprocess an image for ResNet50
def load_and_preprocess_image(img_path):
    """
    Load an image file, resizing it to 224x224 pixels and preprocessing it for ResNet50.
    """
    img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

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
        contrast_enhanced = cv2.equalizeHist(blurred)
        
        # Merge edges back into the original image to highlight them
        enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
        
        # Convert the enhanced image to RGB for plotting
        highlighted_img_colored = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        
        # Draw contours around the edges for better visibility
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(highlighted_img_colored, contours, -1, (0, 255, 0), 2)  # Draw green contours

        return highlighted_img_colored
    except Exception as e:
        print(f"Failed to process image {img_path}: {str(e)}")
        return None

# Utilizing ResNet50 for Object Detection
# Load the ResNet50 model with weights pre-trained on the ImageNet dataset. This model is used here for image classification, 
# but the top predictions can provide insight into the types of objects present in the image.

# Function to predict using ResNet50
def predict_with_resnet(img_path, model_resnet):
    """
    Predict the content of an image using ResNet50 and return the top-3 predictions.
    """
    try:
        processed_image = load_and_preprocess_image(img_path)
        predictions = model_resnet.predict(processed_image)
        return decode_predictions(predictions, top=3)[0]
    except Exception as e:
        print(f"Failed to make prediction for image {img_path}: {str(e)}")
        return []

# Function to display images and predictions
def display_images_and_predictions():
    """
    Display original and edge-highlighted images (in grayscale to emphasize edges)
    along with ResNet50 predictions. Selects a random subset of images from the directory.
    """
    
    model_resnet = ResNet50(weights='imagenet')  # Load the model once for all predictions
    images = [os.path.join(Config.IMAGES_DIRECTORY, img) for img in os.listdir(Config.IMAGES_DIRECTORY) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(images, min(len(images), Config.NUM_IMAGES))

    cols = 3  # Three columns for original, highlighted images, and prediction text
    rows = Config.NUM_IMAGES

    plt.figure(figsize=(20, 6 * rows))  # Adjust figure size dynamically based on the number of images

    for i, img_path in enumerate(selected_images):
        original_img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
        highlighted_img = add_edge_highlights(img_path)
        predictions = predict_with_resnet(img_path, model_resnet)

        # Display the original image
        ax1 = plt.subplot(rows, cols, 3 * i + 1)
        plt.imshow(original_img)
        plt.axis('off')
        ax1.set_title("Original Image")

        # Display the edge-highlighted image in grayscale
        ax2 = plt.subplot(rows, cols, 3 * i + 2)
        if highlighted_img is not None:
            plt.imshow(highlighted_img)
        plt.axis('off')
        ax2.set_title("Edge Highlighted Image")

        # Display prediction texts in a separate subplot
        ax3 = plt.subplot(rows, cols, 3 * i + 3)
        ax3.axis('off')
        prediction_texts = "\n".join([f'{p[1]}: {p[2]*100:.2f}%' for p in predictions])
        plt.text(0.5, 0.5, "Predictions:\n" + prediction_texts, ha='center', va='center', fontsize=12)
        ax3.set_title("Predictions")

    # Adjust subplot parameters to modify spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust horizontal spacing if needed, hspace controls vertical space between subplots
    plt.tight_layout()
    plt.show()

    # Save the model in h5 format
    model_resnet.save(os.path.join(Config.MODEL_SAVE_PATH, 'resnet50_model.h5'))
    print("ResNet50 Model saved successfully.")

if __name__ == "__main__":
    # Execute the function
    display_images_and_predictions()
