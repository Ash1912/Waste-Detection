import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from Model_Development.InceptionV3.config import Config

def load_and_preprocess_image_inception(img_path):
    img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def predict_with_inception(img_path, model):
    processed_image = load_and_preprocess_image_inception(img_path)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=3)[0]

def add_edge_highlights(img_path):
    try:
        original_img = cv2.imread(img_path)
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 150, 250)
        contrast_enhanced = cv2.equalizeHist(blurred)
        enhanced_image = cv2.addWeighted(contrast_enhanced, 0.8, edges, 0.2, 0)
        highlighted_img_colored = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        return highlighted_img_colored
    except Exception as e:
        print(f"Failed to process image {img_path}: {str(e)}")
        return None

def display_images_and_predictions_inception():
    model = InceptionV3(weights='imagenet')
    images = [os.path.join(Config.BASE_DIR, img) for img in os.listdir(Config.BASE_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(images, min(len(images), Config.NUM_IMAGES))

    # Increase figure size and adjust layout
    plt.figure(figsize=(20, 6 * len(selected_images)))  # Modified figure size

    for i, img_path in enumerate(selected_images):
        original_img = image.load_img(img_path, target_size=Config.IMAGE_SIZE)
        highlighted_img = add_edge_highlights(img_path)
        predictions = predict_with_inception(img_path, model)

        ax1 = plt.subplot(len(selected_images), 3, 3 * i + 1)
        plt.imshow(original_img)
        plt.axis('off')
        ax1.set_title("Original Image")

        ax2 = plt.subplot(len(selected_images), 3, 3 * i + 2)
        if highlighted_img is not None:
            plt.imshow(highlighted_img)
        plt.axis('off')
        ax2.set_title("Edge Highlighted Image")

        ax3 = plt.subplot(len(selected_images), 3, 3 * i + 3)
        ax3.axis('off')
        prediction_texts = "\n".join([f'{p[1]}: {p[2]*100:.2f}%' for p in predictions])
        plt.text(0.5, 0.5, "Predictions:\n" + prediction_texts, ha='center', va='center', fontsize=12)
        ax3.set_title("Predictions")

    # Adjust subplot parameters to modify spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Increased vertical space
    plt.tight_layout()
    plt.show()
    
    # Save the model in h5 format
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'inceptionV3_model.h5'))
    print("InceptionV3 Model saved successfully.")
if __name__ == '__main__':
    display_images_and_predictions_inception()
