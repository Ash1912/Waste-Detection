import cv2
import os
import random
import json
import matplotlib.pyplot as plt
from Model_Development.HAAR.config import Config

def load_haar_cascade():
    return cv2.CascadeClassifier(Config.FACE_CASCADE_PATH)

def detect_objects_haar(img_path, cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_bounding_boxes(img_path, objects):
    img = cv2.imread(img_path)
    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return img

def display_detected_images():
    cascade = load_haar_cascade()
    images = [os.path.join(Config.IMAGE_DIRECTORY, img) for img in os.listdir(Config.IMAGE_DIRECTORY) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(images, Config.NUM_IMAGES)
    detection_results = []

    plt.figure(figsize=(20, 12 * len(selected_images)))  # Adjusted for better display
    for i, img_path in enumerate(selected_images):
        objects = detect_objects_haar(img_path, cascade)
        img_with_boxes = draw_bounding_boxes(img_path, objects)
        plt.subplot(len(selected_images), 1, i + 1)
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        object_count = len(objects)
        plt.title(f"Detected {object_count} Objects in Image {i + 1}", pad=20)
        plt.axis('off')
        detection_results.append({'image': os.path.basename(img_path), 'detections': len(objects)})

    plt.subplots_adjust(hspace=0.6)  # Increased spacing
    plt.tight_layout()
    plt.show()

    # Save detection results
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'haar_detection_results.json'), 'w') as f:
        json.dump(detection_results, f)
    print("HAAR Detection results saved successfully.")

if __name__ == '__main__':
    display_detected_images()
