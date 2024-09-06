import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
from Model_Development.COCO.config import Config


def load_yolo_model():
    net = cv2.dnn.readNet(Config.WEIGHTS_PATH, Config.CFG_PATH)
    with open(Config.NAMES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes


def detect_objects_yolo(img_path, net, classes):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes


def draw_bounding_boxes(img_path, boxes, confidences, class_ids, indexes, classes):
    img = cv2.imread(img_path)
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    return img


def display_detected_images():
    net, classes = load_yolo_model()
    images = [os.path.join(Config.IMAGE_DIRECTORY, img) for img in os.listdir(Config.IMAGE_DIRECTORY) if
              img.endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(images, Config.NUM_IMAGES)
    plt.figure(figsize=(15, 10 * len(selected_images)))
    for i, img_path in enumerate(selected_images):
        boxes, confidences, class_ids, indexes = detect_objects_yolo(img_path, net, classes)
        img_with_boxes = draw_bounding_boxes(img_path, boxes, confidences, class_ids, indexes, classes)
        plt.subplot(len(selected_images), 1, i + 1)
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Objects in Image {i + 1}", pad=10)
        plt.axis('off')
    
   
    plt.subplots_adjust(hspace=0.5)  # Adjust spacing based on display needs
    plt.tight_layout()
    plt.show()

    # Save model configuration and classes
    model_config = {
        'classes': classes,
        'cfg_path': Config.CFG_PATH,
        'weights_path': Config.WEIGHTS_PATH
    }
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'yolo_config.json'), 'w') as f:
        json.dump(model_config, f)
    print("Model configuration saved successfully.")


if __name__ == '__main__':
    display_detected_images()
