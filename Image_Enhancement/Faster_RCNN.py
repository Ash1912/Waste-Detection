import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

model_dir = './Models/Faster_RCNN'
image_folder = './Dataset/Water_Trash_Dataset/images/test'
model_path = os.path.join(model_dir, 'faster_rcnn_inception_v2_coco_2018_01_28', 'saved_model')

def load_model(model_path):
    print('Loading model...')
    model = tf.saved_model.load(model_path)
    print('Model loaded successfully.')
    return model

def run_inference(model, image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model.signatures['serving_default'](input_tensor)
    return detections

def display_detections(image_np, detections):
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    for i in range(detection_boxes.shape[0]):
        if detection_scores[i] < 0.3:
            continue
        box = detection_boxes[i]
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        y1, x1, y2, x2 = int(y1 * image_np.shape[0]), int(x1 * image_np.shape[1]), int(y2 * image_np.shape[0]), int(x2 * image_np.shape[1])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    model = load_model(model_path)
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.png'))]
    for image_path in images[:5]:
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        detections = run_inference(model, image_np)
        display_detections(image_np, detections)

if __name__ == '__main__':
    main()
