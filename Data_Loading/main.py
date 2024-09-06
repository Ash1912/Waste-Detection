import os
import cv2
from Data_Loading.config import Config

def load_images_from_subdir(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image {filename}")
    return images

def load_dataset_images(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    train_images = load_images_from_subdir(train_dir)
    test_images = load_images_from_subdir(test_dir)
    print(f"Loaded {len(train_images)} train images.")
    print(f"Loaded {len(test_images)} test images.")
    return train_images, test_images

if __name__ == '__main__':
    train_images, test_images = load_dataset_images(Config.IMAGE_DIR)
