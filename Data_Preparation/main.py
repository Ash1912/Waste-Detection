import os
import zipfile
from Data_Preparation.config import Config

def unzip_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset extracted to: {extract_to}")

def organize_dataset(base_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    train_images_dir = os.path.join(images_dir, 'train')
    test_images_dir = os.path.join(images_dir, 'test')
    train_labels_dir = os.path.join(labels_dir, 'train')
    test_labels_dir = os.path.join(labels_dir, 'test')

    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

    ensure_dir(images_dir)
    ensure_dir(labels_dir)
    ensure_dir(train_images_dir)
    ensure_dir(test_images_dir)
    ensure_dir(train_labels_dir)
    ensure_dir(test_labels_dir)
    print("All necessary directories are set up and ready for use.")

if __name__ == '__main__':
    unzip_dataset(Config.ZIP_PATH, Config.EXTRACTED_FOLDER)
    organize_dataset(Config.BASE_DIR)
