import os

class Config:
    # Directory where the input images are stored
    INPUT_DIR = os.path.join('.', 'Dataset', 'Water_Trash_Dataset')
    # Directory where processed images will be saved
    OUTPUT_DIR = os.path.join('.', 'Dataset', 'Processed_Images_After_Data_Preprocessing')
    # Size to which images are resized
    IMAGE_SIZE = (32, 32)
    # Valid file extensions for image files
    VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    # Path for a specific image to process and display
    IMAGE_PATH = './Dataset/Water_Trash_Dataset/images/train/800wm.jpg'
