# config.py
import os

class Config:
    # Path configurations
    ZIP_PATH = './Dataset/Water_Trash_Dataset.zip'
    EXTRACTED_FOLDER = './Dataset'
    BASE_DIR = './Dataset/Water_Trash_Dataset'

    IMAGE_DIR = './Dataset/Water_Trash_Dataset/images'
    TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, 'train')
    TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, 'test')

    DESTINATION_DIR = "./Dataset/PreProcessed_Images_For_Data_Classification"
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    CLASSES = ['waste', 'no_waste']

    

