import os

class Config:
    IMAGE_DIR = './Dataset/Water_Trash_Dataset/images'
    TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, 'train')
    TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, 'test')
