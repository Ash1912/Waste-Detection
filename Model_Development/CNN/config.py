import os

class Config:
    
    # Image and training configurations
    IMAGE_SIZE = (32, 32)
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 10
    MODEL_SAVE_PATH = './saved_models'
    # Base directory for preprocessed images
    BASE_DIR = "./Dataset/PreProcessed_Images_For_Data_Classification"

