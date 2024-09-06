class Config:
    IMAGE_SIZE = (32, 32)  # Adjust size according to what your model requires
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    PATIENCE = 10
    BASE_DIR = "./Dataset/PreProcessed_Images_For_Data_Classification"
    MODEL_SAVE_PATH = './saved_models'
