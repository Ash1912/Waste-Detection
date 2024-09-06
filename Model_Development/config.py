class Config:
    BASE_DIR = "./Dataset/PreProcessed_Images_For_Data_Classification"
    IMAGE_SIZE = (224, 224)  # Default size, can be overridden by specific models
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 10
    TRAIN_DIR = f"{BASE_DIR}/train"
    TEST_DIR = f"{BASE_DIR}/test"
    NUM_IMAGES = 5  # Common number for display or processing
