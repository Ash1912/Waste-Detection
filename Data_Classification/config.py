class Config:
    BASE_DIR = "./Dataset/Water_Trash_Dataset"
    DESTINATION_DIR = "./Dataset/PreProcessed_Images_For_Data_Classification"
    IMAGE_SIZE = (32, 32)  # Size to which images are resized
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    CLASSES = ['waste', 'no_waste']
