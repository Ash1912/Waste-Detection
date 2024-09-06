class Config:
    IMAGE_SIZE = (32, 32)  # Target image size after processing
    VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    MODEL_PATH = './Models/EDSR_x4.pb'  # Path to super-resolution model
    GAMMA_VALUE = 2.0 # Gamma correction factor
    LUMINANCE_THRESHOLD = 0.5 # Luminance threshold for gamma correction
    SCALE_FACTOR = 1.5
    TRAIN_IMAGE_DIR = './Dataset/Water_Trash_Dataset/images/train'
    TEST_IMAGE_DIR = './Dataset/Water_Trash_Dataset/images/test'