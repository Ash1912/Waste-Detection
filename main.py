from Data_Processing.main import process_images
from Data_Processing.config import Config

if __name__ == '__main__':
    process_images(Config.TRAIN_IMAGE_DIR)