import os
from Dependency.main import run_dependency_checks
from Data_Preparation.main import unzip_dataset, organize_dataset
from Data_Loading.main import load_dataset_images
from Data_Preprocessing.main import preprocess_and_save_images, process_and_display_image
from Data_Shuffling.main import shuffle_data
from Feature_Extraction.main import extract_features
from Data_Classification.main import preprocess_and_classify_images
from Model_Development.main import run_all_models
from Image_Enhancement.main import run_image_enhancement
from config import Config

def main():
    print("Starting the Waste Detection Under Water Project...")

    print("Checking all dependencies...")
    run_dependency_checks()

    print("Unzipping and organizing dataset...")
    unzip_dataset(Config.ZIP_PATH, Config.EXTRACTED_FOLDER)
    organize_dataset(Config.BASE_DIR)

    print("Loading data...")
    load_dataset_images(Config.IMAGE_DIR)

    print("Preprocessing images...")
    preprocessed_images = preprocess_and_save_images()
    process_and_display_image()

    print("Shuffling data...")
    shuffled_images = shuffle_data(preprocessed_images)

    print("Extracting features...")
    extract_features(shuffled_images)

    print("Classifying data...")
    input_dir = os.path.join(Config.BASE_DIR, 'images')
    output_dirs = {
        Config.TRAIN_DIR: os.path.join(Config.DESTINATION_DIR, Config.TRAIN_DIR),
        Config.TEST_DIR: os.path.join(Config.DESTINATION_DIR, Config.TEST_DIR)
    }
    preprocess_and_classify_images(input_dir, output_dirs)

    print("Developing and training models...")
    run_all_models()

    print("Enhancing images for better visualization...")
    run_image_enhancement()

if __name__ == "__main__":
    main()
