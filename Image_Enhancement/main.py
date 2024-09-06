from Image_Enhancement.Image_Super_Resolution import process_and_display_images, load_super_resolution_model, user_decision_to_clear, preprocess_and_save_images, directory_is_empty
from Image_Enhancement.Gamma_Filtration import display_images_with_gamma_correction, user_decision_to_clear, directory_is_empty, preprocess_and_save_images
from Image_Enhancement.Image_Interpolation import display_interpolation_effects
from Image_Enhancement.Image_Augmentation import display_augmentations
from Image_Enhancement.Faster_RCNN import main as run_faster_rcnn
from Image_Enhancement.config import Config

def run_image_enhancement():
    print("Running Image Super Resolution...")

    # Set up directories for input and output
    input_dir = './Dataset/Water_Trash_Dataset'
    output_dir = './Dataset/Processed_Images_After_Image_SuperResolution'
    sr = load_super_resolution_model()  # Load the SR model once and use it

    # Execute the preprocessing function based on user decision
    if user_decision_to_clear():
        print("Processing and saving new images after clearing the directory...")
        preprocess_and_save_images(input_dir, output_dir, sr, True)
    else:
        if not directory_is_empty(output_dir):
            print(f"All images are already processed and saved in this directory: {output_dir}")
        else:
            print("Directory is empty, processing and saving new images...")
            preprocess_and_save_images(input_dir, output_dir, sr, False)

    print("Processing and Displaying Images after Image Super Resolution...")
    sr_model = load_super_resolution_model(Config.MODEL_PATH) 
    process_and_display_images(Config.DATASET_PATH + '/images/train', Config.NUM_IMAGES)

    print("Running Gamma Filtration...")
    
    # Main execution logic
    input_dir = './Dataset/Water_Trash_Dataset'
    output_dir = './Dataset/Processed_Images_With_Gamma_Correction'

    if user_decision_to_clear():
        print("User opted to clear the directory and process new images.")
        preprocess_and_save_images(input_dir, output_dir, force_clear=True)
    else:
        if directory_is_empty(output_dir):
            print("Directory is empty. Starting image processing...")
            preprocess_and_save_images(input_dir, output_dir, force_clear=False)
        else:
            print(f"All images are already processed and saved in this directory: {output_dir}")
    
    print("Processing and Displaying Images after Gamma Filtration...")
    display_images_with_gamma_correction(Config.DATASET_PATH + '/images/train', Config.NUM_IMAGES, Config.GAMMA_VALUE, Config.LUMINANCE_THRESHOLD)

    print("Running Image Interpolation...")
    display_interpolation_effects(Config.DATASET_PATH + '/images/train', Config.NUM_IMAGES, Config.SCALE_FACTOR)

    print("Running Image Augmentation...")
    display_augmentations(Config.DATASET_PATH + '/images/train', Config.NUM_IMAGES)

    print("Running Faster R-CNN Detection...")
    run_faster_rcnn()

if __name__ == '__main__':
    run_image_enhancement()
