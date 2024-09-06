from Model_Development.CNN.main import cnn_model as train_cnn
from Model_Development.Transfer_Learning.main import train_model as train_transfer
from Model_Development.ResNet50.main import display_images_and_predictions as display_resnet
from Model_Development.InceptionV3.main import display_images_and_predictions_inception as display_inception
from Model_Development.VGG16.main import display_images_and_predictions_vgg as display_vgg
from Model_Development.COCO.main import display_detected_images as display_coco
from Model_Development.HAAR.main import display_detected_images as display_haar
from Model_Development.config import Config


def run_all_models():
    # CNN Model
    print("Training CNN Model...")
    train_cnn()

    # Transfer Learning Model
    print("Training Transfer Learning Model...")
    train_transfer()

    # ResNet50 Model
    print("Displaying predictions from ResNet50...")
    display_resnet()

    # InceptionV3 Model
    print("Displaying predictions from InceptionV3...")
    display_inception()

    # VGG16 Model
    print("Displaying predictions from VGG16...")
    display_vgg()

    # COCO Model (YOLOv3)
    print("Displaying detections from COCO (YOLO)...")
    display_coco()

    # HAAR Model
    print("Displaying detections from HAAR...")
    display_haar()


if __name__ == "__main__":
    run_all_models()
