class Config:
    # Paths
    DATASET_PATH = './Dataset/Water_Trash_Dataset'
    MODEL_PATH = './Models/EDSR_x4.pb'
    OUTPUT_DIR = './Dataset/Processed_Images_After_Image_SuperResolution'

    # Image processing parameters
    NUM_IMAGES = 5
    VARIANCE_THRESHOLD = 100
    GAMMA_VALUE = 2.0
    LUMINANCE_THRESHOLD = 0.5
    SCALE_FACTOR = 1.5

    # Faster R-CNN specifics
    FASTER_RCNN_MODEL = './Models/Faster_RCNN/faster_rcnn_inception_v2_coco_2018_01_28/saved_model'
