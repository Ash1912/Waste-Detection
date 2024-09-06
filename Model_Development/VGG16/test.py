import unittest
import os
from Model_Development.VGG16.config import Config
from tensorflow.keras.applications import VGG16
from Model_Development.VGG16.main import load_and_preprocess_image_vgg, predict_with_vgg16

class TestVGG16Model(unittest.TestCase):
    def test_image_loading(self):
        """Test image loading and preprocessing."""
        img_path = os.path.join(Config.IMAGES_DIRECTORY, os.listdir(Config.IMAGES_DIRECTORY)[0])
        processed_img = load_and_preprocess_image_vgg(img_path)
        self.assertEqual(processed_img.shape, (1, 224, 224, 3), "Image preprocessing failed or incorrect dimensions.")

    def test_predictions(self):
        """Test VGG16 predictions."""
        img_path = os.path.join(Config.IMAGES_DIRECTORY, os.listdir(Config.IMAGES_DIRECTORY)[0])
        model = VGG16(weights='imagenet')
        predictions = predict_with_vgg16(img_path, model)
        self.assertTrue(len(predictions) > 0, "Failed to generate predictions.")

if __name__ == '__main__':
    unittest.main()
