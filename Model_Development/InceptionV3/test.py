import unittest
import os
from Model_Development.InceptionV3.main import load_and_preprocess_image_inception, predict_with_inception
from tensorflow.keras.applications import InceptionV3
from Model_Development.InceptionV3.config import Config

class TestInceptionModel(unittest.TestCase):
    def test_image_loading(self):
        """Test image loading and preprocessing."""
        img_path = os.path.join(Config.BASE_DIR, os.listdir(Config.BASE_DIR)[0])
        processed_img = load_and_preprocess_image_inception(img_path)
        self.assertEqual(processed_img.shape, (1, 299, 299, 3), "Image preprocessing failed or incorrect dimensions.")

    def test_predictions(self):
        """Test InceptionV3 predictions."""
        img_path = os.path.join(Config.BASE_DIR, os.listdir(Config.BASE_DIR)[0])
        model = InceptionV3(weights='imagenet')
        predictions = predict_with_inception(img_path, model)
        self.assertTrue(len(predictions) > 0, "Failed to generate predictions.")

if __name__ == '__main__':
    unittest.main()
