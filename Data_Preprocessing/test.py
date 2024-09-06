# test.py
import unittest
import numpy as np
import cv2
import os
from Data_Preprocessing.main import preprocess_image, clear_directory
from Data_Preprocessing.config import Config

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        # Create a dummy image for testing
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        processed_image = preprocess_image(image)
        # Check if the processed image has the correct shape
        self.assertEqual(processed_image.shape, (Config.IMAGE_SIZE[0] * Config.IMAGE_SIZE[1],))
        # Ensure all values in processed image are between 0 and 1
        self.assertTrue((processed_image >= 0).all() and (processed_image <= 1).all())

    def test_clear_directory(self):
        # Setup a test directory
        test_dir = './test_dir'
        # Ensure directory exists for testing
        os.makedirs(test_dir, exist_ok=True)
        # Create a dummy file to test directory clearing
        with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
            f.write("Test")
        # Clear the directory
        clear_directory(test_dir)
        # Check if the directory still exists and is empty
        self.assertTrue(os.path.exists(test_dir))
        self.assertFalse(os.listdir(test_dir))  # Directory should be empty

if __name__ == '__main__':
    unittest.main()
