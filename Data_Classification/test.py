import os
import unittest
import cv2
import numpy as np
from Data_Classification.main import preprocess_image, clear_directory
from Data_Classification.config import Config

class TestDataClassification(unittest.TestCase):
    def test_preprocess_image(self):
        image = cv2.imread('path_to_sample_image.jpg')  # Provide a valid path for testing
        processed_image, is_waste = preprocess_image(image)
        self.assertEqual(len(processed_image), Config.IMAGE_SIZE[0] * Config.IMAGE_SIZE[1])
        self.assertIn(is_waste, [True, False])

    def test_clear_directory(self):
        test_dir = './test_directory'
        clear_directory(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        self.assertEqual(len(os.listdir(test_dir)), 0)

if __name__ == '__main__':
    unittest.main()
