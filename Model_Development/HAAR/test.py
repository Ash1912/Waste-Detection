import unittest
import cv2
import os
from Model_Development.HAAR.main import load_haar_cascade, detect_objects_haar
from Model_Development.HAAR.config import Config

class TestHaarCascadeModel(unittest.TestCase):
    def test_load_cascade(self):
        cascade = load_haar_cascade()
        self.assertIsInstance(cascade, cv2.CascadeClassifier, "Failed to load Haar Cascade model")

    def test_detect_objects(self):
        cascade = load_haar_cascade()
        test_image_path = os.path.join(Config.IMAGE_DIRECTORY, os.listdir(Config.IMAGE_DIRECTORY)[0])
        objects = detect_objects_haar(test_image_path, cascade)
        self.assertIsInstance(objects, tuple, "Detection should return a tuple")

if __name__ == '__main__':
    unittest.main()
