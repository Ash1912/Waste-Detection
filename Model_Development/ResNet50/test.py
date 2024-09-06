import unittest
import os
from Model_Development.ResNet50.config import Config

class TestImageProcessing(unittest.TestCase):
    def test_image_directory(self):
        """Ensure the image directory is not empty."""
        images = os.listdir(Config.IMAGES_DIRECTORY)
        self.assertTrue(len(images) > 0, "No images found in the directory.")

if __name__ == '__main__':
    unittest.main()
