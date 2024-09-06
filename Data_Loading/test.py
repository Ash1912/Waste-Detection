import unittest
import os
from Data_Loading.main import load_images_from_subdir, load_dataset_images
from Data_Loading.config import Config

class TestDataLoading(unittest.TestCase):
    def test_load_images_from_subdir(self):
        images = load_images_from_subdir(Config.TRAIN_IMAGE_DIR)
        self.assertTrue(len(images) > 0)

    def test_load_dataset_images(self):
        train_images, test_images = load_dataset_images(Config.IMAGE_DIR)
        self.assertTrue(len(train_images) > 0)
        self.assertTrue(len(test_images) > 0)

if __name__ == '__main__':
    unittest.main()
