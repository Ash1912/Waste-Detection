import unittest
import os
from Data_Preparation.main import unzip_dataset, organize_dataset
from Data_Preparation.config import Config

class TestDataPreparation(unittest.TestCase):
    def test_unzip_dataset(self):
        unzip_dataset(Config.ZIP_PATH, Config.EXTRACTED_FOLDER)
        self.assertTrue(os.path.exists(Config.EXTRACTED_FOLDER))

    def test_organize_dataset(self):
        organize_dataset(Config.BASE_DIR)
        self.assertTrue(os.path.exists(os.path.join(Config.BASE_DIR, 'images')))
        self.assertTrue(os.path.exists(os.path.join(Config.BASE_DIR, 'labels')))

if __name__ == '__main__':
    unittest.main()
