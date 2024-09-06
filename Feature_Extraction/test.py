import unittest
import numpy as np
from Feature_Extraction.main import extract_features

class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        features = extract_features([dummy_image])
        self.assertTrue(len(features) == 1)
        self.assertTrue(len(features[0]) > 0)  # Check if features are actually extracted

if __name__ == '__main__':
    unittest.main()
