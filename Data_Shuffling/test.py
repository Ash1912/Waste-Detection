import unittest
import numpy as np
from Data_Shuffling.main import shuffle_data

class TestDataShuffling(unittest.TestCase):
    def setUp(self):
        # Setup a known random seed and a test dataset
        np.random.seed(42)
        self.data = np.array([1, 2, 3, 4, 5])

    def test_shuffle_data(self):
        # Ensure that the shuffled data is not the same as the original
        shuffled_data = shuffle_data(self.data.copy())
        self.assertNotEqual(list(self.data), list(shuffled_data),
                            "Shuffled data should not be in the same order as the original data.")

        # Ensure that the shuffled data has the same elements as the original data
        self.assertEqual(sorted(self.data), sorted(shuffled_data),
                         "Shuffled data must contain the same elements as the original data.")

    def test_shuffle_data_consistency(self):
        # Test multiple shuffles to check for consistency in shuffling mechanism
        shuffled_data_first = shuffle_data(self.data.copy())
        shuffled_data_second = shuffle_data(self.data.copy())
        self.assertNotEqual(list(shuffled_data_first), list(shuffled_data_second),
                            "Subsequent shuffles should not produce the same order with a properly set random seed.")

if __name__ == '__main__':
    unittest.main()
