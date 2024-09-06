import unittest
from Model_Development.Transfer_Learning.config import Config
from Model_Development.Transfer_Learning.main import get_data_generators

class TestImageDataGenerators(unittest.TestCase):
    def test_data_generators(self):
        train_gen, test_gen = get_data_generators()
        self.assertGreater(len(train_gen), 0, "Train generator should not be empty")
        self.assertGreater(len(test_gen), 0, "Test generator should not be empty")

if __name__ == '__main__':
    unittest.main()
