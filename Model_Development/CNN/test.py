import unittest
from Model_Development.CNN.main import get_data_generators

class TestImageDataGenerators(unittest.TestCase):
    def test_data_generators(self):
        train_gen, test_gen = get_data_generators()
        self.assertTrue(train_gen.n > 0, "Train generator should not be empty")
        self.assertTrue(test_gen.n > 0, "Test generator should not be empty")

if __name__ == '__main__':
    unittest.main()
