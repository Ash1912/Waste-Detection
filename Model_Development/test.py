import unittest
import Model_Development.CNN.test
import Model_Development.Transfer_Learning.test
import Model_Development.ResNet50.test
import Model_Development.InceptionV3.test
import Model_Development.VGG16.test
import Model_Development.COCO.test
import Model_Development.HAAR.test

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.CNN.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.Transfer_Learning.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.ResNet50.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.InceptionV3.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.VGG16.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.COCO.test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(Model_Development.HAAR.test))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
