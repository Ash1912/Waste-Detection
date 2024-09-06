# test.py
import unittest
from Development.Data_Preparation.test import TestDataPreparation
from Development.Data_Loading.test import TestDataLoading
from Development.Data_Preprocessing.test import TestDataPreprocessing
from Development.Data_Shuffling.test import TestDataShuffling
from Development.Feature_Extraction.test import TestFeatureExtraction
from Development.Data_Classification.test import TestDataClassification
from Development.Model_Development.test import TestModelDevelopment
from Development.Image_Enhancement.test import TestImageEnhancement

def create_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDataPreparation))
    suite.addTest(unittest.makeSuite(TestDataLoading))
    suite.addTest(unittest.makeSuite(TestDataPreprocessing))
    suite.addTest(unittest.makeSuite(TestDataShuffling))
    suite.addTest(unittest.makeSuite(TestFeatureExtraction))
    suite.addTest(unittest.makeSuite(TestDataClassification))
    suite.addTest(unittest.makeSuite(TestModelDevelopment))
    suite.addTest(unittest.makeSuite(TestImageEnhancement))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = create_test_suite()
    runner.run(test_suite)
