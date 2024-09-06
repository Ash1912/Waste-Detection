import unittest
import os
from Model_Development.COCO.main import load_yolo_model, detect_objects_yolo
from Model_Development.COCO.config import Config


class TestYOLOModel(unittest.TestCase):
    def test_load_model(self):
        net, classes = load_yolo_model()
        self.assertIsNotNone(net, "Failed to load YOLO model")
        self.assertGreater(len(classes), 0, "No classes loaded")

    def test_detect_objects(self):
        net, classes = load_yolo_model()
        img_path = os.path.join(Config.IMAGE_DIRECTORY, os.listdir(Config.IMAGE_DIRECTORY)[0])
        boxes, confidences, class_ids, indexes = detect_objects_yolo(img_path, net, classes)
        self.assertIsInstance(boxes, list, "Boxes must be a list")
        self.assertIsInstance(confidences, list, "Confidences must be a list")
        self.assertIsInstance(class_ids, list, "Class IDs must be a list")
        self.assertIsInstance(indexes, list, "Indexes must be a list")


if __name__ == '__main__':
    unittest.main()
