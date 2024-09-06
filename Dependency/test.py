import os
import cv2
import numpy as np
import matplotlib
import shutil
import skimage
import tensorflow as tf
from tensorflow import keras
import keras
import torch
import torchvision
import warnings
import platform
import subprocess
from sklearn import __version__ as sklearn_version
# from imageai.Detection import ObjectDetection

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Function to print version information
def print_versions():
    print(f"NumPy version: {np.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow version (also for Keras): {tf.__version__}")  # Accessing version via TensorFlow
    print(f"Keras version: {keras.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"os: {os.__version__ if hasattr(os, '__version__') else 'builtin module, no version'}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"shutil: {shutil.__version__ if hasattr(shutil, '__version__') else 'builtin module, no version'}")
    print(f"skimage: {skimage.__version__}")
    print(f"Scikit-learn version: {sklearn_version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    # print(f"ImageAI version: {ObjectDetection().__module__.split('.')[0]}")

    # Check the versions of pip and python
    print(f"pip version: {subprocess.run(['pip', '--version'], capture_output=True, text=True).stdout.strip()}")
    print(f"Python version: {platform.python_version()}")

if __name__ == "__main__":
    print_versions()
