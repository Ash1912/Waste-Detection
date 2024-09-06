import os
import cv2
import numpy as np
import matplotlib
import shutil
import skimage
import tensorflow as tf
from tensorflow import keras
import torch
import torchvision
import warnings
import platform
import subprocess
from sklearn import __version__ as sklearn_version
# from imageai.Detection import ObjectDetection

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

def run_dependency_checks():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Check the versions of each library and display versions
    print(f"NumPy version: {np.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version (bundled with TensorFlow): {tf.__version__}")  # Corrected to show TensorFlow version for Keras
    print(f"OpenCV version: {cv2.__version__}")
    print(f"os module: {os.__version__ if hasattr(os, '__version__') else 'builtin module, no version'}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"shutil module: {shutil.__version__ if hasattr(shutil, '__version__') else 'builtin module, no version'}")
    print(f"skimage version: {skimage.__version__}")
    print(f"Scikit-learn version: {sklearn_version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    # print(f"ImageAI version: {ObjectDetection().__module__.split('.')[0]}")

    # Check the versions of pip and python
    pip_version_output = subprocess.run(['pip', '--version'], capture_output=True, text=True).stdout.strip()
    print(f"pip version: {pip_version_output}")
    print(f"Python version: {platform.python_version()}")

if __name__ == "__main__":
    run_dependency_checks()