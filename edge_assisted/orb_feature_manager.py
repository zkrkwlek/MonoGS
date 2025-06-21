

import cv2
import numpy as np
from python_orb_slam3 import ORBExtractor

class OrbFrame:
    def __init__(self, id):
        #self.descriptors = None
        #self.keypoints = None
        #self.id = id
        pass

class OrbFeatureManager:
    def __init(self):
        self.extractor = ORBExtractor()

    def extractor(self, img : np.ndarray, frame : OrbFrame):
        keypoints_2d, descriptors = self.detectAndCompute(img)
        return keypoints_2d, descriptors



