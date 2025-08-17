#XFeat
import sys
sys.path.append('D:/UVR/accelerated_features')
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue

class FeatureManager:
    def __init__(self):
        top_k = 4096
        self.xfeat = XFeat(
            weights='../accelerated_features/weights/xfeat.pt',  # -lighterglue
            top_k=top_k,
            detection_threshold=0.05
        ).eval().cuda()
        self.xfeat.lighterglue = LighterGlue(weights='../accelerated_features/weights/xfeat-lighterglue.pt').eval().cuda()

    def detectAndCompute(self, color):
        return self.xfeat.detectAndCompute(color)[0]

    def match_lightglue(self, d0, d1):
        return self.xfeat.match_lighterglue(d0,d1)