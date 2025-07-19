

class Device:
    def __init__(self, K, D, w, h):
        self.K = K
        self.D = D
        self.w = w
        self.h = h

        self.poses =None
        self.gaussians = None
        self.frame_ids = None
        pass
