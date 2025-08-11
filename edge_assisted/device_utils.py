

class Device:
    def __init__(self, K, D, w, h):
        self.id = None
        self.K = K
        self.D = D
        self.projection_matrix
        self.w = w
        self.h = h

        self.gaussians = None
        self.frame_ids = []   #int or long
        self.kf_windows = None  #int or long

        #tracking information
        self.status = None #
        self.poses = None  # float
        self.prev_id = -1
        self.curr_id = -1


def ConnectDevice(K, D, w, h):
    pass