import numpy as np

class Object:
    def __init__(self,id,R,t):
        self.id = id
        self.initialized = False
        self.used = False
        #self.doingMapping = False
        #추후 스케일, cov 등 추가
        self.T = np.zeros((4, 4), dtype=np.float64)
        self.T[:3, :3] = R
        self.T[:3, 3] = t.flatten()  # 또는 M[:3, 3] = b.squeeze()
        self.T[3, 3] = 1.0

        self.current_window = []
        self.kf_indices = []
        self.first_kf_id = 0