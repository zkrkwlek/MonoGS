from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
import pycolmap
import gtsam
import numpy as np
import cv2
import torch
import atomics

class Device:
    def __init__(self, src, K, D, w, h, bMapper = True, color = [1,0,0], monocular = True):

        self.src = src
        self.id = None
        self.K = K
        self.D = D
        self.w = w
        self.h = h
        self.monocular = monocular
        self.mapper = bMapper #True이면 맵 초기화, False이면 만들어진 맵으로 트래킹

        self.fx = self.K[0][0]
        self.fy = self.K[1][1]
        self.cx = self.K[0][2]
        self.cy = self.K[1][2]

        self.fovx = focal2fov(self.fx, self.w)
        self.fovy = focal2fov(self.fy, self.h)

        self.distorted = self.is_distorted(self.D)
        if self.distorted:
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K,
                self.D,
                np.eye(3),
                self.K,
                (self.w, self.h),
                cv2.CV_32FC1,
            )
        else:
            self.map1x = None
            self.map1y = None

        self.has_depth = True
        self.depth_scale = 1.0

        self.color = color

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.w,
            H=self.h
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device='cuda')
        self.K_gtsam = gtsam.Cal3_S2(self.fx, self.fy, 0.0, self.cx, self.cy)

        self.gaussians = None
        self.frame_ids = []   #int or long
        self.kf_windows = None  #int or long

        #키프레임 관리
        #키프레임 id 집합?

        #tracking information
        self.frames={}
        self.status = None #
        self.poses = None  # float
        self.prev_frame_idx = -1
        self.cur_frame_idx = -1
        self.last_keyframe_idx = -1

        self.requested_keyframe = 0

        #pycolmap
        self.colCam = pycolmap.Camera(
            model="OPENCV",  # 또는 "SIMPLE_RADIAL", "SIMPLE_PINHOLE" 등
            width=self.w,
            height=self.h,
            params=[
                self.fx,  # fx
                self.fy,  # fy
                self.cx,  # cx
                self.cy,  # cy
                *self.D[:4],  # k1, k2, p1, p2 (OPENCV 모델 기준, 필요시 k3 등 추가)
            ]
        )

        col_param = pycolmap.AbsolutePoseEstimationOptions()
        col_param.estimate_focal_length = False
        col_param.ransac.max_error = 4.0
        col_param.ransac.min_inlier_ratio = 0.2
        col_param.ransac.min_num_trials = 500
        col_param.ransac.max_num_trials = 5000
        col_param.ransac.confidence = 0.99
        self.col_param = col_param

        self.used = atomics.atomic(width=4, atype=atomics.INT, init = 0)

    def set_used(self, val = 0):
        self.used.store(val)

    def is_used(self):
        return bool(self.used.load())

    def convert_viewpoint(self, idx):

        frame = self.frames[idx]

        pose = frame.T
        image = frame.color
        depth = frame.depth

        if self.distorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

            if frame.depth is not None:
                depth = cv2.remap(depth, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device='cuda', dtype=torch.float32)
        )
        pose = torch.from_numpy(pose).to(device='cuda')

        return Camera(
            idx,
            image,
            depth,
            pose,
            self.projection_matrix,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.fovx,
            self.fovy,
            self.h,
            self.w,
            device='cuda',
        )

    def convert_depth(self, idx):
        frame = self.frames[idx]
        depth = frame.depth
        if self.distorted and frame.depth is not None:
            depth = cv2.remap(depth, self.map1x, self.map1y, cv2.INTER_LINEAR)
        return depth

    def is_distorted(self, D, tol=1e-6):
        """
        행렬 D가 직교 행렬이 아니면 distorted로 판단하는 함수
        tol: 허용 오차
        """
        # D의 전치행렬
        Dt = D.T
        # D.T * D
        identity_approx = np.dot(Dt, D)
        # 항등행렬
        identity = np.eye(D.shape[0])

        # 차이가 tol 이내면 직교 행렬 -> distorted 아님
        return not np.allclose(identity_approx, identity, atol=tol)

def ConvertFramdId(src, id):
    return src+"_"+str(id)

def ConnectDevice(K, D, w, h):
    pass