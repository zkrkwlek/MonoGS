import torch
import numpy as np
import cv2
from utils.camera_utils import Camera
from edge_assisted.gaussian_feature import project_pc_to_pixel
from utils.pose_utils import SE3_exp

class EdgeFrame:
    def __init__(self,id,color,R,t,depth=None):
        self.id = id
        self.color = color #수정 안함.
        self.depth = depth #수정 안함

        self.T = np.zeros((4, 4), dtype=np.float64)
        self.T[:3, :3] = R
        if t is None:
            t = np.zeros((3,1),dtype=np.float64)
        self.T[:3, 3] = t.flatten()  # 또는 M[:3, 3] = b.squeeze()
        self.T[3, 3] = 1.0

        self.keypoints = None # np 수정 안함.
        self.descriptors = None #np 수정 안함.
        self.gaussianpoints = None #torch, cpu로 내려야 하나? 이것만 수정함. 이걸 통신하자.

    def copy_gaussians_from_frame_matches(self, frame, gaussians, matches):
        #self.gaussianpoints = torch.full((self.keypoints.shape[0],), -1)
        valid = frame.gaussianpoints[matches[:, 0]] > -1
        filted_matches = matches[valid, :]
        g_indices = frame.gaussianpoints[filted_matches[:, 0]]
        self.gaussianpoints[filted_matches[:,1]] = g_indices
        #가우시안 포인트 복사
        #추후 옵저베이션 추가

    def get_correspondence(self, gaussians, R, t, fx, fy, cx, cy, w, h, delta_rot = None, delta_trans= None):

        valid = self.gaussianpoints > -1
        gindex = self.gaussianpoints[valid]
        curr_frame_gaussians = gaussians._xyz[gindex]

        if delta_rot is None:
            delta_rot = torch.zeros(3, requires_grad=True, device=R.device)
            delta_trans = torch.zeros(3, requires_grad=True, device=R.device)
        tau = torch.cat([delta_trans, delta_rot], axis=0)
        T_w2c = torch.eye(4, device=R.device)
        T_w2c[0:3, 0:3] = R
        T_w2c[0:3, 3] = t

        new_w2c = SE3_exp(tau) @ T_w2c

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]
        projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, new_R, new_T,
                                                           fx, fy, cx, cy, w, h)
        points = torch.from_numpy(self.keypoints[valid][valid_projection]).cuda()
        #projection.requires_grad_(True)
        return projection , points

class EdgeFrames:
    def __init__(self, dataset):
        self._dataset = dataset
        self._frames={}
        print("wrapping class")

    def __getattr__(self, attr):
        #dataset의 특성을 그대로 활용
        return getattr(self._dataset, attr)

    def __setitem__(self, key, value):
        #네트워크로 전송받은 데이터를 추가함.
        self._frames[key] = value;

    def __contains__(self, item):
        #데이터를 전송받았는지 확인함.
        return item in self._frames

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, key):
        if isinstance(key, int):
            frame = self._frames[key]
            pose = frame.T
            image = frame.color
            depth = frame.depth

            if self.disorted:
                image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            if frame.depth is not None:
                depth = cv2.remap(depth, self.map1x, self.map1y, cv2.INTER_LINEAR)
            """
            if self.has_depth:
                depth_path = self.depth_paths[idx]
                depth = np.array(Image.open(depth_path)) / self.depth_scale
            """
            image = (
                torch.from_numpy(image / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=self.dtype)
            )
            pose = torch.from_numpy(pose).to(device=self.device)
            return image, depth, pose
        if isinstance(key, str):
            key = int(key)
            if key in self._frames:
                return self._frames[key]
            else:
                print("keyframe",key,"is not in dataset")
                None

def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )