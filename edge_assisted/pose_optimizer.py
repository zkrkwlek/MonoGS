import torch
import pypose as pp
import cv2
from torch import nn
from scipy.spatial.transform import Rotation
from edge_assisted.gaussian_feature import project_pc_to_pixel

class PoseOptimizer2(nn.Module):
    def __init__(self, R, t):
        super(PoseOptimizer2, self).__init__()

        # 6DOF 포즈를 나타내는 파라미터 (3D 회전 + 3D 평행이동)
        # 회전은 axis-angle representation 사용
        # 로드리게스 형태로 전달하기
        rvec, _ = cv2.Rodrigues(R.clone().detach().cpu().numpy())
        rvec = torch.from_numpy(rvec).cuda().float()

        self.rotation = nn.Parameter(rvec)  # axis-angle
        self.translation = nn.Parameter(t.clone().float())

    def forward(self, points, fx, fy, cx, cy, w, h):
        """포인트들에 현재 포즈를 적용"""
        # Axis-angle을 회전 행렬로 변환
        R = axis_angle_to_rotation_matrix(self.rotation)
        #print(R, self.translation, points.dtype)
        # 포인트 변환: R @ points + t
        #transformed_points = torch.mm(points, R.T) + self.translation.unsqueeze(0)
        p, _, v =project_pc_to_pixel(points, R, self.translation, fx, fy, cx, cy, w, h)
        return p,v

    def GetPose(self):
        rvec, _ = cv2.Rodrigues(self.rotation.detach().cpu().numpy())
        R = torch.from_numpy(rvec).cuda()
        t = self.translation.data
        return R, t


def axis_angle_to_rotation_matrix(axis_angle):
    """Axis-angle을 회전 행렬로 변환 (Rodrigues' formula)"""
    angle = torch.norm(axis_angle)

    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

    axis = axis_angle / angle
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]],
                     device=axis_angle.device, dtype=axis_angle.dtype)

    R = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype) + \
        torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.mm(K, K)

    return R

class PoseOptimizer(nn.Module):
    def __init__(self, R, t):
        super().__init__()
        R = R.clone().detach().cpu().numpy()
        t = t.clone().detach().cpu().numpy()
        quat = Rotation.from_matrix(R).as_quat()

        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        tx, ty, tz = t[0], t[1], t[2]

        tvec = torch.tensor([tx, ty, tz])
        quat = torch.tensor([qx, qy, qz, qw])
        pose_init = torch.cat([tvec, quat]).unsqueeze(0).cuda()

        self.pose = pp.Parameter(pp.SE3(pose_init))
        #self.pose = self.pose.cuda()
        #print(self.pose.device)
        #[x, y, z, qx, qy, qz, qw]

    def GetPose(self):
        pose_vec = self.pose.data.squeeze().cpu().numpy()
        t = pose_vec[:3]  # (3,)
        q = pose_vec[3:]  # (4,)

        # 쿼터니언을 회전행렬로 변환
        rot = Rotation.from_quat(q)  # scipy는 [qx, qy, qz, qw] 순서
        R = rot.as_matrix()  # (3, 3)
        print(R, t)
        return R, t

    def forward(self, points3d, K):
        p3d_cam = self.pose @ points3d
        x, y, z = p3d_cam[..., 0], p3d_cam[..., 1], p3d_cam[..., 2]
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        proj = torch.stack([u, v], dim=-1)
        return proj
