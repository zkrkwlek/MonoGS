import torch
import cv2
import numpy as np
from kornia.feature import LoFTR
from ALIKED.nets.aliked import ALIKED
from ALIKED.tracker import SimpleTracker


def visualize_pc(points, colors, R, t, fx, fy, cx, cy, w, h, img = None):
    T = torch.eye(4, device=R.device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    N = points.size()[0]
    points_h = torch.cat([points, torch.ones(N, 1, device=R.device)], dim=1)  # (N,4)

    # 변환 행렬 적용
    points_cam = points_h @ T.T  # (N,4)

    # 카메라 좌표계로 변환 (동차 좌표 마지막 차원으로 나눔)
    points_cam = points_cam[:, :3] / points_cam[:, 3:4]  # (N,3)

    mask = points[:, 2] > 0
    pointsa = points_cam[mask].clone().detach().cpu().numpy()
    colors = colors[mask].clone().detach().squeeze(-1).cpu().numpy()

    X = pointsa[:, 0]
    Y = pointsa[:, 1]
    Z = pointsa[:, 2]
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)

    # 4. OpenCV로 이미지에 점 찍기
    if img is None:
        img = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    img[v[valid], u[valid]] = colors[valid]


    valid_u = u[valid]
    valid_v = v[valid]
    
    for x, y in zip(valid_u, valid_v):
        cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=1)
        
    print(np.count_nonzero(valid), np.count_nonzero(pointsa)/3, np.count_nonzero(colors)/3, np.count_nonzero(img))

    cv2.imshow('Projected PointCloud', img)
    cv2.waitKey(1)

def convert_xyz(rounded_keypoints,color,depth):

    #rounded_keypoints = np.round(keypoints).astype(np.uint32)
    depth_values = depth[rounded_keypoints[:, 1], rounded_keypoints[:, 0]]
    color_values = color[rounded_keypoints[:, 1], rounded_keypoints[:, 0]]
    valid_mask = depth_values > 0

    valid_keypoints = rounded_keypoints[valid_mask]
    valid_depths = depth_values[valid_mask]
    valid_colors = color_values[valid_mask]

    result = torch.cat([
        valid_keypoints.float(),
        valid_depths.unsqueeze(1)
    ], dim=1)

    return result, valid_colors, valid_mask

def pixels_to_pc(points, R, t, fx, fy, cx, cy):
    R = R.type(torch.float32)
    t = t.type(torch.float32)
    u = points[:, 0]
    v = points[:, 1]
    d = points[:, 2]

    Rwc = R.T
    twc = -Rwc@t

    x_c = (u - cx) * d / fx
    y_c = (v - cy) * d / fy
    z_c = d
    P_c = torch.stack([x_c, y_c, z_c], dim=1)

    P_w = P_c@R+twc
    return P_w

#points1 : gaussian
#points2 :

def find_correspondence_with_dist(points1, points2, th = 3.0):
    dists = torch.cdist(points1, points2)  # (N, M)
    min_dists, min_indices = dists.min(dim=1)
    #gauss_matching_indices = min_indices.clone()
    min_indices[min_dists > th] = -1
    return min_indices

def find_correspondence(points1, points2):
    matches_gauss_to_kp = (points1.unsqueeze(1) == points2).all(dim=2)
    gauss_matching_indices = matches_gauss_to_kp.float().argmax(dim=1)
    gauss_matching_indices[~matches_gauss_to_kp.any(dim=1)] = -1

    matches_kp_to_gauss = (points2.unsqueeze(1) == points1).all(dim=2).any(dim=1)
    unmatched_keypoints_mask = ~matches_kp_to_gauss

    return gauss_matching_indices, unmatched_keypoints_mask

def calculate_keypoint_mask(keypoints, w, h):
    """
    keypoints: (N, 2) tensor, 각 행은 (y, x) 좌표
    w, h: 마스크의 너비와 높이
    반환: (1, h, w) boolean mask, keypoints 위치만 True
    """
    mask = torch.zeros((h, w), dtype=torch.bool, device=keypoints.device)
    # keypoints가 이미지 범위 내에 있는 것만 선택 (안전)
    valid = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < h) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < w)
    kp_y = keypoints[valid, 0].int()
    kp_x = keypoints[valid, 1].int()
    mask[kp_y, kp_x] = True
    return mask.unsqueeze(0)

def calculate_feature_mask_with_closest(keypoints, w, h, radius=7):
    N = keypoints.shape[0]

    ys = torch.arange(h, device=keypoints.device).reshape(h, 1)
    xs = torch.arange(w, device=keypoints.device).reshape(1, w)

    kp_y = keypoints[:, 0].reshape(N, 1, 1)
    kp_x = keypoints[:, 1].reshape(N, 1, 1)

    dist2 = (ys - kp_y) ** 2 + (xs - kp_x) ** 2        # (N, h, w)
    patch_mask = dist2 <= radius ** 2                   # (N, h, w)
    mask = patch_mask.any(dim=0, keepdim=True)          # (1, h, w) bool

    # 각 픽셀별 모든 키포인트와의 거리 중 최솟값 인덱스
    dist2_per_pixel = dist2.permute(1, 2, 0)            # (h, w, N)
    closest_idx = torch.argmin(dist2_per_pixel, dim=2)  # (h, w)

    # 반지름 밖은 -1로 처리하여 마스킹
    outside_mask = ~mask[0]                             # (h, w) bool
    closest_idx[outside_mask] = -1

    return mask, closest_idx

def calculate_feature_mask(keypoints,w, h,radius = 7):

    N = keypoints.shape[0]

    # (h, 1), (1, w): 픽셀 좌표 그리드
    ys = torch.arange(h).reshape(h, 1).cuda()
    xs = torch.arange(w).reshape(1, w).cuda()

    # (N, 1, 1): 특징점 좌표 확장
    kp_y = keypoints[:, 0].reshape(N, 1, 1)
    kp_x = keypoints[:, 1].reshape(N, 1, 1)

    # (N, h, w): 각 특징점과 모든 픽셀 간 거리 제곱
    dist2 = (ys - kp_y) ** 2 + (xs - kp_x) ** 2

    # 반지름 이내: True
    patch_mask = dist2 <= radius ** 2  # (N, h, w)

    # 여러 특징점의 패치가 겹치면 True
    mask = patch_mask.any(dim=0, keepdim=True)  # (1, h, w) bool
    return mask

def calculate_bbox_mask(boxes, w, h):

    boxes = torch.tensor([[30, 50, 40, 20], [100, 70, 30, 60]]).cuda()  # (N, 4): [x, y, w, h]

    N = boxes.shape[0]

    # 각 박스의 x1, y1, x2, y2 좌표 계산
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # (h, 1), (1, w): 픽셀 좌표 그리드
    ys = torch.arange(h, device=boxes.device).reshape(h, 1)
    xs = torch.arange(w, device=boxes.device).reshape(1, w)

    # (N, h, w): 각 박스별로 해당 픽셀에 포함되는지 벡터화 계산
    in_box = (
            (ys >= y1[:, None, None]) & (ys < y2[:, None, None]) &
            (xs >= x1[:, None, None]) & (xs < x2[:, None, None])
    )  # (N, h, w) bool

    # 여러 박스가 겹치면 True
    mask = in_box.any(dim=0, keepdim=True)  # (1, h, w) bool

    return mask

def projection(_X, R, t, fx, fy, cx, cy, w, h):
    Xw = _X.to(R.dtype)
    points_cam = R@Xw+t
    X = points_cam[0]
    Y = points_cam[1]
    Z = points_cam[2]
    u = (fx * X / Z + cx)
    v = (fy * Y / Z + cy)

    valid = (Z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def project_pc_to_pixel(points, R, t, fx, fy, cx, cy, w, h):
    T = torch.eye(4, device=R.device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=R.device)

    N = points.size()[0]
    points_h = torch.cat([points, torch.ones(N, 1,device=R.device)], dim=1)  # (N,4)
    points_h = points_h.float()
    # 변환 행렬 적용
    points_cam = points_h @ T.T  # (N,4)

    # 카메라 좌표계로 변환 (동차 좌표 마지막 차원으로 나눔)
    points_cam = points_cam[:, :3] / points_cam[:, 3:4]  # (N,3)

    #mask = points[:, 2] > 0
    #pointsa = points_cam[mask]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]
    u = (fx * X / Z + cx)#.astype(np.int32)
    v = (fy * Y / Z + cy)#.astype(np.int32)

    valid = (Z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    #valid_u = u[valid]
    #alid_v = v[valid]
    ##리턴을 전체로 변경. 에러를 거를려면 직접 valid 수행.
    ##프로젝션, 뎁스, 유효 마스크 크기를 동일하게 함.
    return torch.stack((u, v), axis=1), Z, valid#.detach().cpu().numpy()

def project_3d_to_pixel(self,
                    point_3d: torch.Tensor,
                    camera_pose, camera_intrinsic):
    # point_homo = torch.cat([point_3d, torch.tensor([1.0],device = point_3d.device)])
    # print(point_homo.device, camera_pose.device, camera_intrinsic.device)
    # 카메라 좌표계로 변환
    point_cam = camera_pose @ point_3d

    # 카메라 뒤에 있는 경우
    if point_cam[2] <= 0:
        return None, None, None

    # 2D 프로젝션
    point_2d_homo = camera_intrinsic @ point_cam[:3]
    u = point_2d_homo[0] / point_2d_homo[2]
    v = point_2d_homo[1] / point_2d_homo[2]
    depth = point_cam[2]

    return float(u), float(v), float(depth)

class GaussianPoint:

    center_point: torch.Tensor  # 2D center point [x, y]
    radius: float
    gaussian_indices: torch.Tensor  # 해당 그룹에 속한 가우시안들의 인덱스

    def __init__(self, pt, gaussians = None, radius = 15):
        self.radius = radius
        self.center_point = pt
        self.gaussians = gaussians


class GaussianPointManager:

    def __init__(self):

        self.points = None

        self.feature_model = ALIKED(model_name='aliked-t16',
                  device='cuda:0',
                  top_k=-1,
                  scores_th=0.2,
                  n_limit=3000)
        self.tracker = SimpleTracker()

        pass


def test2(self, image, keypoints, R, t, fx, fy, cx, cy):
    world_to_cam = torch.eye(4, device=R.device, dtype=R.dtype)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = t

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=R.dtype, device = R.device)

    img_draw = cv2.drawKeypoints(image, keypoints, None, color = (0,255,0))

    projected_points = []
    for point in self.points:
        u,v,d = project_3d_to_pixel(point,world_to_cam,K)
        projected_points.append((u,v))
        if u >= 0 and u < 640 and v >= 0 and v < 480 :
            cv2.circle(img_draw, (int(u),int(v)), 3, (255,0,0), 1)



    cv2.imshow("test", img_draw)
    cv2.waitKey(10)

def test(self, keypoints, depth, R,t, fx, fy, cx, cy):

    self.points = []

    dtype = R.dtype
    device = R.device

    #invR = torch.transpose(R,0,1)
    #invT = -invR*t
    #print(invT)

    camera_to_world = torch.eye(4, device=device, dtype=dtype)
    camera_to_world[:3, :3] = R
    camera_to_world[:3, 3] = t
    camera_to_world = torch.inverse(camera_to_world)



    for kp in keypoints:
        u, v = map(int,kp.pt)
        #print(type(u),type(v),depth.size())
        d = depth[0,v,u]

        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d

        # Homogeneous 좌표
        point_cam = torch.tensor([x_cam, y_cam, z_cam, 1.0], device =device, dtype=dtype)

        #print(u.device, v.device, d.device)
        #print(point_cam.device, camera_to_world.device)

        point_world = camera_to_world @ point_cam
        self.points.append(point_world)

