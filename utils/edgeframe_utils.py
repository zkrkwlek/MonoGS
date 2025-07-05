import torch
import torch.nn.functional as F

import time
import numpy as np
import cv2
from utils.camera_utils import Camera
from edge_assisted.gaussian_feature import project_pc_to_pixel
from utils.pose_utils import SE3_exp

class EdgeFrame:
    def __init__(self,id,color,R,t,depth=None):
        self.id = id
        self.kf_id = -1
        self.color = color #수정 안함.
        self.depth = depth #수정 안함

        self.T = np.zeros((4, 4), dtype=np.float64)
        self.T[:3, :3] = R
        if t is None:
            t = np.zeros((3,1),dtype=np.float64)
        self.T[:3, 3] = t.flatten()  # 또는 M[:3, 3] = b.squeeze()
        self.T[3, 3] = 1.0

        self.keypoints = None # np 수정 안함.-> gpu
        self.descriptors = None #np 수정 안함.
        self.gaussianpoints = None #torch, cpu로 내려야 하나? 이것만 수정함. 이걸 통신하자. # 이게 prune 후 frontend로 갈 때 prev, 아직 갱신 안된 키프레임도 처리되어야 함
        self.gaussians = None
        #self.inliers = None #전송안하면, 초기에 넘겨받고 갱신해야 함. 이것도 torch임.

    def extract_patches_differentiable(self, image, center_pts, patch_size=7):
        """
        여러 center_pts (N,2)에 대해 differentiable하게 패치 추출.
        이미지 경계 벗어나는 패치는 제외.
        반환: (추출된 패치 텐서 [M, C, patch_size, patch_size], 사용된 center 인덱스 [M])
        """
        device = image.device
        N = center_pts.shape[0]
        half_size = patch_size // 2
        H, W = image.shape[-2:]
        # center_pts: (N, 2) [x, y]
        xs = center_pts[:, 0]
        ys = center_pts[:, 1]

        # 경계 내에 있는 center만 선택 (bool mask)
        mask = (
                (xs - half_size >= 0) & (xs + half_size < W) &
                (ys - half_size >= 0) & (ys + half_size < H)
        )
        valid_idx = torch.where(mask)[0]  # shape: [M]
        if valid_idx.numel() == 0:
            return None, None

        valid_centers = center_pts[valid_idx]  # [M, 2]
        M = valid_centers.shape[0]

        # 1D grid
        grid_range = torch.arange(patch_size, device=device).float() - half_size

        # (M, patch_size)
        grid_x = valid_centers[:, 0].unsqueeze(1) + grid_range.unsqueeze(0)
        grid_y = valid_centers[:, 1].unsqueeze(1) + grid_range.unsqueeze(0)

        # (M, patch_size, patch_size)
        grid_xx = grid_x.unsqueeze(2).expand(M, patch_size, patch_size)
        grid_yy = grid_y.unsqueeze(1).expand(M, patch_size, patch_size)

        # 정규화
        norm_x = 2.0 * grid_xx / (W - 1) - 1.0
        norm_y = 2.0 * grid_yy / (H - 1) - 1.0

        grid = torch.stack([norm_x, norm_y], dim=-1)  # [M, patch_size, patch_size, 2]

        # 이미지 배치화
        if image.dim() == 3:
            image_batch = image.unsqueeze(0).expand(M, -1, -1, -1)
        else:
            image_batch = image.unsqueeze(0).unsqueeze(0).expand(M, -1, -1, -1)

        patches = F.grid_sample(
            image_batch, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [M, C, patch_size, patch_size]
        return patches, mask

    def calculate_reprojection_error(self, gaussians, keypoints):
        l2 = torch.sum((gaussians - keypoints) ** 2, dim=1)
        return l2

    def update_inlier(self, err, th = 9.0):
        return err < th

    def copy_gaussians_from_frame_matches(self, frame, gaussians, matches):
        #self.gaussianpoints = torch.full((self.keypoints.shape[0],), -1)
        matches = torch.from_numpy(matches).cuda().type(torch.int32)
        valid = frame.gaussianpoints[matches[:, 0]] > -1
        filted_matches = matches[valid, :]
        g_indices = frame.gaussianpoints[filted_matches[:, 0]]
        self.gaussianpoints[filted_matches[:,1]] = g_indices
        #가우시안 포인트 복사
        #추후 옵저베이션 추가

    def get_gaussianpoints(self, gaussians, R, t, fx, fy, cx, cy, w, h, th_radius = 9.0):

        with torch.no_grad():
            valid = self.gaussianpoints > -1
            gindex = self.gaussianpoints[valid]
            curr_frame_gaussians = gaussians._xyz[gindex]
            curr_gaussian_ids = gaussians.unique_gaussian_ids[gindex]
            #현재 프레임의 가우시안의 크기는 밸리드의 크기와 같음.

            projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, R, t,fx, fy, cx, cy, w, h)
            # 프로젝션했을 때 프레임을 벗어나거나, 뎁스가 <0이면 프로젝션의 크기는 가우시안의 크기보다 작음.
            #프로젝션의 크기 = 밸리드 프로젝션보다 작음. 유효한 요소만 남은 것
            # 밸리드 프로젝션의 크기는 가우시안의 크기
            selected_indices = torch.where(valid)[0]
            valid[selected_indices] = torch.from_numpy(valid_projection)
            #print("temp", projection.size()[0], valid_projection.shape[0], selected_indices.size()[0], valid.size()[0])

            #temp_index = torch.where(self.gaussianpoints[valid][valid_projection])[0]
            #inlier 의 수는 projection의 수와 같음. 당연히 가우시안의 수보다 작을 수 있음.
            points = torch.from_numpy(self.keypoints[valid.numpy()]).cuda()
            err = self.calculate_reprojection_error(projection, points)
            inlier = self.update_inlier(err, th = th_radius)
            inlier2 = torch.zeros(selected_indices.size()[0], device='cuda',dtype=bool)
            inlier2[valid_projection] = inlier
            #self.gaussianpoints[valid][~inlier] = -1

        return curr_frame_gaussians[inlier2], points[inlier], torch.where(inlier2)[0]



    def get_frame_gaussians(self, gaussians):
        valid = self.gaussianpoints > -1
        gindex = self.gaussianpoints[valid]
        frame_gaussians = gaussians._xyz[gindex]
        return frame_gaussians, torch.where(valid)[0], gindex.cuda() #N1

    def project_points(self, gaussians, R, t, fx, fy, cx, cy, w, h, delta_rot, delta_trans):
        #delta_rot = torch.zeros(3, requires_grad=True, device=R.device)
        #delta_trans = torch.zeros(3, requires_grad=True, device=R.device)

        """
        tmpR = R.clone().detach().cpu().numpy()
        #tmpT = t.clone().detach().cpu().numpy()
        rvec, _ = cv2.Rodrigues(tmpR)
        rvec = torch.from_numpy(rvec).cuda().squeeze()
        delta_rot.data.copy_(rvec)
        delta_trans.data.copy_(t)
        """

        tau = torch.cat([delta_trans, delta_rot], axis=0)
        T_w2c = torch.eye(4, device=R.device)
        T_w2c[0:3, 0:3] = R
        T_w2c[0:3, 3] = t

        new_w2c = SE3_exp(tau) @ T_w2c

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]
        projection, valid_projection = project_pc_to_pixel(gaussians, new_R, new_T,
                                                           fx, fy, cx, cy, w, h)
        return projection, valid_projection #N2, N1



    def update_outlier_points(self, projection, points, inlier, th_radius = 9.0):
        err = self.calculate_reprojection_error(projection, points)
        tmp_inlier = self.update_inlier(err, th=th_radius)
        return tmp_inlier
        ##외부에서 인라이어를 수정해야 함. 여기서 return은 projection의 크기와 일치함.
        ##inlier param의 크기는 처음 프레임에서 projection view 안의 들어오는 크기라. projection보다 클 수 있음.ㄴ
        """
        selected_indices = torch.where(inlier)[0]

        err = self.calculate_reprojection_error(projection, points)
        print('a', inlier.size(), torch.count_nonzero(inlier), selected_indices.size(), err.size())

        tmp_inlier = self.update_inlier(err, th=th_radius)
        inlier[selected_indices] = tmp_inlier
        #projection = projection[tmp_inlier]
        #points = points[tmp_inlier]
        print('b', inlier.size(), torch.count_nonzero(inlier), projection.size())
        #return inlier #N1
        """

    def update_frame_gaussianpoints(self, indices, inliers):
        self.gaussianpoints[indices][~inliers] = -1

    def update_gaussianpoints(self, gaussians, R, t, fx, fy, cx, cy, w, h, th_radius = 9.0):
        a = time.time()
        with torch.no_grad():
            valid = self.gaussianpoints > -1
            gindex = self.gaussianpoints[valid]
            curr_frame_gaussians = gaussians._xyz[gindex]
            curr_gaussian_ids = gaussians.unique_gaussian_ids[gindex]
            #현재 프레임의 가우시안의 크기는 밸리드의 크기와 같음.

            projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, R, t,fx, fy, cx, cy, w, h)
            # 프로젝션했을 때 프레임을 벗어나거나, 뎁스가 <0이면 프로젝션의 크기는 가우시안의 크기보다 작음.
            #프로젝션의 크기 = 밸리드 프로젝션보다 작음. 유효한 요소만 남은 것
            # 밸리드 프로젝션의 크기는 가우시안의 크기
            selected_indices = torch.where(valid)[0]
            valid[selected_indices] = valid_projection
            #print("temp", projection.size()[0], valid_projection.shape[0], selected_indices.size()[0], valid.size()[0])

            #temp_index = torch.where(self.gaussianpoints[valid][valid_projection])[0]
            #inlier 의 수는 projection의 수와 같음. 당연히 가우시안의 수보다 작을 수 있음.
            #points = torch.from_numpy(self.keypoints).cuda()[valid]
            points = self.keypoints[valid]
            err = self.calculate_reprojection_error(projection, points)
            inlier = self.update_inlier(err, th = th_radius)
            inlier2 = torch.zeros(selected_indices.size()[0], device='cuda',dtype=bool)
            inlier2[valid_projection] = inlier
            self.gaussianpoints[valid][~inlier] = -1

        b = time.time()
        #print('udpate gaussian points', b-a)
        #print(projection.size()[0], points.size()[0], curr_frame_gaussians.size()[0], inlier.size()[0])
        return projection[inlier], points[inlier], curr_frame_gaussians[inlier2].clone(), curr_gaussian_ids[inlier2]

    def get_correspondence_for_visualize(self, gaussians, R, t, fx, fy, cx, cy, w, h):
        with torch.no_grad():
            valid = self.gaussianpoints > -1
            gindex = self.gaussianpoints[valid]
            curr_frame_gaussians = gaussians._xyz[gindex]

            projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, R, t,fx, fy, cx, cy, w, h)

            selected_indices = torch.where(valid)[0]
            valid[selected_indices] = torch.from_numpy(valid_projection)

            #points = torch.from_numpy(self.keypoints[valid]).cuda()
            points = self.keypoints[valid]
        return projection, points

    def get_correspondence_with_frame(self, gaussians, other, matches, R, t, fx, fy, cx, cy, w, h):
        gindex = self.gaussianpoints[matches[:,0]]
        points = other.keypoints[matches[:,1]]
        curr_frame_gaussians = gaussians._xyz[gindex]

        projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, R,t,
                                                           fx, fy, cx, cy, w, h)
        points = points[valid_projection]

        return projection, points

    def get_correspondence(self, gaussians, R, t, fx, fy, cx, cy, w, h, delta_rot = None, delta_trans= None):
        #가우시안 포인트를 gpu로 올리는게 맞을까?

        t1 = time.time()
        valid = self.gaussianpoints > -1
        gindex = self.gaussianpoints[valid]
        curr_frame_gaussians = gaussians._xyz[gindex]
        t2 = time.time()

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
        t3 = time.time()
        projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, new_R, new_T,
                                                           fx, fy, cx, cy, w, h)

        selected_indices = torch.where(valid)[0] #projection 사이즈
        #print(self.gaussianpoints.device,valid.device, gindex.device, valid_projection.device)
        #print(valid.size(), valid_projection.size(), gindex.size())
        valid[selected_indices] = valid_projection#torch.from_numpy(valid_projection)
        #points = torch.from_numpy(self.keypoints).cuda()[valid]
        points = self.keypoints[valid]

        #inlier check
        t4 = time.time()
        #projection.requires_grad_(True)
        #print('corres time','get gaussian', t2-t1, 'projection', t4-t3, t3-t2,)
        return projection , points, torch.where(valid)[0], gindex[valid_projection] #TF와 인덱스 위치.

    def update_pose(camera, viewpoint,converged_threshold=1e-4):
        tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], axis=0)

        """
        T_w2c = torch.eye(4, device=tau.device)
        T_w2c[0:3, 0:3] = camera.R
        T_w2c[0:3, 3] = camera.T
        """
        new_w2c = SE3_exp(tau)

        viewpoint.R = new_w2c[0:3, 0:3]
        viewpoint.T = new_w2c[0:3, 3]

        converged = tau.norm() < converged_threshold
        viewpoint.cam_rot_delta.data.fill_(0)
        viewpoint.cam_trans_delta.data.fill_(0)
        return converged

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