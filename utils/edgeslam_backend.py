import numpy as np
import gc

from utils.slam_win_backend import WinBackEnd
import time

import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity

import cv2
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss, get_reprojection_loss_huber, get_patch_loss, get_loss_gaussian
from edge_assisted.gaussian_feature import unproject_pixel_to_pc, project_pc_to_pixel, projection, calculate_bbox_mask, calculate_feature_mask, find_correspondence_with_dist, calculate_keypoint_mask, get_correspondences_within_threshold, match_ac_from_ab_bc
from utils.edgeframe_utils import EdgeFrame
from utils.pose_utils import SE3_exp, skew_sym_mat, compute_F12

from utils.datahandle_utils import move_camera_to_gpu, move_camera_to_cpu, move_gaussianmodel_to_cpu
from utils.datahandle_utils import move_occ_visibility_to_cpu
#from edge_assisted.gaussian_feature import GaussianPointManager
from edge_assisted.localmap_utils import get_local_gaussians
#from edge_assisted.object_manager import ObjectManager
from edge_assisted.object_loss import ObjectLoss
from edge_assisted.gaussian_orb_model import GaussianOrbModel
from edge_assisted.device_utils import ConvertFramdId
from edge_assisted.pose_optimizer2 import PnPOptimizer

from collections import defaultdict
import psutil
import cProfile

#alike + lightglue
#import sys
#sys.path.append('D:/UVR/LightGlue')
#from lightglue import LightGlue, SuperPoint, ALIKED, DISK
#from lightglue.utils import load_image, rbd, numpy_image_to_torch
#from lightglue import viz2d

#XFeat
#sys.path.append('D:/UVR/accelerated_features')
#from modules.xfeat import XFeat
#from modules.lighterglue import LighterGlue

class EdgeBackEnd(WinBackEnd):
    def __init__(self, config):
        super().__init__(config)
        self.first_kf_id = None
        self.pose_update = None
        self.gs_pose = None
        #self.dataset = None
        self.FeatureManager = None
        self.frames={}
        self.keyframe_ids = {}
        self.neighbor_kfs = {}
        self.weight_reprojection = 0.08
        self.weight_init_rgb = 0.9
        self.weight_init_depth = 0.02

        self.weight_rgb = 0.8
        self.weight_depth = 0.02
        self.weight_ba = 0.03
        self.weight_patch = 0.15
        self.next_kf_id = 0

        #xfeat
        self.feature_manager = None
        self.pose_optimizer = None #맵별
        self.place_recognizer = None

        #object
        self.objects = None
        self.devices = None

        #mapping
        self.bDoingMapping = None
        self.covis_kf_ids = {}
        self.gaussian_kf_ids = []

    def push_to_frontend_with_graph(self, mask, tag = "graph", src = None, th_obs = 2):
        self.last_sent = 0

        """
        if self.gaussians.observation_indices.shape[1] == 2:
            th_obs = 1

        tmp_obs_indices = self.gaussians.observation_indices > -1
        row_sum = tmp_obs_indices.sum(dim=1)
        inlier_mask = (row_sum > th_obs)
        """
        frame_test_gaussians = self.gaussians.get_xyz[mask]
        gaussian_ids = self.gaussians.unique_gaussian_ids[mask]
        print('sync', frame_test_gaussians.shape)#, torch.count_nonzero(tmp_obs_indices).item(), torch.count_nonzero(row_sum > 0).item(), torch.count_nonzero(inlier_mask).item())
        msg = [tag, frame_test_gaussians.detach().cpu(), gaussian_ids.detach().cpu(), src]
        self.frontend_queue.put(msg)

    def push_to_frontend(self, tag=None, first_id = None, prune = None, local_window = None, src = None):

        a = time.time()

        self.last_sent = 0
        keyframes = []

        if local_window is None:
            local_window = self.current_window

        for kf_idx in local_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone().cpu(), kf.T.clone().cpu()))

        b = time.time()

        if tag is None:
            tag = "sync_backend"

        prune_data = None
        if prune is not None:
            prune_data = prune.cpu()
            #print("push_to_frontend::end", self.gaussians.get_xyz.shape)
        msg = [tag, move_gaussianmodel_to_cpu(self.gaussians), move_occ_visibility_to_cpu(self.occ_aware_visibility), (keyframes), (prune_data), src]
        self.frontend_queue.put(msg)
        c = time.time()
        #print('backend::sync', b-a, c-b)

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.frames = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.graph_optimizers = None
        self.keyframe_optimizers = None

        self.first_kf_id = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def preprocessing_add_kf(self):
        self.gaussians.observation_indices = torch.cat([self.gaussians.observation_indices,
                                                        torch.full((self.gaussians.observation_indices.shape[0], 1),
                                                                   -1, device='cuda', dtype=torch.int32)], dim=1)
        self.gaussians.observation_points = torch.cat([self.gaussians.observation_points,
                                                       torch.full((self.gaussians.observation_points.shape[0], 2), -1.0,
                                                                  device='cuda')], dim=1)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, keypoints = None, mask = None, downsample_factor = None):
        print('??????')
        #self.update_gaussian_observation_with_frame(frame)
        if downsample_factor is None:
            if init:
                downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
            else:
                downsample_factor = self.config["Dataset"]["pcd_downsample"]

        if mask is None:
            mask = torch.ones((viewpoint.image_height, viewpoint.image_width), device='cuda', dtype=torch.bool).cpu().numpy()

        print("mask test 0 ", mask)
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,keypoints=keypoints, downsample_factor = downsample_factor, mask = mask
        )
        del mask
        return

    def add_next_kf_with_ba(self, frame, viewpoint, depth_map, kp_gaussian_mask):

        # keypoint mask
        # kp_region_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width, viewpoint.image_height,max_radius=2)

        ##gaussian mask
        # 1) projection
        projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                              viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                              viewpoint.image_width, viewpoint.image_height)
        # 2) masking
        tmp_gaussian_mask = calculate_keypoint_mask(projection[valid_projection], viewpoint.image_width,
                                                    viewpoint.image_height, )  # max_radius=1)#.squeeze(0)
        tmp_gaussian_mask = ~tmp_gaussian_mask
        # gaussian_mask = torch.logical_and(~kp_region_mask, tmp_gaussian_mask)
        gaussian_mask = tmp_gaussian_mask

        unmatched_mask = ~kp_gaussian_mask
        kp_mask = calculate_keypoint_mask(frame.keypoints[unmatched_mask], viewpoint.image_width,
                                          viewpoint.image_height, )
        kp_mask = torch.logical_and(kp_mask, tmp_gaussian_mask)
        N1 = self.gaussians.get_xyz.shape[0]
        #self.add_next_kf(frame.kf_id, viewpoint, depth_map=depth_map, mask=gaussian_mask.squeeze(0).cpu().numpy())
        N2 = self.gaussians.get_xyz.shape[0]
        self.add_next_kf(frame.kf_id, viewpoint, depth_map=depth_map, mask=kp_mask.squeeze(0).cpu().numpy(),
                         keypoints=frame.keypoints, downsample_factor=1.0)
        N3 = self.gaussians.get_xyz.shape[0]
        print('new gaussian', N3-N2, N2-N1, N1)

    def get_gaussian_match_indices(self, gaussian_observation_indices, matches, ref_id, n_start = 0):
        #matches = match[:, idx], match[a, b] 일 때 0이면 a = ref, 1이면 b = ref

        obs_idx_col = gaussian_observation_indices[:, ref_id]  # shape: (N,)
        match_values = matches # shape: (K,)

        eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)
        true_indices = torch.nonzero(eq, as_tuple=False)
        frame_gaussian_index = true_indices[:, 0]
        frame_match_index    = true_indices[:, 1]

        if n_start > 0:
            frame_gaussian_index += n_start
        """
        key_positions = torch.where(eq.any(dim=1), eq.int().argmax(dim=1),
                                    torch.full_like(obs_idx_col, -1, dtype = obs_idx_col.dtype))[0].int()
        included_mask = key_positions != -1

        print(torch.count_nonzero(eq), torch.count_nonzero(included_mask), obs_idx_col.dtype)

        frame_gaussian_index = torch.where(included_mask)[0].int()  # 가우시안의 인덱스
        if n_start > 0:
            frame_gaussian_index += n_start
        frame_match_index = key_positions[included_mask]  # 매치애서 인덱스
        """
        #print('a', obs_idx_col.dtype, obs_idx_col.shape, torch.count_nonzero(obs_idx_col > 0), torch.count_nonzero(included_mask))

        #print('b',frame_match_index.dtype, frame_gaussian_index.dtype, key_positions.dtype)
        return frame_gaussian_index, frame_match_index

    def update_observation(self, frame_gaussian_index, frame_match_index, matches, frame, target_id):
        self.gaussians.observation_indices[frame_gaussian_index, target_id] = matches[frame_match_index]
        self.gaussians.observation_points[frame_gaussian_index, target_id * 2:target_id * 2 + 2] = \
            frame.keypoints[matches[frame_match_index]]

    def check_dist_epipolar_line(self, kp1, kp2, F12, sigma = 1.0):
        # kp1, kp2: (N, 2) shape tensor
        # F12: (3, 3) shape tensor
        # sigma: float scalar

        a = kp1[:, 0] * F12[0, 0] + kp1[:, 1] * F12[1, 0] + F12[2, 0]
        b = kp1[:, 0] * F12[0, 1] + kp1[:, 1] * F12[1, 1] + F12[2, 1]
        c = kp1[:, 0] * F12[0, 2] + kp1[:, 1] * F12[1, 2] + F12[2, 2]

        num = a * kp2[:, 0] + b * kp2[:, 1] + c
        den = a * a + b * b

        dsqr = torch.zeros_like(num)
        valid_mask = den != 0
        dsqr[valid_mask] = num[valid_mask] * num[valid_mask] / den[valid_mask]

        return dsqr < 3.84 * sigma

    def verify_scale_consistency(self, depth_A, depth_B, points1, points2, matches, kp_gaussian_mask, K, R1, t1, R2, t2, width, height, viewpoint, idx):
        """
        depth_A, depth_B: 두 프레임의 깊이 맵
        matches: 특징점 매칭 결과 [(x1,y1), (x2,y2)]
        K: 카메라 내부 파라미터
        R, t: Frame A에서 Frame B로의 회전, 이동
        """
        errors = []

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]

        mpts1, mpts2 = points1[matches[:, 0]], points2[matches[:, 1]]  #: or ...
        depth_values = depth_A[mpts1[:, 1], mpts1[:, 0]]
        valid_depth_mask_A = depth_values > 0

        z2_predicted = depth_B[mpts2[:,1],mpts2[:,0]]
        valid_depth_mask_B = z2_predicted > 0

        tmp_data = torch.cat([
            mpts1.float(),
            depth_values.float().unsqueeze(1)
        ], dim=1).cuda()

        Pw_A = unproject_pixel_to_pc(tmp_data, R1, t1, fx, fy, cx, cy)

        p2_projected, z2_projected, valid_proj = project_pc_to_pixel(Pw_A, R2, t2, fx, fy, cx, cy, width,height)

        valid_mask = valid_depth_mask_B & valid_depth_mask_A & valid_proj

        errors = z2_projected / z2_predicted
        errors = errors[valid_mask]
        diff = p2_projected-mpts2
        diff = diff[valid_mask]
        distances = torch.norm(diff, dim=1)
        kp_gaussian_mask[matches[valid_mask,0][distances > 3.0]] = True
        #print(matches.shape, torch.count_nonzero(distances > 3.0), torch.count_nonzero(distances > 10.0))
        return np.mean(errors.cpu().numpy()), np.std(errors.cpu().numpy())

        Rinv = R1.T
        tinv = -Rinv@t1

        image_np = (
            viewpoint.original_image
                .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                .cpu()  # GPU → CPU
                .numpy()  # NumPy 배열로 변환
        )
        image_np = (image_np * 255.0).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for (x1, y1), (x2, y2) in zip(mpts1, mpts2):
            # Frame A에서 3D 포인트 생성
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))

            z1 = depth_A[y1, x1]
            P1 = np.array([(x1 - K[0, 2]) * z1 / K[0, 0],
                           (y1 - K[1, 2]) * z1 / K[1, 1], z1])

            # Frame B로 변환
            P2_projected = R2@(Rinv @ P1 + tinv) +t2
            z2_projected = P2_projected[2]

            P2_projected = K@P2_projected / z2_projected

            x2_projected = P2_projected[0]
            y2_projected = P2_projected[1]

            # Frame B에서 실제 예측된 깊이
            z2_predicted = depth_B[y2, x2]

            # 오차 계산
            if z2_predicted > 0 and z2_projected > 0:
                #error = abs(z2_projected - z2_predicted) / z2_projected
                error = z2_projected / z2_predicted
                errors.append(error)

                p1 = (int(round(x2_projected)), int(round(y2_projected)))
                p2 = (int(round(x2)), int(round(y2)))

                cv2.line(image_np, p1, p2, (0, 255, 0), 2, lineType=16)
                cv2.circle(image_np, p1, 1, (0, 0, 255), -1, lineType=16)
                cv2.circle(image_np, p2, 1, (255, 0, 0), -1, lineType=16)
        cv2.imwrite('./res/test_ba/d_' + idx +  '.jpg', image_np)

        return np.mean(errors), np.std(errors)

    def get_neighbor_keyframes(self, intersection, kf_id):
        row = intersection[kf_id, :]  # 해당 행 추출
        mask = row > 0  # 0보다 큰 값의 마스크
        values = row[mask]  # 0보다 큰 값만 추출
        indices = torch.where(mask)[0]  # 0보다 큰 값의 인덱스

        # 값 기준으로 내림차순 정렬
        sorted_vals, sort_idx = torch.sort(values, descending=True)  # [1][2][3]

        # 인덱스도 같은 순서로 정렬
        sorted_indices = indices[sort_idx]

        # keyframe id로 변환
        result = [self.keyframe_ids[id_.item()] for id_ in sorted_indices]

        #ids = torch.where(intersection[kf_id,:] > 0)[0]
        #values = [self.keyframe_ids[id_.item()] for id_ in ids]#torch.stack()
        #print('adjacent keyframes',self.keyframe_ids[kf_id],values)
        return result#torch.tensor(result, device = 'cuda')

    def outlier_removal(self, Nold):
        N = 2
        if self.gaussians.observation_indices.shape[1] == 2:
            N = 1
        gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
        obs_indices2 = self.gaussians.observation_indices > -1
        row_sum = obs_indices2.sum(dim=1)
        mask = (row_sum < N) & self.gaussians.isfeatured & (gaussian_indices < Nold)
        mask2 = (row_sum > N) & self.gaussians.isfeatured
        self.gaussians.prune_points(mask)

        print(mask2)
        print('outlier_removal',obs_indices2.shape, torch.count_nonzero(row_sum).item(), torch.count_nonzero(self.gaussians.isfeatured), torch.count_nonzero(mask))
        print('test test', N, Nold, self.gaussians.get_xyz.shape, torch.count_nonzero(row_sum > N), torch.count_nonzero(row_sum < N), torch.count_nonzero((gaussian_indices < Nold)))

    def object_optimization(self, current_window):
        t1 = time.time()
        # 최초 프레임 박스와 가우시안
        cur_idx = current_window[0]
        curr_frame = self.frames[cur_idx]
        curr_viewpoint = self.viewpoints[cur_idx]

        if len(curr_frame.objects) == 0:
            return 0

        loss_object = ObjectLoss()

        obj_gaussians = self.gaussians.get_xyz
        projection, _, valid = project_pc_to_pixel(self.gaussians.get_xyz, curr_viewpoint.R, curr_viewpoint.T,
                                                curr_viewpoint.fx, curr_viewpoint.fy,
                                                curr_viewpoint.cx,curr_viewpoint.cy,
                                                curr_viewpoint.image_width, curr_viewpoint.image_height)

        projection = projection[valid]
        obj_gaussians = obj_gaussians[valid]

        curr_ids = torch.tensor(list(curr_frame.objects.keys()), device='cuda')
        arr = np.array(list(curr_frame.objects.values()))
        curr_bboxes = torch.from_numpy(arr).to('cuda')

        gaussian_obj_ids = loss_object.select_object_gaussians(projection, curr_ids, curr_bboxes)
        box_index = torch.where(gaussian_obj_ids > -1)[0]

        #projection = projection[box_index]
        obj_gaussians = obj_gaussians[box_index]
        gaussian_obj_ids = gaussian_obj_ids[box_index]

        obj_idx = torch.where(valid)[0]
        obj_idx = obj_idx[box_index]
        obj_gaussians_clone = self.gaussians.clone(obj_idx)

        t2 = time.time()

        #각 프레임별 박스 생성
        #loss 계산
        loss = 0
        t3 = 0
        t4 = 0
        t5 = 0
        t6 = 0
        t7 = 0
        t_frame = 0
        for kf_id in current_window[1:]:
            frame = self.frames[kf_id]
            viewpoint = self.viewpoints[kf_id]

            if len(frame.objects) == 0:
                continue

            t_frame += 1
            """
            t3 += time.time()
            tmp_projection, _, valid = project_pc_to_pixel(obj_gaussians, viewpoint.R, viewpoint.T,
                                                    viewpoint.fx, viewpoint.fy,
                                                    viewpoint.cx, viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            tmp_projection = tmp_projection[valid]
            tmp_obj_ids = gaussian_obj_ids[valid]
            t4 += time.time()
            ids = torch.tensor(list(frame.objects.keys()), device='cuda')
            boxes = torch.tensor(list(frame.objects.values()), device='cuda')
            t5 += time.time()
            tmp_boxes = loss_object.convert_index_to_box(tmp_obj_ids, ids, boxes)
            t6 += time.time()
            loss += loss_object.calculate_distance(tmp_projection, tmp_boxes).mean()
            t7 += time.time()
            """
            boxes = torch.tensor(list(frame.objects.values()), device='cuda')
            mask = calculate_bbox_mask(boxes, viewpoint.image_width, viewpoint.image_height)

            render_pkg = render(
                viewpoint, obj_gaussians_clone, self.pipeline_params, self.background
            )
            (
                image,
                depth,
                opacity,
            ) = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            loss_rgb, loss_depth = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, feature_mask = mask
            )
            loss += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

        t_end = time.time()
        print('object mapping', t_frame, t_end-t1, 'test', t4-t3,t5-t4, t6-t5, t7-t6, obj_gaussians.shape, loss)
        return loss

    def bundle_adjustment_with_graph(self, local_gaussian_mask, local_kf_idxs, bPoseUpdate = False, th_obs = 2):
        gaussians_xyz = self.gaussians.get_xyz[local_gaussian_mask]
        obs_points = self.gaussians.observation_points[local_gaussian_mask]
        obs_indices = self.gaussians.observation_indices[local_gaussian_mask]

        loss_ba = 0
        for idx in local_kf_idxs:
            fid = self.covis_kf_ids[idx]
            viewpoint = self.viewpoints[fid]
            frame = self.frames[fid]
            kid = frame.kf_id * 2
            idx = obs_indices[:, frame.kf_id] > -1
            gaussians = gaussians_xyz[idx]

            ##update pose
            if bPoseUpdate:
                tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], axis=0)
                T_w2c = torch.zeros(4, 4, device=viewpoint.R.device)
                T_w2c[0:3, 0:3] = viewpoint.R
                T_w2c[0:3, 3] = viewpoint.T
                T_w2c[3, 3] = 1
                new_w2c = SE3_exp(tau) @ T_w2c

                R = new_w2c[0:3, 0:3]
                t = new_w2c[0:3, 3]
            else:
                R = viewpoint.R
                t = viewpoint.T

            projection, _, valid = project_pc_to_pixel(gaussians, R, t,
                                                       viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                       viewpoint.cy,
                                                       viewpoint.image_width, viewpoint.image_height)
            projection = projection[valid]
            points = obs_points[idx, kid:kid + 2][valid]
            loss_ba += get_reprojection_loss_huber(projection, points).mean()
        return loss_ba

    def bundle_adjustment(self, current_window, bPoseUpdate = False, th_obs = 2):
        t1 = time.time()
        gaussian_indices = torch.where(self.gaussians.isfeatured)[0].cuda()
        gaussians_xyz = self.gaussians.get_xyz[gaussian_indices]
        obs_points = self.gaussians.observation_points[gaussian_indices]
        obs_indices = self.gaussians.observation_indices[gaussian_indices]

        t2= time.time()
        loss_ba = 0
        for fid in current_window:
            viewpoint = self.viewpoints[fid]
            frame = self.frames[fid]
            kid = frame.kf_id*2
            idx = obs_indices[:,frame.kf_id] > -1
            gaussians = gaussians_xyz[idx]

            ##update pose
            if bPoseUpdate:
                tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], axis=0)
                T_w2c = torch.zeros(4,4, device=viewpoint.R.device)
                T_w2c[0:3, 0:3] = viewpoint.R
                T_w2c[0:3, 3] = viewpoint.T
                T_w2c[3, 3] = 1
                new_w2c = SE3_exp(tau) @ T_w2c

                R = new_w2c[0:3, 0:3]
                t = new_w2c[0:3, 3]
            else:
                R = viewpoint.R
                t = viewpoint.T

            projection, _, valid = project_pc_to_pixel(gaussians, R, t,
                                                    viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                    viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            projection = projection[valid]
            points = obs_points[idx, kid:kid + 2][valid]
            loss_ba += get_reprojection_loss_huber(projection, points).mean()
            #print(fid, viewpoint.cam_trans_delta, viewpoint.cam_rot_delta)
        t3 = time.time()
        #print("BA =", t2 - t1, t3 - t2, loss_ba, len(current_window))
        """
        N = 0
        frame_data = defaultdict(lambda: {'xyzs': [], 'pts': []}) #'ids': [], 'count':0
        for gid, xyz, obs in zip(gaussian_indices, gaussians, observations):
            if len(obs) < th_obs:
                continue
            for fid, kpid in obs.items():
                viewpoint = self.viewpoints[fid]
                frame = self.frames[fid]
                #pt = torch.from_numpy(frame.keypoints[kpid]).cuda()
                #frame_data[fid]['ids'].append(gid)
                frame_data[fid]['xyzs'].append(xyz)
                frame_data[fid]['pts'].append(frame.keypoints[kpid])
                #frame_data[fid]['count']+=1

        t2 = time.time()
        for fid in frame_data:
            #print(fid, frame_data[fid]['count'])
            #frame_data[fid]['ids'] = torch.stack(frame_data[fid]['ids'])
            frame_data[fid]['xyzs'] = torch.stack(frame_data[fid]['xyzs'])  # (N, 3)
            frame_data[fid]['pts'] = torch.stack(frame_data[fid]['pts'])  # (N, 2)
        t3 = time.time()

        loss_ba = 0
        for fid in frame_data:
            viewpoint = self.viewpoints[fid]
            projection, valid = project_pc_to_pixel(frame_data[fid]['xyzs'], viewpoint.R, viewpoint.T,
               viewpoint.fx, viewpoint.fy, viewpoint.cx,
               viewpoint.cy,
               viewpoint.image_width, viewpoint.image_height)
            points = frame_data[fid]['pts'][valid]
            loss_ba+=get_reprojection_loss(projection, points).mean()

        t4 = time.time()

        _, _ = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                viewpoint.cy,
                                                viewpoint.image_width, viewpoint.image_height)
        t5 = time.time()
        """
        #print("BA =", t5-t4, t2-t1, t3-t2, t4-t3, loss_ba)

        return loss_ba

    def initialize_map_with_ba(self, cur_frame_idx, viewpoint):
        p = psutil.Process()

        t0 = time.time()
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t6 = 0.0
        t7 = 0.0
        t8 = 0.0
        t9 = 0.0
        t10 = 0.0
        t11 = 0.0
        t12 = 0.0
        nPrune = 0

        curr_frame = self.frames[(cur_frame_idx)]

        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1

            t1 = t1+time.time()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss_rgb, loss_depth = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )

            loss_ba = self.bundle_adjustment([cur_frame_idx])*self.weight_ba

            loss_init = loss_rgb*self.weight_init_rgb+loss_depth*self.weight_init_depth
            loss_init +=loss_ba
            t2 = t2+time.time()

            #projection, points, _, _ = curr_frame.get_correspondence(self.gaussians,viewpoint.R, viewpoint.T,viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,viewpoint.image_width, viewpoint.image_height)
            t3 = t3+time.time()
            #loss_init += get_reprojection_loss(projection,points).mean()*self.weight_reprojection
            t4 = t4+time.time()
            loss_init.backward()
            t5 = t5+time.time()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                t6 = t6+time.time()

                if mapping_iteration % self.init_gaussian_update == 0:
                    t7 = t7+time.time()
                    remove_ids = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    t8 = t8+time.time()
                    """
                    gaussian_indices = torch.arange(self.gaussians._xyz.size()[0])
                    gaussian_obs = self.gaussians.observations[self.gaussians.isfeatured]
                    gaussian_indices = gaussian_indices[self.gaussians.isfeatured]
                    for gaussian_index, obs in zip(gaussian_indices,gaussian_obs):
                        for fid, kpidx in obs.items():
                            frame = self.dataset[str(fid)]
                            frame.gaussianpoints[kpidx] = gaussian_index
                    """
                    #여기서 삭제 된것 + 남은 것 갱신
                    #self.update_gaussian_observation(prune_obs)
                    #self.update_gaussian_observation_after_prune()

                    t9 = t9+time.time()
                    nPrune+=nPrune
                    #print("prune num test", Noldobs, len(prune_obs), torch.count_nonzero(self.gaussians.isfeatured))

                t10 += time.time()
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()
                t11 += time.time()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                t12 += time.time()

            print('initialize_map', self.iteration_count, self.init_itr_num, self.gaussians.get_xyz.shape)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        t13 = time.time()
        Log("Initialized map")
        print("init time = ", t13-t0,'corress', t3-t2,'reprojection',t4-t3,'backword',t5-t4,'update', t6-t5,'prune', t8-t7,'prune update', t9-t8,'other', t11-t10,t12-t11)
        return render_pkg

    def initialize_map(self, cur_frame_idx, viewpoint):
        t0 = time.time()
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t6 = 0.0
        t7 = 0.0
        t8 = 0.0
        t9 = 0.0
        t10 = 0.0
        t11 = 0.0
        t12 = 0.0
        nPrune = 0
        curr_frame = self.frames[(cur_frame_idx)]
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1

            t1 = t1+time.time()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss_rgb, loss_depth = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init = loss_rgb*self.weight_init_rgb+loss_depth*self.weight_init_depth
            t2 = t2+time.time()

            #projection, points, _, _ = curr_frame.get_correspondence(self.gaussians,viewpoint.R, viewpoint.T,viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,viewpoint.image_width, viewpoint.image_height)
            t3 = t3+time.time()
            #loss_init += get_reprojection_loss(projection,points).mean()*self.weight_reprojection
            t4 = t4+time.time()
            loss_init.backward()
            t5 = t5+time.time()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                t6 = t6+time.time()

                if mapping_iteration % self.init_gaussian_update == 0:
                    t7 = t7+time.time()
                    remove_ids = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    t8 = t8+time.time()
                    """
                    gaussian_indices = torch.arange(self.gaussians._xyz.size()[0])
                    gaussian_obs = self.gaussians.observations[self.gaussians.isfeatured]
                    gaussian_indices = gaussian_indices[self.gaussians.isfeatured]
                    for gaussian_index, obs in zip(gaussian_indices,gaussian_obs):
                        for fid, kpidx in obs.items():
                            frame = self.dataset[str(fid)]
                            frame.gaussianpoints[kpidx] = gaussian_index
                    """
                    #여기서 삭제 된것 + 남은 것 갱신
                    #self.update_gaussian_observation(prune_obs)
                    #self.update_gaussian_observation_after_prune()

                    t9 = t9+time.time()
                    nPrune+=nPrune
                    #print("prune num test", Noldobs, len(prune_obs), torch.count_nonzero(self.gaussians.isfeatured))

                t10 += time.time()
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()
                t11 += time.time()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                t12 += time.time()
            print('initialize_map', self.iteration_count, self.init_itr_num, self.gaussians.get_xyz.shape)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        t13 = time.time()
        Log("Initialized map")
        print("init time = ", t13-t0,'corress', t3-t2,'reprojection',t4-t3,'backword',t5-t4,'update', t6-t5,'prune', t8-t7,'prune update', t9-t8,'other', t11-t10,t12-t11)
        return render_pkg

    def update_occ_visibility(self, current_window):
        # update occ
        self.occ_aware_visibility = {}
        with torch.no_grad():
            for cam_idx in current_window:
                viewpoint = self.viewpoints[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    n_touched,
                ) = (
                    render_pkg["n_touched"],
                )
                self.occ_aware_visibility[cam_idx] = (n_touched > 0).long()

    def check_outlier(self, gaussians, keypoints, viewpoint, th = 9.0):
        projections, _, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                   viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                   viewpoint.image_width, viewpoint.image_height)
        #projections = projection[valid]
        #keypoints = keypoints[valid]

        l2 = torch.sum((projections - keypoints) ** 2, dim=1)
        inlier = l2 < th
        #valid[torch.where(valid)[0]] = inlier
        return valid & inlier

        #outlier_idx = gaussian_indices[idx][valid][outlier]
        #Noutlier = outlier_idx.shape[0]
        #self.gaussians.observation_indices[outlier_idx, frame.kf_id] = torch.full((Noutlier,), -1, device='cuda').type(torch.int32)
        #self.gaussians.observation_points[outlier_idx, kid:kid + 2] = torch.full((Noutlier, 2), -1.0, device='cuda')

    def keyframe_matches_with_graph(self, frame, viewpoint, new_kf_idx, keyframe_ids, matches_info, kp_gaussian_mask):
        a = time.time()
        for idx in keyframe_ids:
            kf_idx = self.covis_kf_ids[idx]
            keyframe = self.frames[kf_idx]
            d0 = {'keypoints': frame.keypoints,
                  'descriptors': torch.from_numpy(frame.descriptors).cuda(),
                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
            d1 = {'keypoints': keyframe.keypoints,
                  'descriptors': torch.from_numpy(keyframe.descriptors).cuda(),
                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
            matches = self.feature_manager.match_lightglue(d0, d1).int()
            matches_info[new_kf_idx][kf_idx] = matches
            matches_info[kf_idx][new_kf_idx] = matches[:, [1, 0]]
            kp_gaussian_mask[matches[:, 0]] = False
            #print('kf match', new_kf_idx, kf_idx, matches.shape)
        b = time.time()
        print('match time',b-a)

    def keyframe_matches(self, keyframes, frame, viewpoint, matches_info, tmp_id, kp_gaussian_mask, N_window = 10):

        #match_count = torch.zeros(frame.keypoints.shape[0], dtype= torch.bool, device = 'cuda')
        for idx, kf_idx in enumerate(keyframes[-N_window:-1]):
            keyframe = self.frames[kf_idx]
            d0 = {'keypoints': frame.keypoints,
                  'descriptors': torch.from_numpy(frame.descriptors).cuda(),
                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
            d1 = {'keypoints': keyframe.keypoints,
                  'descriptors': torch.from_numpy(keyframe.descriptors).cuda(),
                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
            matches = self.feature_manager.match_lightglue(d0, d1).int()

            #viewpoint2 = self.viewpoints[kf_idx]
            #F12 = compute_F12(viewpoint.R, viewpoint.T, viewpoint2.R, viewpoint2.T, K_inv, K_inv)
            #res = self.check_dist_epipolar_line(frame.keypoints[matches[:, 0]], keyframe.keypoints[matches[:, 1]],F12)
            #matches = matches[res, :]

            matches_info[tmp_id][kf_idx] = matches
            matches_info[kf_idx][tmp_id] = matches[:, [1, 0]]

            kp_gaussian_mask[matches[:,0]] = False

    def extension_window(self, keyframe, curr_window):
        pass

    #가우시안과 커넥티드 키프레임 전달. 직접 연결
    def update_gaussian_observation_with_graph(self, new_kf_idx, keyframe_idxs, matches_info, K_inv, kp_gaussian_mask = None, Forward = True):
        """
        if Forward:
            start_idx = 0
            end_idx = len(keyframe_idxs) - 1
            step = 1
        else:
            start_idx = len(keyframe_idxs) - 1
            end_idx = 0
            step = -1
        """
        if Forward:
            list = keyframe_idxs
        else:
            list = reversed(keyframe_idxs)

        a = time.time()

        for idx in list:

            kf_idx = self.covis_kf_ids[idx]

            if Forward:
                source_id = kf_idx
                target_id = new_kf_idx
            else:
                source_id = new_kf_idx
                target_id = kf_idx

            keyframe1 = self.frames[source_id]
            keyframe2 = self.frames[target_id]

            viewpoint1 = self.viewpoints[source_id]
            viewpoint2 = self.viewpoints[target_id]

            amatches = matches_info[source_id][target_id]

            # epipoloar constraints
            F12 = compute_F12(viewpoint1.R, viewpoint1.T, viewpoint2.R, viewpoint2.T, K_inv, K_inv)
            res = self.check_dist_epipolar_line(keyframe1.keypoints[amatches[:, 0]],
                                                keyframe2.keypoints[amatches[:, 1]], F12)
            matches = amatches[res, :]

            # 가우시안과 소스 키프레임 사이의 매칭 확인
            agidx, afidx = self.get_gaussian_match_indices(self.gaussians.observation_indices, matches[:, 0],
                                                           keyframe1.kf_id)

            ##overlap mask : 타겟 키프레임에 가우시안이 있는지 확인
            empty_mask = self.gaussians.observation_indices[agidx, keyframe2.kf_id] == -1
            gidx = agidx[empty_mask]
            fidx = afidx[empty_mask]

            ##타겟 키프레임에 reprojection error
            res = self.check_outlier(self.gaussians.get_xyz[gidx], keyframe2.keypoints[matches[fidx, 1]], viewpoint2)
            gidx = gidx[res]
            fidx = fidx[res]

            """
            print('update test::', Forward, source_id, target_id, '=', 'c=',
                  torch.count_nonzero(res).item(),
                  torch.count_nonzero(empty_mask).item(), agidx.shape,
                  'gau', torch.count_nonzero(self.gaussians.observation_indices[:, keyframe1.kf_id] > -1).item(),
                  amatches.shape, matches.shape)
            """
            ##타겟 키프레임에 연관
            self.update_observation(gidx, fidx, matches[:, 1], keyframe2, keyframe2.kf_id)
            if Forward and target_id == new_kf_idx:
                selected_values = matches[fidx, 1]
                kp_gaussian_mask[selected_values] = True

        b = time.time()
        print('update observation time', b-a)


    def update_gaussian_observation(self, keyframes, extension_indices, matches_info, new_kf_id, first_idx, K_inv, N_window = 10, kp_gaussian_mask = None, Forward = True):
        if Forward:
            start_idx = 0
            end_idx = len(keyframes)-1
            step = 1
        else:
            start_idx = len(keyframes)-1
            end_idx = 0
            step = -1

        for idx in range(start_idx, end_idx, step):
        #for idx in extension_indices[start_idx : end_idx : step]:
            next_idx = idx+step
            kf_idx1 = keyframes[idx]
            kf_idx2 = keyframes[next_idx]
            """
            diff = first_idx -idx
            
            if diff < N_window:
                kf_idx2 = new_kf_id
            """
            """
            if Forward:
                source_id = kf_idx1 #가우시안이 있는 프레임
                target_id = kf_idx2 #가우시안을 연결한 프레임
            else:
                source_id = kf_idx2
                target_id = kf_idx1
            """
            source_id = kf_idx1
            target_id = kf_idx2

            #print('update_gaussian_test', Forward, source_id, target_id, '==', idx, next_idx, extension_indices)
            keyframe1 = self.frames[source_id]
            keyframe2 = self.frames[target_id]

            viewpoint1 = self.viewpoints[source_id]
            viewpoint2 = self.viewpoints[target_id]

            amatches = matches_info[kf_idx1][kf_idx2]

            # epipoloar constraints
            F12 = compute_F12(viewpoint1.R, viewpoint1.T, viewpoint2.R, viewpoint2.T, K_inv, K_inv)
            res = self.check_dist_epipolar_line(keyframe1.keypoints[amatches[:, 0]], keyframe2.keypoints[amatches[:, 1]],F12)
            matches = amatches[res, :]

            #가우시안과 소스 키프레임 사이의 매칭 확인
            agidx, afidx = self.get_gaussian_match_indices(self.gaussians.observation_indices, matches[:, 0],keyframe1.kf_id)

            ##overlap mask : 타겟 키프레임에 가우시안이 있는지 확인
            empty_mask = self.gaussians.observation_indices[agidx, keyframe2.kf_id] == -1
            gidx = agidx[empty_mask]
            fidx = afidx[empty_mask]

            ##타겟 키프레임에 reprojection error
            res = self.check_outlier(self.gaussians.get_xyz[gidx], keyframe2.keypoints[matches[fidx, 1]], viewpoint2)
            gidx = gidx[res]
            fidx = fidx[res]

            print('update test::',Forward, source_id, target_id,'=', idx, next_idx, 'c=', torch.count_nonzero(res).item(),
                  torch.count_nonzero(empty_mask).item(), agidx.shape,
                  'gau', torch.count_nonzero(self.gaussians.observation_indices[:, keyframe1.kf_id] > -1).item(), amatches.shape, matches.shape)

            ##타겟 키프레임에 연관
            self.update_observation(gidx, fidx, matches[:, 1], keyframe2, keyframe2.kf_id)
            if Forward and target_id == new_kf_id:
                selected_values = matches[fidx, 1]
                kp_gaussian_mask[selected_values] = True

    def connect_gaussian_and_keyframes(self, keyframes, extension_indices, matches_info, kp_gaussian_mask, new_kf_id, first_idx, K_inv, N_window = 10):

        for idx in extension_indices[:0:-1]:
            kf_idx1 = keyframes[idx]

            diff = first_idx - idx

            if diff < N_window:
                kf_idx2 = new_kf_id
                case = 1
            elif idx % N_window == 0:
                kf_idx2 = keyframes[idx + N_window]
                case = 2
                #print('test % 5 = ', idx, idx + N_window, '=', kf_idx1, kf_idx2)
            else:
                kf_idx2 = keyframes[((idx // N_window) + 1) * N_window]
                case = 3
                #print('test not % 5 = ', idx, ((idx // N_window) + 1) * N_window, '=', kf_idx1, kf_idx2)

            keyframe1 = self.frames[kf_idx1]
            keyframe2 = self.frames[kf_idx2]

            viewpoint1 = self.viewpoints[kf_idx1]
            viewpoint2 = self.viewpoints[kf_idx2]

            matches = matches_info[kf_idx1][kf_idx2]

            #epipoloar constraints
            F12 = compute_F12(viewpoint1.R, viewpoint1.T, viewpoint2.R, viewpoint2.T, K_inv, K_inv)
            res = self.check_dist_epipolar_line(keyframe1.keypoints[matches[:,0]], keyframe2.keypoints[matches[:,1]], F12)
            matches = matches[res,:]

            #a = self.gaussians.observation_indices[:,keyframe1.kf_id]>-1
            #b = torch.unique(self.gaussians.observation_indices[a,keyframe1.kf_id])
            #c = torch.unique(matches[:,0])
            #print('match test', kf_idx1, kf_idx2, matches.shape, keyframe1.keypoints.shape, torch.count_nonzero(c), torch.count_nonzero(self.gaussians.observation_indices[:,keyframe1.kf_id]>-1), torch.count_nonzero(b), torch.count_nonzero(self.gaussians.observation_indices[:,keyframe2.kf_id]>-1))
            gidx, fidx = self.get_gaussian_match_indices(self.gaussians.observation_indices, matches[:, 0], keyframe1.kf_id)

            ##overlap mask
            empty_mask = self.gaussians.observation_indices[gidx, keyframe2.kf_id] == -1
            gidx = gidx[empty_mask]
            fidx = fidx[empty_mask]

            ##reprojection error
            res = self.check_outlier(self.gaussians.get_xyz[gidx], keyframe2.keypoints[matches[fidx,1]], viewpoint2)
            gidx = gidx[res]
            fidx = fidx[res]

            print('update test::forward', kf_idx1, kf_idx2, 'c=', case, torch.count_nonzero(res).item(), torch.count_nonzero(empty_mask).item(),
                  'gau=', torch.count_nonzero(self.gaussians.observation_indices[:, keyframe1.kf_id] > -1).item(),
                  matches.shape)

            self.update_observation(gidx, fidx, matches[:, 1], keyframe2, keyframe2.kf_id)
            if kf_idx2 == new_kf_id:
                selected_values = matches[fidx, 1]
                kp_gaussian_mask[selected_values] = True
        #print("asdfasdf", kp_gaussian_mask.shape[0], torch.count_nonzero(kp_gaussian_mask), matches.shape[0], new_kf_id)

    def select_kf_and_fixed_kf(self, col_id, th_mp = 20, th_obs = 0, nKF = 10):
        #가우시안 선택
        kf_gaussian_ids = self.gaussians.observation_indices[:,col_id] > -1

        #인접 키프레임
        N = self.gaussians.observation_indices.shape[1]
        nKF = min(N, nKF)

        tmp_connected_kfs = self.gaussians.observation_indices[kf_gaussian_ids,:]
        mask_cols = tmp_connected_kfs != -1
        col_counts = mask_cols.sum(dim = 0)

        _, top_n_indices = torch.topk(col_counts, nKF, largest=True, sorted=True)
        top_k_mask = torch.zeros_like(col_counts, dtype=torch.bool)
        top_k_mask[top_n_indices] = True

        connected_kf_mask = (col_counts >= th_mp) & top_k_mask
        #torch.nonzero(col_counts >= N_obs).squeeze(1)

        #포함되지 않는 가우시안 선택
        tmp_local_gaussian_ids = self.gaussians.observation_indices[:, connected_kf_mask]
        mask_rows = (tmp_local_gaussian_ids != -1)
        row_counts = mask_rows.sum(dim=1)
        local_gaussian_mask = row_counts > th_obs
        #print(mask_rows.shape, row_counts.shape, torch.count_nonzero(local_gaussian_mask).item())
        #fixed_keyframes
        tmp_connected_kfs = self.gaussians.observation_indices[local_gaussian_mask, :]
        mask_cols = tmp_connected_kfs != -1
        col_counts = mask_cols.sum(dim=0)
        fixed_kf_mask = torch.logical_or((col_counts>=th_mp), connected_kf_mask)
        #print('connected kfs', b-a)#, connected_kf_mask, fixed_kf_mask, "=", torch.count_nonzero(local_gaussian_mask).item())
        return local_gaussian_mask, connected_kf_mask.nonzero(as_tuple=True)[0].tolist(), fixed_kf_mask.nonzero(as_tuple=True)[0].tolist()

    def map_with_ba(self, current_window, prune=False, iters=1, matches=None, extension_graph=None):
        feature_radius = 9
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = [] #kf_idx를 저장
        frames_to_optimize = self.config["Training"]["pose_window"]
        #frames_to_optimize = len(extension_graph)

        sorted_kf_ids = [self.frames[kf_idx].kf_id for kf_idx in current_window]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(cam_idx)

        remove_ids = None
        last_kf_id = current_window[0]

        curr_patches = None
        curr_patches_valid = None
        curr_rendered_image = None
        kf_patches = defaultdict(lambda: {'patch': None, 'valid': None})

        ##patch consistency
        doPatchConsistency = False
        """
        if False and graph is not None and len(matches) > 0:
            doPatchConsistency = True
            for cam_idx in graph:
                if cam_idx == last_kf_id or cam_idx not in matches:
                    continue
                match = matches[cam_idx]
                viewpoint = self.viewpoints[cam_idx]
                frame = self.frames[cam_idx]
                kf_patches[cam_idx]['patch'], kf_patches[cam_idx]['valid'] = frame.extract_patches_differentiable(
                    viewpoint.original_image, frame.keypoints[match[:, 1]])
        """
        ##patch consistency

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            for cam_idx, kf_idx in enumerate(self.current_window):
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                kf_frame = self.frames[kf_idx]
                loss_rgb, loss_depth = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,  # feature_mask=kf_frame.mapping_mask
                )
                loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
                loss_mapping += loss_kf

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)



            for stack_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                frame_id = random_viewpoint_stack[stack_idx]
                viewpoint = self.viewpoints[frame_id]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                kf_frame = self.frames[frame_id]
                loss_rgb, loss_depth = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, #feature_mask=kf_frame.mapping_mask
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            ##geometric consistency
            if extension_graph is None:
                loss_ba = self.bundle_adjustment(current_window, bPoseUpdate= False)
                loss_mapping += loss_ba * self.weight_ba
            else:
                loss_ba = self.bundle_adjustment(extension_graph, bPoseUpdate=False)
                loss_mapping += loss_ba * self.weight_ba
            """
            if graph is not None:
                #graph / current_window
                #loss_ba = self.bundle_adjustment(current_window)
                #loss_mapping += loss_ba * self.weight_ba

                loss_ba = self.bundle_adjustment2(last_kf_id, current_window)
                loss_mapping += loss_ba * self.weight_ba

                loss_obj = self.object_optimization(current_window)
                loss_mapping += loss_obj * self.weight_ba
            """
            if False and self.gaussians._xyz.grad is not None:
                after_grad_loss_ba = self.gaussians._xyz.grad.clone()
                affected_by_ba = torch.any(after_grad_loss_ba != 0, dim=1)
                print("Gaussians affected by ba:", torch.count_nonzero(affected_by_ba), self.gaussians._xyz.shape,
                      affected_by_ba.nonzero().flatten())

            ##isotropic_scaling
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    to_prune = None
                    if len(current_window) >= self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            #sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_kf_ids[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            # 여기도 직접 수정해야 함.
                            self.gaussians.update_gaussian_observation_before_prune(to_prune.cuda(), self.frames)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                            # self.update_gaussian_observation(prune_obs)
                            # self.update_gaussian_observation_after_prune()
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False, None

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                )

                if update_gaussian:
                    remove_ids = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    for idx in range((len(current_window))):
                        viewpoint = viewpoint_stack[idx]
                        current_idx = current_window[idx]
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background
                        )
                        (
                            n_touched,
                        ) = (
                            render_pkg["n_touched"],
                        )
                        self.occ_aware_visibility[current_idx] = (n_touched > 0).long()

                    # self.update_gaussian_observation(prune_obs)
                    # self.update_gaussian_observation_after_prune()
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    print("before Resetting, ", self.gaussians._xyz.size())
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    print("after Resetting, ", self.gaussians._xyz.size())
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                ## Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.first_kf_id or not self.pose_update:
                        continue
                    update_pose(viewpoint)
                ##graph update
                """
                if graph is not None:
                    self.update_graph(graph, th = 100.0)
                self.update_graph_weights()
                """
                ##graph update

        return gaussian_split, remove_ids

    def map(self, current_window, prune=False, iters=1):
        feature_radius = 9
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        remove_ids = None
        last_kf_id = current_window[0]

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_rgb,loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                )
                loss_kf= loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
                loss_mapping += loss_kf

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_rgb, loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            ##isotropic_scaling
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    to_prune = None
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            #여기도 직접 수정해야 함.
                            self.gaussians.update_gaussian_observation_before_prune(to_prune.cuda(), self.frames)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                            #self.update_gaussian_observation(prune_obs)
                            #self.update_gaussian_observation_after_prune()
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False, None

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every== self.gaussian_update_offset
                )

                if update_gaussian:
                    remove_ids = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    for idx in range((len(current_window))):
                        viewpoint = viewpoint_stack[idx]
                        current_idx = current_window[idx]
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background
                        )
                        (
                            n_touched,
                        ) = (
                            render_pkg["n_touched"],
                        )
                        self.occ_aware_visibility[current_idx] = (n_touched > 0).long()

                    #self.update_gaussian_observation(prune_obs)
                    #self.update_gaussian_observation_after_prune()
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    print("before Resetting, ", self.gaussians._xyz.size())
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    print("after Resetting, ", self.gaussians._xyz.size())
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                ## Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.first_kf_id or not self.pose_update:
                        continue
                    update_pose(viewpoint)

        return gaussian_split, remove_ids

    def run_with_ba(self):

        profiler = cProfile.Profile()

        matches_info = {}
        extension_window = None
        N_last_window = 5
        N_inc_window = 3
        patience = 1

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                ###TEST BA
                """"""
                #best_loss = float('inf')
                wait = 0
                for i in range(5):
                    loss_ba = self.bundle_adjustment(extension_window, bPoseUpdate=True)*self.weight_ba
                    loss_value = loss_ba.item()
                    if loss_value > best_loss:
                        wait += 1
                        if wait >= patience:
                            break
                    else:
                        best_loss = loss_value
                    loss_ba.backward()
                    with torch.no_grad():
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                        # self.gaussians.update_learning_rate(self.iteration_count)
                        self.graph_optimizers.step()
                        self.graph_optimizers.zero_grad(set_to_none=True)
                        for kf_idx in extension_window:
                            viewpoint = self.viewpoints[kf_idx]
                            update_pose(viewpoint)
                ###TEST BA

                s = time.time()
                prune_mask = None
                _, prune_mask1 = self.map_with_ba(self.current_window, extension_graph=extension_window)#matches=kf_matches, graph=current_window)  # matches = kf_matches, graph = recent_keys
                #_, prune_mask1 = self.map(self.current_window)

                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2 = self.map_with_ba(self.current_window, prune=True, iters=10, extension_graph=extension_window)#matches=kf_matches, graph=current_window)  # matches=kf_matches, graph = recent_keys
                    #_, prune_mask2 = self.map(self.current_window, prune=True, iters=10)  # matches=kf_matches, graph=current_window)  # matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2

                    if prune_mask is not None:
                        self.push_to_frontend(prune=prune_mask)
                    else:
                        self.push_to_frontend()
                    e = time.time()

                print("backend = mapping with empty queue", (e-s), self.gaussians.get_xyz.shape[0])
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # print("backend::queue::get::start")
                data = self.backend_queue.get()
                # print("backend::queue::get::end")
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "sync":
                    pass
                elif data[0] == "graph":
                    self.bDoingMapping.store(False)
                    a = time.time()

                    src = data[1]
                    cur_frame_idx = data[2]
                    viewpoint = data[3]
                    depth_map = data[4]
                    f = data[5]
                    device = self.devices[src]
                    tmp_id = ConvertFramdId(src, cur_frame_idx)

                    move_camera_to_gpu(viewpoint)

                    self.viewpoints[tmp_id] = viewpoint

                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = tmp_id
                    self.frames[(tmp_id)] = frame
                    self.covis_kf_ids[frame.kf_id] = tmp_id

                    #reference keyframe
                    ref_kf_id = ConvertFramdId(src, device.last_keyframe_idx)
                    ref_keyframe = self.frames[ref_kf_id]
                    device.last_keyframe_idx = cur_frame_idx

                    ##로컬 맵 구성
                    alocal_gaussians, alocal_kfs_indices, afixed_kfs_indices = self.select_kf_and_fixed_kf(ref_keyframe.kf_id)
                    print('local map', torch.count_nonzero(alocal_gaussians).item(), alocal_kfs_indices, afixed_kfs_indices)
                    self.preprocessing_add_kf()
                    # 가우시안이 새 키프레임에 매칭 되면 True. 하나도 매칭이 없는 영역에서 새로운 가우시안 생성.
                    # 새로운 가우시안은 이전 프레임과 매칭이 될 수 있고, 아예 없을 수도 있음.
                    kp_gaussian_mask = torch.ones(frame.keypoints.shape[0], dtype=torch.bool, device='cuda')

                    last_keys = list(self.frames)
                    # N_window = len(last_keys)
                    matches_info[tmp_id] = {}

                    self.keyframe_matches_with_graph(frame, viewpoint, tmp_id, alocal_kfs_indices, matches_info, kp_gaussian_mask)

                    self.update_gaussian_observation_with_graph(tmp_id, alocal_kfs_indices, matches_info, device.K_inv_gpu, kp_gaussian_mask=kp_gaussian_mask, Forward=True)

                    Nold = self.gaussians._xyz.size()[0]

                    with torch.no_grad():
                        self.add_next_kf_with_ba(frame, viewpoint, depth_map, kp_gaussian_mask)

                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    new_gaussian_mask = gaussian_indices >= Nold
                    self.update_gaussian_observation_with_graph(tmp_id, alocal_kfs_indices, matches_info, device.K_inv_gpu, kp_gaussian_mask=kp_gaussian_mask,Forward=False)

                    """
                    #alocal_kfs_indices.append(frame.kf_id)
                    #afixed_kfs_indices.append(frame.kf_id)

                    ##extension_window
                    #index_map = {value: idx for idx, value in enumerate(last_keys)}
                    #window_indices = [index_map[x] for x in current_window] + [index_map[x] for x in
                    #                                                           last_keys[-N_last_window:-1]]
                    #tmp_extension_indices = list(range(0, max(window_indices), N_inc_window))  # min(window_indices)
                    #extension_indices = list(set(window_indices + tmp_extension_indices))
                    extension_indices = list(range(0,len(last_keys)-1))
                    extension_indices.sort(reverse=True)
                    #extension_window = [last_keys[i] for i in extension_indices]
                    extension_window = last_keys

                    new_kf_idx = extension_indices[0]

                    # 최근 5개 키프레임과 매칭 추가
                    self.keyframe_matches(last_keys, frame, viewpoint, matches_info, tmp_id, kp_gaussian_mask,
                                          N_window=N_last_window + 1)
                    # self.connect_gaussian_and_keyframes(last_keys, extension_indices, matches_info, kp_gaussian_mask, tmp_id, new_kf_idx, device.K_inv_gpu, N_window = N_last_window)
                    self.update_gaussian_observation(last_keys, extension_indices, matches_info, tmp_id, new_kf_idx,
                                                     device.K_inv_gpu, N_window=N_last_window,
                                                     kp_gaussian_mask=kp_gaussian_mask, Forward=True)

                    Nold = self.gaussians._xyz.size()[0]

                    with torch.no_grad():
                        self.add_next_kf_with_ba(frame, viewpoint, depth_map, kp_gaussian_mask)

                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    gaussian_mask = gaussian_indices >= Nold
                    tmp_gaussian_indices = self.gaussians.observation_indices[gaussian_mask]

                    # 새로운 가우시안 포인트를 이전 프레임에 전파
                    self.update_gaussian_observation(last_keys, extension_indices, matches_info, tmp_id, new_kf_idx,
                                                     device.K_inv_gpu, N_window=N_last_window, Forward=False)
                    """
                    ##키프레임 선택
                    local_gaussians, local_kfs_indices, fixed_kfs_indices = self.select_kf_and_fixed_kf(frame.kf_id)

                    ###TEST BA
                    """"""
                    t_ba_1 = time.time()
                    graph_opt_params = []
                    # 윈도우 내의 뷰포인트에 접근해서 최근 뷰포인트는 포즈까지 추가. 나머지는 exposure만 추가
                    for idx in local_kfs_indices:
                        kf_idx = self.covis_kf_ids[idx]
                        if kf_idx == self.first_kf_id:
                            continue
                        win_viewpoint = self.viewpoints[kf_idx]
                        graph_opt_params.append(
                            {
                                "params": [win_viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                      * 0.5,
                                "name": "rot_{}".format(win_viewpoint.uid),
                            }
                        )
                        graph_opt_params.append(
                            {
                                "params": [win_viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"][
                                          "cam_trans_delta"
                                      ]
                                      * 0.5,
                                "name": "trans_{}".format(win_viewpoint.uid),
                            }
                        )

                    self.graph_optimizers = torch.optim.Adam(graph_opt_params)
                    best_loss = float('inf')
                    wait = 0
                    for i in range(5):
                        loss_ba = self.bundle_adjustment_with_graph(local_gaussians, fixed_kfs_indices, bPoseUpdate=True) * self.weight_ba
                        loss_value = loss_ba.item()
                        if loss_value > best_loss:
                            wait += 1
                            if wait >= patience:
                                break
                        else:
                            best_loss = loss_value
                        loss_ba.backward()
                        with torch.no_grad():
                            self.gaussians.optimizer.step()
                            self.gaussians.optimizer.zero_grad(set_to_none=True)
                            # self.gaussians.update_learning_rate(self.iteration_count)
                            self.graph_optimizers.step()
                            self.graph_optimizers.zero_grad(set_to_none=True)
                            for kf_idx in extension_window:
                                aviewpoint = self.viewpoints[kf_idx]
                                keyframe = self.frames[kf_idx]
                                kf_id = keyframe.kf_id * 2
                                idx = self.gaussians.observation_indices[:, keyframe.kf_id] > -1
                                keypoints = self.gaussians.observation_points[idx, kf_id:kf_id + 2]

                                update_pose(aviewpoint)

                                inlier = self.check_outlier(self.gaussians.get_xyz[idx], keypoints, aviewpoint)
                                outlier = ~inlier
                                Noutlier = torch.count_nonzero(outlier)
                                self.gaussians.observation_indices[idx, keyframe.kf_id][outlier] = torch.full(
                                    (Noutlier,), -1, device='cuda').int()
                                self.gaussians.observation_points[idx, kf_id:kf_id + 2][outlier] = torch.full(
                                    (Noutlier, 2), -1, device='cuda').float()

                        t_ba_2 = time.time()
                        print('ba test', t_ba_2 - t_ba_1, loss_ba)
                    self.outlier_removal(Nold)
                    ###TEST BA

                    #self.update_occ_visibility(self.current_window)
                    #self.push_to_frontend("keyframe", src=src)
                    local_gaussians, _, _ = self.select_kf_and_fixed_kf(frame.kf_id, nKF = 5)
                    self.push_to_frontend_with_graph(mask = local_gaussians, src = src)

                    b = time.time()
                    print('backend::graph', tmp_id, b-a, len(alocal_kfs_indices), len(afixed_kfs_indices))
                    self.bDoingMapping.store(True)
                elif data[0] == "init":
                    profiler.enable()
                    src = data[1]
                    cur_frame_idx = data[2]
                    viewpoint = data[3]
                    depth_map = data[4]
                    f = data[5]
                    device = self.devices[src]
                    tmp_id = ConvertFramdId(src, cur_frame_idx)

                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    self.covis_kf_ids[frame.kf_id] = tmp_id

                    move_camera_to_gpu(viewpoint)

                    Log("Resetting the system")
                    self.reset()
                    self.frames[tmp_id] = frame
                    self.first_kf_id = tmp_id
                    self.viewpoints[tmp_id] = viewpoint
                    device.last_keyframe_idx = cur_frame_idx

                    #mask
                    """
                    kp_region_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width,
                                                            viewpoint.image_height, max_radius=2)
                    """
                    kp_mask = calculate_keypoint_mask(frame.keypoints, viewpoint.image_width,
                                                            viewpoint.image_height,)
                    self.preprocessing_add_kf()
                    self.add_next_kf(
                        frame.kf_id, viewpoint, depth_map=depth_map, init=True, mask = (~kp_mask).squeeze(0).cpu().numpy()
                    )
                    self.add_next_kf(
                        frame.kf_id, viewpoint, depth_map=depth_map, init=True,keypoints=frame.keypoints,
                        mask=kp_mask.squeeze(0).cpu().numpy(), downsample_factor=1.0
                    )

                    matches_info[tmp_id] = {}
                    ##test
                    print("init_ba", torch.count_nonzero(self.gaussians.isfeatured), torch.count_nonzero(self.gaussians.observation_points>-1), torch.count_nonzero(self.gaussians.observation_indices > -1), frame.keypoints.shape)

                    self.initialize_map_with_ba(tmp_id, viewpoint)
                    extension_window = [tmp_id]
                    self.push_to_frontend("init", first_id=tmp_id)

                    ##test image save
                    #feature and gaussian
                    image_np = (
                        viewpoint.original_image
                            .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                            .clone().detach().cpu()  # GPU → CPU
                            .numpy()  # NumPy 배열로 변환
                    )
                    image_np = (image_np * 255.0).astype(np.uint8)
                    out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    #gaussian projection
                    projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz[self.gaussians.isfeatured], viewpoint.R,
                                                                          viewpoint.T,
                                                                          viewpoint.fx, viewpoint.fy,
                                                                          viewpoint.cx, viewpoint.cy,
                                                                          viewpoint.image_width,
                                                                          viewpoint.image_height)
                    projection = projection[valid_projection]
                    points1 = projection.detach().cpu().numpy()
                    points2 = frame.keypoints.detach().cpu().numpy()
                    for pt1 in points2:
                        p1 = (int(round(pt1[0])), int(round(pt1[1])))
                        cv2.circle(out, p1, 3, (0, 0, 255), 1, lineType=16)
                    for pt1 in points1:
                        p1 = (int(round(pt1[0])), int(round(pt1[1])))
                        cv2.circle(out, p1, 2, (0, 255, 0), 1, lineType=16)

                    cv2.imwrite('./res/test_ba/' + tmp_id+ '.jpg', out)
                    profiler.disable()
                    #profiler.print_stats(sort='tottime') #tottime, cumtime, percall

                elif data[0] == "keyframe":
                    self.bDoingMapping.store(False)
                    profiler.enable()
                    s = time.time()
                    src = data[1]
                    cur_frame_idx = data[2]
                    viewpoint = data[3]
                    current_window = data[4]
                    depth_map = data[5]
                    f = data[6]
                    device = self.devices[src]
                    tmp_id = ConvertFramdId(src, cur_frame_idx)

                    move_camera_to_gpu(viewpoint)

                    self.viewpoints[tmp_id] = viewpoint
                    self.current_window = current_window

                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = tmp_id
                    self.frames[(tmp_id)] = frame

                    ##match
                    self.preprocessing_add_kf()
                    #가우시안이 새 키프레임에 매칭 되면 True. 하나도 매칭이 없는 영역에서 새로운 가우시안 생성.
                    #새로운 가우시안은 이전 프레임과 매칭이 될 수 있고, 아예 없을 수도 있음.
                    kp_gaussian_mask = torch.ones(frame.keypoints.shape[0], dtype=torch.bool, device='cuda')

                    last_keys = list(self.frames)
                    #N_window = len(last_keys)
                    matches_info[tmp_id]={}

                    ##extension_window
                    index_map = {value: idx for idx, value in enumerate(last_keys)}
                    window_indices = [index_map[x] for x in current_window] + [index_map[x] for x in last_keys[-N_last_window:-1]]
                    tmp_extension_indices = list(range(0, max(window_indices), N_inc_window)) #min(window_indices)
                    extension_indices = list(set(window_indices + tmp_extension_indices))
                    #extension_indices = list(set(window_indices))
                    extension_indices.sort(reverse = True)
                    extension_window = [last_keys[i] for i in extension_indices]

                    #tmp_extension_window.reverse()
                    #extension_indices = [x for x in tmp_extension_window if
                    #                     x not in window_indices]  # 현재 윈도우에 포함 안된 5의 배수 찾기

                    new_kf_idx = extension_indices[0]

                    #최근 5개 키프레임과 매칭 추가
                    self.keyframe_matches(last_keys, frame, viewpoint, matches_info, tmp_id, kp_gaussian_mask, N_window=N_last_window+1)
                    #self.connect_gaussian_and_keyframes(last_keys, extension_indices, matches_info, kp_gaussian_mask, tmp_id, new_kf_idx, device.K_inv_gpu, N_window = N_last_window)
                    self.update_gaussian_observation(last_keys, extension_indices, matches_info, tmp_id, new_kf_idx,device.K_inv_gpu, N_window=N_last_window, kp_gaussian_mask=kp_gaussian_mask, Forward=True)

                    ##새로운 키프레임 포즈 최적화 테스트
                    """
                    prev_kf_key = last_keys[-2]
                    prev_kf_view = self.viewpoints[prev_kf_key]

                    res1, res2 = self.verify_scale_consistency(
                        torch.from_numpy(viewpoint.depth).cuda(),
                        torch.from_numpy(prev_kf_view.depth).cuda(),
                        frame.keypoints.round().int(),self.frames[prev_kf_key].keypoints.round().int(),
                        matches_info[tmp_id][prev_kf_key], kp_gaussian_mask, device.K,
                        viewpoint.R, viewpoint.T,
                        prev_kf_view.R, prev_kf_view.T, viewpoint.image_width, viewpoint.image_height, viewpoint, prev_kf_key)
                    print('depth test', torch.count_nonzero(~kp_gaussian_mask), tmp_id, prev_kf_key, res1, res2)
                    aaa = time.time()
                    kf_opt_params = []
                    kf_opt_params.append(
                        {
                            "params": [viewpoint.cam_rot_delta],
                            "lr": 5e-4,
                            "name": "rot_{}".format(viewpoint.uid),
                        }
                    )
                    kf_opt_params.append(
                        {
                            "params": [viewpoint.cam_trans_delta],
                            "lr": 1e-3,
                            "name": "trans_{}".format(viewpoint.uid),
                        }
                    )
                    kf_optimizers = torch.optim.Adam(kf_opt_params)
                    kf_scheduler = torch.optim.lr_scheduler.StepLR(
                        kf_optimizers, step_size=100, gamma=0.8
                    )
                    loss_kf = self.bundle_adjustment([tmp_id], bPoseUpdate=True) * self.weight_ba
                    loss_kf.backward()
                    with torch.no_grad():
                        kf_optimizers.step()
                        kf_scheduler.step()
                        #kf_optimizers.zero_grad(set_to_none=True)
                        update_pose(viewpoint)
                    bbb = time.time()
                    print('kf pose', bbb-aaa, loss_kf)
                    """
                    ##새로운 키프레임 포즈 최적화 테스트

                    Nold = self.gaussians._xyz.size()[0]

                    with torch.no_grad():
                        self.add_next_kf_with_ba(frame, viewpoint, depth_map, kp_gaussian_mask)

                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    gaussian_mask = gaussian_indices>=Nold
                    tmp_gaussian_indices = self.gaussians.observation_indices[gaussian_mask]

                    #새로운 가우시안 포인트를 이전 프레임에 전파
                    self.update_gaussian_observation(last_keys, extension_indices, matches_info, tmp_id, new_kf_idx, device.K_inv_gpu, N_window=N_last_window, Forward=False)
                    """
                    for idx in extension_indices[1:]:
                        kf_idx1 = last_keys[idx]
                        kf_idx2 = 0
                        diff = new_kf_idx - idx

                        if diff < N_last_window:
                            kf_idx2 = tmp_id
                            case = 1
                        elif idx % N_last_window == 0:
                            kf_idx2 = last_keys[idx + N_last_window]
                            case = 2
                        else:
                            kf_idx2 = last_keys[((idx // N_last_window) + 1) * N_last_window]
                            case = 3

                        keyframe1 = self.frames[kf_idx1]
                        keyframe2 = self.frames[kf_idx2]

                        matches = matches_info[kf_idx2][kf_idx1]
                        viewpoint1 = self.viewpoints[kf_idx1]
                        viewpoint2 = self.viewpoints[kf_idx2]
                        F12 = compute_F12(viewpoint2.R, viewpoint2.T, viewpoint1.R, viewpoint1.T, device.K_inv_gpu, device.K_inv_gpu)
                        res = self.check_dist_epipolar_line(keyframe2.keypoints[matches[:, 0]], keyframe1.keypoints[matches[:, 1]],F12)
                        matches = matches[res, :]

                        gidx, fidx = self.get_gaussian_match_indices(tmp_gaussian_indices, matches[:, 0], keyframe2.kf_id, n_start=Nold)

                        ##overlap mask
                        empty_mask = self.gaussians.observation_indices[gidx, keyframe1.kf_id] == -1
                        gidx = gidx[empty_mask]
                        fidx = fidx[empty_mask]

                        ##outlier check
                        res = self.check_outlier(self.gaussians.get_xyz[gidx], keyframe1.keypoints[matches[fidx,1]], viewpoint1, th = 9)
                        gidx = gidx[res]
                        fidx = fidx[res]

                        print('update test::backward', kf_idx1, kf_idx2, 'c=', case, torch.count_nonzero(res).item(),
                              torch.count_nonzero(empty_mask).item(),
                              'gau', torch.count_nonzero(
                                self.gaussians.observation_indices[gaussian_mask, keyframe2.kf_id] > -1).item(), matches.shape)

                        self.update_observation(gidx, fidx, matches[:, 1], keyframe1, keyframe1.kf_id)
                    """
                    #print('window test',self.current_window)
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    #frames_to_optimize = len(extension_window)
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                                len(self.current_window)
                                == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                    self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    #윈도우 내의 뷰포인트에 접근해서 최근 뷰포인트는 포즈까지 추가. 나머지는 exposure만 추가
                    for idx, kf_idx in enumerate(self.current_window): #self.current_window, extension_window
                        if kf_idx == self.first_kf_id:
                            continue
                        win_viewpoint = self.viewpoints[kf_idx]
                        if self.gs_pose and idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [win_viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                          * 0.5,
                                    "name": "rot_{}".format(win_viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [win_viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                              "cam_trans_delta"
                                          ]
                                          * 0.5,
                                    "name": "trans_{}".format(win_viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [win_viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(win_viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [win_viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(win_viewpoint.uid),
                            }
                        )

                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    ###TEST BA
                    """"""
                    t_ba_1 = time.time()
                    graph_opt_params = []
                    # 윈도우 내의 뷰포인트에 접근해서 최근 뷰포인트는 포즈까지 추가. 나머지는 exposure만 추가
                    for idx, kf_idx in enumerate(extension_window):  # self.current_window, extension_window
                        if kf_idx == self.first_kf_id:
                            continue
                        win_viewpoint = self.viewpoints[kf_idx]
                        graph_opt_params.append(
                            {
                                "params": [win_viewpoint.cam_rot_delta],
                                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                      * 0.5,
                                "name": "rot_{}".format(win_viewpoint.uid),
                            }
                        )
                        graph_opt_params.append(
                            {
                                "params": [win_viewpoint.cam_trans_delta],
                                "lr": self.config["Training"]["lr"][
                                          "cam_trans_delta"
                                      ]
                                      * 0.5,
                                "name": "trans_{}".format(win_viewpoint.uid),
                            }
                        )

                    self.graph_optimizers = torch.optim.Adam(graph_opt_params)
                    best_loss = float('inf')
                    wait = 0
                    for i in range(5):
                        loss_ba = self.bundle_adjustment(extension_window, bPoseUpdate=True)*self.weight_ba
                        loss_value = loss_ba.item()
                        if loss_value > best_loss:
                            wait += 1
                            if wait >= patience:
                                break
                        else:
                            best_loss = loss_value
                        loss_ba.backward()
                        with torch.no_grad():
                            self.gaussians.optimizer.step()
                            self.gaussians.optimizer.zero_grad(set_to_none=True)
                            #self.gaussians.update_learning_rate(self.iteration_count)
                            self.graph_optimizers.step()
                            self.graph_optimizers.zero_grad(set_to_none=True)
                            for kf_idx in extension_window:
                                aviewpoint = self.viewpoints[kf_idx]
                                keyframe = self.frames[kf_idx]
                                kf_id = keyframe.kf_id*2
                                idx = self.gaussians.observation_indices[:,keyframe.kf_id] > -1
                                keypoints = self.gaussians.observation_points[idx, kf_id:kf_id+2]

                                update_pose(aviewpoint)

                                inlier = self.check_outlier(self.gaussians.get_xyz[idx], keypoints, aviewpoint)
                                outlier = ~inlier
                                Noutlier = torch.count_nonzero(outlier)
                                self.gaussians.observation_indices[idx, keyframe.kf_id][outlier] = torch.full((Noutlier,), -1, device='cuda').int()
                                self.gaussians.observation_points[idx, kf_id:kf_id + 2][outlier] = torch.full((Noutlier, 2), -1, device='cuda').float()

                        t_ba_2 = time.time()
                        print('ba test', t_ba_2 - t_ba_1, loss_ba)
                    self.outlier_removal(Nold)
                    ###TEST BA

                    # visualization test
                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    test_gaussian_idx = self.gaussians.observation_indices[:, frame.kf_id] > -1
                    new_idx = gaussian_indices >= Nold
                    old_idx = gaussian_indices < Nold
                    kf_id1 = frame.kf_id * 2
                    for kf_idx in extension_window:  # temp_kf_window
                        # if kf_idx in kf_matches:
                        keyframe = self.frames[kf_idx]
                        viewpoint_kf = self.viewpoints[kf_idx]
                        kf_id2 = keyframe.kf_id * 2

                        tmp_idx = self.gaussians.observation_indices[:, keyframe.kf_id] > -1
                        # tmp_idx = test_gaussian_idx & (self.gaussians.observation_indices[:,keyframe.kf_id] > -1) & (gaussian_indices >= Nold)
                        # tmp_gaussians = self.gaussians.get_xyz[tmp_idx]
                        # tmp_keypoints1 = self.gaussians.observation_points[tmp_idx,kf_id2:kf_id2 + 2]

                        projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint_kf.R,
                                                                              viewpoint_kf.T,
                                                                              viewpoint_kf.fx, viewpoint_kf.fy,
                                                                              viewpoint_kf.cx, viewpoint_kf.cy,
                                                                              viewpoint_kf.image_width,
                                                                              viewpoint_kf.image_height)

                        image_np = (
                            viewpoint_kf.original_image
                                .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                .cpu()  # GPU → CPU
                                .numpy()  # NumPy 배열로 변환
                        )
                        image_np = (image_np * 255.0).astype(np.uint8)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                        tmp_new_idx = tmp_idx & new_idx & valid_projection
                        tmp_old_idx = tmp_idx & old_idx & valid_projection

                        projection_new = projection[tmp_new_idx].detach().cpu().numpy()
                        point_new = self.gaussians.observation_points[tmp_new_idx, kf_id2:kf_id2 + 2].cpu().numpy()

                        projection_old = projection[tmp_old_idx].detach().cpu().numpy()
                        point_old = self.gaussians.observation_points[tmp_old_idx, kf_id2:kf_id2 + 2].cpu().numpy()

                        # projection1 = projection[valid_projection]
                        # points1 = projection1.detach().cpu().numpy()
                        # points_1 = tmp_keypoints1[valid_projection].cpu().numpy()

                        for pt1, pt2 in zip(projection_new, point_new):
                            p1 = (int(round(pt1[0])), int(round(pt1[1])))
                            p2 = (int(round(pt2[0])), int(round(pt2[1])))

                            cv2.line(image_np, p1, p2, (0, 255, 0), 2, lineType=16)
                            cv2.circle(image_np, p1, 1, (0, 0, 255), -1, lineType=16)
                            cv2.circle(image_np, p2, 1, (255, 0, 0), -1, lineType=16)
                        for pt1, pt2 in zip(projection_old, point_old):
                            p1 = (int(round(pt1[0])), int(round(pt1[1])))
                            p2 = (int(round(pt2[0])), int(round(pt2[1])))

                            cv2.line(image_np, p1, p2, (0, 255, 255), 2, lineType=16)
                            cv2.circle(image_np, p1, 1, (255, 0, 255), -1, lineType=16)
                            cv2.circle(image_np, p2, 1, (255, 255, 0), -1, lineType=16)
                        cv2.imwrite('./res/test_ba2/ba2_' + tmp_id + '_' + str(kf_idx) + '.jpg', image_np)
                    # visualization test

                    m1 = time.time()
                    remove_ids = None
                    _, remove_ids1 = self.map_with_ba(self.current_window, iters=iter_per_kf, extension_graph=extension_window)  # graph = temp_kf_window,,   matches=kf_matches, graph=current_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map_with_ba(self.current_window, prune=True, extension_graph=extension_window)  # matches = kf_matches, graph = temp_kf_window,,  matches=kf_matches, graph=current_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    self.update_occ_visibility(self.current_window)

                    if remove_ids is not None:
                        self.push_to_frontend( "keyframe", prune=remove_ids, src = src)
                    else:
                        self.push_to_frontend( "keyframe", src = src)
                    e2 = time.time()
                    print('backend::end', tmp_id, e2 - s, torch.count_nonzero(kp_gaussian_mask), kp_gaussian_mask.shape[0], self.gaussians.get_xyz.shape[0], torch.unique(self.gaussians.unique_gaussian_ids).shape)

                    """
                    ##frame visualization
                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    for kf_idx in current_window:#temp_kf_window
                        #if kf_idx in kf_matches:
                            keyframe = self.frames[kf_idx]
                            viewpoint = self.viewpoints[kf_idx]

                            projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                                          viewpoint.fx, viewpoint.fy,
                                                                          viewpoint.cx, viewpoint.cy,
                                                                          viewpoint.image_width,
                                                                          viewpoint.image_height)
                           
                            image_np = (
                                viewpoint.original_image
                                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                    .cpu()  # GPU → CPU
                                    .numpy()  # NumPy 배열로 변환
                            )
                            image_np = (image_np * 255.0).astype(np.uint8)
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                            projection1 = projection[valid_projection & self.gaussians.isfeatured & (gaussian_indices >= Nold)]
                            points1 = projection1.detach().cpu().numpy()
                            points2 = keyframe.keypoints.detach().cpu().numpy()
                            projection2 = projection[valid_projection & self.gaussians.isfeatured & (gaussian_indices < Nold)]
                            points3 = projection2.detach().cpu().numpy()

                            for pt1 in points2:
                                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                                cv2.circle(image_np, p1, 3, (0, 0, 255), 1, lineType=16)
                            for pt1 in points1:
                                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                                cv2.circle(image_np, p1, 2, (0, 255, 0), 1, lineType=16)
                            for pt1 in points3:
                                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                                cv2.circle(image_np, p1, 2, (255, 0, 0), 1, lineType=16)
                            cv2.imwrite('./res/test_ba/mapping_' + tmp_id + '_' + str(kf_idx) + '.jpg', image_np)
                    """
                    """
                    ##VISUALIZATION TEST
                    indices = torch.where(self.gaussians.observation_indices[:,frame.kf_id] > -1)[0]
                    kf_indices = self.gaussians.observation_indices[indices, frame.kf_id]
                    viewpoint = self.viewpoints[tmp_id]
                    image_np = (
                        viewpoint.original_image
                            .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                            .cpu()  # GPU → CPU
                            .numpy()  # NumPy 배열로 변환
                    )
                    image_np = (image_np * 255.0).astype(np.uint8)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    points1 = frame.keypoints[kf_indices].cpu().numpy()
                    points2 = frame.keypoints.detach().cpu().numpy()
                    for pt1 in points2:
                        p1 = (int(round(pt1[0])), int(round(pt1[1])))
                        cv2.circle(image_np, p1, 3, (0, 0, 255), 1, lineType=16)
                    for pt1 in points1:
                        p1 = (int(round(pt1[0])), int(round(pt1[1])))
                        cv2.circle(image_np, p1, 2, (0, 255, 0), 1, lineType=16)

                    cv2.imwrite('./res/test_ba/' + tmp_id + '.jpg', image_np)
                    ##VISUALIZATION TEST
                    """

                    ##frame visualization
                    image_np_a = (
                        viewpoint.original_image
                            .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                            .cpu()  # GPU → CPU
                            .numpy()  # NumPy 배열로 변환
                    )
                    image_np_a = (image_np_a * 255.0).astype(np.uint8)
                    image_np_a = cv2.cvtColor(image_np_a, cv2.COLOR_RGB2BGR)

                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    test_gaussian_idx = self.gaussians.observation_indices[:, frame.kf_id] > -1
                    kf_id1 = frame.kf_id*2

                    for kf_idx in extension_window:  # temp_kf_window
                        # if kf_idx in kf_matches:
                        keyframe = self.frames[kf_idx]
                        viewpoint_kf = self.viewpoints[kf_idx]

                        tmp_idx = (self.gaussians.observation_indices[:,
                                   keyframe.kf_id] > -1)  # & (gaussian_indices < Nold)
                        tmp_gaussians = self.gaussians.get_xyz[test_gaussian_idx & tmp_idx]
                        tmp_keypoints_idx1 = self.gaussians.observation_indices[test_gaussian_idx & tmp_idx,frame.kf_id]
                        tmp_keypoints_idx2 = self.gaussians.observation_indices[test_gaussian_idx & tmp_idx,keyframe.kf_id]
                        kf_id2 = keyframe.kf_id * 2

                        tmp_keypoints1 = self.gaussians.observation_points[test_gaussian_idx & tmp_idx,kf_id1:kf_id1 + 2]
                        tmp_keypoints2 = self.gaussians.observation_points[test_gaussian_idx & tmp_idx,kf_id2:kf_id2 + 2]
                        #tmp_keypoints = keyframe.keypoints[tmp_keypoints_idx]

                        projection, _, valid_projection = project_pc_to_pixel(tmp_gaussians, viewpoint_kf.R,
                                                                              viewpoint_kf.T,
                                                                              viewpoint_kf.fx, viewpoint_kf.fy,
                                                                              viewpoint_kf.cx, viewpoint_kf.cy,
                                                                              viewpoint_kf.image_width,
                                                                              viewpoint_kf.image_height)

                        image_np = (
                            viewpoint_kf.original_image
                                .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                .cpu()  # GPU → CPU
                                .numpy()  # NumPy 배열로 변환
                        )
                        image_np = (image_np * 255.0).astype(np.uint8)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                        projection1 = projection[valid_projection]
                        points1 = projection1.detach().cpu().numpy()
                        points_1 = tmp_keypoints1[valid_projection].cpu().numpy()
                        points_2 = tmp_keypoints2[valid_projection].cpu().numpy()

                        for pt1, pt2 in zip(points1, points_2):
                            p1 = (int(round(pt1[0])), int(round(pt1[1])))
                            p2 = (int(round(pt2[0])), int(round(pt2[1])))

                            cv2.line(image_np, p1, p2, (0, 255, 0), 2, lineType=16)
                            cv2.circle(image_np, p1, 1, (0, 0, 255), -1, lineType=16)
                            cv2.circle(image_np, p2, 1, (255, 0, 0), -1, lineType=16)
                        cv2.imwrite('./res/test_ba/ba_' + tmp_id + '_' + str(kf_idx) + '.jpg', image_np)

                        """
                        image_np_b = (
                            viewpoint_kf.original_image
                                .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                .cpu()  # GPU → CPU
                                .numpy()  # NumPy 배열로 변환
                        )
                        image_np_b = (image_np_b * 255.0).astype(np.uint8)
                        image_np_b = cv2.cvtColor(image_np_b, cv2.COLOR_RGB2BGR)

                        #F12 = compute_F12(viewpoint.R, viewpoint.T, viewpoint_kf.R, viewpoint_kf.T, device.K_inv_gpu,device.K_inv_gpu)
                        #res = self.check_dist_epipolar_line(frame.keypoints[tmp_keypoints_idx1],
                        #                                    keyframe.keypoints[tmp_keypoints_idx2], F12)
                        matched = torch.stack((tmp_keypoints_idx1, tmp_keypoints_idx2), dim=1).int()

                        out_img = self.FeatureManager.tracker.visualize(image_np_a, image_np_b,
                                                                        matched.cpu().numpy(),
                                                                        frame.keypoints.cpu().numpy(),
                                                                        keyframe.keypoints.cpu().numpy(), mode=True)
                        cv2.imwrite('./res/test_matches2/' + tmp_id + '_' + str(kf_idx) + '.jpg', out_img)
                        """
                    profiler.disable()
                    self.bDoingMapping.store(True)
                    #profiler.print_stats(sort='tottime')
                else:
                    raise Exception("Unprocessed data", data)
                gc.collect()
                torch.cuda.empty_cache()
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def run(self):
        prune_dict = None

        #test
        new_gaussians = GaussianOrbModel(self.gaussians.max_sh_degree, self.gaussians.config)
        new_gaussians.init_lr(6.0)
        new_gaussians.training_setup(self.gaussians.opt_params)
        #test

        p = psutil.Process()

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                s = time.time()

                prune_mask = None
                _, prune_mask1 = self.map(self.current_window,)#matches = kf_matches, graph = recent_keys

                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2= self.map(self.current_window, prune=True, iters=10,)#matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2
                self.update_occ_visibility(self.current_window)

                if prune_mask is not None:
                    self.push_to_frontend(prune=prune_mask)
                else:
                    self.push_to_frontend()

                e = time.time()

            else:
                #print("backend::queue::get::start")
                data = self.backend_queue.get()
                #print("backend::queue::get::end")
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]

                    f = data[4]
                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id+=1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    #frame.gaussianpoints = torch.from_numpy(gaussianpoints)

                    move_camera_to_gpu(viewpoint)

                    Log("Resetting the system")
                    #print("backend init", frame.keypoints, frame.gaussianpoints)
                    self.reset()
                    self.frames[cur_frame_idx] = frame
                    self.first_kf_id = cur_frame_idx
                    self.viewpoints[cur_frame_idx] = viewpoint

                    self.preprocessing_add_kf()
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )

                    self.initialize_map(cur_frame_idx, viewpoint)

                    self.push_to_frontend("init", first_id=cur_frame_idx)

                elif data[0] == "keyframe":
                    s = time.time()
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    move_camera_to_gpu(viewpoint)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window

                    f = data[5]
                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    self.frames[(cur_frame_idx)] = frame

                    self.preprocessing_add_kf()
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]

                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                                len(self.current_window)
                                == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                    self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == self.first_kf_id:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                          * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                              "cam_trans_delta"
                                          ]
                                          * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    m1 = time.time()
                    remove_ids = None
                    _, remove_ids1 = self.map(self.current_window, iters=iter_per_kf,)#graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map(self.current_window, prune=True,)#matches = kf_matches, graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    self.update_occ_visibility(self.current_window)
                    if remove_ids is not None:
                        self.push_to_frontend("keyframe", prune=remove_ids)
                    else :
                        self.push_to_frontend("keyframe")
                    e2 = time.time()
                    print('backend::end', cur_frame_idx, e2-s, frames_to_optimize)
                    ##frame visualization

                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return