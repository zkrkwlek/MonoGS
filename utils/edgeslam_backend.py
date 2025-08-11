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
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss, get_patch_loss, get_loss_gaussian
from edge_assisted.gaussian_feature import project_pc_to_pixel, projection, calculate_bbox_mask, calculate_feature_mask, find_correspondence_with_dist, calculate_keypoint_mask, get_correspondences_within_threshold, match_ac_from_ab_bc
from utils.edgeframe_utils import EdgeFrame
from utils.pose_utils import SE3_exp, skew_sym_mat, compute_F12

from utils.datahandle_utils import move_camera_to_gpu, move_camera_to_cpu, move_gaussianmodel_to_cpu
from utils.datahandle_utils import move_occ_visibility_to_cpu
#from edge_assisted.gaussian_feature import GaussianPointManager
from edge_assisted.localmap_utils import get_local_gaussians
#from edge_assisted.object_manager import ObjectManager
from edge_assisted.object_loss import ObjectLoss
from edge_assisted.gaussian_orb_model import GaussianOrbModel

from collections import defaultdict

#alike + lightglue
import sys
sys.path.append('D:/UVR/LightGlue')
from lightglue import LightGlue, SuperPoint, ALIKED, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d

#XFeat
sys.path.append('D:/UVR/accelerated_features')
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue

class EdgeBackEnd(WinBackEnd):
    def __init__(self, config):
        super().__init__(config)
        self.first_kf_id = None
        self.pose_update = None
        #self.dataset = None
        self.FeatureManager = None
        self.frames={}
        self.keyframe_ids = {}
        self.weight_reprojection = 0.08
        self.weight_init_rgb = 0.9
        self.weight_init_depth = 0.02

        self.weight_rgb = 0.8
        self.weight_depth = 0.02
        self.weight_ba = 0.03
        self.weight_patch = 0.15
        self.next_kf_id = 0

        #object
        self.objects = None
        self.devices = None


    def push_to_frontend(self, tag=None, first_id = None, prune = None, local_window = None):

        self.last_sent = 0
        keyframes = []

        if local_window is None:
            local_window = self.current_window

        for kf_idx in local_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone().cpu(), kf.T.clone().cpu()))


        if tag is None:
            tag = "sync_backend"

        prune_data = None
        if prune is not None:
            prune_data = prune.cpu()
            #print("push_to_frontend::end", self.gaussians.get_xyz.shape)
        msg = [tag, move_gaussianmodel_to_cpu(self.gaussians), move_occ_visibility_to_cpu(self.occ_aware_visibility), (keyframes), (prune_data)]
        self.frontend_queue.put(msg)

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.frames = {}
        self.current_window = []
        self.initialized = not self.monocular
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

        #self.update_gaussian_observation_with_frame(frame)
        if downsample_factor is None:
            if init:
                downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
            else:
                downsample_factor = self.config["Dataset"]["pcd_downsample"]

        if mask is None:
            mask = torch.ones((viewpoint.image_height, viewpoint.image_width), device='cuda', dtype=torch.bool).cpu().numpy()

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,keypoints=keypoints, downsample_factor = downsample_factor, mask = mask
        )
        del mask
        return

    def add_next_kf_from_window(self, new_gaussians, window, ginit = True, init=False, scale=2.0):
        a = time.time()
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        """"""
        if ginit:
            new_gaussians = GaussianOrbModel(self.gaussians.max_sh_degree, self.gaussians.config)
            new_gaussians.init_lr(6.0)
            new_gaussians.training_setup(self.gaussians.opt_params)

        new_gaussians.observation_indices = torch.cat([new_gaussians.observation_indices,
                                                        torch.full((new_gaussians.observation_indices.shape[0], 1),
                                                                   -1, device='cuda')], dim=1)
        new_gaussians.observation_points = torch.cat([new_gaussians.observation_points,
                                                       torch.full((new_gaussians.observation_points.shape[0], 2), -1.0,
                                                                  device='cuda')], dim=1)

        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]

        for idx in window:
            viewpoint = self.viewpoints[idx]
            frame = self.frames[idx]

            gt_img = viewpoint.original_image.cuda()
            valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

            depth_map = torch.from_numpy(viewpoint.depth).unsqueeze(0)
            depth_map[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
            depth_map = depth_map[0].numpy()
            new_gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=idx, init=init, scale=scale, depthmap=depth_map, frame=frame, downsample_factor = downsample_factor
            )
            print(idx, new_gaussians.get_xyz.shape)
        b = time.time()
        print('뎁스화 = Add', b-a, new_gaussians.get_xyz.shape)
        return new_gaussians

    def get_gaussian_match_indices(self, gaussian_observation_indices, matches, ref_id, n_start = 0):
        #matches = match[:, idx], match[a, b] 일 때 0이면 a = ref, 1이면 b = ref
        obs_idx_col = gaussian_observation_indices[:, ref_id]  # shape: (N,)
        match_values = matches # shape: (K,)
        eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)

        print('a', obs_idx_col.dtype, obs_idx_col.shape)

        key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                    torch.full_like(obs_idx_col, -1, dtype = obs_idx_col.dtype))[0].int()
        included_mask = key_positions != -1

        frame_gaussian_index = torch.where(included_mask)[0].int()  # 가우시안의 인덱스
        if n_start > 0:
            frame_gaussian_index += n_start
        frame_match_index = key_positions[included_mask]  # 매치애서 인덱스
        print('b',frame_match_index.dtype, frame_gaussian_index.dtype, key_positions.dtype)
        return frame_gaussian_index, frame_match_index

    def update_observation(self, frame_gaussian_index, frame_match_index, matches, frame, target_id):
        self.gaussians.observation_indices[frame_gaussian_index, target_id] = matches[frame_match_index]
        self.gaussians.observation_points[frame_gaussian_index, target_id * 2:target_id * 2 + 2] = \
            frame.keypoints[matches[frame_match_index]]

    def connect_observation(self, matches, frame, ref_id, target_id, match_row_idx):
        obs_idx_col = self.gaussians.observation_indices[:, ref_id]  # shape: (N,)
        match_values = matches[:, match_row_idx]  # shape: (K,)
        # eq[n, k] == True면 obs_idx_col[n] == match_values[k]
        # 위치: 포함되면 첫 번째 True의 인덱스, 없으면 -1
        # 인클루드 마스크는 가우시안에서 매칭(0)의 키포인트 위치. 즉 매치(0)과 같음., 가우시안 위치를 표현함.
        # 키포지션은 그게 매치 안에서 어디있는지를 알 수 있음.
        eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
        key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                    torch.full_like(obs_idx_col, -1, dtype = obs_idx_col.dtype))
        included_mask = key_positions != -1

        frame_gaussian_index = torch.where(included_mask)[0]
        frame_match_index = key_positions[included_mask]

        #print('asdfasdf', frame_gaussian_index.shape, matches.shape, torch.count_nonzero(obs_idx_col))
        #print(obs_idx_col[obs_idx_col > -1], match_values)

        self.gaussians.observation_indices[frame_gaussian_index, target_id] = matches[frame_match_index, match_row_idx]
        self.gaussians.observation_points[frame_gaussian_index, target_id * 2:target_id * 2 + 2] = \
            frame.keypoints[matches[frame_match_index, match_row_idx]]

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

    def update_gaussian_observation_with_frame(self, frame):
        pass
        """
        # keyframe index
        id = frame.kf_id
        indices = torch.where(frame.gaussianpoints > -1)[0]
        #if indices.shape[0] > 0:
        #print(indices, frame.gaussianpoints[indices], self.gaussians.observation_indices.dtype, indices.dtype)
        self.gaussians.observation_indices[frame.gaussianpoints[indices], id:id + 1] = indices.unsqueeze(1)
        self.gaussians.observation_points[frame.gaussianpoints[indices], id*2:id*2+2] = frame.keypoints[indices]
        self.gaussians.isfeatured[frame.gaussianpoints[indices]] = True
        #print(id, self.gaussians.observation_indices.shape)
        """
        """
        for kp_idx in indices:
            g_idx = frame.gaussianpoints[kp_idx]
            self.gaussians.isfeatured[g_idx] = True
            if self.gaussians.observations[g_idx] is None:
                self.gaussians.observations[g_idx] = {}
            self.gaussians.observations[g_idx][frame.id] = kp_idx.item()
        """

    def update_gaussian_observation(self, prune_obs):
        keys = list(self.frames)
        """
        for idx, obs in prune_obs.items():
            if obs is not None:
                fids = torch.where(obs > -1)[0].cpu().numpy()
                #print(fids.dtype, fids, obs, keys)
                #print(keys[int(fids)],obs[int(fids)])
                for fid, kp in zip(keys[fids], obs[fids]):
                    #print(fid, kp)
                    self.frames[fid].gaussianpoints[kp.itme()] = -1
                #print(idx, obs, self.gaussians.observation_indices.shape)
        """

    def update_gaussian_observation_after_prune(self):
        #print('update_gaussian_observation_after_prune',self.gaussians._xyz.size(), self.gaussians.isfeatured.size(), self.gaussians.observations.shape,torch.count_nonzero(self.gaussians.isfeatured), np.count_nonzero(self.gaussians.observations), np.sum(self.gaussians.observations!=None))
        pass
        """
        keys = list(self.frames)
        for fid in keys:
            frame = self.frames[fid]
            kid = frame.kf_id
            gids = torch.where(self.gaussians.observation_indices[:,kid] > -1)[0]
            kpids = self.gaussians.observation_indices[gids, kid]
            frame.gaussianpoints = torch.full((frame.keypoints.shape[0],),-1, device='cuda')
            frame.gaussianpoints[kpids] = gids
        """


        """
        feature_indices = self.gaussians.isfeatured.clone().cpu().numpy()
        feature_indices = np.where(feature_indices)[0]
        gaussian_obs = list(zip(feature_indices, self.gaussians.observation_indices[feature_indices]))
        
        for gaussian_index, obs in gaussian_obs:
            if obs is None:
                print('obs error', gaussian_index, obs)
                self.gaussians.isfeatured[gaussian_index] = False
                continue
            for fid, kpidx in obs.items():
                #print("update_gaussian frame", fid)
                frame = self.frames[(fid)]
                frame.gaussianpoints[kpidx] = gaussian_index
        """
    def update_graph_weights(self):
        t1 =time.time()
        n = self.gaussians.observation_indices.shape[1]
        res = None
        intersection = None
        if n > 1:
            tmp = (self.gaussians.observation_indices > -1)#.float()
            #res = pairwise_cosine_similarity(tmp.T)

            and_matrix = (tmp.T.unsqueeze(1) & tmp.T.unsqueeze(0))
            intersection = and_matrix.sum(dim=2)
            intersection.fill_diagonal_(0)[1]
            intersection[intersection < 20] = 0

        t2 = time.time()
        #print('update graph weight', n, t2-t1, intersection.shape, intersection)
        return intersection

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

    def update_graph(self, current_window, th = 9.0):
        gaussian_indices = torch.where(self.gaussians.isfeatured)[0].cuda()
        gaussians_xyz = self.gaussians.get_xyz[gaussian_indices]
        obs_points = self.gaussians.observation_points[gaussian_indices]
        obs_indices = self.gaussians.observation_indices[gaussian_indices]

        for fid in current_window:
            viewpoint = self.viewpoints[fid]
            frame = self.frames[fid]
            kid = frame.kf_id*2

            #update pose
            update_pose(viewpoint)

            idx = obs_indices[:, frame.kf_id] > -1
            gaussians = gaussians_xyz[idx]
            projection, _, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                    viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            projection = projection[valid]
            points = obs_points[idx, kid:kid + 2][valid]

            l2 = torch.sum((projection - points) ** 2, dim=1)
            outlier = l2 > th

            outlier_idx = gaussian_indices[idx][valid][outlier]
            Noutlier = outlier_idx.shape[0]
            self.gaussians.observation_indices[outlier_idx, frame.kf_id] = torch.full((Noutlier,  ), -1, device='cuda').type(torch.int32)
            self.gaussians.observation_points[outlier_idx,  kid:kid + 2] = torch.full((Noutlier, 2), -1.0, device='cuda')

        #outlier 관리
        obs_indices2 = self.gaussians.observation_indices[gaussian_indices]+1
        row_sum = obs_indices2.sum(dim=1)
        mask = row_sum <= 0

        if torch.count_nonzero(mask) > 0:
            outlier_indices = gaussian_indices[mask]
            n = torch.count_nonzero(mask)
            self.gaussians.isfeatured[outlier_indices] = False#torch.zeros(n, device='cuda').bool()
            #print('outlier test', outlier_indices.shape, torch.count_nonzero(mask), mask.shape, gaussian_indices.shape)
            """
            print('outlier removal', n,
                  torch.count_nonzero(self.gaussians.isfeatured[gaussian_indices][mask]),
                  torch.count_nonzero(self.gaussians.observation_indices[gaussian_indices][mask] > -1),
                  torch.count_nonzero(self.gaussians.observation_points[gaussian_indices][mask] > -1))
            """

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

    def bundle_adjustment2(self, cur_kf_idx, current_window, th_feature_radius = 7):
        t1 = time.time()
        with torch.no_grad():
            cur_kf_view = self.viewpoints[cur_kf_idx]
            cur_frame = self.frames[cur_kf_idx]
            #현재 프레임에서 가우시안 선택
            projections, depths, valid_proj = project_pc_to_pixel(self.gaussians.get_xyz, cur_kf_view.R, cur_kf_view.T,
                                                                  cur_kf_view.fx, cur_kf_view.fy, cur_kf_view.cx,
                                                                  cur_kf_view.cy,
                                                                  cur_kf_view.image_width, cur_kf_view.image_height)

            # projections = (projections[valid])  # 유효한 프로젝션 결과를 int화 해서 픽셀로 만듬. 정렬하면
            cur_kf_keypoints = (cur_frame.keypoints)
            cur_gaussian_match_idx = find_correspondence_with_dist(projections, cur_kf_keypoints, th=7)  # 약간 시간이 걸림. 0.01 이하

            valid_match = (cur_gaussian_match_idx > -1)  # & self.gaussians.isfeatured[valid]
            valid_gaussian_idx = torch.where(valid_match & valid_proj)[0]
            cur_kf_gaussians = self.gaussians.clone(valid_gaussian_idx)  # self.gaussians.get_xyz[valid][valid_match]

            ###save image
            render_pkg = render(
                cur_kf_view, cur_kf_gaussians, self.pipeline_params, self.background
            )

            image = render_pkg["render"]
            image_np = (
                image
                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    .clone().detach().cpu()  # GPU → CPU
                    .numpy()  # NumPy 배열로 변환
            )

            image_np = (image_np * 255.0).astype(np.uint8)
            out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            points2 = cur_frame.keypoints.detach().cpu().numpy()

            for pt1 in points2:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (255, 255, 0), -1, lineType=16)
            #cv2.imwrite('./res/test/' + str(cur_kf_idx) + '_' + str(cur_kf_idx) + '.jpg', out)

        loss_ba = 0
        for kf_id in current_window[1:]:
            #if kf_id == cur_kf_idx:
            #    continue
            kf_view = self.viewpoints[kf_id]
            kf_frame = self.frames[kf_id]
            kf_feature_mask = calculate_feature_mask(kf_frame.keypoints, kf_view.image_width, kf_view.image_height, max_radius=5)

            render_pkg = render(
                kf_view, cur_kf_gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            loss_rgb,loss_depth = get_loss_mapping(self.config, image, depth, kf_view, opacity, feature_mask=kf_feature_mask)
            loss_kf = (loss_rgb * self.weight_rgb + loss_depth * self.weight_depth)
            loss_ba+=loss_kf

            image_np = (
                image
                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    .clone().detach().cpu()  # GPU → CPU
                    .numpy()  # NumPy 배열로 변환
            )

            image_np = (image_np * 255.0).astype(np.uint8)
            out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            points2 = kf_frame.keypoints.detach().cpu().numpy()
            for pt1 in points2:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 255, 255), -1, lineType=16)
            cv2.imwrite('./res/test/'+str(cur_kf_idx)+'_'+str(kf_id) + '.jpg', out)
        t2 = time.time()
        print('ba', loss_ba, t2-t1)
        return loss_ba

    def bundle_adjustment3(self, current_window, bPoseUpdate=False, th_obs=2):
        t1 = time.time()
        gaussian_indices = torch.where(self.gaussians.isfeatured)[0].cuda()
        gaussians_xyz = self.gaussians.get_xyz[gaussian_indices]
        obs_points = self.gaussians.observation_points[gaussian_indices]
        obs_indices = self.gaussians.observation_indices[gaussian_indices]




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
                T_w2c = torch.eye(4, device=viewpoint.R.device)
                T_w2c[0:3, 0:3] = viewpoint.R
                T_w2c[0:3, 3] = viewpoint.T
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
            #print('aa', fid, frame.kf_id, obs_indices, torch.count_nonzero(obs_indices), torch.count_nonzero(obs_points), torch.count_nonzero(idx))
            #print(projection, points, torch.count_nonzero(gaussian_indices), torch.count_nonzero(gaussians), )
            loss_ba += get_reprojection_loss(projection, points).mean()
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

    def initialize_map_with_mask(self, cur_frame_idx, viewpoint):
        curr_frame = self.frames[(cur_frame_idx)]

        kf_feature_mask = calculate_feature_mask(curr_frame.keypoints, viewpoint.image_width, viewpoint.image_height,
                                                 max_radius=5)
        kf_feature_mask = torch.logical_or(kf_feature_mask, curr_frame.contours_mask)

        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1

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
                self.config, image, depth, viewpoint, opacity, initialization=True, feature_mask=kf_feature_mask
            )
            loss_init = loss_rgb * self.weight_init_rgb + loss_depth * self.weight_init_depth
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if mapping_iteration % self.init_gaussian_update == 0:

                    remove_ids = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                        self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            print('initialize_map', self.iteration_count, self.init_itr_num, self.gaussians.get_xyz.shape)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()

        Log("Initialized map")

        return render_pkg

    def initialize_map_with_ba(self, cur_frame_idx, viewpoint):
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

    def dense_map(self, gaussians, current_window, iters=1):
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        frames_to_optimize = len(current_window)

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        remove_ids = None

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
                    viewpoint, gaussians, self.pipeline_params, self.background
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
                    self.config, image, depth, viewpoint, opacity
                )
                loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
                loss_mapping += loss_kf

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, gaussians, self.pipeline_params, self.background
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
                    self.config, image, depth, viewpoint, opacity
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            ##isotropic_scaling
            scaling = gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                """
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                """
                """
                remove_ids = gaussians.densify_and_prune(
                    self.opt_params.densify_grad_threshold,
                    self.init_gaussian_th,
                    self.init_gaussian_extent,
                    None,
                )
                """
                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(self.iteration_count)
        return gaussian_split, remove_ids

    def update_feature_gaussians(self, current_window, min_kf_window = 4, min_kp_obs = 2):
        a = time.time()
        with torch.no_grad():
            total_visibile_filter = torch.zeros((self.gaussians.get_xyz.shape[0],len(current_window)), dtype = torch.bool, device = 'cuda')
            total_feature_filter = torch.zeros((self.gaussians.get_xyz.shape[0],len(current_window)), dtype = torch.bool, device = 'cuda')
            total_error = torch.zeros((self.gaussians.get_xyz.shape[0], len(current_window)), dtype=torch.float,
                                               device='cuda')
            for idx, cam_idx in enumerate(current_window):
                viewpoint = self.viewpoints[cam_idx]
                keyframe = self.frames[cam_idx]
                ##feature match
                projections, depths, valid_proj = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                                      viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                      viewpoint.cy,
                                                                      viewpoint.image_width, viewpoint.image_height)
                match_idx = find_correspondence_with_dist(projections, keyframe.keypoints, th=7)  # 약간 시간이 걸림. 0.01 이하
                feature_filter = valid_proj & (match_idx > -1)
                ##visible check
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (   image,
                    visibility_filter,
                ) = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                )

                ##error
                err = get_loss_gaussian(self.config, image, viewpoint, projections).squeeze(1)

                total_feature_filter[:,idx] = feature_filter
                total_visibile_filter[:,idx] = visibility_filter
                total_error[:,idx] = err
                #print('update = gaussian test', cam_idx, idx,'f', torch.count_nonzero(feature_filter), 'v', torch.count_nonzero(visibility_filter), total_visibile_filter.shape)
        b = time.time()
        sum_visible = torch.sum(total_visibile_filter, dim = 1)

        #res = torch.sum(total_feature_filter & total_visibile_filter, dim = 1) / sum_visible
        #res = res < 0.1
        if len(current_window) > min_kf_window:
            res = total_feature_filter.sum(dim=1) < min_kp_obs

            if False and self.gaussians.get_xyz.shape[0] > 20000:
                sum_err_count = (total_error < 1000).sum(dim=1, keepdim=True)
                sum_err = (total_error * sum_err_count.float()).sum(dim=1, keepdim=True)
                large_value = 1000.0
                mean_err = torch.where(
                    sum_err_count > 0,
                    sum_err / sum_err_count,
                    torch.full_like(sum_err_count, large_value)
                )
                #print('color err test = ', mean_err[mean_err < 500], sum_err.shape, sum_err_count.shape)

                tmp_val = mean_err < 1000
                print('mean err', mean_err[tmp_val].mean(), torch.max(mean_err[tmp_val]), torch.min(mean_err[tmp_val]))

                res_color = (mean_err > 3.0).squeeze(1)
                res = torch.logical_or(res, res_color)

            self.gaussians.prune_points(res) #~mask : valid, mask : prune
            #print('update = gaussian test=end', b-a, torch.count_nonzero(res))

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

    def select_keyframes(self, keyframes, max_count=8, must_include_last=2, start_offset=0):
        N = len(keyframes)

        # 키프레임 총개수가 max_count 이하면, 복사본 반환
        if N <= max_count:
            return keyframes.copy()

        # 반드시 포함할 최근 키프레임
        must_include = keyframes[-must_include_last:]

        # 나머지에서 뽑을 개수
        remaining_count = max_count - must_include_last
        remaining_range = keyframes[:-must_include_last]  # 최근 제외

        length = len(remaining_range)
        interval = length / remaining_count

        # 시작 offset을 적용해서 인덱스를 계산하고, 범위 내 순환 적용(mod)
        corrected_offset = start_offset % interval

        # 인덱스 계산: i 별 offset 값을 더한 후 interval만큼 곱해서 인덱스 산출,
        # 인덱스가 초과 시 length 내 순환하도록 % length 처리
        indices = ((np.arange(remaining_count) * interval) + corrected_offset).astype(int) % length

        sampled = [remaining_range[idx] for idx in indices]

        # 반드시 포함 키프레임과 결합
        selected = sampled + must_include

        return selected
    """
    def select_keyframes(self, keyframes, max_count=8, must_include_last=2):
        N = len(keyframes)

        # 총 키프레임 적으면 모두 반환
        if N <= max_count:
            return keyframes  # 원본 보존 위해 복사본 반환

        must_include = keyframes[-must_include_last:]  # 최근 must_include_last개 키프레임

        remaining_count = max_count - must_include_last
        remaining_range = keyframes[:-must_include_last]  # 최근 2개 제외 나머지

        interval = len(remaining_range) / remaining_count

        sampled = []
        for i in range(remaining_count):
            idx = int(i * interval)
            sampled.append(remaining_range[idx])

        selected = sampled + must_include
        return selected
    """
    def map_with_ba(self, current_window, prune=False, iters=1, matches=None, extension_graph=None):
        feature_radius = 9
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        #frames_to_optimize = len(current_window)

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

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

                kf_frame = self.frames[viewpoint.uid]
                loss_rgb, loss_depth = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, #feature_mask=kf_frame.mapping_mask
                )
                loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
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

                kf_frame = self.frames[viewpoint.uid]
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
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
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

    def map_with_mask(self, current_window, prune=False, iters=1, matches=None, graph=None):
        feature_radius = 9
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        #frames_to_optimize = len(current_window)

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        remove_ids = None
        last_kf_id = current_window[0]

        curr_patches = None
        curr_patches_valid = None
        curr_rendered_image = None
        kf_patches = defaultdict(lambda: {'patch': None, 'valid': None})

        ##patch consistency
        doPatchConsistency = False
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

        ##patch consistency

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

                kf_frame = self.frames[viewpoint.uid]
                """
                kf_feature_mask = calculate_feature_mask(kf_frame.keypoints, viewpoint.image_width,
                                                         viewpoint.image_height,
                                                         max_radius=feature_radius)
                kf_feature_mask = torch.logical_or(kf_feature_mask, kf_frame.contours_mask)
                """
                loss_rgb, loss_depth = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, feature_mask=kf_frame.mapping_mask
                )
                loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
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

                kf_frame = self.frames[viewpoint.uid]
                """
                kf_feature_mask = calculate_feature_mask(kf_frame.keypoints, viewpoint.image_width,
                                                         viewpoint.image_height,
                                                         max_radius=feature_radius)
                kf_feature_mask = torch.logical_or(kf_feature_mask, kf_frame.contours_mask)
                """
                loss_rgb, loss_depth = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, feature_mask=kf_frame.mapping_mask
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            ##isotropic_scaling
            """
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            """

            ##geometric consistency
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

            ##patch consistency
            if doPatchConsistency:

                loss_patch = 0
                t_patch1 = time.time()

                viewpoint = self.viewpoints[last_kf_id]
                keyframe = self.frames[last_kf_id]

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                ) = (
                    render_pkg["render"],
                )
                curr_patches, curr_patches_valid = keyframe.extract_patches_differentiable(image, keyframe.keypoints)

                for cam_idx in graph:
                    if cam_idx == last_kf_id or cam_idx not in matches:
                        continue
                    match = matches[cam_idx]
                    kf_val = kf_patches[cam_idx]['valid']
                    curr_val = curr_patches_valid[match[:, 0]]
                    valid = torch.logical_and(curr_val, kf_val)

                    cur_patch = curr_patches[match[valid, 0]]
                    kf_patch = kf_patches[cam_idx]['patch'][valid]

                    err = get_patch_loss(cur_patch, kf_patch)
                    loss_patch += err.mean()
                    # print('patch', last_kf_id, cam_idx, torch.count_nonzero(valid), valid.shape, cur_patch.shape)
                loss_mapping += loss_patch * self.weight_patch

                t_patch2 = time.time()
                # print('patch loss = ', loss_patch.mean(), t_patch2-t_patch1)
                if False and self.gaussians._xyz.grad is not None:
                    after_grad_loss_patch = self.gaussians._xyz.grad.clone()
                    affected_by_patch = torch.any(after_grad_loss_patch != 0, dim=1)
                    print("Gaussians affected by patch:", torch.count_nonzero(affected_by_patch),
                          self.gaussians._xyz.shape, affected_by_patch.nonzero().flatten())
            ##patch consistency

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
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
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

    def run2(self):
        max_radius = 5
        local_kf_window = []
        kf_offset = 0

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(local_kf_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                s = time.time()

                #local_kf_window = self.select_keyframes(keyframe_keys, start_offset=kf_offset)
                kf_offset+=1

                prune_mask = None
                _, prune_mask1 = self.map_with_ba(local_kf_window, matches=None,
                                          graph=None)  # matches = kf_matches, graph = recent_keys
                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2 = self.map_with_ba(local_kf_window, prune=True, iters=10, matches=None,
                                              graph=None)  # matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2

                scales_avg = torch.mean(self.gaussians.get_scaling, axis=1, keepdims=True)
                scale_mask1 = scales_avg <= 1.0
                scale_mask2 = scales_avg <= 0.1
                scale_mask3 = scales_avg <= 0.01
                scale_mask4 = scales_avg <= 0.005
                opacity_mask = self.gaussians.get_opacity > 0.9
                obs_mask = self.gaussians.n_obs < 3
                print('obs', torch.count_nonzero(opacity_mask), torch.count_nonzero(obs_mask),'scale test', torch.count_nonzero(scale_mask1),torch.count_nonzero(scale_mask2),
                      torch.count_nonzero(scale_mask3),torch.count_nonzero(scale_mask4))
                print('total', torch.count_nonzero(scale_mask4 & opacity_mask))
                self.gaussians.prune_points(scale_mask4.squeeze(1))

                #self.update_feature_gaussians(local_kf_window)
                self.update_occ_visibility(self.current_window)

                if prune_mask is not None:
                    self.push_to_frontend(prune=prune_mask, local_window=local_kf_window)
                else:
                    self.push_to_frontend(local_window=local_kf_window)
                e = time.time()
                # print("backend = mapping with empty queue", (e-s))

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
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]

                    f = data[4]
                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, frame.contours = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx

                    ##object contour mask
                    mask_bool = np.zeros((viewpoint.image_height, viewpoint.image_width), dtype=np.uint8)
                    cv2.drawContours(mask_bool, frame.contours, -1, color=255, thickness=cv2.FILLED)
                    frame.contours_mask = torch.from_numpy(mask_bool).cuda().bool()
                    ##object contour mask

                    ##feature mask
                    frame.feature_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width,
                                                                viewpoint.image_height, max_radius=max_radius)
                    ##feature mask

                    ##mask 처리
                    frame.mapping_mask = torch.logical_or(frame.feature_mask, frame.contours_mask)

                    move_camera_to_gpu(viewpoint)

                    Log("Resetting the system")
                    # print("backend init", frame.keypoints, frame.gaussianpoints)
                    self.reset()
                    self.frames[cur_frame_idx] = frame
                    self.first_kf_id = cur_frame_idx
                    self.viewpoints[cur_frame_idx] = viewpoint

                    """
                    self.gaussians.observation_indices = torch.cat([self.gaussians.observation_indices,
                                                                    torch.full((
                                                                               self.gaussians.observation_indices.shape[
                                                                                   0], 1),
                                                                               -1, device='cuda')], dim=1)
                    self.gaussians.observation_points = torch.cat([self.gaussians.observation_points,
                                                                   torch.full(
                                                                       (self.gaussians.observation_points.shape[0], 2),
                                                                       -1.0,
                                                                       device='cuda')], dim=1)
                    """
                    self.preprocessing_add_kf()
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )

                    self.initialize_map_with_mask(cur_frame_idx, viewpoint)

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
                    frame.keypoints, frame.descriptors, frame.objects, frame.contours = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    self.frames[(cur_frame_idx)] = frame

                    ##add contour mask
                    mask_bool = np.zeros((viewpoint.image_height, viewpoint.image_width), dtype=np.uint8)
                    cv2.drawContours(mask_bool, frame.contours, -1, color=255, thickness=cv2.FILLED)
                    frame.contours_mask = torch.from_numpy(mask_bool).cuda().bool()
                    ##add contour mask

                    ##feature mask
                    frame.feature_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width, viewpoint.image_height, max_radius=max_radius)
                    ##feature mask

                    ##mask 처리
                    frame.mapping_mask = torch.logical_or(frame.feature_mask, frame.contours_mask)
                    self.preprocessing_add_kf()
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, mask = frame.mapping_mask.squeeze(0).cpu().numpy())

                    new_kf_id = frame.kf_id
                    #local_kf_window = [cur_frame_idx]
                    keyframe_keys = list(self.frames)

                    local_kf_window = keyframe_keys#self.select_keyframes(keyframe_keys, start_offset=kf_offset)
                    kf_offset += 1

                    """
                    for kf_idx in last_keys:
                        if kf_idx == cur_frame_idx:
                            continue
                        keyframe = self.frames[kf_idx]
                        matches = self.FeatureManager.tracker.match(frame.descriptors, keyframe.descriptors)
                        matches = torch.from_numpy(matches).type(torch.int32).cuda()
                        #print('backend=match', cur_frame_idx, kf_idx, matches.shape)
                        if matches.shape[0] > 20:
                            local_kf_window.append(kf_idx)
                    """

                    opt_params = []
                    frames_to_optimize = len(local_kf_window)
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                                len(local_kf_window)
                                == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                    self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(local_kf_window)):
                        if local_kf_window[cam_idx] == self.first_kf_id:
                            continue
                        viewpoint = self.viewpoints[local_kf_window[cam_idx]]
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
                    _, remove_ids1 = self.map_with_ba(local_kf_window, iters=iter_per_kf, matches=None,
                                              graph=None)  # graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map_with_ba(local_kf_window, prune=True, matches=None,
                                              graph=None)  # matches = kf_matches, graph = temp_kf_window

                    #self.update_feature_gaussians(local_kf_window)
                    self.update_occ_visibility(self.current_window)

                    if remove_ids1 is not None:
                        remove_ids = remove_ids1

                    if remove_ids is not None:
                        self.push_to_frontend("keyframe", prune=remove_ids, local_window=local_kf_window)
                    else:
                        self.push_to_frontend("keyframe", local_window=local_kf_window)
                    e2 = time.time()
                    print('backend::end', cur_frame_idx, e2 - s, frames_to_optimize)
                    ##frame visualization

                    for kf_idx in self.current_window:

                        keyframe = self.frames[kf_idx]
                        viewpoint = self.viewpoints[kf_idx]

                        kf_id = keyframe.kf_id
                        tmp_idx = torch.where(self.gaussians.observation_indices[:, kf_id] > -1)[0]
                        tmp_gaussians = self.gaussians.get_xyz[tmp_idx]

                        projection, _, valid_projection = project_pc_to_pixel(tmp_gaussians, viewpoint.R,
                                                                              viewpoint.T,
                                                                              viewpoint.fx, viewpoint.fy,
                                                                              viewpoint.cx, viewpoint.cy,
                                                                              viewpoint.image_width,
                                                                              viewpoint.image_height)
                        projection = projection[valid_projection]
                        tmp_idx = tmp_idx[valid_projection]
                        points = self.gaussians.observation_points[tmp_idx, 2 * kf_id:2 * kf_id + 2]

                        image_np = (
                            viewpoint.original_image
                                .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                .cpu()  # GPU → CPU
                                .numpy()  # NumPy 배열로 변환
                        )
                        image_np = (image_np * 255.0).astype(np.uint8)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        self.FeatureManager.tracker.visualize2(image_np, projection.clone(), points.clone(), delay=1
                                                               , save=True,
                                                               filename='./res/map/mapping_' + str(
                                                                   cur_frame_idx) + '_' + str(kf_idx) + '.jpg')
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def run_with_ba(self):
        prune_dict = None

        # test
        new_gaussians = GaussianOrbModel(self.gaussians.max_sh_degree, self.gaussians.config)
        new_gaussians.init_lr(6.0)
        new_gaussians.training_setup(self.gaussians.opt_params)
        # test

        #lightglue
        light_glue_matcher = LightGlue(features="aliked").eval().cuda()
        extractor = ALIKED(max_num_keypoints=2048).eval().cuda()

        #xfeat
        top_k = 4096
        xfeat = XFeat(
            weights='../accelerated_features/weights/xfeat.pt', #-lighterglue
            top_k=top_k,
            detection_threshold=0.05
        ).eval().cuda()
        xfeat.lighterglue = LighterGlue(weights='../accelerated_features/weights/xfeat-lighterglue.pt').eval().cuda()
        matches_info = {}
        extension_window = None

        K = torch.tensor([[self.dataset.fx, 0, self.dataset.cx],
                      [0, self.dataset.fy, self.dataset.cy],
                      [0, 0, 1]], dtype=torch.float32, device = 'cuda')

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
                _, prune_mask1 = self.map_with_ba(self.current_window, extension_graph=extension_window)#matches=kf_matches, graph=current_window)  # matches = kf_matches, graph = recent_keys
                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2 = self.map_with_ba(self.current_window, prune=True, iters=10, extension_graph=extension_window)#matches=kf_matches, graph=current_window)  # matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2

                if prune_mask is not None:
                    self.push_to_frontend(prune=prune_mask)
                else:
                    self.push_to_frontend()
                e = time.time()
                #print("backend = mapping with empty queue", (e-s), self.gaussians.get_xyz.shape[0])
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
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]

                    f = data[4]
                    frame = EdgeFrame(cur_frame_idx, None, None, None)
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors, frame.objects, _ = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    # frame.gaussianpoints = torch.from_numpy(gaussianpoints)

                    move_camera_to_gpu(viewpoint)

                    Log("Resetting the system")
                    # print("backend init", frame.keypoints, frame.gaussianpoints)
                    self.reset()
                    self.frames[cur_frame_idx] = frame
                    self.first_kf_id = cur_frame_idx
                    self.viewpoints[cur_frame_idx] = viewpoint

                    #mask
                    kp_region_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width,
                                                            viewpoint.image_height, max_radius=2)
                    kp_mask = calculate_keypoint_mask(frame.keypoints, viewpoint.image_width,
                                                            viewpoint.image_height,)
                    self.preprocessing_add_kf()
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True, mask = (~kp_region_mask).squeeze(0).cpu().numpy()
                    )
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True,keypoints=frame.keypoints,
                        mask=kp_mask.squeeze(0).cpu().numpy(), downsample_factor=1.0
                    )
                    matches_info[cur_frame_idx] = {}
                    ##test
                    print("init_ba", torch.count_nonzero(self.gaussians.isfeatured), torch.count_nonzero(self.gaussians.observation_points>-1), torch.count_nonzero(self.gaussians.observation_indices > -1), frame.keypoints.shape)

                    self.initialize_map_with_ba(cur_frame_idx, viewpoint)
                    extension_window = [cur_frame_idx]
                    self.push_to_frontend("init", first_id=cur_frame_idx)

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

                    cv2.imwrite('./res/test_ba/' + str(cur_frame_idx) + '.jpg', out)


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

                    ##match
                    self.preprocessing_add_kf()
                    #가우시안이 새 키프레임에 매칭 되면 True. 하나도 매칭이 없는 영역에서 새로운 가우시안 생성.
                    #새로운 가우시안은 이전 프레임과 매칭이 될 수 있고, 아예 없을 수도 있음.
                    kp_gaussian_mask = torch.zeros(frame.keypoints.shape[0], dtype=torch.bool, device='cuda')

                    last_keys = list(self.frames)
                    N_window = len(last_keys)
                    matches_info[cur_frame_idx]={}

                    index_map = {value: idx for idx, value in enumerate(last_keys)}
                    window_indices = [index_map[x] for x in current_window] + [index_map[x] for x in last_keys[-6:-1]]
                    tmp_extension_indices = list(range(min(window_indices), max(window_indices), 5))
                    extension_indices = list(set(window_indices + tmp_extension_indices))
                    extension_indices.sort(reverse = True)
                    extension_window = [last_keys[i] for i in extension_indices]

                    #tmp_extension_window.reverse()
                    #extension_indices = [x for x in tmp_extension_window if
                    #                     x not in window_indices]  # 현재 윈도우에 포함 안된 5의 배수 찾기

                    new_kf_idx = extension_indices[0]
                    btmp1 = new_kf_idx % 5 == 0

                    #최근 5개 키프레임과 매칭 추가
                    for idx, kf_idx in enumerate(last_keys[-6:-1]):
                        keyframe = self.frames[kf_idx]
                        d0 = {'keypoints': frame.keypoints,
                              'descriptors': torch.from_numpy(frame.descriptors).cuda(),
                              'image_size': (viewpoint.image_width, viewpoint.image_height)}
                        d1 = {'keypoints': keyframe.keypoints,
                              'descriptors': torch.from_numpy(keyframe.descriptors).cuda(),
                              'image_size': (viewpoint.image_width, viewpoint.image_height)}
                        matches = xfeat.match_lighterglue(d0, d1)
                        matches_info[cur_frame_idx][kf_idx] = matches.int()
                        matches_info[kf_idx][cur_frame_idx] = matches[:, [1, 0]].int()

                    ##N의 배수 키프레임 사이의 연관(현재 프레임이 N의 배수인 경우)
                    if btmp1:
                        for i in tmp_extension_indices[-2::-1]:
                            idx1 = i
                            idx2 = i+1
                            kf_idx1 = last_keys[idx1]
                            kf_idx2 = last_keys[idx2]

                            if cur_frame_idx in matches_info[kf_idx2]:

                                match1 = matches_info[kf_idx1][kf_idx2]
                                match2 = matches_info[kf_idx2][cur_frame_idx]
                                #epipolar constarints
                                match_ac = match_ac_from_ab_bc(match1, match2)

                                if match_ac.shape[0] > 10:
                                    matches_info[kf_idx1][cur_frame_idx] = match_ac
                                    matches_info[cur_frame_idx][kf_idx1] = match_ac[:, [1, 0]]

                    ##윈도우 내의 키프레임과 현재 프레임 연결
                    #소스가 배수이거나 배수가 아님
                    #타겟이 배수이거나 배수가 아님
                    #idx1 -> idx2로 가는게 목표임.
                    #idx1 -> tmp_idx1 -> idx2
                    #idx1 -> tmp_idx2 -> idx2
                    #idx1 -> tmp_idx1 -> tmp_idx2 -> idx2
                    #tmp_idx1 = tmp_idx2

                    for idx in extension_indices[:0:-1]:
                        kf_idx1 = last_keys[idx]

                        diff = new_kf_idx - idx
                        print('test ', idx, new_kf_idx, diff)
                        if diff < 6:
                            kf_idx2 = cur_frame_idx
                        elif idx % 5 == 0:
                            kf_idx2 = last_keys[idx + 5]
                            print('test % 5 = ', idx, idx+5, '=', kf_idx1, kf_idx2)
                        else:
                            kf_idx2 = last_keys[((idx // 5) + 1) * 5]
                            print('test not % 5 = ', idx, ((idx // 5) + 1) * 5, '=', kf_idx1, kf_idx2)

                        keyframe1 = self.frames[kf_idx1]
                        keyframe2 = self.frames[kf_idx2]

                        matches = matches_info[kf_idx1][kf_idx2]
                        #print('match', keyframe1.kf_id, max(matches[:,0]),max(self.gaussians.observation_indices[:,keyframe1.kf_id]))
                        gidx, fidx = self.get_gaussian_match_indices(self.gaussians.observation_indices, matches[:, 0],
                                                                     keyframe1.kf_id)
                        self.update_observation(gidx, fidx, matches[:, 1], keyframe2, keyframe2.kf_id)

                        if kf_idx2 == cur_frame_idx:
                            selected_values = matches[fidx, 0]
                            kp_gaussian_mask[selected_values] = True

                    """
                    #delete
                    for idx in extension_window[1:]:
                        kf_idx = last_keys[idx]
                        keyframe = self.frames[kf_idx]
                        view_kf = self.viewpoints[kf_idx]

                        diff = new_kf_idx-idx
                        if diff < 6:
                            d0 = {'keypoints': frame.keypoints,
                                  'descriptors': torch.from_numpy(frame.descriptors).cuda(),
                                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
                            d1 = {'keypoints': keyframe.keypoints,
                                  'descriptors': torch.from_numpy(keyframe.descriptors).cuda(),
                                  'image_size': (viewpoint.image_width, viewpoint.image_height)}
                            matches = xfeat.match_lighterglue(d0, d1)
                        else:
                            btmp2 = idx % 5 == 0
                            if btmp2:
                                source_idx = idx
                            else:
                                source_idx = ((idx // 5) + 1) * 5


                        print('test', idx, kf_idx)
                    """

                    Nold = self.gaussians._xyz.size()[0]

                    with torch.no_grad():
                        a = time.time()
                        #keypoint mask
                        kp_region_mask = calculate_feature_mask(frame.keypoints, viewpoint.image_width, viewpoint.image_height,max_radius=2)
                        aa1 = time.time()
                        ##gaussian mask
                        #1) projection
                        projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T, viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy, viewpoint.image_width, viewpoint.image_height)
                        #2) masking
                        aa2 = time.time()
                        tmp_gaussian_mask = calculate_keypoint_mask(projection[valid_projection], viewpoint.image_width, viewpoint.image_height, )#max_radius=1)#.squeeze(0)
                        aa3 = time.time()
                        tmp_gaussian_mask = ~tmp_gaussian_mask
                        gaussian_mask = torch.logical_and(~kp_region_mask, tmp_gaussian_mask)
                        a1 = time.time()
                        #print('1',aa1-a, aa2-aa1, aa3-aa2, aa3-a1, projection.device, tmp_gaussian_mask.device, gaussian_mask.device)
                        unmatched_mask = ~kp_gaussian_mask
                        kp_mask = calculate_keypoint_mask(frame.keypoints[unmatched_mask], viewpoint.image_width, viewpoint.image_height,)
                        kp_mask = torch.logical_and(kp_mask, tmp_gaussian_mask)
                        b = time.time()
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, mask = gaussian_mask.squeeze(0).cpu().numpy())
                        c = time.time()
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, mask = kp_mask.squeeze(0).cpu().numpy(), keypoints=frame.keypoints, downsample_factor=1.0)
                        d = time.time()

                        """
                        tmp_mask1 = gaussian_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        cv2.imshow("mask1", tmp_mask1)
                        tmp_mask2 = kp_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        cv2.imshow("mask2", tmp_mask2)
                        cv2.waitKey(10)
                        """
                        del tmp_gaussian_mask, kp_region_mask, gaussian_mask, unmatched_mask, projection, valid_projection
                        #print("time test", b-a3, c-b, d-c, 'new gaussian feature', torch.count_nonzero(self.gaussians.observation_indices[:, frame.kf_id]>-1))

                    gaussian_indices = torch.arange(self.gaussians.get_xyz.shape[0]).cuda()
                    gaussian_mask = gaussian_indices>=Nold
                    tmp_gaussian_indices = self.gaussians.observation_indices[gaussian_mask]

                    #새로운 가우시안 포인트를 이전 프레임에 전파
                    for idx in extension_indices[1:]:
                        kf_idx1 = last_keys[idx]
                        kf_idx2 = 0
                        diff = new_kf_idx - idx

                        if diff < 6:
                            kf_idx2 = cur_frame_idx
                        elif idx % 5 == 0:
                            kf_idx2 = last_keys[idx + 5]
                        else:
                            kf_idx2 = last_keys[((idx // 5) + 1) * 5]

                        keyframe1 = self.frames[kf_idx1]
                        keyframe2 = self.frames[kf_idx2]

                        matches = matches_info[kf_idx2][kf_idx1]
                        gidx, fidx = self.get_gaussian_match_indices(tmp_gaussian_indices, matches[:, 0],
                                                                     keyframe2.kf_id, n_start=Nold)
                        self.update_observation(gidx, fidx, matches[:, 1], keyframe1, keyframe1.kf_id)

                    """
                    #delete
                    for kf_idx in last_keys[-6:-1]:
                        #if kf_idx == cur_frame_idx:
                        #    continue
                        keyframe = self.frames[kf_idx]
                        matches = matches_info[cur_frame_idx][kf_idx]

                        gidx,fidx = self.get_gaussian_match_indices(tmp_gaussian_indices, matches[:,0], frame.kf_id, Nold)
                        N1 = torch.count_nonzero(self.gaussians.observation_indices[:, keyframe.kf_id] > -1)
                        self.update_observation(gidx, fidx, matches[:, 1], keyframe, keyframe.kf_id)
                        N2 = torch.count_nonzero(self.gaussians.observation_indices[:, keyframe.kf_id] > -1)
                        print("backend=update_old_kf",kf_idx, N2, N1)
                    """

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
                    _, remove_ids1 = self.map_with_ba(self.current_window, iters=iter_per_kf, extension_graph=extension_window)  # graph = temp_kf_window,,   matches=kf_matches, graph=current_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map_with_ba(self.current_window, prune=True, extension_graph=extension_window)  # matches = kf_matches, graph = temp_kf_window,,  matches=kf_matches, graph=current_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    self.update_occ_visibility(self.current_window)

                    if remove_ids is not None:
                        self.push_to_frontend("keyframe", prune=remove_ids)
                    else:
                        self.push_to_frontend("keyframe")
                    e2 = time.time()
                    print('backend::end', cur_frame_idx, e2 - s, self.gaussians.get_xyz.shape[0], )

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

                            """
                            kf_id = keyframe.kf_id
                            tmp_idx = torch.where(self.gaussians.observation_indices[:, kf_id] > -1)[0]
                            tmp_gaussians = self.gaussians.get_xyz[tmp_idx]

                            projection, _, valid_projection = project_pc_to_pixel(tmp_gaussians, viewpoint.R,
                                                                                  viewpoint.T,
                                                                                  viewpoint.fx, viewpoint.fy,
                                                                                  viewpoint.cx, viewpoint.cy,
                                                                                  viewpoint.image_width,
                                                                                  viewpoint.image_height)
                            projection = projection[valid_projection]
                            tmp_idx = tmp_idx[valid_projection]
                            points = self.gaussians.observation_points[tmp_idx, 2 * kf_id:2 * kf_id + 2]
                            """
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
                            cv2.imwrite('./res/test_ba/mapping_' + str(cur_frame_idx) + '_' + str(kf_idx) + '.jpg', image_np)
                            """"                       
                            self.FeatureManager.tracker.visualize2(image_np, projection.clone(), points.clone(), delay=1
                                                                   , save=True,
                                                                   filename='./res/test_ba/mapping_' + str(
                                                                       cur_frame_idx) + '_' + str(kf_idx) + '.jpg')
                            """
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