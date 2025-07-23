import numpy as np

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
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss, get_patch_loss
from edge_assisted.gaussian_feature import project_pc_to_pixel, projection, calculate_bbox_mask, calculate_feature_mask, find_correspondence_with_dist
from utils.edgeframe_utils import EdgeFrame
from utils.pose_utils import SE3_exp

from utils.datahandle_utils import move_camera_to_gpu, move_camera_to_cpu, move_gaussianmodel_to_cpu
from utils.datahandle_utils import move_occ_visibility_to_cpu
#from edge_assisted.gaussian_feature import GaussianPointManager
from edge_assisted.localmap_utils import get_local_gaussians
#from edge_assisted.object_manager import ObjectManager
from edge_assisted.object_loss import ObjectLoss
from edge_assisted.gaussian_orb_model import GaussianOrbModel

from collections import defaultdict

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

    def push_to_frontend(self, tag=None, first_id = None, prune = None):

        self.last_sent = 0
        keyframes = []

        for kf_idx in self.current_window:
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

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, frame=None):

        self.gaussians.observation_indices = torch.cat([self.gaussians.observation_indices,
                                                        torch.full((self.gaussians.observation_indices.shape[0], 1),
                                                                   -1, device='cuda')], dim=1)
        self.gaussians.observation_points = torch.cat([self.gaussians.observation_points,
                                                       torch.full((self.gaussians.observation_points.shape[0], 2), -1.0,
                                                                  device='cuda')], dim=1)

        #self.update_gaussian_observation_with_frame(frame)

        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,frame=frame, downsample_factor = downsample_factor
        )
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

    def connect_observation(self, matches, frame, new_kf_id, kf_id, match_idx):
        obs_idx_col = self.gaussians.observation_indices[:, new_kf_id]  # shape: (N,)
        match_values = matches[:, match_idx]  # shape: (K,)
        # eq[n, k] == True면 obs_idx_col[n] == match_values[k]
        # 위치: 포함되면 첫 번째 True의 인덱스, 없으면 -1
        # 인클루드 마스크는 가우시안에서 매칭(0)의 키포인트 위치. 즉 매치(0)과 같음., 가우시안 위치를 표현함.
        # 키포지션은 그게 매치 안에서 어디있는지를 알 수 있음.
        eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
        key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                    torch.full_like(obs_idx_col, -1))
        included_mask = key_positions != -1

        frame_gaussian_index = torch.where(included_mask)[0]
        frame_match_index = key_positions[included_mask]

        #print('asdfasdf', frame_gaussian_index.shape, matches.shape, torch.count_nonzero(obs_idx_col))
        #print(obs_idx_col[obs_idx_col > -1], match_values)

        self.gaussians.observation_indices[frame_gaussian_index, kf_id] = matches[frame_match_index, match_idx]
        self.gaussians.observation_points[frame_gaussian_index, kf_id * 2:kf_id * 2 + 2] = \
            frame.keypoints[matches[frame_match_index, match_idx]]

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
            loss_rgb,loss_depth= get_loss_mapping(self.config, image, depth, kf_view, opacity,
                                              feature_mask=kf_feature_mask)
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

    def bundle_adjustment(self, current_window, th_obs = 2):
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
            """"""
            tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], axis=0)
            T_w2c = torch.eye(4, device=viewpoint.R.device)
            T_w2c[0:3, 0:3] = viewpoint.R
            T_w2c[0:3, 3] = viewpoint.T

            new_w2c = SE3_exp(tau) @ T_w2c

            new_R = new_w2c[0:3, 0:3]
            new_T = new_w2c[0:3, 3]

            projection, _, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                    viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                    viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            projection = projection[valid]
            points = obs_points[idx, kid:kid + 2][valid]

            loss_ba += get_reprojection_loss(projection, points).mean()
        t3 = time.time()
        print("BA =", t2 - t1, t3 - t2, loss_ba, len(current_window))
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

    def map(self, current_window, prune=False, iters=1, matches = None, graph = None):
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
        last_kf_id = current_window[0]

        curr_patches = None
        curr_patches_valid = None
        curr_rendered_image = None
        kf_patches = defaultdict(lambda: {'patch': None, 'valid': None})

        ##patch consistency
        doPatchConsistency = False
        if False and graph is not None and len(matches)>0:
            doPatchConsistency = True
            for cam_idx in graph:
                if cam_idx == last_kf_id or cam_idx not in matches:
                    continue
                match = matches[cam_idx]
                viewpoint = self.viewpoints[cam_idx]
                frame = self.frames[cam_idx]
                kf_patches[cam_idx]['patch'], kf_patches[cam_idx]['valid'] = frame.extract_patches_differentiable(viewpoint.original_image, frame.keypoints[match[:,1]])

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
                kf_feature_mask = calculate_feature_mask(kf_frame.keypoints, viewpoint.image_width, viewpoint.image_height,
                                                         max_radius=5)

                loss_rgb,loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, feature_mask=kf_feature_mask
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

                kf_frame = self.frames[viewpoint.uid]
                kf_feature_mask = calculate_feature_mask(kf_frame.keypoints, viewpoint.image_width,
                                                         viewpoint.image_height,
                                                         max_radius=5)

                loss_rgb, loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, feature_mask=kf_feature_mask
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
                print("Gaussians affected by ba:", torch.count_nonzero(affected_by_ba), self.gaussians._xyz.shape, affected_by_ba.nonzero().flatten())

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
                    curr_val = curr_patches_valid[match[:,0]]
                    valid = torch.logical_and(curr_val, kf_val)

                    cur_patch = curr_patches[match[valid,0]]
                    kf_patch = kf_patches[cam_idx]['patch'][valid]

                    err =get_patch_loss(cur_patch,kf_patch)
                    loss_patch += err.mean()
                    #print('patch', last_kf_id, cam_idx, torch.count_nonzero(valid), valid.shape, cur_patch.shape)
                loss_mapping += loss_patch*self.weight_patch

                t_patch2 = time.time()
                #print('patch loss = ', loss_patch.mean(), t_patch2-t_patch1)
                if False and self.gaussians._xyz.grad is not None:
                    after_grad_loss_patch = self.gaussians._xyz.grad.clone()
                    affected_by_patch = torch.any(after_grad_loss_patch != 0, dim=1)
                    print("Gaussians affected by patch:", torch.count_nonzero(affected_by_patch), self.gaussians._xyz.shape, affected_by_patch.nonzero().flatten())
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
                        self.occ_aware_visibility[current_idx] = self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

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
                ##graph update
                if graph is not None:
                    self.update_graph(graph, th = 100.0)
                self.update_graph_weights()

                ##graph update
        return gaussian_split, remove_ids

    def run2(self):
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
                _, prune_mask1 = self.map(self.current_window, matches=kf_matches,
                                          graph=None)  # matches = kf_matches, graph = recent_keys
                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2 = self.map(self.current_window, prune=True, iters=10, matches=kf_matches,
                                              graph=None)  # matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2

                if prune_mask is not None:
                    self.push_to_frontend(prune=prune_mask)
                else:
                    self.push_to_frontend()
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
                    frame.keypoints, frame.descriptors, frame.objects = f
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

                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True, frame=frame
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
                    frame.keypoints, frame.descriptors, frame.objects = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    self.frames[(cur_frame_idx)] = frame

                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, frame=frame)

                    ##add match and observation
                    # get_local_gaussians(self.gaussians, self.current_window)
                    t1 = time.time()
                    # temp_obs = self.gaussians.observations[self.gaussians.isfeatured.clone().cpu().numpy()]
                    a = time.time()
                    temp_kf_window = [x for x in current_window if x != cur_frame_idx]
                    kf_matches = {}
                    new_kf_id = frame.kf_id

                    keyframe_keys = list(self.frames)
                    last_key = keyframe_keys[-2]

                    ##객체
                    """
                    if len(frame.objects) > 0:
                        for oid in frame.objects:
                            obj = self.objects[oid]
                            result = [(key, obj._frames[key]) for key in keyframe_keys if key in obj._frames]
                            print('obj', oid, result)#len(obj._frames), frame.objects[oid])
                    """
                    ##graph 진행중
                    intersection = self.update_graph_weights()

                    if intersection is not None:
                        keys = self.get_neighbor_keyframes(intersection, self.frames[last_key].kf_id)
                        print('keyframes', cur_frame_idx, last_key, keys, temp_kf_window)
                        temp_kf_window = [last_key]
                        temp_kf_window.extend(keys[:10])

                    for kf_idx in temp_kf_window:
                        keyframe = self.frames[kf_idx]
                        matches = self.FeatureManager.tracker.match(frame.descriptors, keyframe.descriptors)
                        matches = torch.from_numpy(matches).type(torch.int32).cuda()
                        if matches.shape[0] > 20:
                            kf_matches[kf_idx] = matches
                            tt1 = time.time()

                            kf_id = keyframe.kf_id

                            self.connect_observation(matches, keyframe, new_kf_id, kf_id, 1)
                            self.connect_observation(matches, frame, kf_id, new_kf_id, 0)
                            """
                            ##새로운 가우시안 포인트에 기존 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, new_kf_id]  # shape: (N,)
                            match_values = matches[:, 0]  # shape: (K,)
                            # eq[n, k] == True면 obs_idx_col[n] == match_values[k]
                            # 위치: 포함되면 첫 번째 True의 인덱스, 없으면 -1
                            # 인클루드 마스크는 가우시안에서 매칭(0)의 키포인트 위치. 즉 매치(0)과 같음., 가우시안 위치를 표현함.
                            # 키포지션은 그게 매치 안에서 어디있는지를 알 수 있음.
                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                    torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, kf_id] = matches[frame_match_index, 1]
                            self.gaussians.observation_points[frame_gaussian_index, kf_id * 2:kf_id * 2 + 2] = \
                            keyframe.keypoints[matches[frame_match_index, 1]]

                            ##기존의 가우시안 포인트에 새로운 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, kf_id]  # shape: (N,)
                            match_values = matches[:, 1]  # shape: (K,)

                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                        torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1 #가우시안 수와 일치함.

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, new_kf_id] = matches[
                                frame_match_index, 0]
                            self.gaussians.observation_points[frame_gaussian_index, new_kf_id * 2:new_kf_id * 2 + 2] = \
                                frame.keypoints[matches[frame_match_index, 0]]

                            tt2 = time.time()
                            #print("gaussian connection", kf_id,tt2-tt1, included_mask.shape, torch.count_nonzero(included_mask), key_positions.shape)
                            #self.gaussians.observation_indices[:, new_kf_id].shape, frame_match_mask.shape, match_values.shape, torch.count_nonzero(frame_match_mask), torch.count_nonzero(frame_match_mask2), torch.count_nonzero(self.gaussians.observation_indices[:, new_kf_id] > -1))
                            """
                        # print('backend::', kf_idx, cur_frame_idx)#, torch.count_nonzero(keyframe.gaussianpoints > -1), keyframe.keypoints.size()[0])

                    b = time.time()
                    # print("kf match test = ", len(temp_kf_window), temp_obs.shape, a-t1, b-a)
                    ##add match and observation

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    frames_to_optimize = len(self.current_window)
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
                        if self.current_window[cam_idx] == 0:
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
                    _, remove_ids1 = self.map(self.current_window, iters=iter_per_kf, matches=kf_matches,
                                              graph=current_window)  # graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map(self.current_window, prune=True, matches=kf_matches,
                                              graph=current_window)  # matches = kf_matches, graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1

                    if remove_ids is not None:
                        need_sync_prune = True
                        self.push_to_frontend("keyframe", prune=remove_ids)
                    e1 = time.time()
                    if need_sync_prune:
                        need_sync_prune = False
                        # self.push_to_frontend("keyframe", prune=remove_ids)
                    else:
                        self.push_to_frontend("keyframe")
                    e2 = time.time()
                    print('backend::end', cur_frame_idx, e2 - s, frames_to_optimize)
                    ##frame visualization

                    for kf_idx in temp_kf_window:
                        if kf_idx in kf_matches:
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

    def run(self):
        prune_dict = None
        need_sync_prune = False

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

                #match 추가
                num_lask_kf = 3
                keys = list(self.frames)
                last_key = keys[-1]
                recent_keys = keys[-num_lask_kf:-1] if len(keys) > 1 else []
                last_keyframe = self.frames[last_key]

                ##graph 진행중
                last_key = keyframe_keys[-2]
                intersection = self.update_graph_weights()
                if intersection is not None:
                    keys = self.get_neighbor_keyframes(intersection, self.frames[last_key].kf_id)
                    #print('keyframes', cur_frame_idx, last_key, keys, temp_kf_window)
                    recent_keys = [last_key]
                    recent_keys.extend(keys[:10])

                kf_matches = {}
                for kf_idx in recent_keys:
                    keyframe = self.frames[kf_idx]
                    matches = self.FeatureManager.tracker.match(last_keyframe.descriptors, keyframe.descriptors)
                    matches = torch.from_numpy(matches).cuda()
                    if matches.shape[0] > 20:
                        kf_matches[kf_idx] = matches

                # match 추가

                prune_mask = None
                _, prune_mask1 = self.map(self.current_window, matches = kf_matches, graph = current_window)#matches = kf_matches, graph = recent_keys
                if prune_mask1 is not None:
                    prune_mask = prune_mask1
                if self.last_sent >= 10:
                    _, prune_mask2= self.map(self.current_window, prune=True, iters=10, matches = kf_matches, graph = current_window)#matches=kf_matches, graph = recent_keys
                    if prune_mask2 is not None:
                        prune_mask = prune_mask2

                if prune_mask is not None:
                    need_sync_prune = True
                    need_sync_prune = False
                    self.push_to_frontend(prune=prune_mask)

                if need_sync_prune:
                    need_sync_prune = False
                    #self.push_to_frontend(prune=prune_mask)
                else:
                    self.push_to_frontend()
                e = time.time()
                #print("backend = mapping with empty queue", (e-s))

                ##뎁스화 TEST
                """
                num_lask_kf = 5
                keys = list(self.frames)
                recent_keys = keys[-num_lask_kf:] if len(keys) > 0 else []

                a = time.time()
                self.dense_map(new_gaussians, recent_keys, iters=1)
                b = time.time()
                print('덴스화 = 매핑', b - a)
                ###save image
                render_pkg = render(
                    viewpoint, new_gaussians, self.pipeline_params, self.background
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
                cv2.imwrite('./res/test_dense/' + str(cur_frame_idx) + '.jpg', out)
                """
                ##뎁스화 TEST

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
                    frame.keypoints, frame.descriptors, frame.objects = f
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

                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True, frame=frame
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
                    frame.keypoints, frame.descriptors, frame.objects = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.keyframe_ids[frame.kf_id] = cur_frame_idx
                    self.frames[(cur_frame_idx)] = frame

                    """
                    if prune_dict is not None:
                        for kpidx, gid in enumerate(frame.gaussianpoints):
                            gid = gid.item()
                            if gid == -1:
                                continue
                            if gid in prune_dict:
                                frame.gaussianpoints[kpidx] = prune_dict[gid]

                        #print('frame equal test', np.equal(frame.gaussianpoints, self.frames[cur_frame_idx].gaussianpoints))
                    """
                    Nold = self.gaussians._xyz.size()[0]
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, frame=frame)

                    ##뎁스화 TEST
                    """
                    num_lask_kf = 5
                    keys = list(self.frames)
                    recent_keys = keys[-num_lask_kf:] if len(keys) > 0 else []
                    
                    new_gaussians = self.add_next_kf_from_window(new_gaussians, recent_keys, ginit = True, init = True, )
                    a = time.time()
                    self.dense_map(new_gaussians, recent_keys, iters= 1)
                    b= time.time()
                    print('덴스화 = 매핑', b-a)
                    ###save image
                    render_pkg = render(
                        viewpoint, new_gaussians, self.pipeline_params, self.background
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
                    cv2.imwrite('./res/test_dense/' + str(cur_frame_idx) + '.jpg', out)
                    """
                    ##뎁스화 TEST

                    ##add match and observation
                    #get_local_gaussians(self.gaussians, self.current_window)
                    t1 = time.time()
                    #temp_obs = self.gaussians.observations[self.gaussians.isfeatured.clone().cpu().numpy()]
                    a = time.time()
                    temp_kf_window = [x for x in current_window if x != cur_frame_idx]
                    kf_matches = {}
                    new_kf_id = frame.kf_id

                    keyframe_keys = list(self.frames)
                    last_key = keyframe_keys[-2]

                    ##객체
                    """
                    if len(frame.objects) > 0:
                        for oid in frame.objects:
                            obj = self.objects[oid]
                            result = [(key, obj._frames[key]) for key in keyframe_keys if key in obj._frames]
                            print('obj', oid, result)#len(obj._frames), frame.objects[oid])
                    """
                    ##graph 진행중
                    intersection = self.update_graph_weights()

                    if intersection is not None:
                        keys = self.get_neighbor_keyframes(intersection, self.frames[last_key].kf_id)
                        print('keyframes',cur_frame_idx, last_key,keys, temp_kf_window)
                        temp_kf_window = [last_key]
                        temp_kf_window.extend(keys[:10])

                    for kf_idx in temp_kf_window:
                        keyframe = self.frames[kf_idx]
                        matches = self.FeatureManager.tracker.match(frame.descriptors, keyframe.descriptors)
                        matches = torch.from_numpy(matches).type(torch.int32).cuda()
                        if matches.shape[0] > 20 :
                            kf_matches[kf_idx] = matches
                            tt1 = time.time()

                            kf_id = keyframe.kf_id

                            self.connect_observation(matches, keyframe, new_kf_id, kf_id, 1)
                            self.connect_observation(matches, frame, kf_id, new_kf_id, 0)
                            """
                            ##새로운 가우시안 포인트에 기존 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, new_kf_id]  # shape: (N,)
                            match_values = matches[:, 0]  # shape: (K,)
                            # eq[n, k] == True면 obs_idx_col[n] == match_values[k]
                            # 위치: 포함되면 첫 번째 True의 인덱스, 없으면 -1
                            # 인클루드 마스크는 가우시안에서 매칭(0)의 키포인트 위치. 즉 매치(0)과 같음., 가우시안 위치를 표현함.
                            # 키포지션은 그게 매치 안에서 어디있는지를 알 수 있음.
                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                    torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, kf_id] = matches[frame_match_index, 1]
                            self.gaussians.observation_points[frame_gaussian_index, kf_id * 2:kf_id * 2 + 2] = \
                            keyframe.keypoints[matches[frame_match_index, 1]]

                            ##기존의 가우시안 포인트에 새로운 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, kf_id]  # shape: (N,)
                            match_values = matches[:, 1]  # shape: (K,)

                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                        torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1 #가우시안 수와 일치함.

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, new_kf_id] = matches[
                                frame_match_index, 0]
                            self.gaussians.observation_points[frame_gaussian_index, new_kf_id * 2:new_kf_id * 2 + 2] = \
                                frame.keypoints[matches[frame_match_index, 0]]

                            tt2 = time.time()
                            #print("gaussian connection", kf_id,tt2-tt1, included_mask.shape, torch.count_nonzero(included_mask), key_positions.shape)
                            #self.gaussians.observation_indices[:, new_kf_id].shape, frame_match_mask.shape, match_values.shape, torch.count_nonzero(frame_match_mask), torch.count_nonzero(frame_match_mask2), torch.count_nonzero(self.gaussians.observation_indices[:, new_kf_id] > -1))
                            """
                        #print('backend::', kf_idx, cur_frame_idx)#, torch.count_nonzero(keyframe.gaussianpoints > -1), keyframe.keypoints.size()[0])


                    b = time.time()
                    #print("kf match test = ", len(temp_kf_window), temp_obs.shape, a-t1, b-a)
                    ##add match and observation

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    frames_to_optimize = len(self.current_window)
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
                        if self.current_window[cam_idx] == 0:
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
                    _, remove_ids1 = self.map(self.current_window, iters=iter_per_kf, matches = kf_matches, graph = current_window )#graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1
                    _, remove_ids1 = self.map(self.current_window, prune=True, matches = kf_matches, graph = current_window)#matches = kf_matches, graph = temp_kf_window
                    if remove_ids1 is not None:
                        remove_ids = remove_ids1

                    if remove_ids is not None:
                        need_sync_prune = True
                        self.push_to_frontend("keyframe", prune=remove_ids)
                    e1 = time.time()
                    if need_sync_prune:
                        need_sync_prune = False
                        #self.push_to_frontend("keyframe", prune=remove_ids)
                    else :
                        self.push_to_frontend("keyframe")
                    e2 = time.time()
                    print('backend::end', cur_frame_idx, e2-s, frames_to_optimize)
                    ##frame visualization

                    for kf_idx in temp_kf_window:
                        if kf_idx in kf_matches:
                            keyframe = self.frames[kf_idx]
                            viewpoint = self.viewpoints[kf_idx]

                            kf_id = keyframe.kf_id
                            tmp_idx = torch.where(self.gaussians.observation_indices[:, kf_id] > -1)[0]
                            tmp_gaussians = self.gaussians.get_xyz[tmp_idx]

                            projection, _, valid_projection = project_pc_to_pixel(tmp_gaussians, viewpoint.R, viewpoint.T,
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
                                   filename='./res/map/mapping_' + str(cur_frame_idx) + '_' + str(kf_idx) + '.jpg')

                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return