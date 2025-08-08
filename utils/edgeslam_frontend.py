import cv2
import torch
import time

import gtsam
import pycolmap

import numpy as np
from scipy.spatial.transform import Rotation
from utils.slam_win_frontend import WinFrontEnd

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_median_depth
from utils.slam_frontend import FrontEnd
from utils.datahandle_utils import move_gaussianpacket_to_gpu, move_gaussianpacket_to_cpu, move_camera_to_cpu, move_camera_to_gpu, move_gaussianmodel_to_gpu
from utils.datahandle_utils import move_occ_visibility_to_gpu
from utils.edgeframe_utils import init_from_dataset
from edge_assisted.gaussian_feature import project_pc_to_pixel,convert_xyz, pixels_to_pc, find_correspondence, find_correspondence_with_dist, calculate_feature_mask, get_correspondences_within_threshold
from edge_assisted.slam_utils import get_loss_tracking, get_reprojection_loss, get_patch_loss, get_reprojection_loss2, get_loss_gaussian
#from edge_assisted.gaussian_feature import GaussianPointManager

from typing import Dict, Tuple, Optional, List
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from edge_assisted.pose_optimizer import PoseOptimizer, PoseOptimizer2

#XFeat
import sys
sys.path.append('D:/UVR/accelerated_features')
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue

class EdgeFrontEnd(WinFrontEnd):
    def __init__(self, config):
        super().__init__(config)

        self.frames = {}

        self.edge_queue = None
        self.tracking_mode = None
        self.tracking_mode_vo = False

        #self.testManager = GaussianPointManager()
        self.testManager = None

    """"""
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        self.frames[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        frame = self.frames[cur_frame_idx]
        f = [frame.keypoints.cpu().clone(), frame.descriptors, frame.objects, frame.contours]
        msg = ["init", cur_frame_idx, move_camera_to_cpu(viewpoint), depth_map, f]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        frame = self.frames[cur_frame_idx]
        f = [frame.keypoints.cpu().clone(), frame.descriptors, frame.objects, frame.contours]
        msg = ["keyframe", cur_frame_idx, move_camera_to_cpu(viewpoint), (current_window), (depthmap),f]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def sync_backend(self, data, prev_frame_idx = None):
        gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]

        move_gaussianmodel_to_gpu(gaussians)
        move_occ_visibility_to_gpu(occ_aware_visibility)
        #move_gaussians_to_gpu(keyframes)

        #copy gaussians from backend
        self.gaussians = gaussians
        self.gaussians._xyz = self.gaussians._xyz.detach()#.requires_grad_(False)
        self.gaussians._features_dc = self.gaussians._features_dc.detach()#.requires_grad_(False)
        self.gaussians._features_rest = self.gaussians._features_rest.detach()#.requires_grad_(False)
        self.gaussians._opacity = self.gaussians._opacity.detach()#.requires_grad_(False)
        self.gaussians._scaling = self.gaussians._scaling.detach()#.requires_grad_(False)
        self.gaussians._rotation = self.gaussians._rotation.detach()#.requires_grad_(False)

        ## 축소 테스트
        """
        x = torch.arange(self.gaussians.get_xyz.shape[0])  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        even_indexed = x[::3]
        self.gaussians = self.gaussians.clone(even_indexed)
        for idx, value in occ_aware_visibility.items():
            occ_aware_visibility[idx] = occ_aware_visibility[idx][even_indexed]
        """
        ## 축소 테스트

        ## 축소 테스트

        self.occ_aware_visibility = occ_aware_visibility
        for idx, tensor in self.occ_aware_visibility.items():
            if tensor.shape[0] != self.gaussians.get_xyz.shape[0]:
                print('occ_aware_visibility : error case', tensor.shape, self.gaussians.get_xyz.shape)

        #키프레임의 포즈 동기화
        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone().to(self.device), kf_T.clone().to(self.device))

        #update frame gaussianpoints
        prune_dict = data[4]

        if prune_dict is not None and prev_frame_idx is not None:
            #remove_ids 로 프론트 엔드 매칭 테이블에서 가우시안을 삭제 해야 함.
            last_kf_id = self.current_window[0]
            update_frame_list = [prev_frame_idx, last_kf_id]

            for fid in update_frame_list:
                frame = self.frames[fid]
                #frame.gaussianpoints = torch.full((frame.keypoints.shape[0],),-1)

                """
                for kpidx, gid in enumerate(frame.gaussianpoints):
                    gid = gid.item()
                    if gid == -1:
                        continue
                    if gid in prune_dict:
                        frame.gaussianpoints[kpidx] = prune_dict[gid]
                        #if prune_dict[gid] == -1:
                        #    frame.inliers[kpidx] = False
                """

            """
            mask = [
                x is not None
                and isinstance(x, dict)
                and (prev_frame_idx in x)
                for x in self.gaussians.observations
            ]
            indices = np.where(mask)[0]
            frame_gaussians = self.gaussians.observations[mask]
            for gidx, obs in zip(indices,frame_gaussians):
                kp_idx = obs[prev_frame_idx]
                frame.gaussianpoints[kp_idx] = gidx
            """
            #print("frontend::update test", data[0], prev_frame_idx, last_kf_id, len(prune_dict))

    #pose를 바로 변경
    def create_camera_matrices(self, K: torch.Tensor, R: torch.Tensor, t : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create view matrix and intrinsics for rendering"""
        # Convert world-to-camera transformation
        #viewmat = torch.inverse(pose)  # Camera-to-world -> World-to-camera

        # Add batch dimension

        T_w2c = torch.eye(4, device=R.device)
        T_w2c[0:3, 0:3] = R
        T_w2c[0:3, 3] = t

        viewmats = T_w2c.unsqueeze(0)  # [1, 4, 4]
        Ks = torch.from_numpy(K).cuda().unsqueeze(0)  # [1, 3, 3]

        return viewmats, Ks

    def tracking_with_pc(self, cur_frame_idx, prev_frame_idx, viewpoint, colCam, col_param, matches):
        prev = self.cameras[prev_frame_idx]
        prev_frame = self.frames[prev_frame_idx]

        viewpoint.update_RT(prev.R, prev.T)
        curr_frame = self.frames[cur_frame_idx]

        ##depth 정렬 테스트
        t_depth_start = time.time()
        projections, depths, valid = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                         viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                         viewpoint.image_width, viewpoint.image_height)
        projections = projections[valid]#torch.round(projections[valid]).int()  # 유효한 프로젝션 결과를 int화 해서 픽셀로 만듬. 정렬하면
        #depths = depths[valid]
        #sorted_indices = torch.argsort(depths)

        prev_frame = self.frames[prev_frame_idx]
        keypoints = prev_frame.keypoints#torch.round(prev_frame.keypoints).int()
        match_idx = get_correspondences_within_threshold(projections, keypoints, th = 2.0)  # 약간 시간이 걸림. 0.01 이하

        #valid 결과에 대해서 수행해야 함.
        valid_match = match_idx > -1 & self.gaussians.isfeatured[valid]
        valid_match_idx = match_idx[valid_match]
        gaussians = self.gaussians.get_xyz[valid][valid_match]

        #prev 키포인트와 대응하는 curr point 획득
        matches = torch.from_numpy(matches).cuda()
        prev_to_curr = torch.full((prev_frame.keypoints.shape[0],), -1, dtype=torch.long, device = 'cuda')
        prev_to_curr[matches[:,0]] = matches[:,1]

        mask = prev_to_curr[valid_match_idx] > -1
        gaussians = gaussians[mask]
        points2d = curr_frame.keypoints[prev_to_curr[valid_match_idx][mask]]

        t_depth_end = time.time()
        print(gaussians.shape,torch.count_nonzero(mask), t_depth_end-t_depth_start)
        #print('depth sort test', t_depth_end - t_depth_start,
        #      torch.count_nonzero(match_idx > -1 & self.gaussians.isfeatured[valid]), sorted_indices.shape,
        #      self.gaussians.get_xyz.shape[0])  # 0.01보다 작음. 거의 0.002정도인데 커질수록 많을 듯
        ##depth 정렬 테스트

        """
        temp_keypoints = torch.round(prev_frame.keypoints).int()
        points = curr_frame.keypoints[matches[:, 1]]
        temp_keypoints = temp_keypoints[matches[:, 0]]

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )

        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        depth = depth.detach().clone().squeeze(0)

        #gaussians, add_colors, valid_depth_mask = convert_xyz(temp_keypoints, viewpoint.original_image, depth)
        depth_values = depth[temp_keypoints[:, 1], temp_keypoints[:, 0]]
        valid_mask = depth_values > 0

        temp_keypoints = temp_keypoints[valid_mask]
        valid_depths = depth_values[valid_mask]
        points2d = points[valid_mask].double()

        gaussians = torch.cat([
            temp_keypoints.float(),
            valid_depths.unsqueeze(1)
        ], dim=1)

        gaussians = pixels_to_pc(gaussians, viewpoint.R, viewpoint.T, viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy)
        """

        ##pose optimization
        model = PoseOptimizer2(viewpoint.R, viewpoint.T).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        best_loss = float('inf')
        best_state = None
        t_po1 = time.time()
        print("최적화 시작...")
        th = 2.447
        for epoch in range(100):

            # 현재 포즈로 포인트 변환
            transformed_points, tmp_valid = model(gaussians,viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                             viewpoint.image_width, viewpoint.image_height)

            # L2 거리 손실
            #loss = torch.mean(torch.sum((transformed_points - points2d) ** 2, dim=1))
            residuals = transformed_points[tmp_valid] - points2d[tmp_valid]
            distances = torch.norm(residuals, dim=1)

            # Huber loss (outlier에 더 robust)
            delta = 0.1
            huber_loss = torch.where(distances < delta,
                                     0.5 * distances ** 2,
                                     delta * distances - 0.5 * delta ** 2)
            loss = torch.mean(huber_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    'rotation': model.rotation.data.clone(),
                    'translation': model.translation.data.clone()
                }
            losses.append(loss.item())

            if epoch % 10 == 0:
                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}", time.time()-t_po1)
        model.rotation.data = best_state['rotation']
        model.translation.data = best_state['translation']
        R, t = model.GetPose()

        ##pypose
        """
        ta = time.time()
        K = torch.tensor([[self.dataset.fx, 0, self.dataset.cx],
                      [0, self.dataset.fy, self.dataset.cy],
                      [0, 0, 1]], dtype=torch.float64, device = 'cuda')
        module = PoseOptimizer(viewpoint.R, viewpoint.T)

        points3d = gaussians.double()
        optimizer = torch.optim.LBFGS(module.parameters(), lr=1.0)
        huber = torch.nn.HuberLoss(delta=1.0, reduction='mean')  # delta는 상황에 따라 조정[1]

        def closure():
            optimizer.zero_grad()
            proj = module(points3d, K)
            loss = huber(proj, points2d)  # Huber loss 적용!
            loss.backward()
            return loss

        for i in range(20):
            optimizer.step(closure)
        R, t = module.GetPose()
        tb = time.time()
        print("asdf", tb-ta)
        """
        ##pypose

        ####COLMAP
        """
        result = pycolmap.estimate_absolute_pose(
            points2d.double().cpu().numpy(), gaussians.double().cpu().numpy(),colCam, col_param
        )

        #if result['success']:
        rigid = result['cam_from_world']
        R = torch.from_numpy(rigid.rotation.matrix()).cuda()
        t = torch.from_numpy(rigid.translation).cuda()
        """
        ####COLMAP

        viewpoint.update_RT(R,t)


        with torch.no_grad():
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        self.median_depth = get_median_depth(depth, opacity)

        ###test
        projections, depths, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                         viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                         viewpoint.image_width, viewpoint.image_height)
        projections = projections[valid]
        points = points2d[valid]
        self.testManager.tracker.visualize2(curr_frame.color, projections, points, delay=10, save=True,
                                            filename='./res/' + str(cur_frame_idx) + '.jpg')

        return render_pkg

    def tracking_with_patch(self, cur_frame_idx, prev_frame_idx, viewpoint, matches):
        prev = self.cameras[prev_frame_idx]
        prev_frame = self.frames[prev_frame_idx]
        viewpoint.update_RT(prev.R, prev.T)
        """
        prev_view = self.cameras[prev_frame_idx]
        prev_render_pkg = render(
            prev_view, self.gaussians, self.pipeline_params, self.background
        )
        prev_render_image = (
            prev_render_pkg["render"],
        )
        """

        curr_frame = self.frames[cur_frame_idx]

        ##patch

        curr_patch, curr_patch_valid = curr_frame.extract_patches_differentiable(
            viewpoint.original_image, curr_frame.keypoints[matches[:, 1]])

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        pose_optimizer = torch.optim.Adam(opt_params)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t_n = 0
        prev_tau = None
        for tracking_itr in range(self.tracking_itr_num):
            t1 = t1+time.time()

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            prev_patch, prev_patch_valid = prev_frame.extract_patches_differentiable(
                image, prev_frame.keypoints[matches[:, 0]])

            t2 = t2 + time.time()

            pose_optimizer.zero_grad()
            t3+=time.time()
            #loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint)
            err = get_patch_loss(prev_patch, curr_patch)
            loss_tracking = err.mean()

            t4 = t4+time.time()
            loss_tracking.backward()

            t5 = t5+time.time()
            t_n = t_n+1
            with torch.no_grad():
                pose_optimizer.step()
                converged, prev_tau = update_pose(viewpoint, tau_prev=prev_tau)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )

            if converged:
                break
        print("tracking processig time", cur_frame_idx, t_n, (t2 - t1), 'proj', (t3 - t2), 'loss', (t4 - t3),
              'backward', (t5 - t4), self.gaussians._xyz.size()[0])

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def tracking_only_feature(self, cur_frame_idx, prev_frame_idx, viewpoint):
        prev = self.cameras[prev_frame_idx]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t_n = 0

        curr_frame = self.frames[cur_frame_idx]

        frame_gaussians, frame_gaussian_indices,_ = curr_frame.get_frame_gaussians(self.gaussians)
        frame_inliers = torch.ones(frame_gaussians.size()[0], device='cuda', dtype=bool)
        #frame_points = torch.from_numpy(curr_frame.keypoints).cuda()[frame_gaussian_indices]
        frame_points = curr_frame.keypoints[frame_gaussian_indices]

        projection, valid_projection = curr_frame.project_points(frame_gaussians, viewpoint.R, viewpoint.T,
                                                                 viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                 viewpoint.cy,
                                                                 viewpoint.image_width, viewpoint.image_height,
                                                                 viewpoint.cam_rot_delta,
                                                                 viewpoint.cam_trans_delta,
                                                                 )
        tmp_proj = torch.from_numpy(valid_projection).cuda()
        frame_inliers = torch.logical_and(frame_inliers, tmp_proj)
        points = frame_points[tmp_proj]

        curr_frame.update_outlier_points(projection, points, frame_inliers, th_radius=100.0)

        temp_opt_inliers = frame_inliers.clone()

        for tracking_itr in range(40):
            t1 += time.time()

            t2 = t2 + time.time()

            t3 = t3 + time.time()

            tmp_gausisans = frame_gaussians[temp_opt_inliers]
            tmp_points = frame_points[temp_opt_inliers]
            projection, valid_projection = curr_frame.project_points(tmp_gausisans, viewpoint.R, viewpoint.T,
                 viewpoint.fx, viewpoint.fy, viewpoint.cx,
                 viewpoint.cy,
                 viewpoint.image_width, viewpoint.image_height,
                 viewpoint.cam_rot_delta,
                 viewpoint.cam_trans_delta,
            )
            tmp_inliers = torch.from_numpy(valid_projection).cuda()
            tmp_points = tmp_points[tmp_inliers]

            pose_optimizer.zero_grad()

            #loss_reprojection = criterion(projection, tmp_points)
            loss_reprojection = get_reprojection_loss(projection, tmp_points).mean()*0.05
            # loss_tracking += loss_reprojection

            t4 = t4 + time.time()
            loss_reprojection.backward()
            t5 = t5 + time.time()
            t_n = t_n + 1
            with torch.no_grad():
                pose_optimizer.step()
                #converged = curr_frame.update_pose(viewpoint)
                converged = update_pose(viewpoint)

            if tracking_itr > 0 and tracking_itr % 10 == 0:
                ##inlier 체크

                temp_opt_inliers = frame_inliers.clone()
                tmp_gausisans = frame_gaussians[temp_opt_inliers]
                tmp_points = frame_points[temp_opt_inliers]

                projection, valid_projection = curr_frame.project_points(tmp_gausisans, viewpoint.R, viewpoint.T,
                                                                         viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                         viewpoint.cy,
                                                                         viewpoint.image_width, viewpoint.image_height,
                                                                         viewpoint.cam_rot_delta,
                                                                         viewpoint.cam_trans_delta,
                                                                         )
                tmp_inliers = torch.from_numpy(valid_projection).cuda()
                tmp_points = tmp_points[tmp_inliers]
                curr_frame.update_outlier_points(projection, tmp_points, temp_opt_inliers, th_radius=9.0)
            if tracking_itr % 10 == 0:

                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )

            if converged:
                break

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )

        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )

        # 최종 가우시안 업데이트
        curr_frame.update_frame_gaussianpoints(frame_gaussian_indices, frame_inliers)
        print("tracking processig time", cur_frame_idx, t_n, (t2 - t1), 'proj', (t3 - t2), 'loss', (t4 - t3),
              'backward', (t5 - t4), self.gaussians._xyz.size()[0])
        self.median_depth = get_median_depth(depth, opacity)

        return render_pkg

    def tracking2(self, cur_frame_idx, prev_frame_idx, viewpoint):

        prev = self.cameras[prev_frame_idx]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t_n = 0

        curr_frame = self.frames[cur_frame_idx]

        frame_gaussians, frame_gaussian_indices, global_gaussian_indices = curr_frame.get_frame_gaussians(self.gaussians)
        frame_inliers = torch.ones(frame_gaussians.size()[0], device='cuda', dtype=bool)
        #frame_points = torch.from_numpy(curr_frame.keypoints).cuda()[frame_gaussian_indices]
        frame_points = curr_frame.keypoints[frame_gaussian_indices]

        projection, valid_projection = curr_frame.project_points(frame_gaussians, viewpoint.R, viewpoint.T,
                 viewpoint.fx, viewpoint.fy, viewpoint.cx,
                 viewpoint.cy,
                 viewpoint.image_width, viewpoint.image_height,
                 viewpoint.cam_rot_delta,
                 viewpoint.cam_trans_delta,
        )
        tmp_proj = torch.from_numpy(valid_projection).cuda()
        frame_inliers = torch.logical_and(frame_inliers, tmp_proj)
        points = frame_points[tmp_proj]

        curr_frame.update_outlier_points(projection, points, frame_inliers, th_radius=100.0)

        temp_opt_inliers = frame_inliers.clone()
        global_gaussian_indices = global_gaussian_indices[temp_opt_inliers]
        frame_test_gaussians = self.gaussians.clone(global_gaussian_indices)

        for tracking_itr in range(self.tracking_itr_num):
            t1 = t1+time.time()

            tmp_gausisans = frame_gaussians[temp_opt_inliers]
            tmp_points = frame_points[temp_opt_inliers]

            render_pkg = render(
                viewpoint, frame_test_gaussians, self.pipeline_params, self.background
            )

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            t2 = t2 + time.time()

            projection, valid_projection = curr_frame.project_points(tmp_gausisans, viewpoint.R, viewpoint.T,
                                                                     viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                     viewpoint.cy,
                                                                     viewpoint.image_width, viewpoint.image_height,
                                                                     viewpoint.cam_rot_delta,
                                                                     viewpoint.cam_trans_delta,
                                                                     )
            tmp_inliers = torch.from_numpy(valid_projection).cuda()
            tmp_points = tmp_points[tmp_inliers]

            pose_optimizer.zero_grad()
            t3 += time.time()
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint)
            loss_tracking += get_reprojection_loss(projection, tmp_points).mean() * 0.05

            t4 = t4+time.time()
            loss_tracking.backward()
            if self.gaussians.get_xyz.grad is not None:
                after_grad_loss_ba = self.gaussians.get_xyz.grad.clone()
                affected_by_ba = torch.any(after_grad_loss_ba != 0, dim=1)
                print("Gaussians affected by tracking:", torch.count_nonzero(affected_by_ba), self.gaussians._xyz.shape, affected_by_ba.nonzero().flatten())
            else:
                print('not affected by tracking')
            t5 = t5+time.time()
            t_n = t_n+1
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )

            if converged:
                break

        print("tracking processig time", cur_frame_idx, t_n,(t2-t1), 'proj',(t3-t2), 'loss', (t4-t3), 'backward', (t5-t4), self.gaussians._xyz.size()[0])
        self.median_depth = get_median_depth(depth, opacity)

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )

        return render_pkg

    def tracking3(self, cur_frame_idx, prev_frame_idx, viewpoint, colCam, col_param, matches_frame = None):

        matches_frame = torch.from_numpy(matches_frame).cuda().int()

        prev = self.cameras[prev_frame_idx]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t_n = 0

        ##depth 정렬 테스트
        with torch.no_grad():
            prev_frame = self.frames[prev_frame_idx]
            cur_frame = self.frames[cur_frame_idx]

            #가우시안 프로젝션
            projections, depths, valid_proj = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                                  viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                  viewpoint.cy,
                                                                  viewpoint.image_width, viewpoint.image_height)

            prev_keypoints = prev_frame.keypoints.detach()
            feature_projections = projections[valid_proj & self.gaussians.isfeatured]
            matches_gaussian = get_correspondences_within_threshold(feature_projections, prev_keypoints)

            mask_gaussian = torch.zeros(prev_keypoints.shape[0], device= 'cuda', dtype = bool)
            mask_gaussian[matches_gaussian[:,1]] = True
            gaussian_idx = -torch.ones(prev_keypoints.shape[0], device ='cuda', dtype = torch.int32)
            gaussian_idx[matches_gaussian[:,1]] = matches_gaussian[:,0]
            prev_kp_idx = -torch.ones(prev_keypoints.shape[0], device ='cuda', dtype = torch.int32)
            prev_kp_idx[matches_gaussian[:, 1]] = matches_gaussian[:, 1]

            mask_frame = torch.zeros(prev_keypoints.shape[0], device='cuda', dtype=bool)
            mask_frame[matches_frame[:, 0]] = True
            curr_kp_idx = -torch.ones(prev_keypoints.shape[0], device='cuda', dtype=torch.int32)
            curr_kp_idx[matches_frame[:, 0]] = matches_frame[:, 1]

            mask = torch.logical_and(mask_frame, mask_gaussian)
            curr_kp = cur_frame.keypoints[curr_kp_idx[mask]]
            feature_gaussians = self.gaussians.get_xyz[valid_proj & self.gaussians.isfeatured][gaussian_idx[mask]]

            matched = torch.stack((prev_kp_idx[mask], curr_kp_idx[mask]), dim=1).int().cpu().numpy()
            #out = self.testManager.tracker.visualize(prev_frame.color, cur_frame.color, matched, prev_frame.keypoints.cpu().numpy(), cur_frame.keypoints.cpu().numpy())
            image_np = (
                viewpoint.original_image
                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    .cpu()  # GPU → CPU
                    .numpy()  # NumPy 배열로 변환
            )
            image_np = (image_np * 255.0).astype(np.uint8)
            out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            points2 = feature_projections[gaussian_idx[mask]].cpu().numpy()#prev_keypoints[prev_kp_idx[mask]].cpu().numpy()
            points3 = cur_frame.keypoints[curr_kp_idx[mask]].cpu().numpy()
            for pt1, pt2 in zip(points2, points3):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(out, p1, p2, (0, 255, 0), 2, lineType=16)
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
                cv2.circle(out, p2, 1, (255, 0, 0), -1, lineType=16)
            cv2.imwrite('./res/test_ba/tracking_' + str(cur_frame_idx) + '.jpg', out)

            ####COLMAP
            """"""
            cam_from_world = pycolmap.Rigid3d(viewpoint.R.cpu().numpy(), viewpoint.T.cpu().numpy())
            inlier_mask = np.ones(curr_kp.shape[0], dtype=bool)
            result = pycolmap.refine_absolute_pose(cam_from_world, curr_kp.cpu().numpy(), feature_gaussians.cpu().numpy(), inlier_mask, colCam, refinement_options=dict(refine_focal_length=True))
            """
            result = pycolmap.estimate_absolute_pose(
                curr_kp.double().cpu().numpy(), feature_gaussians.double().cpu().numpy(),colCam, col_param
            )
            """
            #if result['success']:
            if result is not None:
                rigid = result['cam_from_world']
                R = torch.from_numpy(rigid.rotation.matrix()).cuda()
                t = torch.from_numpy(rigid.translation).cuda()
            else:
                R = prev.R
                t = prev.T

            ####COLMAP

            print('result=solvePnP', torch.count_nonzero(mask_frame&mask_gaussian), result)

            #points2 = points2[match_res[match_mask]].cpu().numpy()
            #points3 = points3[match_mask].cpu().numpy()

            match_idx = find_correspondence_with_dist(projections, prev_keypoints, th=7)  # 약간 시간이 걸림. 0.01 이하
            valid_match = (match_idx > -1)  # & self.gaussians.isfeatured[valid]
            valid_idx = torch.where(valid_match & valid_proj)[0]  # 유효한 이미지 안의 프로젝션 검출. 매치 인덱스 값이 들어갈 곳.
            feature_mask = calculate_feature_mask(cur_frame.keypoints, viewpoint.image_width, viewpoint.image_height,
                                                  max_radius=5)
            viewpoint.update_RT(R, t)

        for tracking_itr in range(self.tracking_itr_num):
            t1 = t1+time.time()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, mask = valid_idx
            )

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            t2 = t2 + time.time()

            t3+=time.time()
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint, feature_mask=feature_mask)

            t4 = t4+time.time()
            pose_optimizer.zero_grad()
            loss_tracking.backward()
            t5 = t5+time.time()
            t_n = t_n+1
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            #gaussians=(gaussians),
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )

            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)

        with torch.no_grad():
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            curr_visibility = (render_pkg["n_touched"] > 0).long()

        print("tracking processig time", cur_frame_idx, 'obj', len(cur_frame.objects), len(cur_frame.contours), t_n,
              'render', (t2 - t1), 'proj', (t3 - t2), 'loss', (t4 - t3), 'backward', (t5 - t4),
              self.gaussians._xyz.size()[0], )

        return render_pkg


    def tracking(self, cur_frame_idx, prev_frame_idx, viewpoint, matches = None):


        prev = self.cameras[prev_frame_idx]
        viewpoint.update_RT(prev.R, prev.T)
        """
        curr_frame = self.frames[cur_frame_idx]
        T = torch.eye(4, device='cuda')
        R = curr_frame.T[:3,:3]
        t = curr_frame.T[:3, 3]
        viewpoint.update_RT(R, t)
        """

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t_n = 0

        ##depth 정렬 테스트
        with torch.no_grad():
            projections, depths, valid_proj = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                             viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                             viewpoint.image_width, viewpoint.image_height)

            prev_frame = self.frames[prev_frame_idx]
            cur_frame = self.frames[cur_frame_idx]
            #projections = (projections[valid])  # 유효한 프로젝션 결과를 int화 해서 픽셀로 만듬. 정렬하면
            keypoints = (prev_frame.keypoints)
            match_idx = find_correspondence_with_dist(projections, keypoints, th=7)  # 약간 시간이 걸림. 0.01 이하

            valid_match = (match_idx > -1)# & self.gaussians.isfeatured[valid]
            valid_idx = torch.where(valid_match & valid_proj)[0] #유효한 이미지 안의 프로젝션 검출. 매치 인덱스 값이 들어갈 곳.
            #valid_idx = valid_idx[valid_match] #매치 값



            #gaussians = self.gaussians.clone(valid_idx) #self.gaussians.get_xyz[valid][valid_match]
            a = time.time()
            feature_mask=calculate_feature_mask(cur_frame.keypoints, viewpoint.image_width, viewpoint.image_height, max_radius=5)
            b = time.time()
            #print('tracking matching test', b-a, torch.count_nonzero(valid_idx), self.gaussians.get_xyz.shape[0],
            #      gaussians._xyz.shape[0])

        for tracking_itr in range(self.tracking_itr_num):
            t1 = t1+time.time()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, mask = valid_idx
            )

            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            t2 = t2 + time.time()

            t3+=time.time()
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint, feature_mask=feature_mask)

            t4 = t4+time.time()
            pose_optimizer.zero_grad()
            loss_tracking.backward()
            t5 = t5+time.time()
            t_n = t_n+1
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            #gaussians=(gaussians),
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        )
                    )
                )

            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)

        ##depth 정렬 테스트
        """
        t_depth_start = time.time()
        projections, depths, valid = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                         viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                         viewpoint.image_width, viewpoint.image_height)

        projections = (projections[valid])  # 유효한 프로젝션 결과를 int화 해서 픽셀로 만듬. 정렬하면
        depths = depths[valid]
        sorted_indices = torch.argsort(depths)

        prev_frame = self.frames[cur_frame_idx]
        keypoints = (prev_frame.keypoints)
        match_idx = find_correspondence_with_dist(projections, keypoints, th = 10)  # 약간 시간이 걸림. 0.01 이하

        valid_match = (match_idx > -1) & self.gaussians.isfeatured[valid]
        gaussians = self.gaussians.get_xyz[valid][valid_match]

        t_depth_end = time.time()
        print('depth sort test', t_depth_end - t_depth_start, 'match=',torch.count_nonzero(match_idx > -1),
              torch.count_nonzero((match_idx > -1) & self.gaussians.isfeatured[valid]), sorted_indices.shape,
              torch.count_nonzero(self.gaussians.isfeatured),self.gaussians.get_xyz.shape[0])  # 0.01보다 작음. 거의 0.002정도인데 커질수록 많을 듯
        """
        ##depth 정렬 테스트
        with torch.no_grad():
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            curr_visibility = (render_pkg["n_touched"] > 0).long()

            ##visualize test
            """"""
            image_np = (
                viewpoint.original_image
                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    .cpu()  # GPU → CPU
                    .numpy()  # NumPy 배열로 변환
            )
            image_np = (image_np * 255.0).astype(np.uint8)
            out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            """
            projections, depths, valid_proj = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                                  viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                  viewpoint.cy,
                                                                  viewpoint.image_width, viewpoint.image_height)
            """
            ##weight = failed
            """
            t1 = time.time()
            cov2d = self.gaussians.convert_cov3d_to_cov2d(viewpoint.R, viewpoint.T, viewpoint.fx, viewpoint.fy)
            t2 = time.time()
            ##weight
            #weight = self.gaussians.compute_2d_gaussian_weights(projections[valid_proj], cov2d[valid_proj], self.gaussians.get_opacity[valid_proj], viewpoint.image_width, viewpoint.image_height)
            t3 = time.time()
            print('test = weight', t2-t1, t3-t2, projections.shape, cov2d.shape)
            """
            ##weight = failed

            ##error test
            #err = get_loss_gaussian(self.config, image, viewpoint, projections).squeeze(1)
            #valid_err = err > 0.8
            ##error test

            """
            points1 = projections[valid_proj].detach().cpu().numpy()
            points2 = cur_frame.keypoints.detach().cpu().numpy()

            test_mask = valid_match & valid_proj
            points3 = projections[test_mask].detach().cpu().numpy()
            points4 = projections[ valid_proj & ~valid_match & valid_err].detach().cpu().numpy()

            #가우시안 : 일반
            for pt1 in points1:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 3, (0, 0, 255), 1, lineType=16)
            #가우시안 : 특징점
            for pt1 in points3:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 2, (255, 0, 0), 1, lineType=16)
            #피쳐
            for pt1 in points2:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 255, 255), -1, lineType=16)
                cv2.circle(out, p1, 5, (0, 255, 255), 1, lineType=16)
            #가우시안 : 에러
            for pt1 in points4:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (255, 255, 0), -1, lineType=16)
            #cv2.imshow("asdfasdfasdf", out)
            #cv2.waitKey(10)
            cv2.imwrite('./res/test_tracking/' + str(cur_frame_idx) + '.jpg', out)
            """
            ##visualize test

            ##visualize test - feature
            projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz,
                                                                  viewpoint.R,
                                                                  viewpoint.T,
                                                                  viewpoint.fx, viewpoint.fy,
                                                                  viewpoint.cx, viewpoint.cy,
                                                                  viewpoint.image_width,
                                                                  viewpoint.image_height)

            points2 = cur_frame.keypoints.detach()
            points3 = projection[valid_projection & self.gaussians.isfeatured]
            match_res = find_correspondence_with_dist(points3, points2)
            match_mask = match_res > -1
            points2 =points2[match_res[match_mask]].cpu().numpy()
            points3 = points3[match_mask].cpu().numpy()

            for pt1, pt2 in zip(points2, points3):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(out, p1, p2, (0, 255, 0), 2, lineType=16)
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
                cv2.circle(out, p2, 1, (255, 0, 0), -1, lineType=16)

            #points1 = projection[valid_projection].detach().cpu().numpy()
            #points2 = cur_frame.keypoints.detach()
            #points3 = projection[valid_projection & self.gaussians.isfeatured]
            #points4 = projection[valid_projection & ~curr_visibility].detach().cpu().numpy()

            #points2 = cur_frame.keypoints.detach().cpu().numpy()
            #points3 = projection[valid_projection & self.gaussians.isfeatured].detach().cpu().numpy()
            #일반 가우시안, 피쳐, 피쳐 가우시안 순으로 파, 빨, 녹

            """
            for pt1 in points1:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (255, 0, 0), 1, lineType=16)
            for pt1 in points2:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 2, (0, 0, 255), 1, lineType=16)
            for pt1 in points3:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 255, 0), 1, lineType=16)
            for pt1 in points4:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 3, (0, 255, 255), 1, lineType=16)
            """
            cv2.imwrite('./res/test_ba/tracking_' + str(cur_frame_idx) + '.jpg', out)
            ##visualize test - feature

            print("tracking processig time", cur_frame_idx, 'obj', len(cur_frame.objects), len(cur_frame.contours), t_n, 'match', torch.count_nonzero(match_res > -1), points2.shape,
                  'render', (t2 - t1), 'proj', (t3 - t2), 'loss', (t4 - t3), 'backward', (t5 - t4),
                  self.gaussians._xyz.size()[0], )

        return render_pkg

    def run(self):
        cur_frame_idx = 0
        prev_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        K = np.array([[self.dataset.fx, 0, self.dataset.cx],
                      [0, self.dataset.fy, self.dataset.cy],
                      [0, 0, 1]], dtype=np.float32)
        D = self.dataset.dist_coeffs

        ##XFeat
        top_k = 4096
        xfeat = XFeat(
            weights='../accelerated_features/weights/xfeat.pt',  # -lighterglue
            top_k=top_k,
            detection_threshold=0.05
        ).eval().cuda()

        ##colmap
        colCam = pycolmap.Camera(
            model="OPENCV",  # 또는 "SIMPLE_RADIAL", "SIMPLE_PINHOLE" 등
            width=self.dataset.width,
            height=self.dataset.height,
            params=[
                K[0][0],  # fx
                K[1][1],  # fy
                K[0][2],  # cx
                K[1][2],  # cy
                *D[:4],  # k1, k2, p1, p2 (OPENCV 모델 기준, 필요시 k3 등 추가)
            ]
        )

        col_param = pycolmap.AbsolutePoseEstimationOptions()
        col_param.estimate_focal_length = False
        col_param.ransac.max_error = 4.0
        col_param.ransac.min_inlier_ratio = 0.2
        col_param.ransac.min_num_trials = 500
        col_param.ransac.max_num_trials = 5000
        col_param.ransac.confidence = 0.99
        ##colmap

        ##solve pnp
        reprojection_threshold = np.float32(9.0)
        confidence = 0.99
        max_iterations = 1000
        ##solve pnp

        #gtsam 설정
        gtsam_prior_noise = gtsam.noiseModel.Diagonal.Variances(np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002]))
        gtsam_closure_noise = gtsam.noiseModel.Diagonal.Variances(np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02]))

        gtsam_finalOptResult = gtsam.Values()
        gtsam_graph = gtsam.NonlinearFactorGraph()
        gtsam_initial_estimates = gtsam.Values()
        """
        gtsamCam = gtsam.Cal3DS2(self.dataset.fx, self.dataset.fy, 0.0, self.dataset.cx, self.dataset.cy,
                                 self.dataset.dist_coeffs[0], self.dataset.dist_coeffs[1],
                                 self.dataset.dist_coeffs[2],self.dataset.dist_coeffs[3])
        """
        """
        gtsam_params = gtsam.ISAM2Params()
        gtsam_params.setFactorization("QR")  # QR 분해 사용
        gtsam_params.setRelinearizeThreshold(0.01)
        isam2 = gtsam.ISAM2(gtsam_params)
        gtsamCam = gtsam.Cal3_S2(self.dataset.fx, self.dataset.fy, 0.0, self.dataset.cx, self.dataset.cy)
        gtsamPointNoise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        gtsamPoseNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1] * 6))  # Pose3: 6D
        current_estimate = None
        """
        # gtsam 설정

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                #print("request kf", self.requested_keyframe)
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if (not self.initialized or not self.tracking_mode or self.tracking_mode_vo) and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                frame_start_time = time.time()
                cur_frame_idx = self.edge_queue.get()

                viewpoint = init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                curr_frame = self.dataset[str(cur_frame_idx)]
                self.frames[cur_frame_idx] = curr_frame

                #검출
                kp_time_start = time.time()
                with torch.inference_mode():
                    #ALIKE
                    #pred = self.testManager.feature_model.run(curr_frame.color)
                    #XFeat
                    pred = xfeat.detectAndCompute(curr_frame.color)[0]
                    keypoints = pred['keypoints'].cpu().numpy()
                    curr_frame.descriptors = pred['descriptors'].cpu().numpy()
                    del pred

                #curr_frame.inliers = np.zeros((keypoints.shape[0], 1), dtype=np.bool)

                kp_time_temp = time.time()

                points_reshaped = keypoints.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(points_reshaped, K, self.dataset.dist_coeffs, None, K)
                curr_frame.keypoints = torch.from_numpy(undistorted.reshape(-1, 2)).cuda()
                del undistorted

                kp_time_end = time.time()
                #print(frame.keypoints)

                ##contour 왜곡 보정

                contours_undistorted = []
                contours_np = [np.array(cnt, dtype=np.int32).reshape((-1, 1, 2)) for cnt in curr_frame.contours]
                for cnt in contours_np:
                    pts = cnt.astype(np.float32)
                    undistorted_pts = cv2.undistortPoints(pts, K, self.dataset.dist_coeffs, None, K)
                    undistorted_pts = undistorted_pts.astype(np.int32)  # 정수형으로 변환
                    contours_undistorted.append(undistorted_pts)
                curr_frame.contours = contours_undistorted

                #mask_bool = np.zeros((viewpoint.image_height, viewpoint.image_width), dtype=np.uint8)
                #cv2.drawContours(mask_bool, contours_undistorted, -1, color=255, thickness=cv2.FILLED)
                #curr_frame.contours_mask = mask_bool.astype(bool)

                ##contour 왜곡 보정

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)

                    #gtsam initialization
                    """
                    gtsamGraph = gtsam.NonlinearFactorGraph()
                    gtsamInitial = gtsam.Values()
                    pose_key = cur_frame_idx

                    R = viewpoint.R.clone().detach().cpu().numpy()
                    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w] 순서
                    w, x, y, z = quat[3], quat[0], quat[1], quat[2]
                    rot = gtsam.Rot3.Quaternion(w, x, y, z)
                    t = viewpoint.T.clone().detach().cpu().numpy()
                    trans = gtsam.Point3(*t)

                    initial_pose = gtsam.Pose3(rot, trans)
                    gtsamGraph.add(gtsam.PriorFactorPose3(pose_key, initial_pose, gtsamPoseNoise))
                    gtsamInitial.insert(pose_key, initial_pose)

                    # iSAM2 초기 업데이트
                    isam2.update(gtsamGraph, gtsamInitial)
                    current_estimate = isam2.calculateEstimate()
                    #gtsam init
                    """
                    prev_frame_idx = cur_frame_idx
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                        len(self.current_window) == self.window_size
                )

                #if self.initialized :
                #    print("tracking ", cur_frame_idx, self.requested_keyframe, len(self.edge_queue))

                # Tracking
                s = time.time()

                if self.tracking_mode:
                    #matching with prev frame
                    match_time_start = time.time()

                    prev_frame = self.frames[(prev_frame_idx)]
                    cur_matches = self.testManager.tracker.match(prev_frame.descriptors,curr_frame.descriptors)

                    """
                    prev_frame = self.frames[(prev_frame_idx)]
                    curr_index = torch.where((prev_frame.gaussianpoints > -1))[0].cpu().numpy() #(curr_frame.gaussianpoints > -1).numpy()#
                    cur_matches = self.testManager.tracker.match(prev_frame.descriptors[curr_index], curr_frame.descriptors)
                    cur_matches[:,0] = curr_index[cur_matches[:,0]]
                    curr_frame.copy_gaussians_from_frame_matches(prev_frame, self.gaussians, cur_matches)

                    #matching with last keyframe
                    last_keyframe = self.frames[self.current_window[0]]
                    kf_index = torch.where((last_keyframe.gaussianpoints > -1))[0].cpu().numpy()
                    kf_matches = self.testManager.tracker.match(last_keyframe.descriptors[kf_index], curr_frame.descriptors)
                    kf_matches[:, 0] = kf_index[kf_matches[:, 0]]
                    curr_frame.copy_gaussians_from_frame_matches(last_keyframe, self.gaussians, kf_matches)
                    """
                    match_time_end = time.time()

                    """
                    if len(self.current_window) > 2:
                        render_pkg = self.tracking_with_pc(cur_frame_idx, prev_frame_idx, viewpoint, colCam, col_param, cur_matches)
                    else:
                    """
                    render_pkg = self.tracking3(cur_frame_idx, prev_frame_idx, viewpoint, colCam, col_param, matches_frame=cur_matches)

                    frame_end_time = time.time()

                else:
                    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    image, depth, opacity = (
                        render_pkg["render"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                    )
                    self.median_depth = get_median_depth(depth, opacity)

                ##test
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    move_gaussianpacket_to_cpu(
                        gui_utils.GaussianPacket(
                            gaussians=(self.gaussians),
                            current_frame=viewpoint,
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    prev_frame_idx = cur_frame_idx
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                            check_time
                            and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )

                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                prev_frame_idx = cur_frame_idx
                cur_frame_idx += 1

                """
                if (
                        self.save_results
                        and self.save_trj
                        and create_kf
                        and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                """
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:

                data = self.frontend_queue.get()

                if data[0] == "sync_backend":
                    self.sync_backend(data, prev_frame_idx=prev_frame_idx)

                elif data[0] == "keyframe":
                    self.sync_backend(data, prev_frame_idx=prev_frame_idx)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break