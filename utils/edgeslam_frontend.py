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
from edge_assisted.gaussian_feature import project_pc_to_pixel
from edge_assisted.slam_utils import get_loss_tracking, get_reprojection_loss, get_reprojection_loss2
#from edge_assisted.gaussian_feature import GaussianPointManager

from typing import Dict, Tuple, Optional, List
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

class EdgeFrontEnd(WinFrontEnd):
    def __init__(self, config):
        super().__init__(config)

        self.frames = {}

        self.edge_queue = None
        self.tracking_mode = None

        #self.testManager = GaussianPointManager()
        self.testManager = None

    """"""
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        frame = self.frames[cur_frame_idx]
        f = [frame.keypoints.cpu().clone(), frame.descriptors]
        msg = ["init", cur_frame_idx, move_camera_to_cpu(viewpoint), depth_map, f]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        frame = self.frames[cur_frame_idx]
        f = [frame.keypoints.cpu().clone(), frame.descriptors]
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

        self.gaussians = gaussians
        self.gaussians._xyz = self.gaussians._xyz.detach().requires_grad_(False)
        self.gaussians._features_dc = self.gaussians._features_dc.detach().requires_grad_(False)
        self.gaussians._features_rest = self.gaussians._features_rest.detach().requires_grad_(False)
        self.gaussians._opacity = self.gaussians._opacity.detach().requires_grad_(False)
        self.gaussians._scaling = self.gaussians._scaling.detach().requires_grad_(False)
        self.gaussians._rotation = self.gaussians._rotation.detach().requires_grad_(False)

        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone().to(self.device), kf_T.clone().to(self.device))

        #update frame gaussianpoints
        prune_dict = data[4]

        if prune_dict is not None and prev_frame_idx is not None:
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

    def render_for_tracking(self, viewpoint, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Specialized render function for tracking with frozen Gaussians

        This function explicitly freezes all Gaussian parameters and only allows
        gradients to flow to camera pose parameters.
        """

        # Get current pose (this WILL have gradients)
        #viewpoint.R, viewpoint.T를 4x4로 변환
        #pose = self.get_camera_pose()
        viewmats, Ks = self.create_camera_matrices(K, viewpoint.R, viewpoint.T)

        # Freeze ALL Gaussian parameters using context manager
        with torch.no_grad():
            # Prepare frozen Gaussian parameters
            frozen_means = self.gaussians._xyz.clone()
            frozen_quats = torch.nn.functional.normalize(self.gaussians._rotation.clone(), dim=-1)
            frozen_scales = torch.exp(self.gaussians._scaling.clone())
            frozen_opacities = torch.sigmoid(self.gaussians._opacity.clone().squeeze())
            frozen_colors = self.gaussians._features_dc.clone()

        # Re-enable gradients only for the frozen tensors we want to use
        # (This is a more explicit way to ensure no gradients flow to Gaussians)
        frozen_means = frozen_means.detach().requires_grad_(False)
        frozen_quats = frozen_quats.detach().requires_grad_(False)
        frozen_scales = frozen_scales.detach().requires_grad_(False)
        frozen_opacities = frozen_opacities.detach().requires_grad_(False)
        frozen_colors = frozen_colors.detach().requires_grad_(False)

        print(frozen_opacities.size())

        # Render with gsplat - only pose gradients will be computed
        rendered_colors, rendered_alphas, info = rasterization(
            means=frozen_means,
            quats=frozen_quats,
            scales=frozen_scales,
            opacities=frozen_opacities,
            colors=frozen_colors,
            viewmats=viewmats,  # This WILL have gradients for pose optimization
            Ks=Ks,
            width=viewpoint.image_width,
            height=viewpoint.image_height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.3,
            packed=False,
            sparse_grad=False,  # No need for sparse grads during tracking
            absgrad=False,  # No need for absgrad during tracking
            render_mode="RGB",  # Only need RGB for tracking
            sh_degree=0
        )

        rendered_image = rendered_colors[0, :, :, :]  # [H, W, 3]
        rendered_alpha = rendered_alphas[0]  # [H, W]

        return {
            'image': rendered_image,
            'alpha': rendered_alpha,
            'info': info
        }

    def render(self, K: torch.Tensor, target_pose: Optional[torch.Tensor] = None,
               freeze_gaussians: bool = False) -> Dict[str, torch.Tensor]:
        """Render current scene from camera viewpoint

        Args:
            K: Camera intrinsics
            target_pose: Optional target pose, uses current pose if None
            freeze_gaussians: If True, detach gaussian parameters from gradient computation
        """

        # Use target pose if provided, otherwise use current estimated pose
        if target_pose is not None:
            pose = target_pose
        else:
            pose = self.get_camera_pose()

        # Create camera matrices
        viewmats, Ks = self.create_camera_matrices(K, pose)

        # Prepare Gaussian parameters
        if freeze_gaussians:
            # Detach gaussian parameters - no gradients will flow to them
            means = self.means.detach()
            quats = torch.nn.functional.normalize(self.quats.detach(), dim=-1)
            scales = torch.exp(self.scales.detach())  # Convert from log space
            opacities = torch.sigmoid(self.opacities.detach())  # Convert from logit space
            colors = self.colors.detach()  # SH coefficients
        else:
            # Normal rendering with gradients
            means = self.means
            quats = torch.nn.functional.normalize(self.quats, dim=-1)
            scales = torch.exp(self.scales)  # Convert from log space
            opacities = torch.sigmoid(self.opacities)  # Convert from logit space
            colors = self.colors  # SH coefficients

        # Render with gsplat
        rendered_colors, rendered_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.config.image_width,
            height=self.config.image_height,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            radius_clip=self.config.radius_clip,
            packed=self.config.use_packed,
            sparse_grad=self.config.use_sparse_grad,
            absgrad=self.config.use_absgrad,
            render_mode="RGB+D",  # Render both color and depth
            sh_degree=0  # Use 3 SH bands, 2
        )

        # Extract rendered image and depth
        rendered_image = rendered_colors[0, :, :, :3]  # [H, W, 3]
        rendered_depth = rendered_colors[0, :, :, 3]  # [H, W]
        rendered_alpha = rendered_alphas[0]  # [H, W]

        return {
            'image': rendered_image,
            'depth': rendered_depth,
            'alpha': rendered_alpha,
            'info': info
        }

    def tracking_with_patch(self, cur_frrame_idx, prev_frame_idx, viewpoint):
        pass

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

    def tracking(self, cur_frame_idx, prev_frame_idx, viewpoint):

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

            t2 = t2 + time.time()

            pose_optimizer.zero_grad()
            t3+=time.time()
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint)

            t4 = t4+time.time()
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
        col_param.ransac.max_error = 9.0
        col_param.ransac.min_inlier_ratio = 0.1
        col_param.ransac.max_num_trials = 1000
        col_param.ransac.confidence = 0.99

        ##solve pnp
        reprojection_threshold = np.float32(9.0)
        confidence = 0.99
        max_iterations = 1000
        ##solve pnp

        #gtsam 설정
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

                if (not self.initialized or not self.tracking_mode) and self.requested_keyframe > 0:
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
                pred = self.testManager.feature_model.run(curr_frame.color)
                keypoints = pred['keypoints']
                curr_frame.descriptors = pred['descriptors']
                #curr_frame.inliers = np.zeros((keypoints.shape[0], 1), dtype=np.bool)

                kp_time_temp = time.time()

                points_reshaped = keypoints.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(points_reshaped, K, self.dataset.dist_coeffs, None, K)
                curr_frame.keypoints = torch.from_numpy(undistorted.reshape(-1, 2)).cuda()

                kp_time_end = time.time()
                #print(frame.keypoints)

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
                    print('frontend::observation', self.gaussians.observation_indices.shape, self.gaussians.observation_points.shape, self.current_window)
                    #matching with prev frame
                    match_time_start = time.time()
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

                    render_pkg = self.tracking(cur_frame_idx, prev_frame_idx, viewpoint)

                    #self.render_for_tracking(viewpoint, K)

                    #아웃 라이어 제거
                    #가우시안 포인트 시각화

                    #매칭 앤 트래킹 테스트
                    #out = self.testManager.tracker.visualize(prev_frame.color, curr_frame.color, cur_matches, prev_frame.keypoints, curr_frame.keypoints)
                    #cv2.imshow("match", out)
                    #cv2.waitKey(1)

                    """
                    #시각화
                    #outlier removal
                    prev = self.cameras[prev_frame_idx]
                    viewpoint.update_RT(prev.R, prev.T)

                    prevR = prev.R.clone().detach().cpu().numpy()
                    prev_rvec,_=cv2.Rodrigues(prevR)
                    prev_tvec = prev.T.clone().detach().cpu().numpy()

                    projection, points, gaussians, _ = curr_frame.update_gaussianpoints(self.gaussians, viewpoint.R, viewpoint.T,
                       viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                       viewpoint.image_width, viewpoint.image_height, th_radius=100.0
                    )
                    gtsam_t1 = time.time()
                    #gaussians = gaussians.detach().cpu().numpy()
                    #points = points.cpu().numpy()
                    """
                    """
                    gtsamGraph = gtsam.NonlinearFactorGraph()
                    gtsamInitial = gtsam.Values()
                    pose_key = cur_frame_idx

                    prevview = self.cameras[prev_frame_idx]
                    R = prevview.R.clone().detach().cpu().numpy()
                    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w] 순서
                    w, x, y, z = quat[3], quat[0], quat[1], quat[2]
                    rot = gtsam.Rot3.Quaternion(w,x,y,z)
                    t = prevview.T.clone().detach().cpu().numpy()
                    trans = gtsam.Point3(*t)

                    initial_pose = gtsam.Pose3(rot, trans)
                    print('pose', initial_pose)
                    #gtsamGraph.add(gtsam.PriorFactorPose3(pose_key, initial_pose, gtsamPoseNoise))
                    gtsamInitial.insert(pose_key, initial_pose)

                    gtsam_t_temp1 = time.time()
                    for i in range(len(gaussians)):
                        landmark_key = gaussian_ids[i].item()+10000
                        point_3d = gtsam.Point3(*gaussians[i])
                        point_2d = gtsam.Point2(*points[i])
                        if not current_estimate .exists(landmark_key):
                            gtsamInitial.insert(landmark_key, point_3d)
                        gtsamGraph.push_back(
                            gtsam.GenericProjectionFactorCal3_S2(
                                point_2d, gtsamPointNoise, pose_key, landmark_key, gtsamCam
                            )
                        )
                        #print('point', landmark_key, point_3d, point_2d)
                    gtsam_t_temp2 = time.time()
                    print('asdf', gtsam_t_temp1-gtsam_t1, gtsam_t_temp2-gtsam_t_temp1)
                    isam2.update(gtsamGraph, gtsamInitial)
                    current_estimate = isam2.calculateEstimate()
                    #gtsamParams = gtsam.LevenbergMarquardtParams()
                    #gtsamOptimizer = gtsam.LevenbergMarquardtOptimizer(gtsamGraph, gtsamInitial, gtsamParams)
                    #gtsamResult = gtsamOptimizer.optimize()
                    optimized_pose = current_estimate.atPose3(pose_key)
                    """

                    """
                    success, rvec, tvec = cv2.solvePnP(
                        gaussians.detach().cpu().numpy(),
                        points.cpu().numpy(),
                        K,
                        D,
                        rvec=prev_rvec,
                        tvec=prev_tvec,
                        useExtrinsicGuess=True,
                        #reprojectionError=reprojection_threshold,
                        #iterationsCount=max_iterations,
                        #confidence=confidence,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    """

                    """
                    result = pycolmap.estimate_absolute_pose(
                        points.cpu().numpy(), gaussians.cpu().numpy(),colCam, col_param
                    )

                    #if result['success']:

                    rigid = result['cam_from_world']
                    #tvec = result['translation']
                    print(rigid.rotation.matrix(), rigid.translation)
                    #quat = [qvec[1], qvec[2], qvec[3], qvec[0]]
                    #rotation = Rotation.from_quat(quat)
                    #R = rotation.as_matrix()


                    #inliers
                    #R, _ = cv2.Rodrigues(rvec)
                    R = torch.from_numpy(rigid.rotation.matrix()).cuda()
                    t = torch.from_numpy(rigid.translation).cuda()

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
                    """
                    gtsam_t2 = time.time()

                    #print("Optimized Camera Pose:\n", rvec,tvec, gtsam_t2-gtsam_t1)
                    """
                    projection, points, gaussians, _ = curr_frame.update_gaussianpoints(self.gaussians, viewpoint.R,
                        viewpoint.T,
                        viewpoint.fx, viewpoint.fy,
                        viewpoint.cx, viewpoint.cy,
                        viewpoint.image_width,
                        viewpoint.image_height,
                    )
                    """
                    frame_end_time = time.time()
                    #print('tracking', cur_frame_idx, projection.size()[0], kp_time_end-kp_time_start, kp_time_end-kp_time_temp, frame_end_time-frame_start_time, match_time_end-match_time_start, gtsam_t2-gtsam_t1)
                    #self.testManager.tracker.visualize2(curr_frame.color, projection, points, delay = 10, save=True, filename='./res/'+str(cur_frame_idx)+'.jpg')

                    #prev = self.cameras[prev_frame_idx]

                    """
                    valid = prev_frame.gaussianpoints > -1
                    gindex = prev_frame.gaussianpoints[valid]
                    prev_frame_gaussians = self.gaussians._xyz[gindex]
                    projection, valid_projection = project_pc_to_pixel(prev_frame_gaussians, prev.R, prev.T,
                                                                       viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                                       viewpoint.cy,
                                                                       viewpoint.image_width, viewpoint.image_height)
                    points = torch.from_numpy(prev_frame.keypoints[valid][valid_projection]).cuda()
                    self.testManager.tracker.visualize2(prev_frame.color, projection,points)
                    """
                    """
                    valid = prev_frame.gaussianpoints[matches[:,0]] > -1
                    filted_matches = matches[valid,:]
                    gindex = prev_frame.gaussianpoints[filted_matches[:,0]]
                    prev_frame_gaussians = self.gaussians._xyz[gindex]
                    projection, valid_projection= project_pc_to_pixel(prev_frame_gaussians, viewpoint.R, viewpoint.T,
                                        viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                        viewpoint.image_width, viewpoint.image_height)
                    points = torch.from_numpy(curr_frame.keypoints[filted_matches[valid_projection,1]]).cuda()
                    """

                    #print(cur_frame_idx, torch.count_nonzero(self.gaussians.isfeatured),
                    #      np.count_nonzero(self.gaussians.observations)
                    #      , self.gaussians._xyz.size(), prev_frame_gaussians.size(), projection.size(), points.shape)

                    #self.testManager.tracker.visualize2(curr_frame.color, projection, points, delay= 1)

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
                #print("test", len(radii), len(opacity), len(n_touched))

                ##render depth test

                t1 = time.time()
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                true_indices = torch.nonzero(visibility_filter).squeeze(dim=1)
                a = self.gaussians._xyz[true_indices]
                t2 = time.time()
                #print(t2-t1, len(true_indices), curr_visibility.size())
                #print("visible test ", curr_visibility.size(), self.gaussians._xyz.size())

                t3 = time.time()
                
                #img_rgb = cv2.cvtColor(image.detach().clone().cpu().numpy(), cv2.COLOR_BGR2RGB)
                #pred = self.testManager.feature_model.run(img_rgb)
                #keypoints = pred['keypoints']
                #descriptors = pred['descriptors']
                #frame = self.dataset[str(cur_frame_idx)]
                #pred = self.testManager.feature_model.run(frame.color)
                #frame.keypoints = pred['keypoints']
                #frame.descriptors = pred['descriptors']
                t4 = time.time()
                """
                if len(descriptors) > 0:
                    matches = self.testManager.tracker.match(descriptors, frame.descriptors)
                    out = self.testManager.tracker.visualize(img_rgb, frame.color, matches, keypoints, frame.keypoints)
                    t5 = time.time
                    #out, N_matches = self.testManager.tracker.update(frame.color, frame.keypoints, frame.descriptors)
                    #print(t4 - t3, t5-t4, self.testManager.feature_model.device, (frame.keypoints.shape))

                    cv2.imshow("match", out)
                    cv2.waitKey(1)
                #print(depth.size())
                """
                """
                if self.testManager.points is None :
                    self.testManager.test(frame.keypoints, depth, viewpoint.R, viewpoint.T,
                                      viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy)
                else:
                    self.testManager.test2(frame.color, frame.keypoints, viewpoint.R, viewpoint.T,
                                      viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy)
                """
                #print(viewpoint.R, viewpoint.T, frame.T)

                ##test

                e = time.time()
                # print("tracking time = ", cur_frame_idx, (e-s), viewpoint.exposure_a, viewpoint.exposure_b)
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
                    #print("test", len(self.current_window), point_ratio, intersection, union)
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
                #s = time.time()

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
                #e = time.time()
                #print("queue tiem = ", (e-s))