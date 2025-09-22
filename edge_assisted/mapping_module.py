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

from utils.camera_utils import Camera

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss, get_reprojection_loss_huber, get_patch_loss, get_loss_gaussian
from edge_assisted.gaussian_feature import unproject_pixel_to_pc, project_pc_to_pixel, projection, calculate_bbox_mask, calculate_feature_mask, find_correspondence_with_dist, calculate_keypoint_mask, get_correspondences_within_threshold, match_ac_from_ab_bc
from utils.edgeframe_utils import EdgeFrame
from utils.pose_utils import SE3_exp, skew_sym_mat, compute_F12

from edge_assisted.gaussian_local_model import LocalGaussianFrame, LocalGaussianModel

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

class MappingModule(WinBackEnd):
    def __init__(self, config):
        super().__init__(config)
        """
        self.first_kf_id = None
        self.pose_update = None
        self.gs_pose = None
        #self.dataset = None
        self.FeatureManager = None
        self.frames={}
        self.keyframe_ids = {}
        self.neighbor_kfs = {}
        """
        self.weight_reprojection = 0.08
        self.weight_init_rgb = 0.9
        self.weight_init_depth = 0.02

        self.weight_rgb = 0.8
        self.weight_depth = 0.02
        self.weight_ba = 0.03
        self.weight_patch = 0.15
        self.next_kf_id = 0

        self.viewpoints = {}
        self.keyframes = None
        self.devices = None

        self.init = False

    def convert_viewpoint(self, device, frame, idx):

        pose = frame.T
        image = frame.color
        depth = frame.depth

        if device.distorted:
            image = cv2.remap(image, device.map1x, device.map1y, cv2.INTER_LINEAR)

            if frame.depth is not None:
                depth = cv2.remap(depth, device.map1x, device.map1y, cv2.INTER_LINEAR)

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
            device.projection_matrix,
            device.fx,
            device.fy,
            device.cx,
            device.cy,
            device.fovx,
            device.fovy,
            device.h,
            device.w,
            device='cuda',
        )

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

    def preprocessing_add_kf(self):
        self.gaussians.observation_indices = torch.cat([self.gaussians.observation_indices,
                                                        torch.full((self.gaussians.observation_indices.shape[0], 1),
                                                                   -1, device='cuda', dtype=torch.int32)], dim=1)
        self.gaussians.observation_points = torch.cat([self.gaussians.observation_points,
                                                       torch.full((self.gaussians.observation_points.shape[0], 2), -1.0,
                                                                  device='cuda')], dim=1)

    def initialize_gaussian(self, kf_id, viewpoint, depth_map):
        self.add_next_kf(kf_id, viewpoint, init = True, depth_map=depth_map, downsample_factor=1)

    def create_gaussian(self, local_gaussian_mask, kf_id, viewpoint, depth_map):
        with torch.no_grad():
            projection, _, valid_projection = project_pc_to_pixel(self.gaussians.get_xyz[local_gaussian_mask], viewpoint.R, viewpoint.T,
                                                                  viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                                  viewpoint.image_width, viewpoint.image_height)

            tmp_gaussian_mask = calculate_keypoint_mask(projection[valid_projection], viewpoint.image_width,
                                                        viewpoint.image_height, )  # max_radius=1)#.squeeze(0)
            tmp_gaussian_mask = ~tmp_gaussian_mask
            Nold = self.gaussians.get_xyz.size()[0]
            self.add_next_kf(kf_id, viewpoint, init = True, depth_map=depth_map, downsample_factor=1, mask=tmp_gaussian_mask.squeeze(0).cpu().numpy())
            #Nnew = self.gaussians.get_xyz.size()[0]
            print('obs test',Nold, kf_id, self.gaussians.observation_indices.shape)
            #self.gaussians.observation_indices[Nold:-1, kf_id] = 1

    def construct_local_frame(self, local_gaussian, frame):
        #추후에는 param이 아닌 tensor로 전달될 것임.
        local_gaussian._xyz = torch.cat((local_gaussian._xyz, frame.frame_gaussian._xyz.data))
        local_gaussian._features_dc = torch.cat((local_gaussian._features_dc, frame.frame_gaussian._features_dc.data))
        local_gaussian._features_rest = torch.cat((local_gaussian._features_rest, frame.frame_gaussian._features_rest.data))
        local_gaussian._opacity = torch.cat((local_gaussian._opacity, frame.frame_gaussian._opacity.data))
        local_gaussian._scaling = torch.cat((local_gaussian._scaling, frame.frame_gaussian._scaling.data))
        local_gaussian._rotation = torch.cat((local_gaussian._rotation, frame.frame_gaussian._rotation.data))

    def construct_loocal_gaussians(self, local_gaussian, kf_ids):

        for id in kf_ids:
            if id in self.keyframes:
                kf = self.keyframes[id]
                self.construct_local_frame(local_gaussian, kf)

        local_gaussian.training_setup(self.opt_params)

        print('local gaussian', local_gaussian.get_xyz.shape[0])


    def select_local_gaussians(self, neigh_kf_ids):
        mask = (self.gaussians.observation_indices[:,neigh_kf_ids] > -1).any(dim=1)
        return mask


    def optimization(self, kf_windows, local_gs, mask = None, iter = 1):

        for i in range(iter):
            loss_mapping = 0

            viewspace_point_tensor_acm = []
            visibility_filter_acm = []

            for id in kf_windows:
                vp = self.viewpoints[id]

                render_pkg = render(
                    vp, local_gs, self.pipeline_params, self.background, mask = mask
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
                    self.config, image, depth, vp, opacity,
                )
                loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth
                loss_mapping += loss_kf

                #viewspace_point_tensor_acm.append(viewspace_point_tensor)
                #visibility_filter_acm.append(visibility_filter)

            scaling = local_gs.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            """
            for idx in range(len(kf_windows)):
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                )
            """

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.gaussians.update_learning_rate(self.iteration_count)

            """
            remove_ids = self.gaussians.densify_and_prune(
                self.opt_params.densify_grad_threshold,
                self.init_gaussian_th,
                self.init_gaussian_extent,
                None,
            )
            """

    def update_sparse_gaussian_map(self, local_sparse_map):
        pass

    def local_gaussian_mapping(self, kf_id, neigh_kf_ids, src):
        device = self.devices[src]
        keyframe = self.keyframes[kf_id]
        viewpoint = self.viewpoints[kf_id]

        a = time.time()

        if neigh_kf_ids is not None:
            local_gaussian = LocalGaussianModel(3, config=self.config, opt_params=self.opt_params)
            self.construct_loocal_gaussians(local_gaussian, neigh_kf_ids)

        a2 = time.time()

        self.preprocessing_add_kf()
        if self.init:
            local_gaussian_mask = self.select_local_gaussians(neigh_kf_ids)
            ##이미지 저장
            render_pkg = render(
                viewpoint, local_gaussian, self.pipeline_params, self.background
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
            image_np = (
                image
                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    .clone().detach().cpu()  # GPU → CPU
                    .numpy()  # NumPy 배열로 변환
            )
            image_np = (image_np * 255.0).astype(np.uint8)
            out = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./res/local_gs/' + str(kf_id) + '.jpg', out)
            cv2.imshow("local gs", out)
            cv2.waitKey(1)

            #calc error
            loss_rgb, loss_depth = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity,
            )
            loss_kf = loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

            Nold = self.gaussians.get_xyz.shape[0]

            ##이미지 저장
            self.create_gaussian(local_gaussian_mask, kf_id, viewpoint, viewpoint.depth)
            Nlg = torch.count_nonzero(local_gaussian_mask).item()

            kf_window = [kf_id] + neigh_kf_ids
            c = time.time()
            self.optimization(kf_window, local_gaussian, iter = 30)
            d = time.time()

            #print(self.gaussians.get_xyz.grad.norm(dim=-1))

            new_gaussian_mask = torch.zeros(self.gaussians.get_xyz.shape[0], device='cuda', dtype=torch.bool)
            new_gaussian_mask[Nold:-1] = True

            keyframe.frame_gaussian = LocalGaussianFrame(self.gaussians.get_xyz[new_gaussian_mask],
               self.gaussians._features_dc[new_gaussian_mask],
               self.gaussians._features_rest[new_gaussian_mask],
               self.gaussians.get_opacity[new_gaussian_mask],
               self.gaussians._scaling[new_gaussian_mask],
               self.gaussians._rotation[new_gaussian_mask])

        else:
            self.initialize_gaussian(kf_id, viewpoint,viewpoint.depth)
            self.init = True
            Nlg = 0
            d = 0
            c = 0

            keyframe.frame_gaussian = LocalGaussianFrame(self.gaussians.get_xyz,
                self.gaussians._features_dc,
                self.gaussians._features_rest,
                self.gaussians.get_opacity,
                self.gaussians._scaling,
                self.gaussians._rotation)

        b = time.time()
        #print(viewpoint.uid, viewpoint.R, viewpoint.T, viewpoint.R.dtype)
        if neigh_kf_ids is None:
            neigh_kf_ids = []
        print('local mapping', kf_id, len(neigh_kf_ids), a2-a, b-a, d-c, Nlg, self.gaussians.get_xyz.shape[0])
