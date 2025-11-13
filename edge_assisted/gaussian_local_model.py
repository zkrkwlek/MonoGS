import numpy
import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from gaussian_splatting.scene.gaussian_model import GaussianModel

from simple_knn._C import distCUDA2
import open3d as o3d
from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH

from edge_assisted.gaussian_feature import calculate_keypoint_mask,calculate_feature_mask,calculate_bbox_mask, project_pc_to_pixel, visualize_pc,convert_xyz, unproject_pixel_to_pc,find_correspondence
from edge_assisted.gaussian_feature import GaussianPointManager

###일단 densification 같은 것에 대응하는지 확인
###이 후 코드 추가

class LocalGaussianFrame:
    def __init__(self, xyz, features_dc, features_rest, opacity, scale, rotation):
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._opacity = opacity
        self._scaling = scale
        self._rotation = rotation

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

class LocalGaussianModel(GaussianModel):

    def __init__(self, sh_degree : int, config = None, opt_params = None):
        super().__init__(sh_degree, config)
        self.init_lr(6.0)
        self.opt_params = opt_params

    def construct_local_gaussians(self, frames, opt_params):
        #가우시안 생성
        #그래프 옵티마이저에 추가
        #densification 참조

        for f in frames:
            pass

        self.training_setup(opt_params)

    def set_requires_true(self):
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._opacity.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)

    # generate gaussian from gaussian_model
    def extend_from_pcd_seq(
            self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, downsample_factor=128, mask=None
    ):

        image_ab = (torch.exp(cam_info.exposure_a)) * cam_info.original_image + cam_info.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))

        return self.extend_from_pcd_seq_partial(cam_info, rgb, mask, kf_id, init=init, scale=scale,
                                                depthmap=depthmap, downsample_factor=downsample_factor)

    def extend_from_pcd_seq_partial(self, cam_info, rgb, mask, kf_id=-1, init=False,
                                    scale=2.0, depthmap=None, downsample_factor=128):

        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, rgb, mask, init, scale=scale, depthmap=depthmap,
                                       downsample_factor=downsample_factor)
        )
        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        return fused_point_cloud, features_dc, features_rest, opacities, scales, rots

    def create_pcd_from_image(self, cam_info, rgb, mask, init=False, scale=2.0, depthmap=None, keypoints=None,
                              boxes=None, downsample_factor=128):
        cam = cam_info

        if depthmap is not None:
            tmp_depth = depthmap.copy()
            tmp_depth[~mask] = 0.0
            depth = o3d.geometry.Image(tmp_depth.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                                    np.ones_like(depth_raw)
                                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                                    * 0.05
                            ) * scale
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init, downsample_factor=downsample_factor)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, downsample_factor=128):

        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()

        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
                .float()
                .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
                torch.clamp_min(
                    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                    0.0000001,
                )
                * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        return fused_point_cloud, features, scales, rots, opacities
    # generate gaussian