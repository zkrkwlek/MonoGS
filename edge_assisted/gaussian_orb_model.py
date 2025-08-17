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

from edge_assisted.gaussian_feature import calculate_keypoint_mask,calculate_feature_mask,calculate_bbox_mask, project_pc_to_pixel, visualize_pc,convert_xyz, pixels_to_pc,find_correspondence
from edge_assisted.gaussian_feature import GaussianPointManager

###일단 densification 같은 것에 대응하는지 확인
###이 후 코드 추가

class GaussianOrbModel(GaussianModel):
    def __init__(self, sh_degree : int, config = None, opt_params = None):
        super().__init__(sh_degree, config)

        self.isfeatured = torch.empty(0, device="cuda").bool()
        self.observation_points = torch.empty((0,0), device="cuda").float()#np.full(0, None, dtype=object) #np dtype:object -> torch N x 2M 으로 변경
        self.observation_indices = torch.empty(0, device='cuda').int()

        self.unique_gaussian_ids = torch.empty(0, device = "cuda").long()
        self.global_gaussian_counter = 0

        self.opt_params = opt_params

    def clone(self, indices):
        new_gaussians = GaussianOrbModel(self.max_sh_degree, self.config)

        new_gaussians._xyz = self._xyz[indices]
        new_gaussians._features_dc = self._features_dc[indices]
        new_gaussians._features_rest = self._features_rest[indices]
        new_gaussians._scaling = self._scaling[indices]
        new_gaussians._rotation = self._rotation[indices]
        new_gaussians._opacity = self._opacity[indices]

        return new_gaussians
    """
    def compute_2d_gaussian_weights(self,
            means_2d,  # (N, 2): 각 2D 가우시안의 중심 (projection된 위치)
            covs_2d,  # (N, 2, 2): 각 2D 가우시안의 2x2 공분산
            opacities,  # (N,): 각 가우시안의 불투명도 (alpha o)
            W, H
    ):
        N = means_2d.shape[0]
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixel_coords = torch.stack([xs, ys], dim=-1).float().cuda()

        # (H, W, 2) → (1, H*W, 2) → (N, H*W, 2)
        pixels = pixel_coords.view(1, -1, 2).expand(N, -1, -1)
        means = means_2d[:, None, :]  # (N, 1, 2)
        diffs = pixels - means  # (N, H*W, 2)

        # (N, 2, 2) → (N, 1, 2, 2)
        covs = covs_2d[:, None, :, :]  # (N, 1, 2, 2)
        covs_inv = torch.linalg.pinv(covs)  # (N, 1, 2, 2)
        covs_det = torch.det(covs)[..., 0]  # (N,)

        # mahalanobis 거리: (N, H*W)
        # (N, H*W, 1, 2) × (N, 1, 2, 2) → (N, H*W, 1, 2) → squeeze(-2): (N, H*W, 2)
        mahal = torch.matmul(diffs.unsqueeze(-2), covs_inv).squeeze(-2)  # (N, H*W, 2)
        mahal = (mahal * diffs).sum(-1)  # (N, H*W)

        norm_factor = 1.0 / (2.0 * torch.pi * torch.sqrt(covs_det + 1e-6))  # (N,)
        norm_factor = norm_factor[:, None]  # (N, 1)

        # 2D 가우시안 PDF
        gauss_pdf = norm_factor * torch.exp(-0.5 * mahal)  # (N, H*W)

        # 픽셀 별 alpha 영향도 계산 (가우시안 opacity 반영)
        weights = gauss_pdf * opacities[:, None]  # (N, H*W)
        weights = weights.view(N, H, W)  # (N, H, W)
        return weights

    def convert_cov3d_to_cov2d(self, R, t, fx, fy):
        means_cam = self.get_xyz @ R.T + t

        x, y, z = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]

        # 자코비안 J: (N, 2, 3)
        J = torch.zeros((x.shape[0], 2, 3), device='cuda', dtype=x.dtype)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * x / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * y / (z * z)

        # 카메라 좌표계로 공분산 이동: (N, 3, 3)

        M = self.quaternion_to_rotation_matrix()
        cov3d = M @ torch.diag_embed(self.get_scaling ** 2) @ M.transpose(-1, -2)
        if R.ndim == 2:
            covs_cam = R @ cov3d @ R.T
        else:  # (N, 3, 3)
            covs_cam = torch.bmm(R, torch.bmm(cov3d, R.transpose(1, 2)))

        # 최종 2D 공분산: (N, 2, 2)
        J_cov = torch.bmm(J, torch.bmm(covs_cam, J.transpose(1, 2)))
        return J_cov

    def quaternion_to_rotation_matrix(self):  # q: (N, 4) -> (N, 3, 3)
        # q: (N, 4) with xyzw order
        q = self.get_rotation
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        N = q.size(0)
        xx = x * x;
        yy = y * y;
        zz = z * z
        ww = w * w;
        xy = x * y;
        xz = x * z;
        yz = y * z
        wx = w * x;
        wy = w * y;
        wz = w * z
        R = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=1).reshape(N, 3, 3)
        return R
    """
    def create_pcd_from_image(self, cam_info, rgb, mask, init=False, scale=2.0, depthmap=None, keypoints=None, boxes = None, downsample_factor = 128):
        cam = cam_info
        #image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        #image_ab = torch.clamp(image_ab, 0.0, 1.0)
        #rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

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

            #rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        """
        fused_point_cloud, features, scales, rots, opacities,new_isfeature, match_index, new_obs =
        if keypoints is not None:
            t_a = time.time()
            tmp_depth = depthmap.copy()
            feature_mask = torch.logical_and(gaussian_mask, feature_mask)
            tmp_depth[(~feature_mask).squeeze(0).cpu().numpy()] = 0.0
            feature_depth = o3d.geometry.Image(tmp_depth.astype(np.float32))
            fused_point_cloud2, features2, scales2, rots2, opacities2, new_isfeature2, match_index2, new_obs2 = self.create_pcd_from_image_and_depth(cam, rgb, feature_depth, init, keypoints=keypoints, downsample_factor = 32)
            t_b = time.time()
            fused_point_cloud = torch.cat((fused_point_cloud, fused_point_cloud2), dim=0)
            features = torch.cat((features, features2), dim=0)
            scales = torch.cat((scales, scales2), dim=0)
            rots = torch.cat((rots, rots2), dim=0)
            opacities = torch.cat((opacities, opacities2), dim=0)
            new_isfeature = torch.cat((new_isfeature, new_isfeature2), dim=0)
            match_index = torch.cat((match_index, match_index2), dim=0)
            new_obs = torch.cat((new_obs, new_obs2), dim=0)
            t_c = time.time()
            #print('new gaussians : feature', t_b-t_a, t_c-t_b)
        if boxes is not None:
            tmp_depth = depthmap.copy()
            box_mask = torch.logical_and(gaussian_mask, box_mask)
            tmp_depth[(~box_mask).squeeze(0).cpu().numpy()] = 0.0
            box_depth = o3d.geometry.Image(tmp_depth.astype(np.float32))
            fused_point_cloud2, features2, scales2, rots2, opacities2, new_isfeature2, match_index2, new_obs2 = self.create_pcd_from_image_and_depth(cam, rgb, box_depth, init, boxes=boxes, downsample_factor = 32)

            fused_point_cloud = torch.cat((fused_point_cloud, fused_point_cloud2), dim=0)
            features = torch.cat((features, features2), dim=0)
            scales = torch.cat((scales, scales2), dim=0)
            rots = torch.cat((rots, rots2), dim=0)
            opacities = torch.cat((opacities, opacities2), dim=0)
            new_isfeature = torch.cat((new_isfeature, new_isfeature2), dim=0)
            match_index = torch.cat((match_index, match_index2), dim=0)
            new_obs = torch.cat((new_obs, new_obs2), dim=0)
        """
        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init, keypoints = keypoints, boxes = boxes, downsample_factor = downsample_factor)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, keypoints=None, boxes = None, downsample_factor = 128):

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

        #farthest_point_down_sample(3000)
        #pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.03)
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

        N1 = fused_point_cloud.shape[0]
        if keypoints is not None:
            temp_points, _, temp_valid = project_pc_to_pixel(fused_point_cloud, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy,
                                                             cam.image_width, cam.image_height)
            temp_points = torch.round(temp_points).int()
            temp_keypoints = torch.round(keypoints).int()
            match_index, unmatch_mask = find_correspondence(temp_points, temp_keypoints)

            temp_index = torch.where(match_index > -1)[0]  # 매칭 된 결과값

            new_isfeature = torch.zeros(N1, device='cuda')
            new_isfeature[temp_index] = True

            new_obs = torch.full((N1, 2), -1.0, device="cuda")
            new_obs[temp_index] = keypoints[match_index[temp_index]]
            #print('new gaussian feature test', torch.count_nonzero(new_obs > -1), fused_point_cloud.shape, temp_points.shape, torch.count_nonzero(temp_index), match_index.shape, unmatch_mask.shape, temp_keypoints.shape)
        else:
            new_isfeature = torch.zeros(N1, device='cuda')
            new_obs = torch.full((N1, 2), -1.0, device="cuda")
            match_index = torch.full((N1,), -1.0, device="cuda")

        if new_xyz.shape[0] != fused_point_cloud.shape[0]:
            print('error new gaussian', new_xyz.shape, fused_point_cloud.shape)

        return fused_point_cloud, features, scales, rots, opacities,new_isfeature, match_index, new_obs

    #frame 정보가 추가 전송
    def extend_from_pcd_seq_partial(self, cam_info, rgb, mask, kf_id=-1, init=False, scale=2.0, depthmap=None, keypoints=None, boxes = None, downsample_factor = 128):

        fused_point_cloud, features, scales, rots, opacities, \
        new_isfeature, tmp_new_observation_indices, new_observation_points = (
            self.create_pcd_from_image(cam_info, rgb, mask, init, scale=scale, depthmap=depthmap, keypoints=keypoints, boxes=boxes,
                                       downsample_factor=downsample_factor)
        )

        # feature 관련 갱신
        Nnew = fused_point_cloud.size()[0]
        Ncol1 = max(0, self.observation_indices.shape[1] - 1)
        Ncol2 = max(0, self.observation_points.shape[1] - 2)
        new_prev_obs_indices = torch.full((Nnew, Ncol1), -1, dtype=torch.int32, device='cuda')
        tmp_new_observation_indices = tmp_new_observation_indices.type(torch.int32)
        new_prev_obs_points = torch.full((Nnew, Ncol2), -1.0, device='cuda')

        new_observation_indices = torch.cat([new_prev_obs_indices, tmp_new_observation_indices.unsqueeze(1)], dim=1)
        new_observation_points = torch.cat([new_prev_obs_points, new_observation_points], dim=1)
        # feature 관련 갱신

        # global gaussian id
        old_gaussian = self.global_gaussian_counter
        self.global_gaussian_counter += Nnew
        new_global_ids = torch.arange(old_gaussian, self.global_gaussian_counter).cuda()
        # global gaussian id

        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id, isfeatures=new_isfeature,
            observation_indices=new_observation_indices,
            observation_points=new_observation_points,
            global_ids=new_global_ids
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, keypoints=None, downsample_factor = 128, mask = None
    ):

        """
        #포인트 처리
        if frame is not None:
            keypoints = frame.keypoints
        else:
            keypoints =None
        #가우시안 처리
        if len(frame.objects) == 0:
            boxes = None
        else:
            arr = np.array(list(frame.objects.values()))
            boxes = torch.from_numpy(arr).to('cuda')
        """


        image_ab = (torch.exp(cam_info.exposure_a)) * cam_info.original_image + cam_info.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))

        self.extend_from_pcd_seq_partial(cam_info, rgb, mask, kf_id, init=init, scale=scale, depthmap=depthmap,
                                         keypoints=keypoints, boxes=None, downsample_factor=downsample_factor)

        return
        #완전 특징점
        #특징점 주변
        #박스
        #가우시안 없는 곳
        #에러 있는 곳

        ##가우시안 위치 마스크
        if self.get_xyz.shape[0] > 0 :
            projections, depths, valid_proj = project_pc_to_pixel(self.get_xyz, cam_info.R, cam_info.T,
                                                              cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy,
                                                              cam_info.image_width, cam_info.image_height)
            gaussian_mask = calculate_keypoint_mask(projections[valid_proj], cam_info.image_width, cam_info.image_height).squeeze(0)
            gaussian_mask = ~gaussian_mask
        else:
            gaussian_mask = torch.ones((cam_info.image_height, cam_info.image_width), device='cuda', dtype=torch.bool)
        ##가우시안 위치 마스크

        feature_mask = torch.zeros((cam_info.image_height, cam_info.image_width), device='cuda', dtype=torch.bool)
        if keypoints is not None and False:
            feature_mask = calculate_keypoint_mask(keypoints, cam_info.image_width, cam_info.image_height)
            feature_region_mask = calculate_feature_mask(keypoints, cam_info.image_width, cam_info.image_height, max_radius=7)
            #feature_mask = feature_mask.cpu()

        box_mask = torch.zeros((cam_info.image_height, cam_info.image_width), device='cuda', dtype=torch.bool)
        if boxes is not None and False:
            box_mask = calculate_bbox_mask(boxes, cam_info.image_width, cam_info.image_height)
            #box_mask = box_mask.cpu()


        base_mask = torch.logical_and(gaussian_mask, torch.logical_and(feature_mask == 0, box_mask == 0)).squeeze(0).cpu().numpy()
        #base_mask = gaussian_mask.cpu().numpy()
        """
        tmp_mask = base_mask.astype(np.uint8) * 255
        cv2.imshow("base-mask", tmp_mask)
        cv2.waitKey(10)
        """

        ##마스크화해서 넘기기
        self.extend_from_pcd_seq_partial(cam_info, rgb, base_mask, kf_id, init=init, scale=scale, depthmap=depthmap, keypoints=None, boxes=None, downsample_factor=downsample_factor)

        if keypoints is not None and False:
            feature_mask = torch.logical_and(gaussian_mask, feature_region_mask)
            feature_mask = feature_mask.squeeze(0).cpu().numpy()

            """"""
            tmp_mask = feature_mask.astype(np.uint8) * 255
            cv2.imshow("asdf-mask", tmp_mask)
            cv2.waitKey(10)

            self.extend_from_pcd_seq_partial(cam_info, rgb, feature_mask, kf_id, init=init, scale=scale, depthmap=depthmap,
                                             keypoints=None, boxes=None, downsample_factor=downsample_factor)
        if boxes is not None and False:
            box_mask = torch.logical_and(gaussian_mask, box_mask)
            box_mask = box_mask.squeeze(0).cpu().numpy()
            self.extend_from_pcd_seq_partial(cam_info, rgb, box_mask, kf_id, init=init, scale=scale, depthmap=depthmap,
                                             keypoints=None, boxes=boxes, downsample_factor=downsample_factor)

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id, isfeatures = None, observation_indices = None, observation_points = None, global_ids = None
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_isfeatures=isfeatures,
            new_observation_indices = observation_indices,
            new_observation_points = observation_points,
            new_global_ids = global_ids
        )

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_isfeatures=None,
        new_observation_indices=None,
        new_observation_points=None,
        new_global_ids=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        if new_isfeatures is not None:
            self.isfeatured = torch.cat((self.isfeatured, new_isfeatures)).bool()
        if new_observation_indices is not None:
            self.observation_indices = torch.cat((self.observation_indices, new_observation_indices)).int()
        if new_observation_points is not None:
            self.observation_points = torch.cat((self.observation_points, new_observation_points)).float()
        if new_global_ids is not None:
            self.unique_gaussian_ids = torch.cat((self.unique_gaussian_ids, new_global_ids)).long()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        #selected_pts_mask = torch.logical_and(
        #    selected_pts_mask, ~self.isfeatured)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        new_isfeatures = self.isfeatured[selected_pts_mask].repeat(N)
        new_observation_indices = self.observation_indices[selected_pts_mask].repeat(N,1)
        new_observation_points = self.observation_points[selected_pts_mask].repeat(N,1)

        #global id
        Nnew = new_isfeatures.size()[0]
        old_gaussian = self.global_gaussian_counter
        self.global_gaussian_counter += Nnew
        new_global_ids = torch.arange(old_gaussian, self.global_gaussian_counter).cuda()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_isfeatures=new_isfeatures,
            new_observation_indices=new_observation_indices,
            new_observation_points=new_observation_points,
            new_global_ids = new_global_ids
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        remove_split_ids = self.unique_gaussian_ids[prune_filter]
        self.prune_points(prune_filter)

        return remove_split_ids

    def  densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        #selected_pts_mask = torch.logical_and(
        #    selected_pts_mask,~self.isfeatured)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]

        new_isfeatures = self.isfeatured[selected_pts_mask]
        new_observation_indices = self.observation_indices[selected_pts_mask]
        new_observation_points = self.observation_points[selected_pts_mask]

        Nnew = new_isfeatures.size()[0]
        old_gaussian = self.global_gaussian_counter
        self.global_gaussian_counter += Nnew
        new_global_ids = torch.arange(old_gaussian, self.global_gaussian_counter).cuda()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_isfeatures=new_isfeatures,
            new_observation_indices = new_observation_indices,
            new_observation_points = new_observation_points,
            new_global_ids = new_global_ids
        )
        return

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):

        Nold = self._xyz.size()[0]

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)

        split_ids =self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        p_opa1 = torch.count_nonzero(prune_mask).item()
        p_opa2 = torch.count_nonzero(prune_mask & self.isfeatured).item()
        p_vs = 0
        p_ws = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            p_vs = torch.count_nonzero(big_points_vs & self.isfeatured).item()
            p_ws = torch.count_nonzero(big_points_ws & self.isfeatured).item()
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        print('prune_points', self.get_xyz.shape[0], torch.count_nonzero(prune_mask).item(), torch.count_nonzero(prune_mask & (self.isfeatured)).item(), '=', p_opa1, p_opa2, p_vs, p_ws)
        ## remove gaussian ids
        removed_ids = self.unique_gaussian_ids[prune_mask]
        removed_ids = torch.cat((split_ids, removed_ids), axis = 0)
        self.prune_points(prune_mask)

        ##test
        #aa = torch.isin(self.unique_gaussian_ids, removed_ids)
        #print('aa',torch.count_nonzero(aa))

        return removed_ids

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

        self.isfeatured = self.isfeatured[valid_points_mask]
        self.observation_indices = self.observation_indices[valid_points_mask]
        self.observation_points = self.observation_points[valid_points_mask]
        self.unique_gaussian_ids = self.unique_gaussian_ids[valid_points_mask]
        #observation도 처리 필요

    """
    def update_gaussian_observation_before_prune(self, mask):

        feature_indices = self.isfeatured.clone().cpu().numpy()
        gaussian_indices = torch.arange(self._xyz.size()[0])

        prune_mask = (mask).cpu().numpy()

        gaussian_obs = self.observations[feature_indices & prune_mask]
        gaussian_indices = gaussian_indices[feature_indices & prune_mask]

        for gaussian_index, obs in zip(gaussian_indices, gaussian_obs):
            if obs is None:
                #print(gaussian_index, obs)
                self.isfeatured[gaussian_index] = False
                continue
            for fid, kpidx in obs.items():
                #print("update_gaussian frame", fid)
                frame = frames[(fid)]
                #temp_gidx = frame.gaussianpoints[kpidx]
                #if gaussian_index == temp_gidx:
                frame.gaussianpoints[kpidx] = -1
    """

def create_gaussian_orb_model(base_model, config: Dict = None) -> GaussianOrbModel:
    """TrackableGaussianModel 생성"""
    default_config = {
        'orb_features': 1000,
        'orb_scale_factor': 1.2,
        'orb_levels': 8,
        'orb_edge_threshold': 31,
        'orb_fast_threshold': 20,
        'min_track_confidence': 0.3,
        'max_projection_distance': 15.0,
        'min_opacity_threshold': 0.05
    }

    if config:
        default_config.update(config)

    return GaussianOrbModel(base_model, default_config)