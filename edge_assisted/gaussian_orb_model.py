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

from edge_assisted.gaussian_feature import calculate_feature_mask,calculate_bbox_mask, project_pc_to_pixel, visualize_pc,convert_xyz, pixels_to_pc,find_correspondence
from edge_assisted.gaussian_feature import GaussianPointManager

###일단 densification 같은 것에 대응하는지 확인
###이 후 코드 추가

class GaussianOrbModel(GaussianModel):
    def __init__(self, sh_degree : int, config = None):
        super().__init__(sh_degree, config)

        self.isfeatured = torch.empty(0, device="cuda").bool()
        self.observation_points = torch.empty((0,0), device="cuda").float()#np.full(0, None, dtype=object) #np dtype:object -> torch N x 2M 으로 변경
        self.observation_indices = torch.empty(0, device='cuda').int()

        self.unique_gaussian_ids = torch.empty(0, device = "cuda").long()
        self.global_gaussian_counter = 0

    def clone(self, indices):
        new_gaussians = GaussianOrbModel(self.max_sh_degree, self.config)

        new_gaussians._xyz = self._xyz[indices].clone()
        new_gaussians._features_dc = self._features_dc[indices].clone()
        new_gaussians._features_rest = self._features_rest[indices].clone()
        new_gaussians._scaling = self._scaling[indices].clone()
        new_gaussians._rotation = self._rotation[indices].clone()
        new_gaussians._opacity = self._opacity[indices].clone()

        return new_gaussians

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, keypoints=None, boxes = None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
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

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        if keypoints is not None:
            pass

        if boxes is not None:
            pass

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init,keypoints=keypoints, boxes = boxes)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, keypoints=None, boxes = None):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
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

        if boxes is not None:
            mask = calculate_bbox_mask(boxes, cam.image_width, cam.image_height)
            rgbd_object = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth[mask],
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )

            pcd_tmp_object = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_object,
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
            print('object test', pcd_tmp_object.shape, torch.count_nonzero(mask))

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

        if keypoints is not None:

            mask = calculate_feature_mask(keypoints, cam.image_width, cam.image_height, radius=1)
            rgbd_feature = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth[mask],
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )

            pcm_tmp_feature = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_feature,
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
            print('feature test', pcm_tmp_feature.shape, torch.count_nonzero(mask))


            #프로젝션
            #키포인트 없으면 추가
            #temp_keypoints =torch.round(torch.from_numpy(np.asarray(keypoints)).cuda()).int()
            #temp_backup_points는 pcd로 생성한 포인트
            temp_keypoints = torch.round(keypoints).int()
            temp_backup_points = np.asarray(pcd_tmp.points)
            temp_backup_colors = np.asarray(pcd_tmp.colors)

            #temp_valid는 temp_backup_points에서 이미지 안의 포인트를 의미함. = pcd의 수보다 작을 수 있음.
            temp_points=torch.from_numpy(temp_backup_points).cuda()
            temp_points, _, temp_valid = project_pc_to_pixel(temp_points, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy, cam.image_width, cam.image_height)
            temp_points = torch.round(temp_points[temp_valid]).int()

            rgb = torch.from_numpy(np.asarray(rgb)).cuda()
            depth = torch.from_numpy(np.asarray(depth)).cuda()

            #새로 생성하는 gaussian 포인트에서 매칭 대응쌍, 키포인트에서 매칭 안된 index mask
            #pcd와 키포인트 중 매치 된 포인트와 안된 키포인트를 의미
            #유니크 키포인트는 매치가 안된 키포인트임.
            match_index, unmatch_mask = find_correspondence(temp_points, temp_keypoints)
            unique_keypoints = temp_keypoints[unmatch_mask]
            temp_unmatched_keypoints_index = torch.arange(0,unmatch_mask.size()[0]).cuda()[unmatch_mask]

            #매치 안된 유니크 키포인트 만큼 추가
            add_points, add_colors, valid_depth_mask = convert_xyz(unique_keypoints, rgb, depth)
            add_points = pixels_to_pc(add_points, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy)

            add_points = add_points.cpu().detach().numpy()
            add_colors = add_colors.cpu().detach().numpy()/255

            temp_valid = temp_valid.cpu().numpy()
            new_xyz = np.concatenate((temp_backup_points[temp_valid], add_points), axis=0)
            new_rgb = np.concatenate((temp_backup_colors[temp_valid], add_colors),axis=0)

            N1 = np.count_nonzero(temp_valid) #기존에 랜덤하게 생성된 포인트
            N2 = torch.count_nonzero(valid_depth_mask)#unique_keypoints.shape[0]    #랜덤 생성된 값을 제외한 키포인트
            #Ncol = self.observations.shape[1]


            #그리고 pcd로 생성된 수와 같아야 함. 그 중에서 temp_valid가 true인 애들
            #isfeature, obs, match_index를 추가해야 함. 그 수는 유니크 키포인트와 같음.
            #add_point에서 valid_mask를 고려해야 할 듯. 이게 가끔 에러가 있음.
            temp_index = torch.where(match_index > -1)[0] #매칭 된 결과값

            new_isfeature = torch.zeros(N1, device = 'cuda')
            new_isfeature[temp_index] = True

            #기존 pcd 데이터
            new_obs = torch.full((N1, 2), -1.0, device="cuda")
            new_obs[temp_index] = keypoints[match_index[temp_index]]

            temp_unmatched_keypoints_index = temp_unmatched_keypoints_index[valid_depth_mask]
            new_isfeature = torch.cat((new_isfeature,torch.ones(N2, device='cuda')), axis = 0)
            new_obs = torch.cat((new_obs, keypoints[temp_unmatched_keypoints_index]), axis=0)
            match_index = torch.concatenate((match_index, temp_unmatched_keypoints_index), axis = 0)

            if new_obs.shape[0] != new_xyz.shape[0]:
                print('new gaussian error case', new_obs.shape[0], match_index.shape[0], new_isfeature.shape[0])
                print(torch.count_nonzero(valid_depth_mask),add_points.shape[0],temp_unmatched_keypoints_index.shape[0])

            #if N1 != new_obs.shape[0]:
            #    print('err new gaussians : asdf', N1, new_obs.shape, temp_index.shape, temp_unmatched_keypoints_index.shape, torch.count_nonzero(temp_unmatched_keypoints_index > -1))

            #print('bb',match_index.shape,new_xyz.shape, torch.count_nonzero(match_index > -1), torch.count_nonzero(new_obs > -1)/2)
            #print('aa', temp_valid.shape, temp_backup_points[temp_valid].shape,temp_unmatched_keypoints_index.shape, temp_keypoints.shape)
            #matched_mask = match_index > -1

        else:
            new_xyz = np.asarray(pcd_tmp.points)
            new_rgb = np.asarray(pcd_tmp.colors)

            N1 = new_xyz.shape[0]
            new_isfeature = torch.zeros(N1, device='cuda')
            new_obs = torch.full((N1, 2), -1.0, device="cuda")
            match_index = torch.full((new_xyz.shape[0]),-1)

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
        if new_xyz.shape[0] != fused_point_cloud.shape[0]:
            print('error new gaussian', new_xyz.shape, fused_point_cloud.shape)
        return fused_point_cloud, features, scales, rots, opacities,new_isfeature, match_index, new_obs

    #frame 정보가 추가 전송
    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, frame=None
    ):

        if frame is not None:
            keypoints = frame.keypoints
        else:
            keypoints =None
        if len(frame.objects) == 0:
            boxes = None
        else:
            boxes = torch.tensor(list(frame.objects.values()), device='cuda')

        fused_point_cloud, features, scales, rots, opacities, new_isfeature, tmp_new_observation_indices, new_observation_points = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, keypoints=keypoints, boxes = boxes)
        )

        # global gaussian id
        Nnew = fused_point_cloud.size()[0]
        Ncol1 = max(0, self.observation_indices.shape[1] - 1)
        Ncol2 = max(0, self.observation_points.shape[1] - 2)
        new_prev_obs_indices = torch.full((Nnew, Ncol1), -1, dtype = torch.int32, device = 'cuda')
        tmp_new_observation_indices = tmp_new_observation_indices.type(torch.int32)
        new_prev_obs_points = torch.full((Nnew, Ncol2), -1.0, device = 'cuda')
        #print(new_prev_obs_points.shape, new_prev_obs_indices.shape, new_observation_indices.shape, new_observation_points.shape)

        if new_prev_obs_indices.shape[0] != tmp_new_observation_indices.shape[0]:
            print('err new gaussians bbbb', Nnew, tmp_new_observation_indices.shape)

        new_observation_indices = torch.cat([new_prev_obs_indices, tmp_new_observation_indices.unsqueeze(1)], dim=1)
        new_observation_points = torch.cat([new_prev_obs_points, new_observation_points], dim=1)

        #print(new_observation_indices.shape, new_observation_points.shape)
        old_gaussian = self.global_gaussian_counter
        self.global_gaussian_counter += Nnew
        new_global_ids = torch.arange(old_gaussian, self.global_gaussian_counter).cuda()
        Nold = self.get_xyz.shape[0]
        """
        new_isfeature = torch.zeros(Nnew,device="cuda").bool()
        new_observations = np.full(Nnew, None, dtype=object)

        if frame is not None:
            for gauss_idx, point in enumerate(fused_point_cloud):
                kp_idx = match_index[gauss_idx]
                if kp_idx > -1:
                    frame.gaussianpoints[kp_idx] = gauss_idx + Nold
                    new_isfeature[gauss_idx] = True
                    new_observations[gauss_idx] = {frame.id:kp_idx.item()}
                    #new_observations[gauss_idx][frame.id] = kp_idx.numpy()
        """

        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id, isfeatures=new_isfeature,
            observation_indices = new_observation_indices,
            observation_points = new_observation_points,
            global_ids=new_global_ids
        )




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
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
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