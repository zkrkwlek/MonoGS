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

from edge_assisted.gaussian_feature import project_pc_to_pixel, visualize_pc,convert_xyz, pixels_to_pc,find_correspondence
from edge_assisted.gaussian_feature import GaussianPointManager

####추후 수정
@dataclass
class ORBFeatureData:
    """ORB 특징점 데이터 저장"""
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: np.ndarray = None
    view_ids: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def add_feature(self, keypoint: cv2.KeyPoint, descriptor: np.ndarray,
                    view_id: int, confidence: float = 1.0):
        """새로운 특징점 추가"""
        self.keypoints.append(keypoint)
        self.view_ids.append(view_id)
        self.confidences.append(confidence)

        if self.descriptors is None:
            self.descriptors = descriptor.reshape(1, -1)
        else:
            self.descriptors = np.vstack([self.descriptors, descriptor])

    def get_latest_descriptor(self) -> Optional[np.ndarray]:
        """가장 최근 디스크립터 반환"""
        return self.descriptors[-1] if self.descriptors is not None else None

    def get_average_descriptor(self) -> Optional[np.ndarray]:
        """평균 디스크립터 반환"""
        if self.descriptors is None:
            return None
        return np.mean(self.descriptors, axis=0).astype(np.uint8)

    def cleanup_old_data(self, max_history: int = 5):
        """오래된 데이터 정리"""
        if len(self.keypoints) > max_history:
            self.keypoints = self.keypoints[-max_history:]
            self.view_ids = self.view_ids[-max_history:]
            self.confidences = self.confidences[-max_history:]
            if self.descriptors is not None:
                self.descriptors = self.descriptors[-max_history:]

###일단 densification 같은 것에 대응하는지 확인
###이 후 코드 추가

class GaussianOrbModel(GaussianModel):
    def __init__(self, sh_degree : int, config = None):
        super().__init__(sh_degree, config)

        self.isfeatured = torch.empty(0, device="cuda").bool()
        self.observations = np.full(0, None, dtype=object) #np dtype:object
        #self.unique_gaussian_ids = torch.empty(0).long()
        #self.global_gaussian_counter = 0

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, keypoints=None):
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

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init,keypoints=keypoints)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, keypoints=None):
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
        """
        temp_xyz = np.asarray(pcd_tmp.points)
        temp_pc = torch.from_numpy(temp_xyz).float().cuda()
        temp_colors = torch.from_numpy(np.asarray(pcd_tmp.colors)).float().cuda()
        visualize_pc(temp_pc,temp_colors, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy, cam.image_width, cam.image_height)
        """
        #farthest_point_down_sample(3000)
        #pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.03)
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)

        if keypoints is not None:
            #프로젝션
            #키포인트 없으면 추가
            temp_keypoints =torch.round(torch.from_numpy(np.asarray(keypoints)).cuda()).int()

            temp_backup_points = np.asarray(pcd_tmp.points)
            temp_backup_colors = np.asarray(pcd_tmp.colors)

            temp_points=torch.from_numpy(temp_backup_points).cuda()
            temp_points, temp_valid = project_pc_to_pixel(temp_points, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy, cam.image_width, cam.image_height)
            temp_points = torch.round(temp_points).int()
            rgb_np =np.asarray(rgb)
            rgb = torch.from_numpy(np.asarray(rgb)).cuda()
            depth = torch.from_numpy(np.asarray(depth)).cuda()

            #새로 생성하는 gaussian 포인트에서 매칭 대응쌍, 키포인트에서 매칭 안된 index mask
            match_index, unmatch_mask = find_correspondence(temp_points, temp_keypoints)
            #is_in_points = (temp_keypoints.unsqueeze(1) == temp_points).all(dim=2).any(dim=1)
            #not_in_points = ~is_in_points
            unique_keypoints = temp_keypoints[unmatch_mask]
            temp_unmatched_keypoints_index = torch.arange(0,unmatch_mask.size()[0]).cuda()[unmatch_mask]

            add_points, add_colors, valid_depth_mask = convert_xyz(unique_keypoints, rgb, depth)
            add_points = pixels_to_pc(add_points, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy)

            #print("init test",match_index.size(), temp_points.size(), add_points.size(), unique_keypoints.size(), torch.count_nonzero(valid_depth_mask), temp_unmatched_keypoints_index.size())

            """
            print(temp_unmatched_keypoints_index.size(), temp_keypoints.size())

            visualize_pc(add_points, add_colors, cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy
                         , cam.image_width, cam.image_height, rgb_np)

            #print("overlap test", add_points.size(), unique_keypoints.size(), not_in_points.size())

            
            if True:

                temp_points = project_pc_to_pixel(add_points.clone(), cam.R, cam.T, cam.fx, cam.fy, cam.cx, cam.cy,
                                                  cam.image_width, cam.image_height)
                temp_points = torch.round(temp_points).int()
                is_in_points2 = (temp_keypoints.unsqueeze(1) == temp_points.unsqueeze(0)).all(dim=2).any(dim=1)
                not_in_points2 = ~is_in_points2
                print("overlap test = ", torch.count_nonzero(is_in_points2))
            """
            add_points = add_points.cpu().detach().numpy()
            add_colors = add_colors.cpu().detach().numpy()/255

            new_xyz = np.concatenate((temp_backup_points[temp_valid], add_points), axis=0)
            new_rgb = np.concatenate((temp_backup_colors[temp_valid], add_colors),axis=0)
            match_index = torch.concatenate((match_index, temp_unmatched_keypoints_index), axis = 0)

            matched_mask = match_index > -1
            matched_indices = match_index[matched_mask]
            #print("asdf", temp_backup_points[temp_valid].shape, temp_points.size())
            #print("frame gaussian test",new_xyz.shape, match_index.size(), matched_indices.size(), temp_keypoints.size(), torch.count_nonzero(valid_depth_mask))

            #new_xyz = np.asarray(pcd_tmp.points)
            #new_rgb = np.asarray(pcd_tmp.colors)
            #print("sampling", new_xyz.shape, add_points.shape, torch.count_nonzero(is_in_points))
        else:
            new_xyz = np.asarray(pcd_tmp.points)
            new_rgb = np.asarray(pcd_tmp.colors)
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

        return fused_point_cloud, features, scales, rots, opacities,match_index

    #frame 정보가 추가 전송
    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, frame=None
    ):

        if frame is not None:
            #convert_xyz(frame.keypoints, cam_info.original_image, cam_info.depth)
            keypoints = frame.keypoints
        else:
            keypoints =None

        fused_point_cloud, features, scales, rots, opacities, match_index = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, keypoints=keypoints)
        )
        #프레임의 키포인트가 비어있는 곳 확인해야 함.
        #fused_point_cloud를 특징점과 연결
        #다만, 카메라 좌표계에서 프로젝션하는지, 월드 좌표계에서 프로젝션하는지 확인필요
        #cam_info가 viewpoint임. 그렇다면, 월드 좌표계라고 생각하면 편함.
        #다만, 그냥 그대로 가우시안을 생성하는 거 같은데? overlap 되는 거 고려 안하고?

        #두 개의 파라메터 추가
        #N = fused_point_cloud의 수 만큼
        #if frame is not None:
        Nold = self._xyz.size()[0]
        Nnew = fused_point_cloud.size()[0]
        new_isfeature = torch.zeros(Nnew,device="cuda").bool()
        new_observations = np.full(Nnew, None, dtype=object)

        if frame is not None:
            #매치 인덱스를 이용해서 가우시안 인덱스 넣기
            #기존 가우시안 + N을 해야 함.
            #-1인 경우에만
            #frame.gaussianpoints = torch.full((frame.keypoints.shape[0],),-1)
            for gauss_idx, point in enumerate(fused_point_cloud):
                kp_idx = match_index[gauss_idx]
                if kp_idx > -1:
                    frame.gaussianpoints[kp_idx] = gauss_idx + Nold
                    new_isfeature[gauss_idx] = True
                    new_observations[gauss_idx] = {frame.id:kp_idx.cpu().numpy()}
                    #new_observations[gauss_idx][frame.id] = kp_idx.numpy()
            """            
            points = project_pc_to_pixel(fused_point_cloud,
                                         cam_info.R, cam_info.T, cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy,
                                         cam_info.image_width, cam_info.image_height)
            out = copy.deepcopy(frame.color)
            
            for gauss_idx, point in enumerate(points):
                kp_idx = match_index[gauss_idx]
                if kp_idx > 0:
                    pt1 = (int(torch.round(point[0])), int(torch.round(point[1])))
                    kp = frame.keypoints[kp_idx]
                    pt2 = (int(round(kp[0])), int(round(kp[1])))
                    cv2.line(out,pt1,pt2,(0,255,0), 2)

            cv2.imshow("asdf", out)
            cv2.waitKey((1))
            """
        """
        radius = 1
        if frame is not None:
            points = project_pc_to_pixel(fused_point_cloud,
                                cam_info.R, cam_info.T,cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy,
                                cam_info.image_width, cam_info.image_height)

            out = copy.deepcopy(frame.color)
            mask = np.zeros((cam_info.image_height, cam_info.image_width), dtype=np.uint16)
            for idx, kp in enumerate(frame.keypoints):
                u, v = (int(round(kp[0])), int(round(kp[1])))
                cv2.circle(mask, (u,v), radius, idx+1, -1)
                cv2.circle(out, (u,v), 3, (0,255,0), 1)

            inside=[]
            for idx, kp in enumerate(points):
                x, y = (int(torch.round(kp[0])), int(torch.round(kp[1])))
                if x < 0 or x >= cam_info.image_width or y < 0 or y >= cam_info.image_height:
                    continue
                if mask[int(y), int(x)] > 0:
                    inside.append((x, y))
                cv2.circle(out, (x, y), 3, (255, 0, 0), 1)
            print("matching test = ", len(inside))
            cv2.imshow("asdf", out)
            cv2.waitKey((1))
        """

        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id, isfeatures=new_isfeature, observations=new_observations
        )

        """
        # 코드 확인
        valid = frame.gaussianpoints > -1
        gindex = frame.gaussianpoints[valid]
        prev_frame_gaussians = self._xyz[gindex]
        projection, valid_projection = project_pc_to_pixel(prev_frame_gaussians, cam_info.R, cam_info.T,
                                                           cam_info.fx, cam_info.fy, cam_info.cx,
                                                           cam_info.cy,
                                                           cam_info.image_width, cam_info.image_height)
        points = torch.from_numpy(frame.keypoints[valid][valid_projection]).cuda()

        out = copy.deepcopy(frame.color)
        points1 = projection.detach().cpu().numpy()
        points2 = points.detach().cpu().numpy()
        for pt1, pt2 in zip(points1, points2):
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))

            cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
            cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
            cv2.circle(out, p2, 1, (255, 0, 0), -1, lineType=16)
        cv2.imshow("asdfasdfasdf123412341234", out)
        cv2.waitKey(1)
        """

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id, isfeatures = None, observations = None
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
            new_observations=observations
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
        new_observations=None,
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
        if new_observations is not None:
            self.observations = np.concatenate((self.observations, new_observations))

    def densify_and_split(self, grads, grad_threshold, scene_extent, frames, N=2):
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

        new_isfeatures = self.isfeatured[selected_pts_mask.cpu()].repeat(N)
        new_observations = self.observations[selected_pts_mask.cpu().numpy()].repeat(N)

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
            new_observations=new_observations
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.update_gaussian_observation_before_prune(prune_filter, frames)
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
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

        new_isfeatures = self.isfeatured[selected_pts_mask.cpu()]
        new_observations = self.observations[selected_pts_mask.cpu().numpy()]

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
            new_observations=new_observations
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, frames:dict):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        #print("densify_and_prune::start", self._xyz.size())
        self.densify_and_clone(grads, max_grad, extent)
        #print("densify_and_prune::clone", self._xyz.size())
        self.densify_and_split(grads, max_grad, extent, frames)
        #print("densify_and_prune::split", self._xyz.size())
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        self.update_gaussian_observation_before_prune(prune_mask, frames)
        self.prune_points(prune_mask)
        #print("densify_and_prune::end", self._xyz.size(), self.isfeatured.size(), torch.count_nonzero(self.isfeatured),
        #      self.observations.shape, np.count_nonzero(self.observations), torch.count_nonzero(prune_mask))
        return prune_mask

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

        self.isfeatured = self.isfeatured[valid_points_mask.cpu()]
        self.observations = self.observations[valid_points_mask.cpu().numpy()]
        #observation도 처리 필요

    def update_gaussian_observation_before_prune(self, mask, frames:dict):

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
                frame.gaussianpoints[kpidx] = -1


class GaussianOrbWarppingModel:
    """
    ORB 특징점 트래킹이 가능한 가우시간 모델
    기존 GaussianModel을 래핑하여 트래킹 기능 추가
    """

    def __init__(self, base_gaussian_model, config: Dict = None):
        """
        Args:
            base_gaussian_model: 기존 GaussianModel 인스턴스
            config: 설정 딕셔너리
        """
        self.base_model = base_gaussian_model
        self.config = config or {}

        # ORB 검출기 설정
        """
        self.orb_detector = cv2.ORB_create(
            nfeatures=self.config.get('orb_features', 1000),
            scaleFactor=self.config.get('orb_scale_factor', 1.2),
            nlevels=self.config.get('orb_levels', 8),
            edgeThreshold=self.config.get('orb_edge_threshold', 31),
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.config.get('orb_fast_threshold', 20)
        )
        """

        # 트래킹 데이터: 가우시간 인덱스 -> ORB 특징점 데이터
        self.orb_features: Dict[int, ORBFeatureData] = {}

        # 트래킹 관련 메타데이터
        self.tracking_metadata = {
            'last_frame_id': -1,
            'total_tracked_gaussians': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'last_update_time': 0.0
        }

        # 필터링 파라미터
        self.min_track_confidence = self.config.get('min_track_confidence', 0.3)
        self.max_projection_distance = self.config.get('max_projection_distance', 15.0)
        self.min_opacity_threshold = self.config.get('min_opacity_threshold', 0.05)

        # 성능 모니터링
        self.performance_stats = {
            'feature_extraction_time': 0.0,
            'association_time': 0.0,
            'filtering_time': 0.0,
            'total_processing_time': 0.0
        }

        # 상태 추적 (동기화용)
        self._last_gaussian_count = 0
        self._sync_required = False

    def __getattr__(self, name):
        """기존 GaussianModel의 모든 속성/메서드에 투명하게 접근"""
        if hasattr(self.base_model, name):
            attr = getattr(self.base_model, name)

            # 가우시간 구조를 변경하는 메서드들을 감지
            if name in ['prune_points', 'densify_and_split', 'densify_and_clone',
                        'densify_and_prune', 'extend_from_pcd']:
                return self._wrap_structural_method(attr, name)

            return attr

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """속성 설정 시 base_model로 위임할지 결정"""
        # GaussianOrbModel 고유 속성들
        orb_model_attrs = {
            'base_model', 'config', 'orb_features', 'tracking_metadata',
            'min_track_confidence', 'max_projection_distance', 'min_opacity_threshold',
            'performance_stats', '_last_gaussian_count', '_sync_required'
        }

        if name in orb_model_attrs:
            # GaussianOrbModel의 고유 속성
            super().__setattr__(name, value)
        else:
            # base_model이 존재하고 해당 속성을 가지고 있으면 base_model에 설정
            if hasattr(self, 'base_model') and hasattr(self.base_model, name):
                setattr(self.base_model, name, value)
            else:
                # 그렇지 않으면 자신에게 설정
                super().__setattr__(name, value)

    def __getstate__(self):
        """pickling을 위한 상태 반환"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """unpickling을 위한 상태 복원"""
        self.__dict__.update(state)

    def _wrap_structural_method(self, original_method, method_name: str):
        """구조 변경 메서드들을 래핑하여 동기화 처리"""

        def wrapped_method(*args, **kwargs):
            # 변경 전 상태 저장
            pre_count = self.base_model.get_xyz.shape[0]
            pre_orb_features = copy.deepcopy(self.orb_features)

            # 원본 메서드 실행
            result = original_method(*args, **kwargs)

            # 변경 후 동기화
            post_count = self.base_model.get_xyz.shape[0]
            if pre_count != post_count:
                self._synchronize_after_structural_change(
                    method_name, pre_count, post_count, pre_orb_features, args, kwargs
                )

            return result

        return wrapped_method

    def _synchronize_after_structural_change(self, method_name: str, pre_count: int,
                                             post_count: int, pre_orb_features: Dict,
                                             args: tuple, kwargs: Dict):
        """구조 변경 후 ORB 특징점 데이터 동기화"""
        print(f"[TrackableGaussian] Syncing after {method_name}: {pre_count} -> {post_count}")

        if method_name == 'prune_points':
            self._sync_after_prune(args[0], pre_orb_features)  # mask가 첫 번째 인자
        elif method_name in ['densify_and_split', 'densify_and_clone']:
            self._sync_after_densification(method_name, pre_count, post_count, pre_orb_features)
        elif method_name == 'densify_and_prune':
            self._sync_after_complex_operation(pre_count, post_count, pre_orb_features)
        elif method_name == 'extend_from_pcd':
            self._sync_after_extension(pre_count, post_count)

        # 통계 업데이트
        self.tracking_metadata['total_tracked_gaussians'] = len(self.orb_features)

    def _sync_after_prune(self, mask: torch.Tensor, pre_orb_features: Dict):
        """Pruning 후 동기화"""
        valid_points_mask = ~mask

        # 새로운 인덱스 매핑 생성
        new_orb_features = {}
        new_index = 0

        for old_index in range(len(mask)):
            if valid_points_mask[old_index] and old_index in pre_orb_features:
                new_orb_features[new_index] = pre_orb_features[old_index]
                # Pruning으로 인한 신뢰도 약간 감소
                for i in range(len(new_orb_features[new_index].confidences)):
                    new_orb_features[new_index].confidences[i] *= 0.95

            if valid_points_mask[old_index]:
                new_index += 1

        self.orb_features = new_orb_features

    def _sync_after_densification(self, method_name: str, pre_count: int,
                                  post_count: int, pre_orb_features: Dict):
        """Densification 후 동기화"""
        if method_name == 'densify_and_clone':
            # Clone: 기존 데이터 유지 + 복제된 것들에 대한 새 데이터
            added_count = post_count - pre_count
            new_orb_features = copy.deepcopy(pre_orb_features)

            # 복제된 가우시간들에 대한 특징점 데이터는 별도 처리 필요
            # (실제로는 어떤 가우시간이 복제되었는지 추적 필요)

        elif method_name == 'densify_and_split':
            # Split: 일부 제거 + 새로운 것들 추가
            # 복잡한 인덱스 재매핑 필요
            self._rebuild_orb_features_after_split(pre_orb_features)

        self.orb_features = self.orb_features  # 임시

    def _sync_after_complex_operation(self, pre_count: int, post_count: int,
                                      pre_orb_features: Dict):
        """복잡한 작업 후 동기화 (위치 기반 매칭)"""
        if not pre_orb_features:
            return

        # 현재 가우시간 위치 가져오기
        current_positions = self.base_model.get_xyz.detach().cpu().numpy()

        # 거리 기반으로 가장 가까운 매칭 찾기
        new_orb_features = {}
        used_indices = set()

        for old_idx, orb_data in pre_orb_features.items():
            if not orb_data.keypoints:
                continue

            # 마지막 관측된 3D 위치 추정 (카메라 파라미터 필요, 여기서는 근사)
            last_kp = orb_data.keypoints[-1]

            # 현재 가우시간들 중 가장 가까운 것 찾기 (단순 거리)
            best_match_idx = None
            best_distance = float('inf')

            for new_idx in range(current_positions.shape[0]):
                if new_idx in used_indices:
                    continue

                # 실제로는 3D 위치와 2D 특징점 위치 간의 매칭이 필요
                # 여기서는 단순화된 버전
                distance = np.random.random()  # 실제 구현에서는 정확한 거리 계산

                if distance < best_distance and distance < 0.5:  # 임계값
                    best_distance = distance
                    best_match_idx = new_idx

            if best_match_idx is not None:
                new_orb_features[best_match_idx] = orb_data
                used_indices.add(best_match_idx)

                # 거리에 따른 신뢰도 조정
                confidence_factor = max(0.5, 1.0 - best_distance)
                for i in range(len(orb_data.confidences)):
                    orb_data.confidences[i] *= confidence_factor

        self.orb_features = new_orb_features

    def _sync_after_extension(self, pre_count: int, post_count: int):
        """확장 후 동기화 (새 가우시간들에는 특징점 데이터 없음)"""
        # 기존 ORB 특징점 데이터는 그대로 유지
        # 새로 추가된 가우시간들은 다음 update에서 특징점 할당됨
        pass

    def _rebuild_orb_features_after_split(self, pre_orb_features: Dict):
        """Split 후 ORB 특징점 데이터 재구성"""
        # Split의 경우 복잡한 재매핑이 필요
        # 여기서는 단순화된 버전
        self.orb_features = {}  # 임시로 초기화, 다음 update에서 재구성

    """입력 파라메터 수정 필요"""
    def _project_gaussians_to_2d(self, camera_params: Dict, pose: np.ndarray) -> np.ndarray:
        """가우시안을 2D로 투영"""
        # 카메라 파라미터 추출
        fx, fy = camera_params['fx'], camera_params['fy']
        cx, cy = camera_params['cx'], camera_params['cy']

        # 가우시안 위치를 카메라 좌표계로 변환
        xyz_world = self.base_model._xyz.detach().cpu().numpy()

        # 동차좌표로 변환
        xyz_homo = np.hstack([xyz_world, np.ones((xyz_world.shape[0], 1))])

        # 카메라 좌표계로 변환
        xyz_cam = (pose @ xyz_homo.T).T[:, :3]

        # 카메라 뒤쪽 점들 필터링
        valid_mask = xyz_cam[:, 2] > 0.1

        # 2D 투영
        projected_2d = np.zeros((xyz_world.shape[0], 4))  # [x, y, scale, opacity]

        if np.any(valid_mask):
            valid_xyz = xyz_cam[valid_mask]

            # 투영
            u = fx * valid_xyz[:, 0] / valid_xyz[:, 2] + cx
            v = fy * valid_xyz[:, 1] / valid_xyz[:, 2] + cy

            # 스케일과 투명도 정보 추가
            scaling = torch.exp(self.base_model._scaling[valid_mask]).detach().cpu().numpy()
            opacity = torch.sigmoid(self.base_model._opacity[valid_mask]).detach().cpu().numpy()

            projected_2d[valid_mask, 0] = u
            projected_2d[valid_mask, 1] = v
            projected_2d[valid_mask, 2] = np.mean(scaling, axis=1)  # 평균 스케일
            projected_2d[valid_mask, 3] = opacity.flatten()

        return projected_2d

    """이 함수는 대폭 변경될 수 있음."""
    def _associate_features_with_gaussians(self, keypoints: List[cv2.KeyPoint],
                                           descriptors: np.ndarray,
                                           projected_gaussians: np.ndarray,
                                           frame_id: int):
        """특징점과 가우시안 연결"""
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            # 특징점 주변의 가우시안 찾기
            nearby_indices = self._find_nearby_gaussians(kp.pt, projected_gaussians)

            if len(nearby_indices) == 0:
                continue

            # 최적 가우시간 선택
            best_idx = self._select_best_gaussian(kp, nearby_indices, projected_gaussians)

            if best_idx is not None:
                # 트래킹 메타데이터 업데이트
                self._update_tracking_metadata(best_idx, kp, desc, frame_id)

    def _find_nearby_gaussians(self, feature_point: Tuple[float, float],
                               projected_gaussians: np.ndarray,
                               radius: float = 15.0) -> List[int]:
        """특징점 주변 가우시안 찾기"""
        fx, fy = feature_point
        nearby_indices = []

        for idx, (px, py, scale, opacity) in enumerate(projected_gaussians):
            if opacity < self.opacity_threshold:
                continue

            distance = np.sqrt((fx - px) ** 2 + (fy - py) ** 2)
            effective_radius = radius * max(scale, 0.5) * opacity

            if distance < effective_radius and distance < self.max_tracking_distance:
                nearby_indices.append(idx)

        return nearby_indices

    def _select_best_gaussian(self, keypoint: cv2.KeyPoint,
                              nearby_indices: List[int],
                              projected_gaussians: np.ndarray) -> Optional[int]:
        """최적 가우시안 선택"""
        best_idx = None
        best_score = -1

        fx, fy = keypoint.pt

        for idx in nearby_indices:
            px, py, scale, opacity = projected_gaussians[idx]

            # 거리 스코어
            distance = np.sqrt((fx - px) ** 2 + (fy - py) ** 2)
            distance_score = 1.0 / (1.0 + distance)

            # 투명도 스코어
            opacity_score = opacity

            # 스케일 스코어 (너무 크거나 작으면 패널티)
            scale_score = 1.0 / (1.0 + abs(scale - 1.0))

            # 기존 트래킹 히스토리 스코어
            history_score = 1.0
            if idx in self.tracking_metadata:
                history_score = min(self.tracking_metadata[idx].track_confidence, 1.0)

            # 종합 스코어
            total_score = distance_score * opacity_score * scale_score * history_score

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        return best_idx if best_score > 0.2 else None



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