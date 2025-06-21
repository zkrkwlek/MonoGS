import torch
import numpy as np

from typing import Dict, List, Tuple, Optional, Set

from utils.ObjectMap.bbox_mapping_utils import MultiViewBBoxMapping
from utils.ObjectMap.bbox_utils import BoundingBox
from gaussian_splatting.scene.gaussian_model import GaussianModel

from utils.logging_utils import Log

import open3d as o3d
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.general_utils import inverse_sigmoid
from simple_knn._C import distCUDA2

class SelectiveGaussianModel(GaussianModel):
    """Gaussian Model with selective update and multi-view consistency"""

    def __init__(self, sh_degree: int, object_id: int, config=None):
        super().__init__(sh_degree, config)
        self.object_id = object_id

        # Multi-view bbox mapping
        self.multiview_mappings: Dict[str, MultiViewBBoxMapping] = {}  # track_id -> mapping
        self.gaussian_to_track: torch.Tensor = torch.empty(0, dtype=torch.long)  # gaussian_idx -> track_id hash
        self.track_ids: List[str] = []  # List of active track IDs
        self.track_id_to_hash: Dict[str, int] = {}  # track_id -> hash for tensor indexing

        # Selective update control
        self.update_mask: torch.Tensor = torch.empty(0, dtype=torch.bool)  # Which Gaussians to update
        self.freeze_unselected = True  # Freeze non-selected Gaussians during optimization

        # Multi-view consistency
        self.multiview_constraint_weight = 1.0
        self.bbox_consistency_threshold = 0.3  # IoU threshold for consistency
        self.max_views_for_constraint = 5  # Maximum views to consider for constraint

        # Projection cache for efficiency
        self.projection_cache: Dict[int, torch.Tensor] = {}  # view_id -> 2D projections

    def add_multiview_bbox_mapping(self, bbox: BoundingBox, frame_id: int, view_id: int) -> str:
        """Add bbox mapping with multi-view support"""
        track_id = bbox.track_id

        if track_id not in self.multiview_mappings:
            # Create new mapping
            track_hash = len(self.track_ids)
            self.track_id_to_hash[track_id] = track_hash
            self.track_ids.append(track_id)

            self.multiview_mappings[track_id] = MultiViewBBoxMapping(
                track_id=track_id,
                gaussian_indices=torch.empty(0, dtype=torch.long),
                bbox_history={},
                creation_frame=frame_id,
                last_update_frame=frame_id,
                confidence_history=[bbox.confidence],
                active_views=set()
            )
            Log(f"Created new multi-view mapping for track_id: {track_id}")

        # Check consistency before adding
        mapping = self.multiview_mappings[track_id]
        if mapping.is_consistent_across_views(view_id, bbox, self.bbox_consistency_threshold):
            mapping.add_view_bbox(view_id, bbox)
            mapping.last_update_frame = frame_id
            Log(f"Added view {view_id} bbox for track_id: {track_id}")
        else:
            Log(f"Warning: Inconsistent bbox for track_id {track_id} in view {view_id}")

        return track_id

    def project_gaussians_to_view(self, viewpoint, view_id: int = None) -> torch.Tensor:
        """Project 3D Gaussians to 2D view coordinates"""
        if view_id is not None and view_id in self.projection_cache:
            return self.projection_cache[view_id]

        gaussians_3d = self.get_xyz
        if len(gaussians_3d) == 0:
            return torch.empty(0, 2)

        # Transform to camera coordinates
        world_to_view = viewpoint.world_view_transform
        proj_matrix = viewpoint.full_proj_transform

        # Project 3D points to 2D
        gaussians_homo = torch.cat([gaussians_3d, torch.ones(gaussians_3d.shape[0], 1, device=gaussians_3d.device)], dim=1)
        projected = torch.matmul(gaussians_homo, proj_matrix.T)

        # Perspective divide and convert to pixel coordinates
        x_2d = (projected[:, 0] / (projected[:, 3] + 1e-8) + 1.0) * 0.5 * viewpoint.image_width
        y_2d = (projected[:, 1] / (projected[:, 3] + 1e-8) + 1.0) * 0.5 * viewpoint.image_height

        projected_2d = torch.stack([x_2d, y_2d], dim=1)

        # Cache projection
        if view_id is not None:
            self.projection_cache[view_id] = projected_2d

        return projected_2d

    def check_multiview_consistency(self, track_id: str, viewpoints: List, view_ids: List[int]) -> torch.Tensor:
        """Check if Gaussians are consistent across multiple views for a track"""
        if track_id not in self.multiview_mappings:
            return torch.ones(0, dtype=torch.bool)

        mapping = self.multiview_mappings[track_id]
        gaussian_indices = mapping.gaussian_indices

        if len(gaussian_indices) == 0 or len(viewpoints) == 0:
            return torch.ones(len(gaussian_indices), dtype=torch.bool)

        consistency_mask = torch.ones(len(gaussian_indices), dtype=torch.bool, device=self.device)

        for i, (viewpoint, view_id) in enumerate(zip(viewpoints, view_ids)):
            # Get bbox for this view
            bbox = mapping.get_bbox_for_view(view_id)
            if bbox is None:
                continue

            # Project Gaussians to this view
            projected_2d = self.project_gaussians_to_view(viewpoint, view_id)

            if len(projected_2d) == 0:
                continue

            # Check which Gaussians are inside bbox
            gaussian_projections = projected_2d[gaussian_indices]
            expanded_bbox = bbox.expand(1.1)  # Allow slight expansion

            inside_bbox = (
                    (gaussian_projections[:, 0] >= expanded_bbox.x1) &
                    (gaussian_projections[:, 0] <= expanded_bbox.x2) &
                    (gaussian_projections[:, 1] >= expanded_bbox.y1) &
                    (gaussian_projections[:, 1] <= expanded_bbox.y2)
            )

            # Update consistency mask (Gaussian must be inside bbox in ALL views)
            consistency_mask = consistency_mask & inside_bbox

        return consistency_mask

    def set_selective_update_mask(self, target_track_ids: List[str]):
        """Set which Gaussians should be updated based on track IDs"""
        if len(self.get_xyz) == 0:
            self.update_mask = torch.empty(0, dtype=torch.bool)
            return

        self.update_mask = torch.zeros(len(self.get_xyz), dtype=torch.bool, device=self.device)

        for track_id in target_track_ids:
            if track_id in self.multiview_mappings:
                mapping = self.multiview_mappings[track_id]
                if len(mapping.gaussian_indices) > 0:
                    self.update_mask[mapping.gaussian_indices] = True

        Log(f"Set selective update mask: {self.update_mask.sum().item()}/{len(self.update_mask)} Gaussians selected")

    def selective_extend_from_bbox(self, cam_info, bbox: BoundingBox, kf_id: int, view_id: int,
                                   init=False, scale=2.0, depthmap=None):
        """Selectively extend Gaussians for specific bbox with multi-view consistency"""
        # Add multi-view bbox mapping
        track_id = self.add_multiview_bbox_mapping(bbox, kf_id, view_id)

        # Create point cloud from bbox region
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_bbox_region(cam_info, bbox, init, scale=scale, depthmap=depthmap))

        # Get starting index for new Gaussians
        start_idx = self.get_xyz.shape[0] if hasattr(self, '_xyz') else 0
        num_new_gaussians = fused_point_cloud.shape[0]

        # Extend Gaussians
        self.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id)

        # Update mapping with new Gaussian indices
        if num_new_gaussians > 0:
            new_indices = torch.arange(start_idx, start_idx + num_new_gaussians, dtype=torch.long)
            mapping = self.multiview_mappings[track_id]
            mapping.gaussian_indices = torch.cat([mapping.gaussian_indices, new_indices])

            # Update gaussian-to-track mapping
            track_hash = self.track_id_to_hash[track_id]
            new_track_assignments = torch.full((num_new_gaussians,), track_hash, dtype=torch.long)
            if len(self.gaussian_to_track) == 0:
                self.gaussian_to_track = new_track_assignments
            else:
                self.gaussian_to_track = torch.cat([self.gaussian_to_track, new_track_assignments])

            # Update selective mask to include new Gaussians
            if len(self.update_mask) == 0:
                self.update_mask = torch.ones(num_new_gaussians, dtype=torch.bool, device=self.device)
            else:
                new_mask = torch.ones(num_new_gaussians, dtype=torch.bool, device=self.device)
                self.update_mask = torch.cat([self.update_mask, new_mask])

        Log(f"Selectively extended {num_new_gaussians} Gaussians for track {track_id} in view {view_id}")
        return track_id

    def create_pcd_from_bbox_region(self, cam_info, bbox: BoundingBox,
                                    init=False, scale=2.0, depthmap=None):
        """Create point cloud from bounding box region"""
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        # Crop to bbox region
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(rgb_raw.shape[1], x2)
        y2 = min(rgb_raw.shape[0], y2)

        rgb_cropped = rgb_raw[y1:y2, x1:x2]

        if depthmap is not None:
            depth_cropped = depthmap[y1:y2, x1:x2]
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.ones((cam.image_height, cam.image_width)) * scale
            depth_cropped = depth_raw[y1:y2, x1:x2]

        if rgb_cropped.size == 0 or depth_cropped.size == 0:
            # Return empty point cloud if invalid crop
            from gaussian_splatting.utils.graphics_utils import BasicPointCloud
            from gaussian_splatting.utils.sh_utils import RGB2SH
            from gaussian_splatting.utils.general_utils import inverse_sigmoid

            empty_points = np.empty((0, 3))
            empty_colors = np.empty((0, 3))
            pcd = BasicPointCloud(points=empty_points, colors=empty_colors, normals=empty_points)

            fused_point_cloud = torch.empty(0, 3, device="cuda")
            features = torch.empty(0, 3, (self.max_sh_degree + 1) ** 2, device="cuda")
            scales = torch.empty(0, 3 if not self.isotropic else 1, device="cuda")
            rots = torch.empty(0, 4, device="cuda")
            opacities = torch.empty(0, 1, device="cuda")

            return fused_point_cloud, features, scales, rots, opacities

        rgb = o3d.geometry.Image(rgb_cropped.astype(np.uint8))
        depth = o3d.geometry.Image(depth_cropped.astype(np.float32))

        return self.create_pcd_from_bbox_image_and_depth(cam, rgb, depth, bbox, init)

    def create_pcd_from_bbox_image_and_depth(self, cam, rgb, depth, bbox: BoundingBox, init=False):
        """Create point cloud from cropped bbox image and depth"""
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]

        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                depth_array = np.asarray(depth)
                if depth_array.size > 0:
                    point_size = min(0.05, point_size * np.median(depth_array))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)

        # Adjust camera intrinsics for cropped region
        fx, fy = cam.fx, cam.fy
        cx_cropped = cam.cx - bbox.x1
        cy_cropped = cam.cy - bbox.y1

        # Ensure valid intrinsics
        if cx_cropped < 0 or cy_cropped < 0:
            cx_cropped = bbox.width / 2
            cy_cropped = bbox.height / 2

        from gaussian_splatting.utils.graphics_utils import getWorld2View2
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()

        try:
            pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    bbox.width, bbox.height, fx, fy, cx_cropped, cy_cropped),
                extrinsic=W2C,
                project_valid_depth_only=True,
            )

            if len(pcd_tmp.points) > 0:
                pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)

        except:
            pcd_tmp = o3d.geometry.PointCloud()

        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        if len(new_xyz) == 0:
            # Create single point at bbox center if no points found
            bbox_center_3d = np.array([[0, 0, 2]], dtype=np.float32)  # Default depth
            bbox_center_color = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
            new_xyz = bbox_center_3d
            new_rgb = bbox_center_color



        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3)))
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
                    .float().cuda())
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if len(fused_point_cloud) > 1:
            dist2 = (torch.clamp_min(
                distCUDA2(fused_point_cloud), 0.0000001) * point_size)
        else:
            dist2 = torch.ones(len(fused_point_cloud), device="cuda") * point_size

        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1),
                             dtype=torch.float, device="cuda"))

        return fused_point_cloud, features, scales, rots, opacities

    def enforce_multiview_consistency(self, viewpoints: List, view_ids: List[int]):
        """Enforce multi-view consistency by pruning inconsistent Gaussians"""
        if len(self.multiview_mappings) == 0 or len(viewpoints) <= 1:
            return

        points_to_prune = torch.zeros(len(self.get_xyz), dtype=torch.bool, device=self.device)

        for track_id, mapping in self.multiview_mappings.items():
            if len(mapping.gaussian_indices) == 0:
                continue

            # Check consistency across views
            consistency_mask = self.check_multiview_consistency(track_id, viewpoints, view_ids)

            # Mark inconsistent Gaussians for pruning
            if len(consistency_mask) > 0:
                inconsistent_indices = mapping.gaussian_indices[~consistency_mask]
                if len(inconsistent_indices) > 0:
                    points_to_prune[inconsistent_indices] = True
                    Log(f"Marking {len(inconsistent_indices)} inconsistent Gaussians for pruning in track {track_id}")

        # Prune inconsistent points
        if points_to_prune.any():
            self.prune_points(points_to_prune)
            self._update_all_mappings_after_pruning(points_to_prune)
            Log(f"Pruned {points_to_prune.sum().item()} inconsistent Gaussians")

    def _update_all_mappings_after_pruning(self, pruned_mask: torch.Tensor):
        """Update all mappings after pruning"""
        # Create index mapping from old to new indices
        remaining_indices = torch.where(~pruned_mask)[0]
        old_to_new = torch.full((len(pruned_mask),), -1, dtype=torch.long)
        old_to_new[remaining_indices] = torch.arange(len(remaining_indices))

        # Update all mappings
        for track_id, mapping in self.multiview_mappings.items():
            if len(mapping.gaussian_indices) > 0:
                # Get new indices for remaining Gaussians
                old_indices = mapping.gaussian_indices
                mask = old_to_new[old_indices] >= 0
                new_indices = old_to_new[old_indices[mask]]
                mapping.gaussian_indices = new_indices

        # Update selective mask
        if len(self.update_mask) > 0:
            self.update_mask = self.update_mask[~pruned_mask]

        # Update gaussian-to-track mapping
        if len(self.gaussian_to_track) > 0:
            self.gaussian_to_track = self.gaussian_to_track[~pruned_mask]

    def selective_training_step(self):
        """Perform training step only on selected Gaussians"""
        if self.freeze_unselected and len(self.update_mask) > 0:
            # Store gradients of selected Gaussians only
            selected_params = {}
            for param_group in self.optimizer.param_groups:
                param = param_group["params"][0]
                if param.grad is not None:
                    # Apply mask to gradients
                    if param.grad.shape[0] == len(self.update_mask):
                        selected_grad = param.grad.clone()
                        selected_grad[~self.update_mask] = 0  # Zero out gradients for unselected
                        param.grad = selected_grad

        # Perform optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def get_selective_update_statistics(self) -> Dict:
        """Get statistics about selective updates"""
        total_gaussians = len(self.get_xyz) if hasattr(self, '_xyz') else 0
        selected_gaussians = self.update_mask.sum().item() if len(self.update_mask) > 0 else 0

        stats = {
            'total_gaussians': total_gaussians,
            'selected_gaussians': selected_gaussians,
            'selection_ratio': selected_gaussians / total_gaussians if total_gaussians > 0 else 0,
            'track_details': {}
        }

        for track_id, mapping in self.multiview_mappings.items():
            stats['track_details'][track_id] = {
                'gaussian_count': len(mapping.gaussian_indices),
                'active_views': len(mapping.active_views),
                'avg_confidence': np.mean(mapping.confidence_history) if mapping.confidence_history else 0
            }

        return stats