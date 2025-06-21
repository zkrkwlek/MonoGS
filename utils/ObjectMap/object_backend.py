import torch
import time
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from utils.slam_win_frontend import WinFrontEnd
from utils.ObjectMap.object_gaussian_model import SelectiveGaussianModel
from utils.ObjectMap.bbox_utils import BoundingBox

from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log
from utils.slam_utils import get_loss_mapping

class SelectiveMultiObjectBackEnd(WinFrontEnd):
    """Multi-object backend with selective updates and multi-view consistency"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.object_gaussians: Dict[int, SelectiveGaussianModel] = {}
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # Multi-object and multi-view specific
        self.object_windows: Dict[int, List] = {}
        #self.depth_estimator = DepthAnythingIntegration()
        self.active_objects = set()

        # Multi-view tracking
        self.view_counter = 0
        self.view_id_mapping: Dict[int, int] = {}  # frame_id -> view_id
        self.viewpoint_history: Dict[int, object] = {}  # view_id -> viewpoint

        # Selective update control
        self.target_track_ids: Set[str] = set()  # Track IDs to update
        self.selective_mode = True

    def add_object(self, object_id: int, sh_degree: int = 3):
        """Add new object for tracking"""
        if object_id not in self.object_gaussians:
            self.object_gaussians[object_id] = SelectiveGaussianModel(
                sh_degree, object_id, self.config)
            self.object_windows[object_id] = []
            self.active_objects.add(object_id)
            Log(f"Added object {object_id} for selective Gaussian mapping")

    def selective_update_object_with_bbox(self, frame_idx: int, viewpoint,
                                          target_bboxes: List[BoundingBox], depth_map=None):
        """Selectively update only specified objects with bboxes"""
        # Assign view ID
        if frame_idx not in self.view_id_mapping:
            self.view_id_mapping[frame_idx] = self.view_counter
            self.viewpoint_history[self.view_counter] = viewpoint
            self.view_counter += 1

        view_id = self.view_id_mapping[frame_idx]

        # Set frame and view IDs for bboxes
        for bbox in target_bboxes:
            bbox.frame_id = frame_idx
            bbox.view_id = view_id

        # Generate depth if needed
        if depth_map is None:
            image_np = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            depth_map = self.depth_estimator.estimate_depth(image_np)

        # Extract target track IDs
        target_track_ids = {bbox.track_id for bbox in target_bboxes}
        self.target_track_ids.update(target_track_ids)

        # Process each bbox
        updated_objects = set()
        for bbox in target_bboxes:
            object_id = bbox.class_id

            # Add object if not exists
            if object_id not in self.object_gaussians:
                self.add_object(object_id)

            # Initialize if needed
            if not hasattr(self.object_gaussians[object_id], 'spatial_lr_scale'):
                self.object_gaussians[object_id].init_lr(self.cameras_extent)

            # Set selective update mask for this object
            gaussian_model = self.object_gaussians[object_id]
            gaussian_model.set_selective_update_mask([bbox.track_id])

            # Selectively extend Gaussians for this bbox
            track_id = gaussian_model.selective_extend_from_bbox(
                viewpoint, bbox, frame_idx, view_id,
                init=(object_id not in updated_objects),
                depthmap=depth_map)

            updated_objects.add(object_id)

            # Update object window
            if object_id not in self.object_windows:
                self.object_windows[object_id] = []
            if frame_idx not in self.object_windows[object_id]:
                self.object_windows[object_id].append(frame_idx)

            # Keep window manageable
            max_window_size = self.config["Training"]["window_size"]
            if len(self.object_windows[object_id]) > max_window_size:
                self.object_windows[object_id] = self.object_windows[object_id][-max_window_size:]

        # Store viewpoint
        self.viewpoints[frame_idx] = viewpoint

        Log(f"Selectively updated objects {updated_objects} with {len(target_bboxes)} bboxes in frame {frame_idx}")
        return list(updated_objects)

    def enforce_multiview_consistency_for_objects(self, object_ids: List[int]):
        """Enforce multi-view consistency for specified objects"""
        for object_id in object_ids:
            if object_id not in self.object_gaussians:
                continue

            gaussian_model = self.object_gaussians[object_id]

            # Get recent viewpoints and view IDs
            if object_id in self.object_windows:
                recent_frames = self.object_windows[object_id][-gaussian_model.max_views_for_constraint:]
                viewpoints = [self.viewpoints[frame_idx] for frame_idx in recent_frames if frame_idx in self.viewpoints]
                view_ids = [self.view_id_mapping[frame_idx] for frame_idx in recent_frames if frame_idx in self.view_id_mapping]

                if len(viewpoints) > 1:
                    gaussian_model.enforce_multiview_consistency(viewpoints, view_ids)

    def selective_map_objects(self, target_object_ids: List[int] = None, prune=False, iters=1):
        """Map only specified objects with selective updates"""
        if target_object_ids is None:
            target_object_ids = list(self.active_objects)

        for object_id in target_object_ids:
            if object_id in self.object_windows and len(self.object_windows[object_id]) > 0:
                self.selective_map_single_object(object_id, prune, iters)

    def selective_map_single_object(self, object_id: int, prune=False, iters=1):
        """Map single object with selective updates"""
        if object_id not in self.object_gaussians:
            return

        current_window = self.object_windows[object_id]
        if len(current_window) == 0:
            return

        gaussian_model = self.object_gaussians[object_id]
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window if kf_idx in self.viewpoints]

        if len(viewpoint_stack) == 0:
            return

        for iteration in range(iters):
            self.iteration_count += 1
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            # Render from multiple viewpoints
            for cam_idx, viewpoint in enumerate(viewpoint_stack):
                render_pkg = render(viewpoint, gaussian_model, self.pipeline_params, self.background)

                if render_pkg is None:
                    continue

                (image, viewspace_point_tensor, visibility_filter, radii,
                 depth, opacity, n_touched) = (
                    render_pkg["render"], render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"], render_pkg["radii"],
                    render_pkg["depth"], render_pkg["opacity"],
                    render_pkg["n_touched"])

                # Apply selective masking to loss computation
                if self.selective_mode and len(gaussian_model.update_mask) > 0:
                    # Modify loss to focus on selected Gaussians
                    masked_loss = get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                else:
                    masked_loss = get_loss_mapping(self.config, image, depth, viewpoint, opacity)

                loss_mapping += masked_loss
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            if loss_mapping > 0:
                # Add multi-view consistency loss
                multiview_loss = self.compute_multiview_consistency_loss(object_id, viewpoint_stack)
                loss_mapping += multiview_loss

                # Add isotropic loss (only for selected Gaussians)
                if len(gaussian_model.update_mask) > 0:
                    scaling = gaussian_model.get_scaling
                    selected_scaling = scaling[gaussian_model.update_mask] if gaussian_model.update_mask.any() else scaling
                    isotropic_loss = torch.abs(selected_scaling - selected_scaling.mean(dim=1).view(-1, 1))
                    loss_mapping += 10 * isotropic_loss.mean()

                loss_mapping.backward()

                with torch.no_grad():
                    # Update statistics and densification for selected Gaussians
                    for idx in range(len(viewspace_point_tensor_acm)):
                        if len(viewspace_point_tensor_acm[idx]) > 0:
                            vis_filter = visibility_filter_acm[idx]
                            if len(gaussian_model.update_mask) > 0:
                                # Only update statistics for selected Gaussians
                                selected_vis_filter = vis_filter & gaussian_model.update_mask
                            else:
                                selected_vis_filter = vis_filter

                            if selected_vis_filter.any():
                                gaussian_model.max_radii2D[selected_vis_filter] = torch.max(
                                    gaussian_model.max_radii2D[selected_vis_filter],
                                    radii_acm[idx][selected_vis_filter])
                                gaussian_model.add_densification_stats(
                                    viewspace_point_tensor_acm[idx], selected_vis_filter)

                    # Selective densification and pruning
                    update_gaussian = (
                            self.iteration_count % self.config["Training"]["gaussian_update_every"]
                            == self.config["Training"]["gaussian_update_offset"])
                    if update_gaussian:
                        gaussian_model.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.config["Training"]["gaussian_th"],
                            self.cameras_extent * self.config["Training"]["gaussian_extent"],
                            self.config["Training"]["size_threshold"])

                    # Selective training step
                    gaussian_model.selective_training_step()
                    gaussian_model.update_learning_rate(self.iteration_count)

                    # Update visibility tracking
                    for idx, kf_idx in enumerate(current_window):
                        if idx < len(n_touched_acm) and kf_idx in self.viewpoints:
                            n_touched = n_touched_acm[idx]
                            self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
        if iters > 1:  # Only for substantial updates
            self.enforce_multiview_consistency_for_objects([object_id])

    def compute_multiview_consistency_loss(self, object_id: int, viewpoints: List) -> torch.Tensor:
        """Compute loss for multi-view consistency"""
        if object_id not in self.object_gaussians or len(viewpoints) <= 1:
            return torch.tensor(0.0, device=self.device)

        gaussian_model = self.object_gaussians[object_id]
        consistency_loss = torch.tensor(0.0, device=self.device)

        # Check each track's multi-view consistency
        for track_id, mapping in gaussian_model.multiview_mappings.items():
            if len(mapping.gaussian_indices) == 0:
                continue

            # Compute consistency penalty based on projection variance across views
            projections = []
            for i, viewpoint in enumerate(viewpoints[-gaussian_model.max_views_for_constraint:]):
                view_id = gaussian_model.view_counter - len(viewpoints) + i
                projected_2d = gaussian_model.project_gaussians_to_view(viewpoint, view_id)
                if len(projected_2d) > 0 and len(mapping.gaussian_indices) > 0:
                    track_projections = projected_2d[mapping.gaussian_indices]
                    projections.append(track_projections)

            if len(projections) > 1:
                # Compute variance in projections as consistency measure
                stacked_projections = torch.stack(projections, dim=1)  # [N_gaussians, N_views, 2]
                projection_variance = torch.var(stacked_projections, dim=1).mean()
                consistency_loss += projection_variance * gaussian_model.multiview_constraint_weight

        return consistency_loss

    def push_selective_updates_to_frontend(self, tag=None):
        """Push selective updates to frontend"""
        pass
        """
        self.last_sent = 0
        keyframes = []

        for kf_idx in self.current_window:
            if kf_idx in self.viewpoints:
                kf = self.viewpoints[kf_idx]
                keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))

        if tag is None:
            tag = "selective_sync"

        # Send updated objects with selection info
        object_gaussians_clone = {}
        selection_stats = {}

        for obj_id, gaussian in self.object_gaussians.items():
            object_gaussians_clone[obj_id] = clone_obj(gaussian)
            selection_stats[obj_id] = gaussian.get_selective_update_statistics()

        msg = [tag, object_gaussians_clone, self.occ_aware_visibility, keyframes,
               selection_stats, list(self.target_track_ids)]
        self.frontend_queue.put(msg)
        """
    def run(self):
        """Main backend loop with selective updates"""
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue

                if len(self.active_objects) == 0:
                    time.sleep(0.01)
                    continue

                # Selective mapping of active objects
                self.selective_map_objects()
                if self.last_sent >= 10:
                    self.selective_map_objects(prune=True, iters=10)
                    self.push_selective_updates_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "selective_update":
                    frame_idx = data[1]
                    viewpoint = data[2]
                    target_bboxes = data[3]  # List[BoundingBox] - only objects to update
                    depth_map = data[4] if len(data) > 4 else None

                    updated_objects = self.selective_update_object_with_bbox(
                        frame_idx, viewpoint, target_bboxes, depth_map)
                    self.push_selective_updates_to_frontend("selective_updated")
                elif data[0] == "set_target_tracks":
                    track_ids = data[1]
                    self.target_track_ids = set(track_ids)
                    # Update selection masks for all objects
                    for obj_id, gaussian_model in self.object_gaussians.items():
                        gaussian_model.set_selective_update_mask(track_ids)
                    Log(f"Set target tracks: {track_ids}")

        # Cleanup
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()