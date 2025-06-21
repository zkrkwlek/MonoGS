import numpy as np

from utils.slam_win_backend import WinBackEnd
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
#from utils.slam_utils import get_loss_mapping
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss
from edge_assisted.gaussian_feature import project_pc_to_pixel
from utils.edgeframe_utils import EdgeFrame

from utils.datahandle_utils import move_camera_to_gpu, move_camera_to_cpu, move_gaussianmodel_to_cpu
from utils.datahandle_utils import move_occ_visibility_to_cpu
#from edge_assisted.gaussian_feature import GaussianPointManager

class EdgeBackEnd(WinBackEnd):
    def __init__(self, config):
        super().__init__(config)
        self.first_kf_id = None
        self.pose_update = None
        #self.dataset = None
        self.FeatureManager = None
        self.frames={}

    def push_to_frontend(self, tag=None, first_id = None):

        self.last_sent = 0
        keyframes = []
        frames = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone().cpu(), kf.T.clone().cpu()))

        #처음 가우시안 포인트 전송 용
        if first_id is not None:
            self.current_window.append(first_id)
        for kf_idx in self.current_window:
            f = self.frames[kf_idx]
            frames.append((kf_idx, f.gaussianpoints.clone()))
        if first_id is not None:
            self.current_window = []

        if tag is None:
            tag = "sync_backend"
        #print("push_to_frontend::end", len(frames))
        msg = [tag, move_gaussianmodel_to_cpu(self.gaussians), move_occ_visibility_to_cpu(self.occ_aware_visibility), (keyframes), (frames)]
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
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,frame=frame
        )

    def update_gaussian_observation_after_prune(self):
        print('update_gaussian_observation_after_prune',self.gaussians._xyz.size(), self.gaussians.isfeatured.size(), self.gaussians.observations.shape,
              torch.count_nonzero(self.gaussians.isfeatured), np.count_nonzero(self.gaussians.observations), np.sum(self.gaussians.observations!=None))
        feature_indices = self.gaussians.isfeatured.clone().cpu().numpy()
        gaussian_indices = torch.arange(self.gaussians._xyz.size()[0])
        gaussian_obs = self.gaussians.observations[feature_indices]
        gaussian_indices = gaussian_indices[feature_indices]
        #Ng = self.gaussians._xyz.size()[0]

        for gaussian_index, obs in zip(gaussian_indices, gaussian_obs):
            if obs is None:
                #print(gaussian_index, obs)
                self.gaussians.isfeatured[gaussian_index] = False
                continue
            #if gaussian_index > Ng :
            #    print('gaussian index error', gaussian_index)
            for fid, kpidx in obs.items():
                #print("update_gaussian frame", fid)
                frame = self.frames[(fid)]
                frame.gaussianpoints[kpidx] = gaussian_index

    def initialize_map(self, cur_frame_idx, viewpoint):

        curr_frame = self.frames[(cur_frame_idx)]

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
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            """
            valid = curr_frame.gaussianpoints > -1
            gindex = curr_frame.gaussianpoints[valid]
            curr_frame_gaussians = self.gaussians._xyz[gindex]
            projection, valid_projection = project_pc_to_pixel(curr_frame_gaussians, viewpoint.R, viewpoint.T,
                                                               viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                               viewpoint.cy,
                                                               viewpoint.image_width, viewpoint.image_height)
            points = torch.from_numpy(curr_frame.keypoints[valid][valid_projection]).cuda()
            """
            projection, points = curr_frame.get_correspondence(self.gaussians, viewpoint.R, viewpoint.T,
                viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                viewpoint.image_width, viewpoint.image_height)
            loss_init += get_reprojection_loss(projection,points)

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
                    prune_mask = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                        self.frames
                    )
                    """
                    gaussian_indices = torch.arange(self.gaussians._xyz.size()[0])
                    gaussian_obs = self.gaussians.observations[self.gaussians.isfeatured]
                    gaussian_indices = gaussian_indices[self.gaussians.isfeatured]
                    for gaussian_index, obs in zip(gaussian_indices,gaussian_obs):
                        for fid, kpidx in obs.items():
                            frame = self.dataset[str(fid)]
                            frame.gaussianpoints[kpidx] = gaussian_index
                    """
                    self.update_gaussian_observation_after_prune()

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

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
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                ###reprojection error 추가
                #print("mapping test", viewpoint.uid)
                keyframe = self.frames[(viewpoint.uid)]
                if keyframe is not None:
                    projection, points = keyframe.get_correspondence(self.gaussians, viewpoint.R, viewpoint.T,
                                                                       viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                                       viewpoint.image_width, viewpoint.image_height,
                                                                     delta_rot=viewpoint.cam_rot_delta,
                                                                     delta_trans=viewpoint.cam_trans_delta,
                                                                     )
                    loss_mapping += get_reprojection_loss(projection, points)
                ###reprojection error 추가

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
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                ###reprojection error 추가
                keyframe = self.frames[(viewpoint.uid)]
                if keyframe is not None:
                    projection, points = keyframe.get_correspondence(self.gaussians, viewpoint.R, viewpoint.T,
                                                                     viewpoint.fx, viewpoint.fy, viewpoint.cx,viewpoint.cy,
                                                                     viewpoint.image_width, viewpoint.image_height,
                                                                     delta_rot=viewpoint.cam_rot_delta,
                                                                     delta_trans=viewpoint.cam_trans_delta,
                                                                     )
                    loss_mapping += get_reprojection_loss(projection, points)
                ###reprojection error 추가

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

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
                            self.gaussians.update_gaussian_observation_before_prune(to_prune.cuda(), self.frames)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                            self.update_gaussian_observation_after_prune()
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    prune_mask = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                        self.frames
                    )
                    self.update_gaussian_observation_after_prune()
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.first_kf_id or not self.pose_update:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def run(self):
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
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
                e = time.time()
                #print("backend = mapping with empty queue", (e-s))
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
                    frame.keypoints, frame.descriptors, frame.gaussianpoints = f
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

                    self.initialize_map(cur_frame_idx, viewpoint)

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
                    frame.keypoints, frame.descriptors, frame.gaussianpoints = f
                    self.frames[(cur_frame_idx)] = frame

                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, frame=frame)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
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
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    e1 = time.time()
                    self.push_to_frontend("keyframe")
                    e2 = time.time()
                    print(cur_frame_idx, self.gaussians._xyz.size(), 'mapping time = ', (e1-s), (e2-e1), e1-m1)
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return