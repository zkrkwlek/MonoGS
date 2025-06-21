import cv2
import torch
import time

import numpy as np
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
from edge_assisted.slam_utils import get_loss_tracking, get_reprojection_loss
#from edge_assisted.gaussian_feature import GaussianPointManager

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
        f = [frame.keypoints, frame.descriptors, frame.gaussianpoints.clone()]
        msg = ["init", cur_frame_idx, move_camera_to_cpu(viewpoint), depth_map, f]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        frame = self.frames[cur_frame_idx]
        f = [frame.keypoints, frame.descriptors, frame.gaussianpoints.clone()]
        msg = ["keyframe", cur_frame_idx, move_camera_to_cpu(viewpoint), (current_window), (depthmap),f]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def sync_backend(self, data, prev_frame_idx = None):
        gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        frames = data[4]
        move_gaussianmodel_to_gpu(gaussians)
        move_occ_visibility_to_gpu(occ_aware_visibility)
        #move_gaussians_to_gpu(keyframes)

        self.gaussians = gaussians
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone().to(self.device), kf_T.clone().to(self.device))
        for kf_id, gaussianpoints in frames:
            self.frames[kf_id].gaussianpoints = gaussianpoints
            #mask = gaussianpoints > 1
            #print(kf_id, self.frames[kf_id].gaussianpoints, torch.count_nonzero(mask))

        #update frame gaussianpoints
        if prev_frame_idx is not None:
            print('frame update', prev_frame_idx)
            frame = self.frames[prev_frame_idx]
            frame.gaussianpoints =  torch.full((frame.keypoints.shape[0],),-1)
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
        t_n = 0

        curr_frame = self.frames[cur_frame_idx]

        for tracking_itr in range(self.tracking_itr_num):
            t1 = t1+time.time()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            t2 = t2+time.time()
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()

            projection, points = curr_frame.get_correspondence(self.gaussians, viewpoint.R, viewpoint.T,
                                                               viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                               viewpoint.image_width, viewpoint.image_height,
                                                               delta_rot= viewpoint.cam_rot_delta,
                                                               delta_trans=viewpoint.cam_trans_delta,
                                                               )
            #print(projection.device, points.device, projection.grad_fn)
            #print(projection.requires_grad, points.requires_grad)
            loss_tracking = get_reprojection_loss(projection, points)
            #print('before', loss_tracking)
            loss_tracking += get_loss_tracking(self.config, image, depth, opacity, viewpoint)
            #print('after',loss_tracking)
            t3 = t3+time.time()
            loss_tracking.backward()
            t4 = t4+time.time()
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
        print("tracking processig time", t_n,(t2-t1),(t3-t2), (t4-t3))
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
                cur_frame_idx = self.edge_queue.get()

                viewpoint = init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )

                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                curr_frame = self.dataset[str(cur_frame_idx)]
                self.frames[cur_frame_idx] = curr_frame
                """
                if True:
                    image = cv2.remap(frame.color, self.dataset.map1x, self.dataset.map1y, cv2.INTER_LINEAR)
                else:
                    image = frame.color
                """
                pred = self.testManager.feature_model.run(curr_frame.color)
                keypoints = pred['keypoints']
                curr_frame.descriptors = pred['descriptors']

                points_reshaped = keypoints.reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(points_reshaped, K, self.dataset.dist_coeffs, None, K)
                curr_frame.keypoints = undistorted.reshape(-1, 2)
                curr_frame.gaussianpoints = torch.full((curr_frame.keypoints.shape[0],),-1)
                #print(frame.keypoints)

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
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
                    prev_frame = self.frames[(prev_frame_idx)]
                    matches = self.testManager.tracker.match(prev_frame.descriptors, curr_frame.descriptors)
                    curr_frame.copy_gaussians_from_frame_matches(prev_frame, self.gaussians, matches)

                    render_pkg = self.tracking(cur_frame_idx, prev_frame_idx, viewpoint)


                    #매칭 앤 트래킹 테스트
                    out = self.testManager.tracker.visualize(prev_frame.color, curr_frame.color, matches, prev_frame.keypoints, curr_frame.keypoints)
                    t5 = time.time
                    # out, N_matches = self.testManager.tracker.update(frame.color, frame.keypoints, frame.descriptors)
                    # print(t4 - t3, t5-t4, self.testManager.feature_model.device, (frame.keypoints.shape))

                    cv2.imshow("match", out)
                    cv2.waitKey(1)
                    #print(prev_frame.gaussianpoints.size(), prev_frame.keypoints.shape)

                    prev = self.cameras[prev_frame_idx]

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