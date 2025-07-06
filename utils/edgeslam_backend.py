import numpy as np

from utils.slam_win_backend import WinBackEnd
import time

import torch
import torch.nn.functional as F

import cv2
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from edge_assisted.slam_utils import get_loss_mapping, get_reprojection_loss, get_patch_loss
from edge_assisted.gaussian_feature import project_pc_to_pixel, projection
from utils.edgeframe_utils import EdgeFrame

from utils.datahandle_utils import move_camera_to_gpu, move_camera_to_cpu, move_gaussianmodel_to_cpu
from utils.datahandle_utils import move_occ_visibility_to_cpu
#from edge_assisted.gaussian_feature import GaussianPointManager
from edge_assisted.localmap_utils import get_local_gaussians
from collections import defaultdict

class EdgeBackEnd(WinBackEnd):
    def __init__(self, config):
        super().__init__(config)
        self.first_kf_id = None
        self.pose_update = None
        #self.dataset = None
        self.FeatureManager = None
        self.frames={}
        self.weight_reprojection = 0.08
        self.weight_init_rgb = 0.9
        self.weight_init_depth = 0.02

        self.weight_rgb = 0.8
        self.weight_depth = 0.02
        self.weight_ba = 0.03
        self.weight_patch = 0.15
        self.next_kf_id = 0

    def push_to_frontend(self, tag=None, first_id = None, prune = None):

        self.last_sent = 0
        keyframes = []

        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone().cpu(), kf.T.clone().cpu()))

        if tag is None:
            tag = "sync_backend"

        prune_data = None
        if prune is not None:
            prune_data = prune
        #print("push_to_frontend::end", len(frames))
        msg = [tag, move_gaussianmodel_to_cpu(self.gaussians), move_occ_visibility_to_cpu(self.occ_aware_visibility), (keyframes), (prune_data)]
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
        self.gaussians.observation_indices = torch.cat([self.gaussians.observation_indices,
                                                        torch.full((self.gaussians.observation_indices.shape[0], 1),
                                                                   -1, device='cuda')], dim=1)
        self.gaussians.observation_points = torch.cat([self.gaussians.observation_points,
                                                       torch.full((self.gaussians.observation_points.shape[0], 2), -1.0,
                                                                  device='cuda')], dim=1)

        #self.update_gaussian_observation_with_frame(frame)

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,frame=frame
        )
        return

    def update_gaussian_observation_with_frame(self, frame):
        pass
        """
        # keyframe index
        id = frame.kf_id
        indices = torch.where(frame.gaussianpoints > -1)[0]
        #if indices.shape[0] > 0:
        #print(indices, frame.gaussianpoints[indices], self.gaussians.observation_indices.dtype, indices.dtype)
        self.gaussians.observation_indices[frame.gaussianpoints[indices], id:id + 1] = indices.unsqueeze(1)
        self.gaussians.observation_points[frame.gaussianpoints[indices], id*2:id*2+2] = frame.keypoints[indices]
        self.gaussians.isfeatured[frame.gaussianpoints[indices]] = True
        #print(id, self.gaussians.observation_indices.shape)
        """
        """
        for kp_idx in indices:
            g_idx = frame.gaussianpoints[kp_idx]
            self.gaussians.isfeatured[g_idx] = True
            if self.gaussians.observations[g_idx] is None:
                self.gaussians.observations[g_idx] = {}
            self.gaussians.observations[g_idx][frame.id] = kp_idx.item()
        """

    def update_gaussian_observation(self, prune_obs):
        keys = list(self.frames)
        """
        for idx, obs in prune_obs.items():
            if obs is not None:
                fids = torch.where(obs > -1)[0].cpu().numpy()
                #print(fids.dtype, fids, obs, keys)
                #print(keys[int(fids)],obs[int(fids)])
                for fid, kp in zip(keys[fids], obs[fids]):
                    #print(fid, kp)
                    self.frames[fid].gaussianpoints[kp.itme()] = -1
                #print(idx, obs, self.gaussians.observation_indices.shape)
        """

    def update_gaussian_observation_after_prune(self):
        #print('update_gaussian_observation_after_prune',self.gaussians._xyz.size(), self.gaussians.isfeatured.size(), self.gaussians.observations.shape,torch.count_nonzero(self.gaussians.isfeatured), np.count_nonzero(self.gaussians.observations), np.sum(self.gaussians.observations!=None))
        pass
        """
        keys = list(self.frames)
        for fid in keys:
            frame = self.frames[fid]
            kid = frame.kf_id
            gids = torch.where(self.gaussians.observation_indices[:,kid] > -1)[0]
            kpids = self.gaussians.observation_indices[gids, kid]
            frame.gaussianpoints = torch.full((frame.keypoints.shape[0],),-1, device='cuda')
            frame.gaussianpoints[kpids] = gids
        """


        """
        feature_indices = self.gaussians.isfeatured.clone().cpu().numpy()
        feature_indices = np.where(feature_indices)[0]
        gaussian_obs = list(zip(feature_indices, self.gaussians.observation_indices[feature_indices]))
        
        for gaussian_index, obs in gaussian_obs:
            if obs is None:
                print('obs error', gaussian_index, obs)
                self.gaussians.isfeatured[gaussian_index] = False
                continue
            for fid, kpidx in obs.items():
                #print("update_gaussian frame", fid)
                frame = self.frames[(fid)]
                frame.gaussianpoints[kpidx] = gaussian_index
        """
    def update_graph(self, current_window, th = 9.0):
        gaussian_indices = torch.where(self.gaussians.isfeatured)[0].cuda()
        gaussians_xyz = self.gaussians.get_xyz[gaussian_indices]
        obs_points = self.gaussians.observation_points[gaussian_indices]
        obs_indices = self.gaussians.observation_indices[gaussian_indices]

        for fid in current_window:
            viewpoint = self.viewpoints[fid]
            frame = self.frames[fid]
            kid = frame.kf_id*2

            idx = obs_indices[:, frame.kf_id] > -1
            gaussians = gaussians_xyz[idx]
            projection, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                    viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            points = obs_points[idx, kid:kid + 2][valid]

            l2 = torch.sum((projection - points) ** 2, dim=1)
            outlier = l2 > th

            outlier_idx = gaussian_indices[idx][valid][outlier]
            Noutlier = outlier_idx.shape[0]
            self.gaussians.observation_indices[outlier_idx, frame.kf_id] = torch.full((Noutlier,  ), -1, device='cuda').type(torch.int32)
            self.gaussians.observation_points[outlier_idx,  kid:kid + 2] = torch.full((Noutlier, 2), -1.0, device='cuda')

        #outlier 관리
        obs_indices2 = self.gaussians.observation_indices[gaussian_indices]+1
        row_sum = obs_indices2.sum(dim=1)
        mask = row_sum <= 0

        if torch.count_nonzero(mask) > 0:
            outlier_indices = gaussian_indices[mask]
            n = torch.count_nonzero(mask)
            self.gaussians.isfeatured[outlier_indices] = False#torch.zeros(n, device='cuda').bool()
            print('outlier test', outlier_indices.shape, torch.count_nonzero(mask), mask.shape, gaussian_indices.shape)
            """
            print('outlier removal', n,
                  torch.count_nonzero(self.gaussians.isfeatured[gaussian_indices][mask]),
                  torch.count_nonzero(self.gaussians.observation_indices[gaussian_indices][mask] > -1),
                  torch.count_nonzero(self.gaussians.observation_points[gaussian_indices][mask] > -1))
            """

    def bundle_adjustment(self, current_window, th_obs = 2):
        t1 = time.time()
        gaussian_indices = torch.where(self.gaussians.isfeatured)[0].cuda()
        gaussians_xyz = self.gaussians.get_xyz[gaussian_indices]
        obs_points = self.gaussians.observation_points[gaussian_indices]
        obs_indices = self.gaussians.observation_indices[gaussian_indices]

        t2= time.time()
        loss_ba = 0
        for fid in current_window:
            viewpoint = self.viewpoints[fid]
            frame = self.frames[fid]
            kid = frame.kf_id*2
            idx = obs_indices[:,frame.kf_id] > -1
            gaussians = gaussians_xyz[idx]
            projection, valid = project_pc_to_pixel(gaussians, viewpoint.R, viewpoint.T,
                                                    viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                    viewpoint.cy,
                                                    viewpoint.image_width, viewpoint.image_height)
            points = obs_points[idx, kid:kid + 2][valid]

            loss_ba += get_reprojection_loss(projection, points).mean()
        t3 = time.time()
        print("BA =", t2 - t1, t3 - t2, loss_ba, len(current_window))
        """
        N = 0
        frame_data = defaultdict(lambda: {'xyzs': [], 'pts': []}) #'ids': [], 'count':0
        for gid, xyz, obs in zip(gaussian_indices, gaussians, observations):
            if len(obs) < th_obs:
                continue
            for fid, kpid in obs.items():
                viewpoint = self.viewpoints[fid]
                frame = self.frames[fid]
                #pt = torch.from_numpy(frame.keypoints[kpid]).cuda()
                #frame_data[fid]['ids'].append(gid)
                frame_data[fid]['xyzs'].append(xyz)
                frame_data[fid]['pts'].append(frame.keypoints[kpid])
                #frame_data[fid]['count']+=1

        t2 = time.time()
        for fid in frame_data:
            #print(fid, frame_data[fid]['count'])
            #frame_data[fid]['ids'] = torch.stack(frame_data[fid]['ids'])
            frame_data[fid]['xyzs'] = torch.stack(frame_data[fid]['xyzs'])  # (N, 3)
            frame_data[fid]['pts'] = torch.stack(frame_data[fid]['pts'])  # (N, 2)
        t3 = time.time()

        loss_ba = 0
        for fid in frame_data:
            viewpoint = self.viewpoints[fid]
            projection, valid = project_pc_to_pixel(frame_data[fid]['xyzs'], viewpoint.R, viewpoint.T,
               viewpoint.fx, viewpoint.fy, viewpoint.cx,
               viewpoint.cy,
               viewpoint.image_width, viewpoint.image_height)
            points = frame_data[fid]['pts'][valid]
            loss_ba+=get_reprojection_loss(projection, points).mean()

        t4 = time.time()

        _, _ = project_pc_to_pixel(self.gaussians.get_xyz, viewpoint.R, viewpoint.T,
                                                viewpoint.fx, viewpoint.fy, viewpoint.cx,
                                                viewpoint.cy,
                                                viewpoint.image_width, viewpoint.image_height)
        t5 = time.time()
        """
        #print("BA =", t5-t4, t2-t1, t3-t2, t4-t3, loss_ba)

        return loss_ba

    def initialize_map(self, cur_frame_idx, viewpoint):
        t0 = time.time()
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t6 = 0.0
        t7 = 0.0
        t8 = 0.0
        t9 = 0.0
        t10 = 0.0
        t11 = 0.0
        t12 = 0.0
        nPrune = 0
        curr_frame = self.frames[(cur_frame_idx)]
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1

            t1 = t1+time.time()
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
            loss_rgb, loss_depth = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init = loss_rgb*self.weight_init_rgb+loss_depth*self.weight_init_depth
            t2 = t2+time.time()

            #projection, points, _, _ = curr_frame.get_correspondence(self.gaussians,viewpoint.R, viewpoint.T,viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,viewpoint.image_width, viewpoint.image_height)
            t3 = t3+time.time()
            #loss_init += get_reprojection_loss(projection,points).mean()*self.weight_reprojection
            t4 = t4+time.time()
            loss_init.backward()
            t5 = t5+time.time()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                t6 = t6+time.time()

                if mapping_iteration % self.init_gaussian_update == 0:
                    t7 = t7+time.time()
                    gaussian_indices, prune_mask, prune_obs = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    t8 = t8+time.time()
                    """
                    gaussian_indices = torch.arange(self.gaussians._xyz.size()[0])
                    gaussian_obs = self.gaussians.observations[self.gaussians.isfeatured]
                    gaussian_indices = gaussian_indices[self.gaussians.isfeatured]
                    for gaussian_index, obs in zip(gaussian_indices,gaussian_obs):
                        for fid, kpidx in obs.items():
                            frame = self.dataset[str(fid)]
                            frame.gaussianpoints[kpidx] = gaussian_index
                    """
                    #여기서 삭제 된것 + 남은 것 갱신
                    #self.update_gaussian_observation(prune_obs)
                    self.update_gaussian_observation_after_prune()

                    t9 = t9+time.time()
                    nPrune+=nPrune
                    #print("prune num test", Noldobs, len(prune_obs), torch.count_nonzero(self.gaussians.isfeatured))

                t10 += time.time()
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()
                t11 += time.time()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                t12 += time.time()

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        t13 = time.time()
        Log("Initialized map")
        print("init time = ", t13-t0,'corress', t3-t2,'reprojection',t4-t3,'backword',t5-t4,'update', t6-t5,'prune', t8-t7,'prune update', t9-t8,'other', t11-t10,t12-t11)
        return render_pkg

    def map(self, current_window, prune=False, iters=1, matches = None):
        if len(current_window) == 0:
            return None

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        prune_mask = None
        gaussian_indices = None
        prune_obs = None
        last_kf_id = current_window[0]

        curr_patches = None
        curr_patches_valid = None
        curr_rendered_image = None
        kf_patches = defaultdict(lambda: {'patch': None, 'valid': None})

        ##patch consistency
        doPatchConsistency = False
        if matches is not None and len(matches)>0:
            doPatchConsistency = True
            for cam_idx in current_window:
                if cam_idx == last_kf_id or cam_idx not in matches:
                    continue
                match = matches[cam_idx]
                viewpoint = self.viewpoints[cam_idx]
                frame = self.frames[cam_idx]
                kf_patches[cam_idx]['patch'], kf_patches[cam_idx]['valid'] = frame.extract_patches_differentiable(viewpoint.original_image, frame.keypoints[match[:,1]])

        ##patch consistency

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

                loss_rgb,loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

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
                loss_rgb, loss_depth= get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                loss_mapping += loss_rgb * self.weight_rgb + loss_depth * self.weight_depth

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            ##isotropic_scaling
            """
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            """

            ##geometric consistency
            loss_ba = self.bundle_adjustment(current_window)
            loss_mapping+=loss_ba*self.weight_ba

            if False and self.gaussians._xyz.grad is not None:
                after_grad_loss_ba = self.gaussians._xyz.grad.clone()
                affected_by_ba = torch.any(after_grad_loss_ba != 0, dim=1)
                print("Gaussians affected by ba:", torch.count_nonzero(affected_by_ba), self.gaussians._xyz.shape, affected_by_ba.nonzero().flatten())

            ##patch consistency
            if doPatchConsistency:

                loss_patch = 0
                t_patch1 = time.time()

                viewpoint = self.viewpoints[last_kf_id]
                keyframe = self.frames[last_kf_id]
                
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                ) = (
                    render_pkg["render"],
                )
                curr_patches, curr_patches_valid = keyframe.extract_patches_differentiable(image, keyframe.keypoints)

                for cam_idx in current_window:
                    if cam_idx == last_kf_id or cam_idx not in matches:
                        continue
                    match = matches[cam_idx]
                    kf_val = kf_patches[cam_idx]['valid']
                    curr_val = curr_patches_valid[match[:,0]]
                    valid = torch.logical_and(curr_val, kf_val)

                    cur_patch = curr_patches[match[valid,0]]
                    kf_patch = kf_patches[cam_idx]['patch'][valid]

                    err =get_patch_loss(cur_patch,kf_patch)
                    loss_patch += err.mean()
                    #print('patch', last_kf_id, cam_idx, torch.count_nonzero(valid), valid.shape, cur_patch.shape)
                loss_mapping += loss_patch*self.weight_patch

                t_patch2 = time.time()
                #print('patch loss = ', loss_patch.mean(), t_patch2-t_patch1)
                if False and self.gaussians._xyz.grad is not None:
                    after_grad_loss_patch = self.gaussians._xyz.grad.clone()
                    affected_by_patch = torch.any(after_grad_loss_patch != 0, dim=1)
                    print("Gaussians affected by patch:", torch.count_nonzero(affected_by_patch), self.gaussians._xyz.shape, affected_by_patch.nonzero().flatten())
            ##patch consistency

            ##isotropic_scaling
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
                    to_prune = None
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
                            #여기도 직접 수정해야 함.
                            self.gaussians.update_gaussian_observation_before_prune(to_prune.cuda(), self.frames)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                            #self.update_gaussian_observation(prune_obs)
                            self.update_gaussian_observation_after_prune()
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False, None, None, None

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
                    gaussian_indices, prune_mask, prune_obs = self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    #self.update_gaussian_observation(prune_obs)
                    self.update_gaussian_observation_after_prune()
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    print("before Resetting, ", self.gaussians._xyz.size())
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    print("after Resetting, ", self.gaussians._xyz.size())
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                ## Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.first_kf_id or not self.pose_update:
                        continue
                    update_pose(viewpoint)
                ##graph update
                self.update_graph(current_window)

                for cam_idx in range(len(current_window)):
                    viewpoint = viewpoint_stack[cam_idx]
                    keyframe = self.frames[(viewpoint.uid)]

                    #print(torch.count_nonzero(inlier), inlier.device, gaussian_index.size())
                    """
                    keyframe.gaussianpoints[keypoint_index[~inlier]] = -1
                    gaussian_index = gaussian_index[~inlier]
                    keypoint_index = keypoint_index[~inlier]
                    for kidx, idx in zip(keypoint_index,gaussian_index):
                        kidx = kidx.item()
                        idx = idx.item()
                        obs = self.gaussians.observations[idx]
                        if obs is not None and keyframe.id in obs:
                            if kidx == self.gaussians.observations[idx][keyframe.id]:
                                del self.gaussians.observations[idx][keyframe.id]
                            else:
                                print('del error', idx, keyframe.id,self.gaussians.observations[idx][keyframe.id], kidx)
                            if not self.gaussians.observations[idx]:
                                self.gaussians.isfeatured[idx] = False
                    """
                    """
                    image_np = (
                        viewpoint.original_image
                            .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                            .cpu()  # GPU → CPU
                            .numpy()  # NumPy 배열로 변환
                    )
                    image_np = (image_np * 255.0).astype(np.uint8)
                    self.FeatureManager.tracker.visualize2(image_np, projection.clone(), points.clone(), delay=1
                                                           , save=True,
                                                           filename='./res/mapping_' + str(viewpoint.uid) +'_'+str(last_kf_id)+'.jpg')
                    """
                ##graph update
        return gaussian_split, prune_mask, gaussian_indices, prune_obs

    def run(self):
        prune_dict = None
        need_sync_prune = False
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
                Nold = self.gaussians._xyz.size()[0]

                #match 추가
                num_lask_kf = 3
                keys = list(self.frames)
                last_key = keys[-1]
                recent_keys = keys[-num_lask_kf:-1] if len(keys) > 1 else []
                last_keyframe = self.frames[last_key]

                kf_matches = {}
                for kf_idx in recent_keys:
                    keyframe = self.frames[kf_idx]
                    matches = self.FeatureManager.tracker.match(last_keyframe.descriptors, keyframe.descriptors)
                    matches = torch.from_numpy(matches).cuda()
                    if matches.shape[0] > 20:
                        kf_matches[kf_idx] = matches

                # match 추가

                prune_idxs = None
                prune_mask = None
                prune_obs = None
                _, prune_mask1, prune_idxs1, prune_obs1 = self.map(self.current_window, matches = kf_matches)
                if prune_mask1 is not None:
                    prune_idxs = prune_idxs1
                    prune_mask = prune_mask1
                    prune_obs  = prune_obs1
                if self.last_sent >= 10:
                    _, prune_mask2, prune_idxs2, prune_obs2 = self.map(self.current_window, prune=True, iters=10, matches=kf_matches)
                    if prune_mask2 is not None:
                        prune_idxs = prune_idxs2
                        prune_mask = prune_mask2
                        prune_obs  = prune_obs2

                if prune_mask is not None:
                    old_idxs = prune_idxs[:Nold]
                    old_masks = prune_mask[:Nold]

                    Nnew = torch.count_nonzero(~old_masks)
                    new_idxs = torch.arange(Nold)
                    new_idxs[~old_masks] = torch.arange(Nnew)
                    new_idxs[old_masks] = -1

                    prune_dict = {k.item(): v.item() for k, v in zip(old_idxs, new_idxs) if self.gaussians.isfeatured[v] and v < Nold}
                    for key, _ in prune_obs.items():
                        if key <= Nold:
                            prune_dict[key] = -1
                    need_sync_prune = True
                    #print('update dict', len(prune_dict), torch.unique(new_idxs).size()[0], Nnew, self.gaussians.isfeatured[-1], self.gaussians.isfeatured.size(), self.gaussians.observations.shape)

                if need_sync_prune:
                    need_sync_prune = False
                    self.push_to_frontend(prune=prune_dict)
                else:
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
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id+=1
                    frame.keypoints, frame.descriptors = f
                    frame.keypoints = frame.keypoints.cuda()
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
                    frame.kf_id = self.next_kf_id
                    self.next_kf_id += 1
                    frame.keypoints, frame.descriptors = f
                    frame.keypoints = frame.keypoints.cuda()
                    self.frames[(cur_frame_idx)] = frame

                    """
                    if prune_dict is not None:
                        for kpidx, gid in enumerate(frame.gaussianpoints):
                            gid = gid.item()
                            if gid == -1:
                                continue
                            if gid in prune_dict:
                                frame.gaussianpoints[kpidx] = prune_dict[gid]

                        #print('frame equal test', np.equal(frame.gaussianpoints, self.frames[cur_frame_idx].gaussianpoints))
                    """
                    Nold = self.gaussians._xyz.size()[0]
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, frame=frame)

                    ##add match and observation
                    #get_local_gaussians(self.gaussians, self.current_window)
                    t1 = time.time()
                    #temp_obs = self.gaussians.observations[self.gaussians.isfeatured.clone().cpu().numpy()]
                    a = time.time()
                    temp_kf_window = [x for x in current_window if x != cur_frame_idx]
                    kf_matches = {}
                    new_kf_id = frame.kf_id

                    for kf_idx in temp_kf_window:
                        keyframe = self.frames[kf_idx]

                        matches = self.FeatureManager.tracker.match(frame.descriptors, keyframe.descriptors)
                        matches = torch.from_numpy(matches).type(torch.int32).cuda()
                        if matches.shape[0] > 20 :
                            kf_matches[kf_idx] = matches
                            tt1 = time.time()

                            kf_id = keyframe.kf_id

                            ##새로운 가우시안 포인트에 기존 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, new_kf_id]  # shape: (N,)
                            match_values = matches[:, 0]  # shape: (K,)
                            # eq[n, k] == True면 obs_idx_col[n] == match_values[k]
                            # 위치: 포함되면 첫 번째 True의 인덱스, 없으면 -1
                            # 인클루드 마스크는 가우시안에서 매칭(0)의 키포인트 위치. 즉 매치(0)과 같음., 가우시안 위치를 표현함.
                            # 키포지션은 그게 매치 안에서 어디있는지를 알 수 있음.
                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                    torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, kf_id] = matches[frame_match_index, 1]
                            self.gaussians.observation_points[frame_gaussian_index, kf_id * 2:kf_id * 2 + 2] = \
                            keyframe.keypoints[matches[frame_match_index, 1]]

                            ##기존의 가우시안 포인트에 새로운 키프레임의 키포인트와 연결
                            obs_idx_col = self.gaussians.observation_indices[:, kf_id]  # shape: (N,)
                            match_values = matches[:, 1]  # shape: (K,)

                            eq = obs_idx_col.unsqueeze(1) == match_values.unsqueeze(0)  # (N, K)
                            key_positions = torch.where(eq.any(dim=1), eq.float().argmax(dim=1),
                                                        torch.full_like(obs_idx_col, -1))
                            included_mask = key_positions != -1 #가우시안 수와 일치함.

                            frame_gaussian_index = torch.where(included_mask)[0]
                            frame_match_index = key_positions[included_mask]

                            self.gaussians.observation_indices[frame_gaussian_index, new_kf_id] = matches[
                                frame_match_index, 0]
                            self.gaussians.observation_points[frame_gaussian_index, new_kf_id * 2:new_kf_id * 2 + 2] = \
                                frame.keypoints[matches[frame_match_index, 0]]

                            tt2 = time.time()
                            print("aaa", kf_id,tt2-tt1, included_mask.shape, torch.count_nonzero(included_mask), key_positions.shape)
                            #self.gaussians.observation_indices[:, new_kf_id].shape, frame_match_mask.shape, match_values.shape, torch.count_nonzero(frame_match_mask), torch.count_nonzero(frame_match_mask2), torch.count_nonzero(self.gaussians.observation_indices[:, new_kf_id] > -1))

                            """
                            tmp_valid = frame.gaussianpoints[matches[:,0]] > -1
                            filtered_match = matches[tmp_valid]
                            tmp_gaussian_idx = frame.gaussianpoints[filtered_match[:,0]]
                            self.gaussians.observation_indices[tmp_gaussian_idx, kf_id] = filtered_match[:,1]
                            self.gaussians.observation_points[tmp_gaussian_idx, kf_id*2:kf_id*2+2] = keyframe.keypoints[filtered_match[:,1]]

                            tmp_valid2 = keyframe.gaussianpoints[matches[:, 1]] > -1
                            filtered_match2 = matches[tmp_valid2]
                            tmp_gaussian_idx2 = keyframe.gaussianpoints[filtered_match2[:, 1]]
                            self.gaussians.observation_indices[tmp_gaussian_idx2, new_kf_id] = filtered_match2[:, 0]
                            self.gaussians.observation_points[tmp_gaussian_idx2, new_kf_id * 2:new_kf_id * 2 + 2] = \
                            frame.keypoints[filtered_match2[:, 0]]
                            """
                            #obs = self.gaussians.observations[,kf_id:kf_id+2]

                            """
                            for obs in temp_obs:
                                fidx = -1;
                                kidx = -1;
                                if kf_idx in obs:
                                    kidx = obs[kf_idx]
                                    # print('test1', obs[kf_idx])
                                if cur_frame_idx in obs:
                                    fidx = obs[cur_frame_idx]
                                    if fidx in matches[:, 0]:
                                        print('test',obs[cur_frame_idx])

                                    continue
                                    # print('test2', obs[cur_frame_idx])
                            """
                            """
                            for idx1, idx2 in matches:
                                if frame.gaussianpoints[idx1] >= 0:
                                    gidx = frame.gaussianpoints[idx1].item() #int
                                    
                                    if kf_idx in self.gaussians.observations[gidx]:
                                        print("check", idx2, self.gaussians.observations[gidx][kf_idx])
                                    else:
                                        self.gaussians.observations[gidx][kf_idx] = idx2.item()
                                        #print(gidx, len(self.gaussians.observations[gidx]))
                            """
                        print('backend::', kf_idx)#, torch.count_nonzero(keyframe.gaussianpoints > -1), keyframe.keypoints.size()[0])



                    b = time.time()
                    #print("kf match test = ", len(temp_kf_window), temp_obs.shape, a-t1, b-a)
                    ##add match and observation

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
                    prune_idxs = None
                    prune_mask = None
                    prune_obs = None
                    _, prune_mask1, prune_idxs1, prune_obs1 = self.map(self.current_window, matches = kf_matches, iters=iter_per_kf)
                    if prune_mask1 is not None:
                        prune_idxs = prune_idxs1
                        prune_mask = prune_mask1
                        prune_obs  = prune_obs1
                    _, prune_mask1, prune_idxs1, prune_obs1 = self.map(self.current_window, prune=True)
                    if prune_mask1 is not None:
                        prune_idxs = prune_idxs1
                        prune_mask = prune_mask1
                        prune_obs  = prune_obs1

                    if prune_mask is not None:
                        old_idxs = prune_idxs[:Nold]
                        old_masks = prune_mask[:Nold]

                        #여기 데이터 축소 필요
                        Nnew = torch.count_nonzero(~old_masks)
                        new_idxs = torch.arange(Nold)
                        new_idxs[~old_masks] = torch.arange(Nnew)
                        new_idxs[old_masks] = -1
                        prune_dict = {k.item(): v.item() for k, v in zip(old_idxs, new_idxs) if
                                      self.gaussians.isfeatured[v] and v < Nold}
                        for key, _ in prune_obs.items():
                            if key <= Nold:
                                prune_dict[key] = -1
                        #print('update dict', len(prune_dict), torch.unique(new_idxs).size()[0], Nnew, self.gaussians.isfeatured.size(), self.gaussians.observations.shape)
                        need_sync_prune = True
                    e1 = time.time()
                    if need_sync_prune:
                        need_sync_prune = False
                        self.push_to_frontend("keyframe", prune=prune_dict)
                    else :
                        self.push_to_frontend("keyframe")
                    e2 = time.time()

                    ##frame visualization
                    """
                    for kf_idx in temp_kf_window:
                        if kf_idx in kf_matches:
                            matches = kf_matches[kf_idx]
                            keyframe = self.frames[kf_idx]
                            viewpoint = self.viewpoints[kf_idx]
                            mask = frame.gaussianpoints[matches[:,0]] > -1
                            matches = matches[mask.cpu().numpy(),:]
                            projection, points= frame.get_correspondence_with_frame(self.gaussians, keyframe, matches, viewpoint.R, viewpoint.T,
                                viewpoint.fx, viewpoint.fy, viewpoint.cx, viewpoint.cy,
                                viewpoint.image_width, viewpoint.image_height)

                            image_np = (
                                viewpoint.original_image
                                    .permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                                    .cpu()  # GPU → CPU
                                    .numpy()  # NumPy 배열로 변환
                            )
                            image_np = (image_np * 255.0).astype(np.uint8)
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                            self.FeatureManager.tracker.visualize2(image_np, projection.clone(), points.clone(), delay=1
                                   , save=True,
                                   filename='./res/map/mapping_' + str(cur_frame_idx) + '_' + str(kf_idx) + '.jpg')
                    """

                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return