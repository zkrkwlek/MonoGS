import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import cv2
import torch
#import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
#from gui import gui_utils, slam_gui
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
#from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.objslsam_backend import ObjectBackEnd
#from utils.slam_frontend import FrontEnd
from utils.camera_utils import Camera

import time
#from multiprocessing import Queue
from utils.queue_utils import PeekableQueue

#gui
from utils.multiprocessing_utils import FakeQueue
from gui import gui_utils, obj_slam_gui
#from multiprocessing import Process
import threading
from utils.datahandle_utils import move_gaussianpacket_to_cpu

class Object:
    def __init__(self,id,R,t):
        self.id = id
        self.initialized = False
        self.used = False
        #self.doingMapping = False
        #추후 스케일, cov 등 추가
        self.T = np.zeros((4, 4), dtype=np.float64)
        self.T[:3, :3] = R
        self.T[:3, 3] = t.flatten()  # 또는 M[:3, 3] = b.squeeze()
        self.T[3, 3] = 1.0

        self.current_window = []
        self.kf_indices = []
        self.first_kf_id = 0

class Frame:
    def __init__(self,id,color,R,t,depth=None):
        self.id = id
        self.color = color
        self.depth = depth
        self.T = np.zeros((4, 4), dtype=np.float64)
        self.T[:3, :3] = R
        self.T[:3, 3] = t.flatten()  # 또는 M[:3, 3] = b.squeeze()
        self.T[3, 3] = 1.0

class Frames:
    def __init__(self, dataset):
        self._dataset = dataset
        self._frames={}
        print("wrapping class")

    def __getattr__(self, attr):
        #dataset의 특성을 그대로 활용
        return getattr(self._dataset, attr)

    def __setitem__(self, key, value):
        #네트워크로 전송받은 데이터를 추가함.
        self._frames[key] = value;

    def __contains__(self, item):
        #데이터를 전송받았는지 확인함.
        return item in self._frames

    def __getitem__(self, key):
        ##dataset 처럼, color, depth, pose return
        """
        if isinstance(key, str):
            return self._frames[key]
        elif isinstance(key, int):
            #print("not implemented", key)
            return self._dataset[key]
        """
        frame = self._frames[key]
        pose = frame.T
        image = frame.color
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
        """
        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale
        """
        image = (
            torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose

class ObjectSLAM():
    def __init__(self, config, device, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        #dataset을 warrping하여 이미지 통신에 대응할 수 있도록 변경함.
        self.dataset = Frames(
            load_dataset(
            model_params, model_params.source_path, config=config
        ))

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.backend = ObjectBackEnd(self.config)

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        #self.backend.frontend_queue = frontend_queue
        #self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.cameras={}
        self.objects = {}
        self.device = device
        self.MappingQueue = PeekableQueue()
        self.FrameQueue = PeekableQueue()
        self.running = False

        self.backend.cameras = self.cameras
        self.backend.objects = self.objects

        ##GUI
        self.use_gui = self.config["Results"]["use_gui"]
        q_main2vis = PeekableQueue() if self.use_gui else FakeQueue()
        q_vis2main = PeekableQueue() if self.use_gui else FakeQueue()
        self.backend.q_main2vis = q_main2vis
        self.backend.q_vis2main = q_vis2main

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        #obj_slam_gui.run(self.params_gui)
        if self.use_gui:
            gui_process = threading.Thread(target=obj_slam_gui.run, args=(self.params_gui,)) #mp.Process(target=obj_slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            print('gui.run start')
        #torch.cuda.synchronize()

    def run(self):

        while True:
            if self.FrameQueue.empty():
                time.sleep(0.1)
            else:
                ##객체 맵이 없으면 초기화
                ##객체 매핑 중이면 패스
                ##아무것도 안하면 키프레임 체크
                data = self.FrameQueue.peek()
                obj = self.objects[data[1]]
                if not obj.used:
                    self.FrameQueue.get()
                    if obj.initialized:
                        bKF = self.KeyFrameProcess(data[1], data[2])
                        if bKF:
                            data[0] = "keyframe"
                            self.MappingQueue.put(data)
                    else:
                        data[0] = "init"
                        self.MappingQueue.put(data)
            if self.MappingQueue.empty():
                time.sleep(0.1)
            else:
                data = self.MappingQueue.get(timeout=0.1)
                oid = data[1]
                fid = data[2]
                #print(self.FrameQueue.qsize(),data)
                if data[0] == "init":
                    self.ObjectMapInitialization(oid, fid)
                elif data[0] == "keyframe":
                    bbox = data[3]
                    self.ObjectMapUpdate(oid, fid, bbox)

            """
            if self.queue.empty():
                time.sleep(0.1)
                continue
            else:
                data = self.queue.get(timeout=0.1)
                oid = data[1]
                fid = data[2]
                print(data)
                if data[0] == "init":
                    self.ObjectMapInitialization(oid, fid)
                elif data[0] =="frame":
                    bKF = self.KeyFrameProcess(oid,fid)
                    if bKF:
                        data[0] = "keyframe"
                        self.queue.put(data)
                elif data[0] == "keyframe":
                    try:
                        bbox = data[3]
                        self.ObjectMapUpdate(oid, fid, bbox)
                    except Exception as e:
                        print(e)
                continue
            """
        return

    def AddObject(self, oid, R, t):
        self.objects[oid] = Object(oid, R, t)

    def AddFrame(self, fid, img, R, t):
        self.dataset[fid] = Frame(fid, img, R, t)

    def CheckFrame(self, fid):
        return (fid) in self.dataset

    def IsInitialized(self, oid):
        return self.objects[oid].initialized
    def IsUsed(self, oid):
        return self.objects[oid].used

    def ConvertViewePoint(self, fid, projection_matrix):
        #image 타입 확인 필요
        img, depth, pose = self.dataset[fid]

        viewpoint = Camera(fid, img, depth, pose, projection_matrix,
            self.dataset.fx,
            self.dataset.fy,
            self.dataset.cx,
            self.dataset.cy,
            self.dataset.fovx,
            self.dataset.fovy,
            self.dataset.height,
            self.dataset.width,
            device=self.dataset.device,
        )
        viewpoint.compute_grad_mask(self.config)
        viewpoint.update_RT(viewpoint.R_gt,viewpoint.T_gt)
        viewpoint.cam_rot_delta.data.fill_(0)
        viewpoint.cam_trans_delta.data.fill_(0)
        #print("pose",viewpoint.R, viewpoint.T)
        self.cameras[fid] = viewpoint

    #추후 이것도 오브젝트 단위로 키프레임 체크가 가능해야 함.
    def KeyFrameProcess(self, oid, fid):
        return self.backend.KeyFrameProcess(oid,fid)

    def ObjectMapUpdate(self, oid, fid, bbox):
        obj = self.objects[oid]
        obj.used = True

        opt_params = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        iter_per_kf = self.backend.mapping_itr_num if self.backend.single_thread else 10
        if not obj.initialized:
            if (
                    len(obj.current_window)
                    == self.config["Training"]["window_size"]
            ):
                frames_to_optimize = (
                        self.config["Training"]["window_size"] - 1
                )
                iter_per_kf = 50 if self.live_mode else 300
                Log("Performing initial BA for initialization")
            else:
                iter_per_kf = self.backend.mapping_itr_num
        for cam_idx in range(len(obj.current_window)):
            if obj.current_window[cam_idx] == 0:
                continue
            viewpoint = self.backend.viewpoints[obj.current_window[cam_idx]]

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
        bPoseUpdate = True
        self.backend.keyframe_optimizers = torch.optim.Adam(opt_params)
        self.backend.map(obj.current_window, first_kf_id=obj.first_kf_id, iters=iter_per_kf, pose_update= bPoseUpdate)
        for i in range(10):
            self.backend.map(obj.current_window, first_kf_id=obj.first_kf_id, iters=10,prune =  True,
                             pose_update=bPoseUpdate)
        self.backend.map(obj.current_window, first_kf_id=obj.first_kf_id, prune=True, pose_update= bPoseUpdate)

        #print("Object Map Update", oid, fid, len(self.objects[oid].kf_indices), len(self.objects[oid].current_window), len(self.gaussians._xyz), iter_per_kf, frames_to_optimize)
        #print(type(self.gaussians._xyz))
        #self.backend.render(oid,fid)
        #obj.doingMapping = False

        current_window_dict = {}
        current_window_dict[obj.current_window[0]] = obj.current_window[1:]
        keyframes = [self.cameras[kf_idx] for kf_idx in obj.current_window]
        viewpoint = self.cameras[fid]
        self.backend.q_main2vis.put(
            move_gaussianpacket_to_cpu(
                gui_utils.GaussianPacket(
                    gaussians=(self.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    kf_window=current_window_dict,
                    gtcolor=viewpoint.original_image,
                    gtdepth=viewpoint.depth
                    if not self.monocular
                    else np.zeros((viewpoint.image_height, viewpoint.image_width))
                )
            )
        )
        obj.used = False

    def ObjectMapInitialization(self, oid, fid):
        #queue에 메세지를 넣는 역할로
        obj = self.objects[oid]
        if obj.initialized:
            return
        print("initialization start", oid, fid)
        obj.used = True
        obj.first_kf_id = fid
        viewpoint = self.cameras[fid]
        self.backend.viewpoints[fid] = viewpoint

        depth = self.backend.add_new_keyframe(fid, init = True)
        obj.current_window.append(fid)
        obj.kf_indices.append(fid)

        self.backend.add_next_kf(
            fid, viewpoint, depth_map=depth, init=True
        )

        render_pkg = self.backend.initialize_map(fid, viewpoint)

        print("gaussians = ", len(self.gaussians._xyz))
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
        """
        img_cv=image.permute(1, 2, 0).cpu().detach().numpy()
        img_cv = (img_cv * 255).astype(np.uint8)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./res/init.png',img_cv)
        """

        self.backend.q_main2vis.put(
            move_gaussianpacket_to_cpu(
                gui_utils.GaussianPacket(
                    gaussians=(self.gaussians),
                    current_frame=viewpoint,
                    #keyframes=keyframes,
                    #kf_window=current_window_dict,
                )
            )
        )

        obj.initialized = True
        obj.used = False