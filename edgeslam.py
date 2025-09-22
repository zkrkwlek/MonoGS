import multiprocessing

import torch
from munch import munchify
import threading
import time
from multiprocessing import Queue

from slam_win import SLAM_WIN
from utils.dataset import load_dataset
from gui import gui_utils,slam_win_gui
from utils.queue_utils import PeekableQueue
from utils.edgeframe_utils import EdgeFrame, EdgeFrames
from utils.edgeslam_frontend import EdgeFrontEnd
from utils.edgeslam_backend import EdgeBackEnd
from gaussian_splatting.scene.gaussian_model import GaussianModel
from edge_assisted.gaussian_orb_model import GaussianOrbModel, create_gaussian_orb_model
from edge_assisted.gaussian_feature import GaussianPointManager
from utils.multiprocessing_utils import FakeQueue
from edge_assisted.device_utils import Device
from edge_assisted.object_manager import Object, ObjectManager

from edge_assisted.feature_manager import FeatureManager
from edge_assisted.pose_optimizer2 import PnPOptimizer
from edge_assisted.place_recognizer import PlaceRecognizer

from atomicx import AtomicBool

from edge_assisted.mapping_module import MappingModule

import yappi

class EdgeGSSLAM(SLAM_WIN):
    def __init__(self, config, tracking_mode=False, mapping_update_pose = False, gs_pose = False, save_dir=None, UseXfeat = False):
        super().__init__(config, save_dir)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

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
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianOrbModel(model_params.sh_degree, config=self.config, opt_params=opt_params)
        #self.gaussian s = create_gaussian_orb_model(GaussianModel(model_params.sh_degree, config=self.config), self.config)

        self.gaussians.init_lr(6.0)
        self.dataset = EdgeFrames(load_dataset(
            model_params, model_params.source_path, config=config
        ))

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        q_main2vis = Queue() if self.use_gui else FakeQueue()
        q_vis2main = Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.tracking_mode = tracking_mode
        self.object_map_mode = False

        self.dataset.depth_scale = 1.0
        print("depth scale", self.dataset.depth_scale)

        self.frontend = EdgeFrontEnd(self.config)
        self.backend = EdgeBackEnd(self.config)
        self.mapping_module = MappingModule(self.config)

        #gs rasterization으로 pose까지 변경할지 체크
        self.frontend.gs_pose = gs_pose
        self.backend.gs_pose = gs_pose

        ##mapping atomic bool
        bDoingMapping = AtomicBool()
        bDoingMapping.store(True)
        self.frontend.bDoingMapping = bDoingMapping
        self.backend.bDoingMapping = bDoingMapping

        #안쓰임
        FeatureManagerA = GaussianPointManager()
        self.frontend.testManager = FeatureManagerA
        self.backend.FeatureManager = FeatureManagerA

        #xfeat
        if UseXfeat:
            XFeat = FeatureManager()
            self.frontend.feature_manager = XFeat
            self.backend.feature_manager = XFeat

        #gtsam
        #PoseOptimizer = PnPOptimizer()
        #self.frontend.pose_optimizer = PoseOptimizer
        #self.backend.pose_optimizer = PoseOptimizer

        #salad
        _PlaceRecognizer = PlaceRecognizer()
        self.frontend.place_recognizer = _PlaceRecognizer
        self.backend.place_recognizer = _PlaceRecognizer

        frontend_queue = Queue()
        backend_queue = Queue()
        self.edge_queue = PeekableQueue()

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.edge_queue = self.edge_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()
        self.frontend.tracking_mode = self.tracking_mode

        self.backend.dataset = self.dataset
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.pose_update = mapping_update_pose

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        self.object_manager = ObjectManager()
        self.backend.objects = self.object_manager

        #device 정보
        self.devices={}
        self.backend.devices  = self.devices
        self.frontend.devices = self.devices
        self.mapping_module.devices = self.devices

        #mapping module 초기화
        self.mapping_module.dataset = self.dataset
        self.mapping_module.gaussians = self.gaussians
        self.mapping_module.background = self.background
        self.mapping_module.cameras_extent = 6.0
        self.mapping_module.pipeline_params = self.pipeline_params
        self.mapping_module.opt_params = self.opt_params
        self.mapping_module.frontend_queue = frontend_queue
        self.mapping_module.backend_queue = backend_queue
        self.mapping_module.live_mode = self.live_mode
        self.mapping_module.pose_update = mapping_update_pose
        self.mapping_module.set_hyperparams()

        #local gs 수정 필요
        self.keyframes = {}
        self.mapping_module.keyframes = self.keyframes

    def UpdateSparseMap(self, local_sparse_map):
        #id, pos
        self.mapping_module.update_sparse_gaussian_map(local_sparse_map)
        pass

    def GenerateLocalMap(self, target_kf_id, neighbor_kf_ids, src):
        #새로운 KF
        #그 안의 스파스 맵
        #인접한 키프레임 정보

        #select local gaussian
        #self.mapping_module.select_local_gaussians(target_kf_id, neighbor_kf_ids)
        #generate local gaussian
        #optimize local gaussian
        self.mapping_module.local_gaussian_mapping(target_kf_id, neighbor_kf_ids, src)

        pass

    def AddDevice(self, src, K, D, w, h, bMapper = True):
        device = Device(src, K, D, w, h, bMapper = bMapper)
        self.devices[src] = device

    def Alignment(self, src, idx):
        device = self.devices[src]
        p = threading.Thread(target = self.frontend.coordinate_alignment, args=(device, idx))
        p.start()

    def AddDepth(self, src, idx, depth):
        device = self.devices[src]
        f = device.frames[idx]
        f.depth = depth
        p = threading.Thread(target=self.frontend.after_depth, args=(device, idx))
        p.start()

    def AddKeyFrame(self, kf_id, img, keypoints, depth, T, src):
        f = EdgeFrame(kf_id, img, None,None, depth=depth, T = T)
        f.keypoints = keypoints
        self.keyframes[kf_id] = f
        device = self.devices[src]
        viewpoint = self.mapping_module.convert_viewpoint(device, f, kf_id)
        T = torch.from_numpy(f.T).cuda()
        #T의 타입 설정이 중요
        viewpoint.R = T[:3, :3]
        viewpoint.T = T[:3, 3]
        self.mapping_module.viewpoints[kf_id] = viewpoint

    def AddFrame(self, fid, img, R, t, depth = None, src = None, ts = None):
        device = self.devices[src]
        if fid in device.frames:
            #f = self.dataset[str(fid)]
            f = device.frames[fid]
            f.color = img
            f.depth = depth
            f.ts = ts
            f.UpdatePose(R,t)
        else:
            f = EdgeFrame(fid, img, None, None, depth=depth, src=src, ts = ts)
            device.frames[fid] = f
            #self.dataset[fid] = f
        return f

    def AddPlaceRecogDesc(self, fid, desc, src = None):
        device = self.devices[src]
        if fid in device.frames:
            f = device.frames[fid]
        else:
            f = EdgeFrame(fid, None, None, None, src=src)
            device.frames[fid] = f
        f.pr_desc = torch.from_numpy(desc.copy()).unsqueeze(0)
        if device.poses is None and not device.mapper:
            #self.frontend.relocalization(device, fid)
            p = threading.Thread(target=self.frontend.relocalization, args=(device, fid))
            p.start()


    def AddContours(self, fid, contours, src=None):
        device = self.devices[src]
        if fid in device.frames:
            #f = self.dataset[str(fid)]
            f = device.frames[fid]
        else:
            f = EdgeFrame(fid, None, None, None, src=src)
            device.frames[fid] = f
        f.contours =  contours

    def AddObjectBBox(self, fid, oid, bbox, src=None):
        device = self.devices[src]
        if fid in device.frames:
            #f = self.dataset[str(fid)]
            f = device.frames[fid]
        else:
            f = EdgeFrame(fid, None,None,None,src=src)
            #self.dataset[fid] = f
            device.frames[fid] = f
        if oid not in self.object_manager:
            self.object_manager[oid] = Object(oid)
        obj = self.object_manager[oid]
        obj[fid] = bbox
        f.AddObject(oid,bbox)


    def SetDepth(self, fid, depth):
        self.dataset[str(fid)].depth = depth

    def CheckFrame(self, fid):
        return (fid) in self.dataset

    def run(self):
        backend_process = threading.Thread(target=self.backend.run_with_ba)
        if self.use_gui:
            gui_process = threading.Thread(target=slam_win_gui.run, args=(self.params_gui,))
            gui_process.start()
            print("gui start")
            time.sleep(1)

        backend_process.start()
        print("backend start")
        frontend_process = threading.Thread(target=self.frontend.run_with_graph)
        frontend_process.start()
        print("frontend start")
        #self.frontend.backend_queue.put(["pause"])