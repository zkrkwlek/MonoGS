
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

from edge_assisted.object_manager import Object, ObjectManager


class EdgeGSSLAM(SLAM_WIN):
    def __init__(self, config, tracking_mode=False, mapping_update_pose = False, save_dir=None):
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

        self.gaussians = GaussianOrbModel(model_params.sh_degree, config=self.config)
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

        FeatureManager = GaussianPointManager()
        self.frontend.testManager = FeatureManager
        self.backend.FeatureManager = FeatureManager

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

    def AddFrame(self, fid, img, R, t, depth = None):
        if fid in self.dataset:
            f = self.dataset[str(fid)]
            f.color = img
            f.depth = depth
        else:
            f = EdgeFrame(fid, img, R, t, depth=depth)
            self.dataset[fid] = f
        return f

    def AddObjectBBox(self, fid, oid, bbox):

        if fid in self.dataset:
            f = self.dataset[str(fid)]
        else:
            f = EdgeFrame(fid, None,None,None)
            self.dataset[fid] = f
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
        backend_process = threading.Thread(target=self.backend.run)
        if self.use_gui:
            gui_process = threading.Thread(target=slam_win_gui.run, args=(self.params_gui,))
            gui_process.start()
            print("gui start")
            time.sleep(1)

        backend_process.start()
        print("backend start")
        frontend_process = threading.Thread(target=self.frontend.run)
        frontend_process.start()
        print("frontend start")
        #self.frontend.backend_queue.put(["pause"])