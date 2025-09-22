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

from edge_assisted.gaussian_feature import calculate_keypoint_mask,calculate_feature_mask,calculate_bbox_mask, project_pc_to_pixel, visualize_pc,convert_xyz, unproject_pixel_to_pc,find_correspondence
from edge_assisted.gaussian_feature import GaussianPointManager

###일단 densification 같은 것에 대응하는지 확인
###이 후 코드 추가

class LocalGaussianFrame:
    def __init__(self, xyz, features_dc, features_rest, opacity, scale, rotation):
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._opacity = opacity
        self._scaling = scale
        self._rotation = rotation

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

class LocalGaussianModel(GaussianModel):

    def __init__(self, sh_degree : int, config = None, opt_params = None):
        super().__init__(sh_degree, config)
        self.init_lr(6.0)

        #self.isfeatured = torch.empty(0, device="cuda").bool()
        #self.observation_points = torch.empty((0,0), device="cuda").float()#np.full(0, None, dtype=object) #np dtype:object -> torch N x 2M 으로 변경
        #self.observation_indices = torch.empty(0, device='cuda').int()

        #self.unique_gaussian_ids = torch.empty(0, device = "cuda").long()
        #self.global_gaussian_counter = 0

        #self.opt_params = opt_params

    def construct_local_gaussians(self, frames, opt_params):
        #가우시안 생성
        #그래프 옵티마이저에 추가
        #densification 참조

        for f in frames:
            pass


        self.training_setup(opt_params)