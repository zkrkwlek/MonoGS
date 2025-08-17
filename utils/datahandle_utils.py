import torch
import copy
from torch import nn
import numpy as np
from typing import Union, Dict, Any, Optional

"""
def move_gaussianpackets_to_cpu(gaussians):
    cpu_gaussians = copy.deepcopy(gaussians)
    for attr_name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'unique_kfIDs', 'n_obs']:
        if hasattr(cpu_gaussians, attr_name):
            tensor = getattr(cpu_gaussians, attr_name)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_gaussians, attr_name, tensor.detach().clone().cpu())
    return cpu_gaussians

def move_gaussianpackets_to_gpu(cpu_gaussians, device="cuda"):
    for attr_name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'unique_kfIDs', 'n_obs']:
        if hasattr(cpu_gaussians, attr_name):
            tensor = getattr(cpu_gaussians, attr_name)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_gaussians, attr_name, tensor.to(device))
    #return cpu_gaussians  # 이제 GPU에 있는 gaussians
"""


def detect_gaussian_model_type(model) -> str:
    """
    GaussianModel 타입 자동 감지
    Returns:
        str: 'trackable', 'base', 'unknown'
    """
    # TrackableGaussianModel 감지
    if hasattr(model, 'base_model') and hasattr(model, 'orb_features'):
        return 'trackable'

    # 기존 GaussianModel 감지
    if hasattr(model, '_xyz') and hasattr(model, '_opacity') and hasattr(model, 'get_xyz'):
        return 'base'

    return 'unknown'


def move_orb_features_to_cpu(orb_features: Dict) -> Dict:
    """ORB 특징점 데이터를 CPU로 이동"""
    cpu_orb_features = {}

    for gaussian_idx, orb_data in orb_features.items():
        # ORBFeatureData 복사
        cpu_orb_data = copy.deepcopy(orb_data)

        # NumPy 배열들은 이미 CPU에 있으므로 그대로 유지
        # cv2.KeyPoint 객체들도 그대로 유지

        cpu_orb_features[gaussian_idx] = cpu_orb_data

    return cpu_orb_features


def move_orb_features_to_gpu(orb_features: Dict, device: str = "cuda") -> Dict:
    """ORB 특징점 데이터를 GPU로 이동 (실제로는 CPU에 유지)"""
    # ORB 특징점 데이터는 OpenCV/NumPy 기반이므로 CPU에서 처리
    # GPU로 "이동"이라고 하지만 실제로는 참조만 유지
    return orb_features

def move_gaussianpacket_to_cpu(packet):
    cpu_packet = copy.deepcopy(packet)

    # GaussianModel 변환
    if packet.has_gaussians:
        tensor_attrs = ['get_xyz', 'get_opacity', 'get_scaling', 'get_rotation',
                        'get_features', '_rotation', 'unique_kfIDs', 'n_obs']

        for attr in tensor_attrs:
            if hasattr(packet, attr):
                tensor = getattr(packet, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(cpu_packet, attr, tensor.detach().clone().cpu())

    # Camera 객체들 변환
    if packet.keyframe is not None:
        cpu_packet.keyframe = move_camera_to_cpu(packet.keyframe)

    if packet.current_frame is not None:
        cpu_packet.current_frame = move_camera_to_cpu(packet.current_frame)

    # Camera 배열 변환
    if packet.keyframes is not None:
        cpu_packet.keyframes = [move_camera_to_cpu(kf) for kf in packet.keyframes]

    # 이미지 텐서들 변환
    for attr in ['gtcolor', 'gtdepth', 'gtnormal']:
        if hasattr(packet, attr):
            tensor = getattr(packet, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_packet, attr, tensor.detach().clone().cpu())

    return cpu_packet


def move_gaussianpacket_to_gpu(cpu_packet, device="cuda", training_args=None):
    # GaussianModel 변환
    if cpu_packet.has_gaussians:
        tensor_attrs = ['get_xyz', 'get_opacity', 'get_scaling', 'get_rotation',
                        'get_features', '_rotation', 'unique_kfIDs', 'n_obs']

        for attr in tensor_attrs:
            if hasattr(cpu_packet, attr):
                tensor = getattr(cpu_packet, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(cpu_packet, attr, tensor.to(device))

    # Camera 객체들 변환
    if cpu_packet.keyframe is not None:
        move_camera_to_gpu(cpu_packet.keyframe, device)

    if cpu_packet.current_frame is not None:
        move_camera_to_gpu(cpu_packet.current_frame, device)

    # Camera 배열 변환
    if cpu_packet.keyframes is not None:
        for kf in cpu_packet.keyframes:
            move_camera_to_gpu_gui(kf, device)

    # 이미지 텐서들 변환
    for attr in ['gtcolor', 'gtdepth', 'gtnormal']:
        if hasattr(cpu_packet, attr):
            tensor = getattr(cpu_packet, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_packet, attr, tensor.to(device))

def move_frame_to_cpu(camera):
    cpu_camera = copy.deepcopy(camera)

    # 텐서 속성들
    """
    tensor_attrs = ['gaussianpoints']
    for attr in tensor_attrs:
        if hasattr(cpu_camera, attr):
            tensor = getattr(cpu_camera, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_camera, attr, tensor.clone())
    """
    # nn.Parameter들
    """
    np_attrs = ['cam_rot_delta', 'cam_trans_delta', 'exposure_a', 'exposure_b']
    for attr in np_attrs:
        if hasattr(cpu_camera, attr):
            param = getattr(cpu_camera, attr)
            if isinstance(tensor, np.ndarray):
                setattr(cpu_gaussians, attr, copy.deepcopy(tensor))
            #if isinstance(param, np.ndarray):
            #    setattr(cpu_camera, attr, nn.Parameter(param.detach().clone().cpu()))
    """
    #cpu_camera.device = "cpu"
    return cpu_camera


def move_camera_to_cpu(camera):
    cpu_camera = copy.deepcopy(camera)

    # 텐서 속성들
    tensor_attrs = ['R', 'T', 'R_gt', 'T_gt', 'original_image', 'depth', 'projection_matrix', 'grad_mask']
    for attr in tensor_attrs:
        if hasattr(cpu_camera, attr):
            tensor = getattr(cpu_camera, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_camera, attr, tensor.detach().clone().cpu())

    # nn.Parameter들
    param_attrs = ['cam_rot_delta', 'cam_trans_delta', 'exposure_a', 'exposure_b']
    for attr in param_attrs:
        if hasattr(cpu_camera, attr):
            param = getattr(cpu_camera, attr)
            if isinstance(param, nn.Parameter):
                setattr(cpu_camera, attr, nn.Parameter(param.detach().clone().cpu()))

    cpu_camera.device = "cpu"
    return cpu_camera


def move_camera_to_gpu(cpu_camera, device="cuda"):
    # 텐서 속성들
    tensor_attrs = ['R', 'T', 'R_gt', 'T_gt', 'original_image', 'depth', 'projection_matrix']
    for attr in tensor_attrs:
        if hasattr(cpu_camera, attr):
            tensor = getattr(cpu_camera, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_camera, attr, tensor.to(device))

    # nn.Parameter들
    param_attrs = ['cam_rot_delta', 'cam_trans_delta', 'exposure_a', 'exposure_b']
    for attr in param_attrs:
        if hasattr(cpu_camera, attr):
            param = getattr(cpu_camera, attr)
            if isinstance(param, nn.Parameter):
                setattr(cpu_camera, attr, nn.Parameter(param.to(device)))

    cpu_camera.device = device

def move_camera_to_cpu_gui(camera):
    cpu_camera = copy.deepcopy(camera)

    # 텐서 속성들
    tensor_attrs = ['R', 'T', 'R_gt', 'T_gt','projection_matrix']
    for attr in tensor_attrs:
        if hasattr(cpu_camera, attr):
            tensor = getattr(cpu_camera, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_camera, attr, tensor.detach().clone().cpu())
    cpu_camera.device = "cpu"
    return cpu_camera


def move_camera_to_gpu_gui(cpu_camera, device="cuda"):
    # 텐서 속성들
    tensor_attrs = ['R', 'T', 'R_gt', 'T_gt','projection_matrix']
    for attr in tensor_attrs:
        if hasattr(cpu_camera, attr):
            tensor = getattr(cpu_camera, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_camera, attr, tensor.to(device))
    cpu_camera.device = device

def move_gaussianmodel_to_cpu(gaussians):
    cpu_gaussians = copy.deepcopy(gaussians)

    # 주요 텐서 속성들
    tensor_attrs = [
        '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
        'max_radii2D', 'xyz_gradient_accum', 'unique_kfIDs', 'n_obs', 'denom' ,'isfeatured','observation_indices','observation_points','unique_gaussian_ids'
    ]

    for attr in tensor_attrs:
        if hasattr(cpu_gaussians, attr):
            tensor = getattr(cpu_gaussians, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_gaussians, attr, tensor.detach().clone().cpu())
            elif isinstance(tensor, nn.Parameter):
                setattr(cpu_gaussians, attr, nn.Parameter(tensor.detach().clone().cpu()))
            elif isinstance(tensor, np.ndarray):
                setattr(cpu_gaussians, attr, copy.deepcopy(tensor))

    # optimizer는 None으로 (재생성 필요)
    cpu_gaussians.optimizer = None
    return cpu_gaussians

def move_gaussianmodel_to_gpu(cpu_gaussians, device="cuda"):
    tensor_attrs = [
        '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
        'max_radii2D', 'xyz_gradient_accum', 'unique_kfIDs', 'n_obs', 'denom', 'isfeatured','observation_indices','observation_points','unique_gaussian_ids'
    ]

    for attr in tensor_attrs:
        if hasattr(cpu_gaussians, attr):
            tensor = getattr(cpu_gaussians, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(cpu_gaussians, attr, tensor.to(device))
            elif isinstance(tensor, nn.Parameter):
                setattr(cpu_gaussians, attr, nn.Parameter(tensor.to(device)))
            elif isinstance(tensor, np.ndarray):
                setattr(cpu_gaussians, attr, copy.deepcopy(tensor))

    # optimizer는 training_setup으로 재생성 필요

def move_occ_visibility_to_cpu(occ_aware_visibility):
    cpu_occ_visibility = {}
    for idx, tensor in occ_aware_visibility.items():
        if isinstance(tensor, torch.Tensor):
            cpu_occ_visibility[idx] = tensor.detach().clone().cpu()
        else:
            cpu_occ_visibility[idx] = tensor
    return cpu_occ_visibility

def move_occ_visibility_to_gpu(cpu_occ_visibility, device="cuda"):
    for idx, tensor in cpu_occ_visibility.items():
        if isinstance(tensor, torch.Tensor):
            cpu_occ_visibility[idx] = tensor.to(device)


def move_trackable_gaussian_to_cpu(trackable_model) -> Any:
    """TrackableGaussianModel을 CPU로 이동"""
    # 전체 모델 복사
    cpu_model = copy.deepcopy(trackable_model)

    # 1. 기본 GaussianModel을 CPU로 이동
    cpu_model.base_model = move_gaussianmodel_to_cpu(cpu_model.base_model)

    # 2. ORB 특징점 데이터를 CPU로 이동
    cpu_model.orb_features = move_orb_features_to_cpu(cpu_model.orb_features)

    # 3. ORB detector는 이미 CPU 기반이므로 그대로 유지
    # 4. 성능 통계 및 메타데이터는 그대로 유지

    return cpu_model


def move_trackable_gaussian_to_gpu(cpu_model, device: str = "cuda") -> None:
    """TrackableGaussianModel을 GPU로 이동 (in-place)"""

    # 1. 기본 GaussianModel을 GPU로 이동
    move_gaussianmodel_to_gpu(cpu_model.base_model, device)

    # 2. ORB 특징점 데이터 처리 (실제로는 CPU에 유지)
    cpu_model.orb_features = move_orb_features_to_gpu(cpu_model.orb_features, device)

    # 3. ORB detector는 CPU 기반이므로 그대로 유지

def move_gaussian_to_cpu(model) -> Any:
    """
    자동 감지하여 적절한 방법으로 GaussianModel을 CPU로 이동

    Args:
        model: GaussianModel 또는 TrackableGaussianModel

    Returns:
        CPU로 이동된 모델
    """
    model_type = detect_gaussian_model_type(model)

    if model_type == 'trackable':
        print("[DeviceTransfer] Moving TrackableGaussianModel to CPU...")
        return move_trackable_gaussian_to_cpu(model)
    elif model_type == 'base':
        print("[DeviceTransfer] Moving GaussianModel to CPU...")
        return move_gaussianmodel_to_cpu(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}. "
                         f"Expected GaussianModel or TrackableGaussianModel")


def move_gaussian_to_gpu(cpu_model, device: str = "cuda") -> None:
    """
    자동 감지하여 적절한 방법으로 GaussianModel을 GPU로 이동 (in-place)

    Args:
        cpu_model: CPU에 있는 GaussianModel 또는 TrackableGaussianModel
        device: 목표 디바이스
    """
    model_type = detect_gaussian_model_type(cpu_model)

    if model_type == 'trackable':
        print(f"[DeviceTransfer] Moving TrackableGaussianModel to {device}...")
        move_trackable_gaussian_to_gpu(cpu_model, device)
    elif model_type == 'base':
        print(f"[DeviceTransfer] Moving GaussianModel to {device}...")
        move_gaussianmodel_to_gpu(cpu_model, device)
    else:
        raise ValueError(f"Unknown model type: {type(cpu_model)}. "
                         f"Expected GaussianModel or TrackableGaussianModel")