import torch

def get_patch_loss(patch, gt_patch):
    return torch.abs(patch-gt_patch)

def get_reprojection_loss2(gaussians, keypoints):
    return gaussians-keypoints

def get_reprojection_loss(gaussians, keypoints):
    # 휴버 로스 추가
    #l1 = keypoints-gaussians
    #loss = torch.abs(gaussians-keypoints)
    l2_dist = torch.sum((gaussians - keypoints) ** 2, dim=1)
    #l1_dist = torch.sqrt(l2_dist + 1e-8)
    #condition = l1_dist < 1.0
    #loss = torch.where(condition, 0.5 * l2_dist, l1_dist - 0.5)
    #print(l1, l1.mean())
    return l2_dist


def get_reprojection_loss_huber(projected_points, observed_keypoints, delta=3.0):
    """
    Huber loss를 사용한 reprojection loss (outlier에 더 robust)
    """
    assert projected_points.shape == observed_keypoints.shape
    assert projected_points.shape[1] == 2

    diff = projected_points - observed_keypoints
    l2_dist = torch.sum(diff ** 2, dim=1)
    l1_dist = torch.sqrt(l2_dist + 1e-8)  # 수치적 안정성을 위한 epsilon

    # Huber loss
    condition = l1_dist < delta
    loss = torch.where(condition,
                       0.5 * l2_dist,
                       delta * (l1_dist - 0.5 * delta))
    return loss

def get_loss_gaussian(config, image, viewpoint, projection):
    gt_image = viewpoint.original_image.cuda()
    n = projection.shape[0]

    # 1) 전체 에러 텐서 초기화 (크기 n, RGB 차이 대신 스칼라 에러로 가정)
    error = torch.full((n,1), 1000.0, device=projection.device, dtype=torch.float)

    # 2) 좌표 정수화 및 유효 포인트 마스크
    kps = torch.round(projection).int()
    valid = (kps[:, 0] >= 0) & (kps[:, 0] < viewpoint.image_width) & (kps[:, 1] >= 0) & (kps[:, 1] < viewpoint.image_height)

    # 3) 유효 포인트 좌표 추출
    kps_valid = kps[valid]
    x, y = kps_valid[:, 0], kps_valid[:, 1]

    # 4) 두 이미지에서 픽셀 RGB 추출
    rgb1 = image[:,y, x].permute(1, 0).float()
    rgb2 = gt_image[:,y, x].permute(1, 0).float()

    # 5) 유효 포인트 RGB 차이 계산 (예: L2 norm)
    rgb_diff = rgb1 - rgb2
    diff_norm = torch.norm(rgb_diff, dim=1, keepdim=True)  # shape (m,), m = 유효 포인트 개수
    #print(rgb1.shape,rgb_diff.shape, diff_norm.shape, error.shape)
    # 6) 전체 에러  텐서에서 유효한 인덱스 위치에만 값 반영
    error[valid] = diff_norm
    
    return error

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, monocular = True, feature_mask = None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if monocular:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint, feature_mask = feature_mask)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, feature_mask = feature_mask)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, feature_mask = None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    if feature_mask is not None:
        rgb_pixel_mask = rgb_pixel_mask * feature_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False, feature_mask = None
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, feature_mask = feature_mask)
    depth_mask = depth_pixel_mask * opacity_mask
    if feature_mask is not None:
        depth_mask = depth_mask * feature_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, feature_mask = None):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint, feature_mask = feature_mask)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, feature_mask = feature_mask)


def get_loss_mapping_rgb(config, image, depth, viewpoint, feature_mask = None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    if feature_mask is not None:
        rgb_pixel_mask = rgb_pixel_mask * feature_mask
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False, feature_mask = None):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    if feature_mask is not None:
        rgb_pixel_mask = rgb_pixel_mask * feature_mask
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    if feature_mask is not None:
        depth_pixel_mask = depth_pixel_mask * feature_mask

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    #print("loss", l1_depth.mean(), l1_rgb.mean(), l1_rgb.shape, image.shape, rgb_pixel_mask.shape)
    return l1_rgb.mean(), l1_depth.mean()#alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


