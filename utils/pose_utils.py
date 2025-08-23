import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    ssm = torch.tensor([[zero, -x[2], x[1]],
                        [x[2], zero, -x[0]],
                        [-x[1], x[0], zero]], device=x.device, dtype=x.dtype)
    """
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    """
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    angle_sq = torch.sum(theta * theta)

    W = skew_sym_mat(theta)
    W2 = W @ W

    I = torch.eye(3, device=device, dtype=dtype)

    if angle_sq < 1e-8:
        return I + W + 0.5 * W2
    else:
        angle = torch.sqrt(angle_sq)
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / angle_sq) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device

    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W

    #angle = torch.norm(theta)
    angle_sq = torch.sum(theta * theta)

    if angle_sq < 1e-8:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        angle = torch.sqrt(angle_sq)
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / angle_sq)
            + W2 * ((angle - torch.sin(angle)) / (angle_sq*angle))
        )
    return V


def SE3_exp(tau):
    #dtype = tau.dtype
    #device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    #T = torch.eye(4, device=device, dtype=dtype)
    #T[:3, :3] = R
    #T[:3, 3] = t
    T = torch.zeros(4,4, device=tau.device, dtype = R.dtype)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1
    return T

def compute_F12(R1, t1, R2, t2, K1, K2):
    # R1, R2: (3,3) tensor
    # t1, t2: (3,) tensor
    # K1, K2: (3,3) tensor
    R2 = R2.float()
    t2 = t2.float()
    R12 = R1 @ R2.t()
    t12 = -R1 @ R2.t() @ t2 + t1
    t12x = skew_sym_mat(t12)
    M = torch.linalg.inv(K1.t()) @ t12x @ R12 @ torch.linalg.inv(K2)
    return M

def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    converged = tau.norm() < converged_threshold

    T_w2c = torch.zeros(4,4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T
    T_w2c[3, 3] = 1

    new_w2c = SE3_exp(tau) @ T_w2c
    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
