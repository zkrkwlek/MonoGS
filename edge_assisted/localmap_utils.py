import torch
import time
from edge_assisted.gaussian_orb_model import GaussianOrbModel, create_gaussian_orb_model

def get_local_gaussians(gaussians, current_window):
    a = time.time()
    temp_observations = gaussians.observations[gaussians.isfeatured.clone().detach().cpu().numpy()]
    temp_index = torch.where(gaussians.isfeatured)[0]
    local_mask = torch.zeros((gaussians._xyz.size()[0], 1), device="cuda")
    """
    for gidx, obs in zip(temp_index,temp_observations):
        if obs is not None:
            for fidx, kpidx in obs.items():
                if fidx in current_window:
                    local_mask[gidx] = True
                    break
    """
    print('backend',current_window)
    b = time.time()
    print("local gaussians time", b-a, torch.count_nonzero(local_mask))


