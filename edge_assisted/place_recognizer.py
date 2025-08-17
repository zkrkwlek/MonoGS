import torch
from torch.nn import functional as F

class PlaceRecognizer:
    def __init__(self):
        pass

    def calculate_similarity(self, descriptor1, descriptor2):
        descriptor1 = torch.nn.functional.normalize(descriptor1, dim=1)
        descriptor2 = torch.nn.functional.normalize(descriptor2, dim=1)
        similarity = torch.mm(descriptor1, descriptor2.t())  # Cosine similarity
        return similarity.item()

    def place_recognition(self, desc, keyframes, similarity_threshold = 0.7, similarity_threshold_inter = 0.85):
        highestScore = 0
        highestOther = 0
        best_other_idx = None
        best_kf_idx = None

        for index, kf_idx in enumerate(keyframes):
            keyframe = keyframes[kf_idx]
            simScore = self.calculate_similarity(desc, keyframe.pr_desc)
            #print('sim', kf_idx, simScore)
            if simScore > similarity_threshold and simScore > highestScore:
                highestScore = simScore
                best_kf_idx = kf_idx
            if simScore > similarity_threshold_inter: #check same keyframe
                if simScore > highestOther:
                    highestOther = simScore
                    best_other_idx = kf_idx
        if best_other_idx is not None:
            best_kf_idx = best_other_idx
            # if there exists loop closure, we will adjust the trajectory based on the highest similarity score
        return best_kf_idx