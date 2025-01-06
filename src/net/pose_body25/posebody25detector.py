import numpy as np
import torch
from model import PoseBody25, Skeleton
import util

_try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseBody25Detector:
    def __init__(self, weights_path: str):
        self.model = PoseBody25()
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.num_joints = 25
        self.num_heatmap = 26
        self.num_paf = 52

    def __call__(self, ori_img: np.ndarray, output_heatmap_paf: bool = False, device: torch.device = _try_cuda) -> list[Skeleton]:
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [0.5]
        box_size = 368
        threshold_joint = 0.1
        threshold_limb = 0.05
        multiplier = [x * box_size / ori_img.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((self.num_heatmap, ori_img.shape[0], ori_img.shape[1]))
        paf_avg = np.zeros((self.num_paf, ori_img.shape[0], ori_img.shape[1]))

        self.model.eval()
        for m in multiplier:
            scale = m
            data, pad = util.preprocess_image2tensor(ori_img, scale)

            with torch.no_grad():
                data = data.to(device)
                self.model.to(device)
                heatmap, paf = self.model(data)
                heatmap = heatmap.cpu().numpy()
                paf = paf.cpu().numpy()

            heatmap, paf = util.postprocess_heatmap_paf(heatmap, paf, pad, ori_img.shape)
            if output_heatmap_paf:
                util.show_heatmaps_paf(heatmap, paf)
            heatmap_avg += heatmap
            paf_avg += + paf
        else:
            heatmap_avg /= len(multiplier)
            paf_avg /= len(multiplier)

        joints = util.get_joints_from_heatmaps(heatmap_avg, threshold_joint)
        candidate_limbs = util.get_limbs_from_paf(joints, paf_avg, ori_img.shape[0], threshold_limb)
        return util.get_skeletons_from_limbs(candidate_limbs)
