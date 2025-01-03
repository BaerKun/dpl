import numpy as np
import torch
from model import PoseBody25, Skeleton
import util

_try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PoseBody25Detector:
    def __init__(self, weights_path: str):
        self.model = PoseBody25()
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))

        self.num_joint = 25
        self.num_heatmap = 26
        self.num_paf = 52

        # 从颈部向四肢和头部关节点的拓扑排序，保证遍历时，当前连接不可能指向已遍历过的关节点
        self.limb_map2joints = [[1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                                [10, 11], [8, 12], [12, 13], [13, 14], [0, 15], [0, 16], [15, 17], [16, 18],
                                [11, 24], [11, 22], [14, 21], [14, 19], [22, 23], [19, 20]]
        self.limb_map_paf_xy = [[30, 31], [14, 15], [16, 17], [18, 19], [22, 23], [24, 25], [26, 27], [0, 1], [6, 7],
                                [2, 3], [4, 5], [8, 9], [10, 11], [12, 13], [32, 33], [34, 35], [36, 37], [38, 39],
                                [50, 51], [46, 47], [44, 45], [40, 41], [48, 49], [42, 43]]

    def __call__(self, ori_img: np.ndarray, device: torch.device = _try_cuda) -> list[Skeleton]:
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

            heatmap_avg += heatmap
            paf_avg += + paf
        else:
            heatmap_avg /= len(multiplier)
            paf_avg /= len(multiplier)

        joints = util.get_joints_from_heatmaps(heatmap_avg, threshold_joint)

        candidate_limbs = util.get_limbs_from_paf(joints, paf_avg, self.limb_map2joints,
                                                  self.limb_map_paf_xy, ori_img.shape[0], threshold_limb)

        return util.get_skeletons_from_limbs(candidate_limbs, self.limb_map2joints)
