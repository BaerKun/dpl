import cv2
import numpy as np
import math

from scipy.ndimage.filters import gaussian_filter
import torch
from model import Joint, Skeleton


def draw_body_pose(img, skeletons: list[Skeleton]):
    stick_width = 4

    limbs = [[1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
             [10, 11], [8, 12], [12, 13], [13, 14], [0, 15], [0, 16], [15, 17], [16, 18],
             [11, 24], [11, 22], [14, 21], [14, 19], [22, 23], [19, 20]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 255, 0], [255, 255, 85], [255, 255, 170],
              [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255]]
    for skel in skeletons:
        for joint, color in zip(skel.joints, colors):
            if joint is None:
                continue
            cv2.circle(img, joint.xy, 4, color, thickness=-1)

    for skel in skeletons:
        for limb, color in zip(limbs, colors):
            joint0 = skel[limb[0]]
            joint1 = skel[limb[1]]
            if joint0 is None or joint1 is None:
                continue
            cur_canvas = img.copy()
            x0, y0 = joint0.xy
            x1, y1 = joint1.xy
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            polygon = cv2.ellipse2Poly(((x0 + x1) // 2, (y0 + y1) // 2), (int(length / 2), stick_width), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)

    return img


def pad_down_right_corner(img: np.ndarray) -> (np.ndarray, (int, int)):
    stride = 8
    pad_value = 128
    h, w, _ = img.shape
    pad_d = 0 if h % stride == 0 else stride - h % stride
    pad_r = 0 if w % stride == 0 else stride - w % stride

    if pad_d == 0 and pad_r == 0:
        padded_img = img
    else:
        padded_img = cv2.copyMakeBorder(img, 0, pad_d, 0, pad_r, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, (pad_d, pad_r)


def preprocess_image2tensor(img: np.ndarray, scale: float) -> (torch.Tensor, (int, int)):
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded_img, pad = pad_down_right_corner(resized_img)
    im = np.transpose(np.float32(padded_img), (2, 0, 1)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    data = torch.from_numpy(im).float()
    data.unsqueeze_(0)
    return data, pad


def postprocess_heatmap_paf(heatmap: np.ndarray, paf: np.ndarray, pad: (int, int), output_shape: (int, int)):
    stride = 8

    def __process(_x: np.ndarray):
        _y = np.transpose(np.squeeze(_x), (1, 2, 0))
        _y = cv2.resize(_y, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        _y = _y[:_y.shape[0] - pad[0], :_y.shape[1] - pad[1]]
        _y = cv2.resize(_y, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)
        return _y.transpose(2, 0, 1)

    return __process(heatmap), __process(paf)


def get_joints_from_heatmaps(heatmaps: np.ndarray, threshold: float) -> list[list[Joint]]:
    joints = []

    for heatmap in heatmaps[:25]:
        smooth_heatmap = gaussian_filter(heatmap, sigma=3)

        map_left = np.zeros(smooth_heatmap.shape)
        map_left[1:, :] = smooth_heatmap[:-1, :]
        map_right = np.zeros(smooth_heatmap.shape)
        map_right[:-1, :] = smooth_heatmap[1:, :]
        map_up = np.zeros(smooth_heatmap.shape)
        map_up[:, 1:] = smooth_heatmap[:, :-1]
        map_down = np.zeros(smooth_heatmap.shape)
        map_down[:, :-1] = smooth_heatmap[:, 1:]

        # 求热图中超过阈值的峰值点，作为关节点的候选点
        peaks_binary = np.logical_and.reduce(
            (smooth_heatmap > threshold, smooth_heatmap >= map_left, smooth_heatmap >= map_right,
             smooth_heatmap >= map_up,
             smooth_heatmap >= map_down))  # 逻辑与

        peaks = np.argwhere(peaks_binary)
        candidate_joints = [Joint(x, y, heatmap[y, x].item()) for y, x in peaks]
        joints.append(candidate_joints)

    return joints


def get_limbs_from_paf(joints: list[list[Joint]], paf: np.ndarray, map2joints, map2paf, ori_img_w, threshold) -> \
        list[(Joint, Joint, float)]:
    matched_limbs = []
    mid_num = 10

    # 用关节点向量和paf匹配，得到候选躯干
    for joints_idx, paf_idx in zip(map2joints, map2paf):
        score_mid = paf[paf_idx, :, :]
        candidate_joint0 = joints[joints_idx[0]]
        candidate_joint1 = joints[joints_idx[1]]
        num_joint0 = len(candidate_joint0)
        num_joint1 = len(candidate_joint1)
        if num_joint0 != 0 and num_joint1 != 0:
            candidate_connection = []
            for joint0 in candidate_joint0:
                for joint1 in candidate_joint1:
                    vec = np.subtract(joint1.xy, joint0.xy)
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0.:
                        continue
                    vec = np.divide(vec, norm)

                    start_end = zip(np.linspace(joint0.x, joint1.x, num=mid_num),
                                    np.linspace(joint0.y, joint1.y, num=mid_num))

                    vec_paf = np.array([score_mid[:, int(round(y)), int(round(x))] for x, y in start_end])

                    score_midpoints = np.multiply(vec_paf, vec).sum(axis=1)  # cos <vec, vec_paf>
                    score_with_dist_prior = (score_midpoints.mean().item() +
                                             min(ori_img_w / 2. / norm - 1., 0.))

                    if (score_with_dist_prior > 0. and
                            len(np.argwhere(score_midpoints > threshold)) > 0.8 * mid_num):
                        candidate_connection.append((joint0, joint1, score_with_dist_prior))

            candidate_connection = sorted(candidate_connection, key=lambda x: x[2], reverse=True)
            connection = []
            matched_joints_id = []
            for joint0, joint1, score in candidate_connection:
                if id(joint0) not in matched_joints_id and id(joint1) not in matched_joints_id:
                    connection.append((joint0, joint1, score))
                    matched_joints_id.append(id(joint0))
                    matched_joints_id.append(id(joint1))
                    if len(connection) >= min(num_joint0, num_joint1):
                        break

            matched_limbs.append(connection)
        else:
            matched_limbs.append([])

    return matched_limbs


def get_skeletons_from_limbs(limbs: list[(Joint, Joint, float)], map_joints: list[tuple[int, int]])->list[Skeleton]:
    candidate_skeleton = []

    # 躯干尝试搭建骨架
    for (joint0_idx, joint1_idx), limb in zip(map_joints, limbs):
        if not limb:
            continue
        for joint0, joint1, score in limb:  # = 1:size(temp,1)
            for cand_skel in candidate_skeleton:  # 1:size(subset,1):
                if cand_skel[joint0_idx] is joint0:
                    # 因为拓扑排序，只可能是id0出现重复，
                    # 而一般id1不可能出现在之前的骨架中，
                    # 除非允许两个骨架的同一个关节位置共用一个关节点
                    cand_skel[joint1_idx] = joint1
                    cand_skel.num_joints += 1
                    cand_skel.score += score
                    break
            else:
                cand_skel = Skeleton()
                cand_skel[joint0_idx] = joint0
                cand_skel[joint1_idx] = joint1
                cand_skel.num_joints = 2
                cand_skel.score = joint0.score + joint1.score + score
                candidate_skeleton.append(cand_skel)

    skeletons = []
    for cand_skel in candidate_skeleton:
        if cand_skel.score >= 4. and cand_skel.num_joints / cand_skel.score >= 0.4:
            skeletons.append(cand_skel)

    return skeletons
