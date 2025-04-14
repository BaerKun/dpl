import torch, cv2
import numpy as np


def visualize_tensor(tensor: torch.Tensor, delta=None, label: str = None, prob: float = None,
                     key_points: np.ndarray = None) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    im = tensor.cpu().numpy().transpose(1, 2, 0)
    if delta is not None:
        im += delta

    if label is not None or prob is not None:
        cv2.putText(im, f"{label if label is not None else ''} {prob if prob is not None else ''}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))

    for kp in key_points:
        cv2.circle(im, (int(kp[0]), int(kp[1])), 3, (1, 0, 0), -1)

    return im
