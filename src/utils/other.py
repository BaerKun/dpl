import torch
import numpy as np

def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    img_np = tensor.permute(1, 2, 0).numpy()
    img_np = img_np * 255
    return img_np.astype(np.uint8)


def show_image(image: np.ndarray, label: str=None, prob: float = None) -> bool:
    import cv2

    if label is None:
        label = ''
    if prob is not None:
        label += f" {prob:.2f}"
    cv2.putText(image, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("test", image)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        return False
    return True