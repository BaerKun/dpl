import torch
import openvino as ov

def convert2openvino(model, save_path, weight_path=None):
    if weight_path:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    ov_model = ov.convert_model(model)
    ov.save_model(ov_model, save_path)