# printerid/utils/grad_cam.py
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np, cv2


def grad_cam(model, img_tensor, target_layer="blocks.12.norm1"):
    """
    Returns heatmap (H,W) as numpy array in [0,1].
    Works with timm models exposing .blocks (ViT, Xception pre-act name hack).
    """
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict([*model.named_modules()])[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    pred = model(img_tensor.unsqueeze(0))
    class_idx = pred.argmax().item()
    score = pred[0, class_idx]
    score.backward()

    acts = activations["value"][0]          # [C,H,W]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(0).clamp(min=0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam.cpu().numpy(), (img_tensor.shape[2], img_tensor.shape[1]))
    h1.remove(); h2.remove()
    return cam, class_idx
