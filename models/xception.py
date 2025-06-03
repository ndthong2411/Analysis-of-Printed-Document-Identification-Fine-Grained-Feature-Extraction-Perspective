# printerid/models/xception.py
from __future__ import annotations
import timm
import torch.nn as nn


def build_model(num_classes: int, name: str = "xception41_preact", pretrained=False):
    """
    Returns timm model with an output head sized to `num_classes`.
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    # timm already adjusts final layer if num_classes given.
    return model
