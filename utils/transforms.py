# printerid/utils/transforms.py
from __future__ import annotations
import numpy as np
import cv2
import torch
from torchvision import transforms as T


def _to_tensor(img: np.ndarray):
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0


class CV2ToTensor:
    def __call__(self, img: np.ndarray):
        return _to_tensor(img)


def build_train_transforms(size: int):
    return T.Compose(
        [
            T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)),  # keep CV2 friendly
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def build_val_transforms(size: int):
    return T.Compose(
        [
            T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)),
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
