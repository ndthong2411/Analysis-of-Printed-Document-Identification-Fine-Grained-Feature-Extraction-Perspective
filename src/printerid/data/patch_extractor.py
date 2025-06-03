# printerid/data/patch_extractor.py
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def extract_edge_patches(
    img_path: Path,
    out_dir: Path,
    patch_size: int = 299,
    stride: int = 150,
    max_patches: int = 80,
):
    """
    Simple but effective: run Canny, slide a window, keep windows whose
    mean edge density > threshold. Saves RGB patches as PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    h, w = gray.shape
    patches = 0
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            roi_edges = edges[y : y + patch_size, x : x + patch_size]
            density = roi_edges.mean() / 255.0
            if density < 0.05:  # skip flat areas
                continue
            roi_rgb = img[y : y + patch_size, x : x + patch_size]
            fname = out_dir / f"{img_path.stem}_{patches:03d}.png"
            cv2.imwrite(str(fname), roi_rgb)
            patches += 1
            if patches >= max_patches:
                return patches
    return patches
