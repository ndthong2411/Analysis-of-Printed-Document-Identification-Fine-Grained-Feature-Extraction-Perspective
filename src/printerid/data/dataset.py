# printerid/data/dataset.py
from __future__ import annotations
import glob, json, random, re
from pathlib import Path
from typing import Tuple, List

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from .patch_extractor import extract_edge_patches
from ..utils.transforms import build_train_transforms, build_val_transforms


class VIPPrintPatchDataset(Dataset):
    """
    Loads ROI patches *already extracted* by preprocess_patches.py.
    Directory structure:
        root/patches/<printer_id>/<image_id>_<patch_idx>.png
    """

    def __init__(
        self,
        patch_root: Path,
        split_json: Path,
        train: bool,
        input_size: int = 299,
    ):
        self.patch_root = Path(patch_root)
        self.train = train
        self.input_size = input_size

        # list[dict]: {"path": str, "label": int}
        self.samples = json.loads(Path(split_json).read_text())

        self.transform = (
            build_train_transforms(input_size)
            if train
            else build_val_transforms(input_size)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        img = cv2.imread(item["path"], cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR→RGB
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = self.transform(img)
        return img, item["label"]


def make_splits(
    full_patch_dir: Path,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Splits *by original digital IMAGE ID* to avoid content leakage.
    Assumes patch filenames = <imageid>_<patchidx>.png
    """
    rng = random.Random(seed)

    all_files = list(full_patch_dir.glob("*/*.png"))
    print(f"Found {len(all_files):,} patches")

    # extract image id
    id_pat = re.compile(r"(\d+)_\d+\.png$")
    id_to_files = {}
    for f in all_files:
        match = id_pat.search(f.name)
        if match is None:
            continue
        img_id = match.group(1)
        id_to_files.setdefault(img_id, []).append(f)

    img_ids = list(id_to_files)
    rng.shuffle(img_ids)
    n = len(img_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train : n_train + n_val])

    def flatten(id_set):
        out = []
        for iid in id_set:
            for file_ in id_to_files[iid]:
                printer_label = int(file_.parent.name.replace("printer", ""))
                out.append({"path": str(file_), "label": printer_label})
        return out

    train_samples = flatten(train_ids)
    val_samples = flatten(val_ids)
    test_samples = flatten(set(img_ids) - train_ids - val_ids)

    return train_samples, val_samples, test_samples
