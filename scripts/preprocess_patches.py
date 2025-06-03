#!/usr/bin/env python
"""Extracts ROI patches from each scanned page."""
import multiprocessing as mp
from pathlib import Path
from printerid.data.patch_extractor import extract_edge_patches

SRC = Path("data/VIPPrint/scans")           # after extraction
DST = Path("data/VIPPrint_patches")

def worker(img_path):
    printer = img_path.parent.name         # e.g. printer0
    out_dir = DST / printer
    extract_edge_patches(img_path, out_dir)

if __name__ == "__main__":
    imgs = list(Path(SRC).rglob("*.png"))
    with mp.Pool() as pool:
        for _ in pool.imap_unordered(worker, imgs):
            pass
