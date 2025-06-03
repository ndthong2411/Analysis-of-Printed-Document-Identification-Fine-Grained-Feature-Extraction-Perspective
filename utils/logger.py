# printerid/utils/logger.py
from __future__ import annotations
import csv, time
from pathlib import Path


class CSVLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "train_log.csv"
        self.file = self.path.open("w", newline="")
        self.writer = None

    def log(self, **kwargs):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=kwargs.keys())
            self.writer.writeheader()
        self.writer.writerow(kwargs)
        self.file.flush()

    def close(self):
        self.file.close()
