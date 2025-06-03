# printerid/training/train.py
from __future__ import annotations
import argparse, yaml, random, time
from pathlib import Path

import numpy as np, torch
from torch.utils.data import DataLoader

from ..data.dataset import VIPPrintPatchDataset, make_splits
from ..models.xception import build_model
from ..utils.metrics import batch_accuracy
from ..utils.logger import CSVLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main(cfg):
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])

    root = Path(cfg["data"]["patch_dir"])
    split_file = root / "splits.json"
    if not split_file.exists():
        # first run → create splits on the fly
        train_s, val_s, test_s = make_splits(
            root,
            cfg["data"]["train_split"],
            cfg["data"]["val_split"],
            cfg["seed"],
        )
        import json, random

        json.dump(train_s + val_s + test_s, split_file.open("w"))
    else:
        import json, itertools

        all_s = json.load(split_file.open())
        # restore lists
        train_s = [x for x in all_s if x.get("split") == "train"]
        val_s = [x for x in all_s if x.get("split") == "val"]

    train_ds = VIPPrintPatchDataset(
        root,
        split_file,
        train=True,
        input_size=cfg["data"]["input_size"],
    )
    val_ds = VIPPrintPatchDataset(
        root, split_file, train=False, input_size=cfg["data"]["input_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg["optim"]["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["optim"]["epochs"]
    )

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    logger = CSVLogger(out_dir)

    best_acc = 0.0
    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += batch_accuracy(logits, labels)

            if (step + 1) % cfg["logging"]["log_interval"] == 0:
                print(
                    f"Epoch[{epoch+1}/{cfg['optim']['epochs']}], "
                    f"Step[{step+1}/{len(train_loader)}] "
                    f"Loss:{loss.item():.4f}"
                )

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)

        # ---- validation ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=scheduler.get_last_lr()[0],
        )
        if val_acc > best_acc and cfg["logging"]["save_best"]:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        print(
            f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, best={best_acc:.3f}"
        )

    logger.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train printer ID model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
