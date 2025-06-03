# printerid/training/evaluate.py
import argparse, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader

from ..data.dataset import VIPPrintPatchDataset
from ..models.xception import build_model
from ..utils.metrics import report


def main(cfg):
    device = torch.device(cfg["device"])

    ds = VIPPrintPatchDataset(
        cfg["data"]["patch_dir"],
        cfg["data"]["patch_dir"] + "/splits.json",
        train=False,
        input_size=cfg["data"]["input_size"],
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

    model = build_model(
        cfg["model"]["num_classes"], name=cfg["model"]["name"], pretrained=False
    )
    model.load_state_dict(torch.load(Path(cfg["logging"]["out_dir"]) / "best.pt"))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    class_names = [f"printer{i}" for i in range(cfg["model"]["num_classes"])]
    rep = report(y_true, y_pred, class_names)
    print(rep["report"])
    print("Macro-F1:", rep["macro_f1"])
    rep["confusion"].to_csv(Path(cfg["logging"]["out_dir"]) / "confusion.csv")
    print("Saved confusion matrix CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(yaml.safe_load(open(args.config)))
