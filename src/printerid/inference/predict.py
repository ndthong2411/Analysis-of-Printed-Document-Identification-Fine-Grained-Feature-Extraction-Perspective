# printerid/inference/predict.py
import argparse, yaml, torch, cv2
from pathlib import Path
from torchvision import transforms as T

from ..models.xception import build_model
from ..utils.transforms import build_val_transforms
from ..data.patch_extractor import extract_edge_patches


def load_model(cfg):
    model = build_model(
        cfg["model"]["num_classes"],
        name=cfg["model"]["name"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(Path(cfg["logging"]["out_dir"]) / "best.pt"))
    model.eval().to(cfg["device"])
    return model


def predict_single(img_path: Path, cfg, model):
    tmp_dir = Path(".tmp_patches")
    tmp_dir.mkdir(exist_ok=True)
    num = extract_edge_patches(img_path, tmp_dir, patch_size=cfg["data"]["input_size"])
    transform = build_val_transforms(cfg["data"]["input_size"])
    votes = torch.zeros(cfg["model"]["num_classes"])

    for patch in tmp_dir.glob(f"{img_path.stem}_*.png"):
        img = cv2.imread(str(patch))[:, :, ::-1]
        img = transform(img).unsqueeze(0).to(cfg["device"])
        with torch.no_grad():
            logits = model(img)
        votes += logits.squeeze().cpu().softmax(0)

    printer = votes.argmax().item()
    confidence = votes.max().item()
    # clean tmp
    for p in tmp_dir.glob(f"{img_path.stem}_*.png"):
        p.unlink()
    return printer, confidence


def main(cfg):
    parser = argparse.ArgumentParser(description="Predict printer of a scanned image.")
    parser.add_argument("image", type=str)
    args = parser.parse_args()

    model = load_model(cfg)
    printer, conf = predict_single(Path(args.image), cfg, model)
    print(f"Predicted: printer{printer}  (confidence={conf:.3f})")


if __name__ == "__main__":
    main(yaml.safe_load(open("config.yaml")))
