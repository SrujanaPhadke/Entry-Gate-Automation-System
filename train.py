from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Read command-line options for Ultralytics YOLO training."""
    parser = argparse.ArgumentParser(description="Train a YOLO number plate detector with Ultralytics.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to Ultralytics data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, for example yolov8n.pt or yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", default=None, help="Device, for example cpu, 0, or 0,1")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics output project directory")
    parser.add_argument("--name", default="number_plate", help="Run name under the project directory")
    parser.add_argument("--copy-best", action="store_true", help="Copy trained best.pt to models/best.pt")
    return parser.parse_args()


def validate_dataset(data_path: Path) -> None:
    """Fail early when the dataset configuration cannot be found."""
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit(f"Dataset file must be YAML: {data_path}")


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    data_path = Path(args.data)
    validate_dataset(data_path)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_path}")

    if args.copy_best:
        target = Path("models") / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, target)
        print(f"Copied {best_path} to {target}")


if __name__ == "__main__":
    main()
