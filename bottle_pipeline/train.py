"""Train one YOLO model on the prepared bottle datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("segment", "classify"))
    parser.add_argument("--datasets", type=Path, default=root / "artifacts" / "datasets")
    parser.add_argument("--model", type=Path, help="Initial YOLO .pt weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", help="Ultralytics device, e.g. 0 or cpu")
    args = parser.parse_args()
    is_seg = args.task == "segment"
    weights = args.model or root.parent / "modelZoo" / ("yolo26n-seg.pt" if is_seg else "yolo26n-cls.pt")
    data = args.datasets.resolve() / args.task
    if is_seg:
        data = data / "data.yaml"
    if not data.exists() or not weights.is_file():
        parser.error(f"Check the prepared data and initial weights: {data}, {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != args.task:
        parser.error(f"Requested {args.task} but {weights} is a {model.task} model")
    model.train(
        data=str(data),
        project=str(root / "artifacts" / "runs" / args.task),
        name="bottles",
        imgsz=args.imgsz if args.imgsz is not None else (640 if is_seg else 224),
        batch=args.batch if args.batch is not None else (8 if is_seg else 32),
        epochs=args.epochs,
        device=args.device,
        workers=4,
    )


if __name__ == "__main__":
    main()
