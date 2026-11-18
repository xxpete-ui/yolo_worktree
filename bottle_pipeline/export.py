"""Export a classification or single-class bottle segmentation checkpoint for the pipeline."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from bottle_pipeline.classifier import Classifier
from bottle_pipeline.segmentation import Segmenter


def export(weights, output, imgsz=None):
    """Stage a checkpoint, export at its training size, and verify one deployment inference before publishing."""
    from ultralytics import YOLO

    weights, output = Path(weights).resolve(strict=True), Path(output).resolve()
    if weights.suffix.lower() != ".pt" or output.suffix.lower() != ".onnx":
        raise ValueError("Expected a .pt checkpoint and an .onnx destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        # Ultralytics writes next to its input checkpoint. Stage it to keep training runs untouched.
        checkpoint = shutil.copyfile(weights, Path(temporary) / weights.name)
        model = YOLO(checkpoint)
        if model.task not in ("classify", "segment"):
            raise ValueError(f"Unsupported task {model.task!r}")
        kwargs = {} if imgsz is None else {"imgsz": imgsz}
        path = Path(model.export(format="onnx", dynamic=True, simplify=False, nms=None, device="cpu", **kwargs))
        if model.task == "classify":
            runtime = Classifier(path)
            runtime.predict([np.zeros((*runtime.size, 3), np.uint8)])
        else:
            runtime = Segmenter(path)
            list(runtime.predict(np.zeros((*runtime.size, 3), np.uint8)))
        path.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, help="Square export size (defaults to the checkpoint training size)")
    args = parser.parse_args()
    print(export(args.weights, args.output, args.imgsz))
