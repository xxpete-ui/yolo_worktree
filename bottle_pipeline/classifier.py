"""Ultralytics ONNX classification for the BGR bottle canvases produced by the pipeline."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Classifier:
    """Read the class order, canvas size and batch constraints from a classification model."""

    def __init__(self, model, labels=None, size=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one RGB image input and one classification probability output")
        self.input, self.output = inputs[0], outputs[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3 or self.input.type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError(f"Expected a floating-point NCHW RGB input, got {self.input.type} {shape}")
        if len(self.output.shape) != 2:
            raise ValueError(f"Expected class scores [batch, classes], got {self.output.shape}")
        self.dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32
        self.batch = shape[0] if isinstance(shape[0], int) else None
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("task", "classify") != "classify":
            raise ValueError("Expected a classification model")
        if size is None and "imgsz" in metadata:
            size = literal_eval(metadata["imgsz"])
        if isinstance(size, int):
            size = (size, size)
        if size is None:
            size = shape[2:]
        if len(size) != 2 or not all(isinstance(n, int) and n > 0 for n in size):
            raise ValueError("Specify classifier height and width for an ONNX with dynamic size and no imgsz metadata")
        if any(isinstance(axis, int) and axis != n for axis, n in zip(shape[2:], size)):
            raise ValueError(f"Classifier size {size} disagrees with ONNX input {shape}")
        self.size = tuple(size)

        names = literal_eval(metadata["names"]) if "names" in metadata else None
        if isinstance(names, dict):
            by_id = {int(index): value for index, value in names.items()}
            if set(by_id) != set(range(len(names))):
                raise ValueError("Class IDs must be contiguous starting at zero")
            names = [by_id[index] for index in range(len(names))]
        if labels is not None:
            supplied = Path(labels).read_text(encoding="utf-8-sig").splitlines()
            if names is not None and supplied != names:
                raise ValueError("Labels file disagrees with the ONNX class order")
            names = supplied
        if not isinstance(names, list) or not names or not all(isinstance(s, str) and s.strip() for s in names):
            raise ValueError("Supply a UTF-8 labels file, one class per line in model output order")
        self.names = names
        classes = self.output.shape[1]
        if isinstance(classes, int) and classes != len(names):
            raise ValueError(f"Found {len(names)} class names for {classes} ONNX output scores")

    def scores(self, crops):
        """Return all model probabilities. Each crop is an already-sized uint8 BGR canvas."""
        if not len(crops):
            return np.empty((0, len(self.names)), dtype=self.dtype)
        if any(crop.dtype != np.uint8 or crop.shape != (*self.size, 3) for crop in crops):
            raise ValueError(f"Expected uint8 BGR classifier canvases of size {self.size} (height, width)")
        chunks = []
        batch_size = self.batch or len(crops)
        for offset in range(0, len(crops), batch_size):
            batch = list(crops[offset : offset + batch_size])
            count = len(batch)
            if count < batch_size:
                batch.extend([batch[-1]] * (batch_size - count))
            rgb = np.stack(batch)[..., ::-1].transpose(0, 3, 1, 2)
            tensor = np.ascontiguousarray(rgb, dtype=self.dtype) / self.dtype(255)
            values = self.session.run([self.output.name], {self.input.name: tensor})[0]
            if values.shape != (batch_size, len(self.names)):
                raise ValueError(
                    f"Expected [{batch_size}, {len(self.names)}] classification scores, got {values.shape}"
                )
            normalized = np.allclose(values.sum(1, dtype=np.float32), 1, atol=1e-3)
            if not normalized or not np.all((values >= 0) & (values <= 1)):
                raise ValueError("Expected single-label class probabilities in [0, 1] summing to one per crop")
            chunks.append(values[:count])
        return np.concatenate(chunks)

    def predict(self, crops):
        """Return (class ID, model class name, probability) in input order."""
        predictions = []
        for row in self.scores(crops):
            index = int(row.argmax())
            predictions.append((index, self.names[index], float(row[index])))
        return predictions
