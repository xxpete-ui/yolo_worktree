"""Single-class raw YOLO instance segmentation and the deployment camera's rotated-rectangle crop."""

from __future__ import annotations

import ast
import math

import cv2
import numpy as np
import onnxruntime as ort


class Segmenter:
    """Load a raw bottle ONNX once and yield (xyxy, score, original-size binary mask) per instance."""

    def __init__(self, model, size=None, confidence=0.25, iou=0.45, mask_threshold=0.5):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError("Expected one RGB input and two raw segmentation outputs")
        self.input = inputs[0]
        shape = self.input.shape
        if self.input.type != "tensor(float)" or len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected float32 NCHW RGB segmentation input, got {shape} {self.input.type}")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "names" in metadata and ast.literal_eval(metadata["names"]) not in ({0: "bottle"}, ["bottle"]):
            raise ValueError("The segmenter must have one class named 'bottle'")
        if size is None:
            size = shape[2:] if all(isinstance(x, int) for x in shape[2:]) else ast.literal_eval(metadata["imgsz"])
        self.size = tuple(size)
        if len(self.size) != 2 or any(x < 1 or x % 32 for x in self.size):
            raise ValueError("Segmentation height and width must be positive multiples of 32")
        if any(isinstance(a, int) and a != b for a, b in zip(shape, (1, 3, *self.size))):
            raise ValueError(f"Input size {self.size} is incompatible with ONNX input {shape}")
        if sorted(len(o.shape) for o in outputs) != [3, 4]:
            raise ValueError("Expected rank-three predictions and rank-four mask prototypes")
        self.output_names = [o.name for o in sorted(outputs, key=lambda o: len(o.shape))]
        if not all(0 <= x <= 1 for x in (confidence, iou, mask_threshold)):
            raise ValueError("Thresholds must be in [0, 1]")
        self.confidence, self.iou, self.mask_threshold = confidence, iou, mask_threshold

    def predict(self, image):
        """Restore masks through the actual integer letterbox geometry, as in Segmentation.cpp."""
        h, w = image.shape[:2]
        ih, iw = self.size
        scale = min(iw / w, ih / h)
        rw, rh = max(1, math.floor(w * scale + 0.5)), max(1, math.floor(h * scale + 0.5))
        left, top = (iw - rw) // 2, (ih - rh) // 2
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[top : top + rh, left : left + rw] = cv2.resize(image, (rw, rh))
        tensor = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, swapRB=True)
        raw, protos = self.session.run(self.output_names, {self.input.name: tensor})
        proto = protos[0]
        channels, ph, pw = proto.shape
        if raw.shape[:2] != (1, 5 + channels) or protos.shape[0] != 1:
            raise ValueError(f"Expected raw single-class xywh + score + coefficients, got {raw.shape}, {protos.shape}")
        rows = raw[0].T
        rows = rows[rows[:, 4] > self.confidence]
        if not np.isfinite(rows).all() or np.any(rows[:, 4] > 1):
            raise ValueError("Invalid segmentation outputs")
        # Float64 reproduces the C++ Rect2d arithmetic, including coordinates near integer boundaries.
        xywh = rows[:, :4].astype(np.float64)
        xywh[:, :2] -= xywh[:, 2:] / 2
        xyxy = xywh.copy()
        xyxy[:, 2:] += xyxy[:, :2]
        coords = (xyxy - [left, top, left, top]) / [rw / w, rh / h, rw / w, rh / h]
        coords = np.clip(coords, 0, [w, h, w, h])
        valid = (coords[:, 2:] > coords[:, :2]).all(axis=1)
        rows, xywh, xyxy, coords = rows[valid], xywh[valid], xyxy[valid], coords[valid]
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), rows[:, 4].tolist(), self.confidence, self.iou)
        for i in np.asarray(kept, dtype=int).ravel():
            x1, y1, x2, y2 = np.ceil(xyxy[i] * [pw / iw, ph / ih, pw / iw, ph / ih]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, pw)
            y1, y2 = np.clip([y1, y2], 0, ph)
            probability = np.zeros((ph, pw), np.float32)
            # Sampling uses a half-open crop at prototype resolution, before both linear resizes.
            logits = np.einsum("c,cyx->yx", rows[i, 5:], proto[:, y1:y2, x1:x2])
            probability[y1:y2, x1:x2] = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))
            network = cv2.resize(probability, (iw, ih))
            restored = cv2.resize(network[top : top + rh, left : left + rw], (w, h))
            box = np.concatenate((np.floor(coords[i, :2]), np.ceil(coords[i, 2:]))).astype(int)
            yield tuple(map(int, box)), float(rows[i, 4]), cv2.compare(restored, self.mask_threshold, cv2.CMP_GT)


def classification_crop(image, mask, box, size):
    """Mask the largest contour's rotated rectangle, preserve its angle, and center it on the classifier canvas."""
    x1, y1, x2, y2 = box
    contours, _ = cv2.findContours(mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    (cx, cy), sides, angle = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(((cx + x1, cy + y1), sides, angle))
    h, w = image.shape[:2]
    x, y = np.maximum(0, np.floor(corners.min(axis=0)).astype(int))
    right, bottom = np.minimum([w, h], np.ceil(corners.max(axis=0)).astype(int) + 1)
    poly_mask = np.zeros((bottom - y, right - x), np.uint8)
    cv2.fillConvexPoly(poly_mask, corners.astype(np.int32) - np.int32([x, y]), 255)
    roi = cv2.bitwise_and(image[y:bottom, x:right], image[y:bottom, x:right], mask=poly_mask)
    height, width = size
    rh, rw = roi.shape[:2]
    scale = min(width / rw, height / rh)
    if scale < 1:
        rw, rh = max(1, round(rw * scale)), max(1, round(rh * scale))
        roi = cv2.resize(roi, (rw, rh))
    x, y = width // 2 - rw // 2, height // 2 - rh // 2
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[y : y + rh, x : x + rw] = roi
    return canvas
