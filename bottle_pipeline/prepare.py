"""Prepare traceable YOLO bottle segmentation and folder-labelled classification datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SEG_SOURCE = Path(r"D:\Data\seg_data\hei_huizong_0812")
CLS_SOURCE = Path(r"D:\Data\clsData\huizon-xlj-0604")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff", ".webp"}
FRAME = re.compile(r"(?<!\d)(20\d{16})(?!\d)")


@dataclass
class Sample:
    path: Path
    category: str
    group: str
    digest: str = ""
    labels: tuple = ()
    reason: str = ""
    detail: str = ""


def label_lines(annotation, size):
    """Convert one complete LabelMe frame. A broken bottle annotation invalidates the whole frame."""
    if (annotation["imageWidth"], annotation["imageHeight"]) != size:
        raise ValueError("annotation dimensions disagree with the image")
    shapes = annotation["shapes"]
    if not isinstance(shapes, list):
        raise TypeError("expected a list of shapes")
    lines, corrections = [], []
    for index, shape in enumerate(shapes):
        if shape.get("label") != "bottle" or shape.get("shape_type") != "polygon" or shape.get("group_id") is not None:
            raise ValueError(f"shape {index}: expected an individual bottle polygon")
        points = np.asarray(shape["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"shape {index}: invalid polygon coordinates")
        clipped = np.clip(points, 0, size)
        excursion = float(np.max(np.abs(points - clipped)))
        if excursion > 0.5:
            raise ValueError(f"shape {index}: polygon extends {excursion:.4f} pixels outside image")
        if excursion:
            corrections.append(f"shape {index} projected to boundary (max {excursion:.4f} px)")
            points = clipped
        x, y = points.T
        if abs(x @ np.roll(y, 1) - y @ np.roll(x, 1)) < 1e-8:
            raise ValueError(f"shape {index}: zero-area polygon")
        lines.append("0 " + " ".join(f"{c:.8f}" for c in (points / size).flat))
    if len(set(lines)) < len(lines):
        corrections.append(f"removed {len(lines) - len(set(lines))} duplicate polygon rows")
    return tuple(sorted(set(lines))), " | ".join(corrections)


def inspect_image(path, annotation=None):
    is_seg = annotation is not None
    frame = FRAME.search(path.stem)
    sample = Sample(path, "bottle" if is_seg else path.parent.name, path.stem if is_seg else frame[1] if frame else "")
    try:
        with Image.open(path) as im:
            im.load()
            if min(im.size) < 10:
                raise ValueError("image smaller than the trainer minimum of 10 pixels")
            if is_seg and im.getexif().get(274, 1) != 1:
                raise ValueError("EXIF orientation needs matching polygon coordinates")
            if im.format == "JPEG":
                with path.open("rb") as source:
                    source.seek(-2, 2)
                    if source.read() != b"\xff\xd9":
                        raise ValueError("JPEG end marker missing")
            size = im.size
            digest = hashlib.sha256(str(size).encode())
            digest.update(im.convert("RGB").tobytes())
            sample.digest = digest.hexdigest()
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        sample.reason, sample.detail = "invalid_image", str(exc)
        return sample
    if is_seg:
        try:
            data = json.loads(annotation.read_bytes())
            if Path(data["imagePath"].replace("\\", "/")).name.casefold() != path.name.casefold():
                raise ValueError(f"imagePath disagrees: {data['imagePath']}")
            sample.labels, sample.detail = label_lines(data, size)
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            sample.reason, sample.detail = "invalid_annotation", str(exc)
    return sample


def split_for(group, seed):
    # Unknown source frames stay in training. All crops from a known frame get the same split across categories.
    if not group:
        return "train"
    bucket = int.from_bytes(hashlib.sha256(f"{seed}/{group}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def write_dataset(samples, destination, task, classes, seed):
    by_pixels = defaultdict(list)
    for sample in samples:
        if sample.digest:
            by_pixels[sample.digest].append(sample)
    for copies in by_pixels.values():
        valid = [s for s in copies if not s.reason]
        if not valid:
            continue
        if len({(s.category, s.labels) for s in valid}) > 1:
            for s in valid:
                s.reason, s.detail = "conflicting_labels", valid[0].digest
        else:
            # Keep a traceable crop when an identical untraceable copy is available.
            valid.sort(key=lambda s: not bool(s.group))
            for s in valid[1:]:
                s.reason, s.detail = "duplicate_pixels", str(valid[0].path)

    dest = destination / task
    dest.mkdir()
    split_counts, rejected, instances = Counter(), Counter(), Counter()
    no_frame = corrected_frames = 0
    with (dest / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("source", "category", "source_frame", "split", "output", "reason", "detail"))
        for s in samples:
            split, output = "", ""
            if s.reason:
                rejected[s.reason] += 1
            else:
                split = split_for(s.group, seed)
                if task == "segment":
                    image = dest / "images" / split / s.path.name
                    label = dest / "labels" / split / (s.path.stem + ".txt")
                    label.parent.mkdir(exist_ok=True, parents=True)
                    label.write_text("".join(line + "\n" for line in s.labels), encoding="utf-8")
                    instances[split] += len(s.labels)
                    corrected_frames += int(bool(s.detail))
                else:
                    image = dest / split / s.category / s.path.name
                    no_frame += int(not s.group)
                image.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(s.path, image)
                output = image.relative_to(dest).as_posix()
                split_counts[(s.category, split)] += 1
            writer.writerow((s.path, s.category, s.group, split, output, s.reason, s.detail))
    counts = {c: {split: split_counts[c, split] for split in ("train", "val", "test")} for c in classes}
    if not counts or any(n == 0 for row in counts.values() for n in row.values()):
        raise ValueError(f"One or more classes have an empty split. Check source diversity and seed. Counts: {counts}")
    if task == "segment" and any(instances[k] == 0 for k in ("train", "val", "test")):
        raise ValueError(f"Segmentation requires labelled bottles in every split. Found {instances}")
    return {
        "input_records": len(samples),
        "kept": sum(split_counts.values()),
        "excluded": dict(rejected),
        "images_per_class": counts,
        **(
            {"bottles": dict(instances), "kept_with_annotation_corrections": corrected_frames}
            if task == "segment"
            else {
                "input_without_frame_id": sum(not s.group for s in samples),
                "kept_without_frame_id_train_only": no_frame,
            }
        ),
    }


def prepare_segment(seg, stage, output, seed):
    seg_images = sorted(
        p for p in (seg / "JPEGImages").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    annotations = {p.stem: p for p in (seg / "Annotations").glob("*.json")}
    stems = Counter(p.stem for p in seg_images)

    def inspect_frame(path):
        if stems[path.stem] > 1 or path.stem not in annotations:
            return Sample(path, "bottle", path.stem, reason="ambiguous_or_missing_pair")
        return inspect_image(path, annotations[path.stem])

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(inspect_frame, seg_images))
    for stem in sorted(annotations.keys() - stems.keys()):
        frames.append(Sample(annotations[stem], "bottle", stem, reason="missing_image"))
    summary = write_dataset(frames, stage, "segment", ["bottle"], seed)
    spec = {"path": str(output / "segment"), "names": {0: "bottle"}}
    spec.update({split: f"images/{split}" for split in ("train", "val", "test")})
    (stage / "segment" / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return summary


def prepare(seg, cls, output, seed):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output directory: {output}")
    if any(source.resolve() == output or source.resolve() in output.parents for source in (seg, cls)):
        raise ValueError("Output must be outside both original datasets")
    classes = sorted(d.name for d in cls.iterdir() if d.is_dir())
    crops = [
        p for c in classes for p in sorted((cls / c).iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(prefix=".bottle-prepare-", dir=output.parent) as temp:
        stage = Path(temp) / "datasets"
        stage.mkdir()
        summary = {"seed": seed, "segment": prepare_segment(seg, stage, output, seed)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            summary["classify"] = write_dataset(list(pool.map(inspect_image, crops)), stage, "classify", classes, seed)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        stage.rename(output)
    print(output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg", type=Path, default=SEG_SOURCE)
    parser.add_argument("--cls", type=Path, default=CLS_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.seg, args.cls, args.output, args.seed)
