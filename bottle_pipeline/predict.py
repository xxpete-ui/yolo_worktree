"""Batch bottle segmentation and classification, with per-image evidence for manual inspection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .classifier import Classifier
from .segmentation import Segmenter, classification_crop

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def save_image(path, image):
    """OpenCV's memory encoder supports Unicode filenames on Windows."""
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    data.tofile(path)


def inspect_image(image, directory, segmenter, classifier):
    """Classify each usable instance, retaining empty contours and the exact classifier input for review."""
    overlay = image.copy()
    objects, crops, indices = [], [], []
    for index, (box, score, mask) in enumerate(segmenter.predict(image), 1):
        mask_name = f"mask_{index}.png"
        save_image(directory / mask_name, mask)
        obj = {"id": index, "xyxy": box, "seg_score": score, "mask": mask_name}
        objects.append(obj)
        crop = classification_crop(image, mask, box, classifier.size)
        if crop is None:
            obj["status"] = "empty_contour"
            continue
        obj.update(status="classified", crop=f"cls_input_{index}.png")
        save_image(directory / obj["crop"], crop)
        indices.append(index - 1)
        crops.append(crop)
        x, y, w, h = cv2.boundingRect(mask)
        roi = overlay[y : y + h, x : x + w]
        tinted = cv2.addWeighted(roi, 0.65, np.full_like(roi, (0, 255, 0)), 0.35, 0)
        cv2.copyTo(tinted, mask[y : y + h, x : x + w], roi)
    for index, (class_id, name, score) in zip(indices, classifier.predict(crops)):
        objects[index].update(class_id=class_id, label=name, cls_score=score)
    for obj in objects:
        x1, y1, x2, y2 = obj["xyxy"]
        text = f"#{obj['id']}"
        color = (0, 180, 255)
        if "class_id" in obj:
            # OpenCV Hershey fonts are ASCII only. The index and JSON retain the original Unicode name.
            name = obj["label"] if obj["label"].isascii() else f"class {obj['class_id']}"
            text += f" {name} {obj['cls_score']:.3f}"
            color = (0, 255, 0)
        else:
            text += " empty contour"
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        pos = (max(0, min(x1, image.shape[1] - cv2.getTextSize(text, 0, 0.55, 1)[0][0])), max(18, y1 - 5))
        cv2.putText(overlay, text, pos, 0, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, pos, 0, 0.55, color, 1, cv2.LINE_AA)
    save_image(directory / "overlay.jpg", overlay)
    return objects


def write_reports(output, records, settings):
    """Store the original image, numerical results and corresponding lossless crops together by filename."""
    counts = Counter(obj["status"] for rec in records for obj in rec.get("objects", []))
    summary = {
        "images": len(records),
        "unreadable": sum("error" in r for r in records),
        "without_detections": sum(r.get("objects") == [] for r in records),
        "instances": sum(counts.values()),
        "classified": counts["classified"],
        "empty_contours": counts["empty_contour"],
    }
    (output / "report.json").write_text(
        json.dumps({"summary": summary, "settings": settings, "images": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = ["image", "id", "xyxy", "seg_score", "status", "class_id", "label", "cls_score", "mask", "crop"]
    with (output / "instances.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            for obj in rec.get("objects", []):
                row = dict(image=rec["image"], **obj)
                for key in ("mask", "crop"):
                    if key in row:
                        row[key] = f"{rec['directory']}/{row[key]}"
                writer.writerow(row)
    rows = []
    for rec in records:
        image_name = escape(rec["image"])
        if "error" in rec:
            rows.append(f"<tr><td>{image_name}</td><td>{escape(rec['error'])}</td></tr>")
            continue
        directory = quote(rec["directory"])
        overlay = f"{directory}/overlay.jpg"
        links = []
        for obj in rec["objects"]:
            result = escape(obj["label"]) + f" {obj['cls_score']:.4f}" if "label" in obj else "无有效轮廓"
            crop = f'<a href="{directory}/{obj["crop"]}">分类输入</a>' if "crop" in obj else ""
            links.append(f'<li>#{obj["id"]} {result} · <a href="{directory}/{obj["mask"]}">掩码</a> {crop}</li>')
        rows.append(
            f'<tr><td><a href="{overlay}"><img loading="lazy" width="320" src="{overlay}" alt=""></a><br>'
            f'<a href="{escape(Path(rec["source"]).as_uri(), quote=True)}">{image_name}</a></td>'
            f"<td><ul>{''.join(links) or '<li>未检出瓶子</li>'}</ul></td></tr>"
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>瓶子人工复核</title>'
        "<style>body{font:16px system-ui;margin:2rem}td{padding:.7rem;border-bottom:1px solid #bbb}"
        "a{color:#17568c}ul{padding-left:1.4em}</style><h1>瓶子人工复核</h1>"
        f"<p>{summary['images']} 张大图 · {summary['instances']} 个实例 · "
        f"{summary['classified']} 个分类结果 · {summary['without_detections']} 张无检出 · "
        f"{summary['empty_contours']} 个无有效轮廓 · {summary['unreadable']} 张读取失败</p>"
        '<p><a href="instances.csv">逐瓶 CSV</a> · <a href="report.json">模型设置和逐图明细</a></p>'
        "<table><thead><tr><th>大图与分割结果</th><th>实例与分类置信度</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("D:/Data/seg_testdata"))
    parser.add_argument("--segmenter", type=Path, default=Path("D:/Project/yolo26/modelZoo/hei_huizong_250428.onnx"))
    parser.add_argument("--classifier", type=Path, default=ROOT / "artifacts/models/classifier.onnx")
    parser.add_argument("--labels", type=Path, help="UTF-8 names in output order for a classifier without metadata")
    parser.add_argument("--seg-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--cls-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="New run directory")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    images = (
        [source]
        if source.is_file()
        else sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
    )
    if not images:
        parser.error(f"No images found in {source}")
    output = (args.output or ROOT / "artifacts/predictions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if source.is_dir() and (source == output or source in output.parents):
        parser.error("Choose an output directory outside the source image tree")
    segmenter = Segmenter(args.segmenter, args.seg_size, args.confidence, args.iou, args.mask_threshold)
    classifier = Classifier(args.classifier, args.labels, args.cls_size)
    settings = {
        "source": str(source),
        "seg_hw": segmenter.size,
        "cls_hw": classifier.size,
        "classes": classifier.names,
        "confidence": args.confidence,
        "iou": args.iou,
        "mask_threshold": args.mask_threshold,
    }
    for key, model in (("segmenter", args.segmenter), ("classifier", args.classifier)):
        settings[key] = {"file": str(model.resolve()), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, path in enumerate(images, 1):
        name = path.name if source.is_file() else path.relative_to(source).as_posix()
        record = {"image": name, "source": str(path), "directory": f"images/{name}"}
        records.append(record)
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (OSError, cv2.error) as error:
            image = None
            record["error"] = str(error)
        if image is None:
            record.setdefault("error", "Image could not be decoded")
            continue
        directory = output / record["directory"]
        directory.mkdir(parents=True)
        record["objects"] = inspect_image(image, directory, segmenter, classifier)
        if i % 25 == 0:
            print(f"{i}/{len(images)}", flush=True)
    summary = write_reports(output, records, settings)
    print(output, summary, flush=True)
    if summary["unreadable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
