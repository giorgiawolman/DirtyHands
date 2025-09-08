#!/usr/bin/env python3
"""
run_eval_from_config.py — one-command evaluation driven by a YAML config.

Usage:
  python run_eval_from_config.py --cfg eval_config.yaml

Notes
- If yolo_seg_eval.py is present with write_data_yaml/run_eval/sweep_thresholds,
  this script will use them. Otherwise it falls back to local implementations.
- 'train'/'val'/'test' in your config should point to the *images* folders.
- Use forward slashes (C:/...) or single-quoted/escaped backslashes in YAML.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

# Try to import helper module (optional)
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
try:
    import yolo_seg_eval as yse  # may or may not have needed functions
except Exception:
    yse = None  # we'll fall back


# ---------------------------
# Fallbacks (used if missing)
# ---------------------------
def _fallback_write_data_yaml(out_path: Path, train: str, val: str, test: str | None, names_csv: str):
    """Local minimal writer for data.yaml."""
    import yaml as _yaml
    names = [n.strip() for n in str(names_csv).split(",") if n.strip()]
    y = {
        "train": str(Path(train).resolve()),
        "val":   str(Path(val).resolve()),
        "names": names,
        "nc": len(names),
    }
    if test:
        y["test"] = str(Path(test).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(y, f, sort_keys=False)
    return out_path


def _fallback_run_eval(weights: Path, data_yaml: Path, out_dir: Path, imgsz: int,
                       conf: float, nms_iou: float, device: str, split: str, tta: bool):
    """Local minimal evaluator using Ultralytics directly."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(
            "Ultralytics is required. Install: pip install 'ultralytics>=8.2.0'\n"
            f"Import error: {e}"
        )
    model = YOLO(str(weights))
    run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        iou=nms_iou,
        device=device,
        split=split,
        half=False,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
        augment=tta,
    )
    save_dir = Path(getattr(results, "save_dir", out_dir / run_name))
    # summarize key scalars
    summary = {}
    for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)",
              "metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/precision(M)", "metrics/recall(M)"]:
        try:
            summary[k] = float(results.results_dict.get(k, np.nan))
        except Exception:
            pass
    out_json = {
        "weights": str(weights),
        "data_yaml": str(Path(data_yaml).resolve()),
        "save_dir": str(save_dir.resolve()),
        "imgsz": imgsz,
        "conf": conf,
        "nms_iou": nms_iou,
        "split": split,
        "tta": tta,
        "summary": summary,
    }
    with open(save_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    return save_dir


def _fallback_sweep_thresholds(weights: Path, data_yaml: Path, out_dir: Path, imgsz: int,
                               device: str, split: str,
                               conf_list: list[float], iou_list: list[float]):
    """Local conf/NMS sweep."""
    from ultralytics import YOLO
    rows = []
    model = YOLO(str(weights))
    for conf in conf_list:
        for iou in iou_list:
            r = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou,
                          device=device, split=split, half=False, plots=False, save_json=False, verbose=False)
            d = r.results_dict
            rows.append({
                "conf": conf,
                "nms_iou": iou,
                "mAP50_mask": float(d.get("metrics/mAP50(M)", float("nan"))),
                "mAP50_95_mask": float(d.get("metrics/mAP50-95(M)", float("nan"))),
                "precision_mask": float(d.get("metrics/precision(M)", float("nan"))),
                "recall_mask": float(d.get("metrics/recall(M)", float("nan"))),
            })
    df = pd.DataFrame(rows).sort_values(["mAP50_mask", "mAP50_95_mask"], ascending=False)
    out_csv = Path(out_dir) / "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


# Resolve which implementations to use
def write_data_yaml(*a, **kw):
    if yse and hasattr(yse, "write_data_yaml"):
        return yse.write_data_yaml(*a, **kw)
    return _fallback_write_data_yaml(*a, **kw)

def run_eval(*a, **kw):
    if yse and hasattr(yse, "run_eval"):
        return yse.run_eval(*a, **kw)
    return _fallback_run_eval(*a, **kw)

def sweep_thresholds(*a, **kw):
    if yse and hasattr(yse, "sweep_thresholds"):
        return yse.sweep_thresholds(*a, **kw)
    return _fallback_sweep_thresholds(*a, **kw)


# ---------------------------
# Utility
# ---------------------------
def resolve_cfg_path(cfg_arg: str) -> Path:
    """Absolute → as-is; relative → try CWD then script folder."""
    p = Path(cfg_arg)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = HERE / cfg_arg
    if candidate.exists():
        return candidate.resolve()
    return p  # will fail later


def ensure_images_path(p: Path) -> Path:
    """
    Ensure dataset path points to an 'images' folder.
    If user passed split root (.../train), use ./images if it exists.
    """
    if p is None:
        return p
    if p.name.lower() == "images":
        return p
    images_child = p / "images"
    if images_child.exists():
        return images_child
    return p


# ---------------------------
# Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cfg", type=str, default="eval_config.yaml", help="Path to config YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_cfg_path(args.cfg)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Required paths
    try:
        weights = Path(cfg["weights"])
    except KeyError:
        raise SystemExit("Missing 'weights' in config YAML.")

    out_dir = Path(cfg.get("out_dir", "./eval_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build or use data.yaml
    data_yaml_value = cfg.get("data", None)
    if data_yaml_value:
        data_yaml = Path(data_yaml_value).resolve()
        if not data_yaml.exists():
            raise SystemExit(f"--data file not found: {data_yaml}")
        print(f"[cfg] Using existing data.yaml: {data_yaml}")
    else:
        ds = cfg.get("dataset", {})
        train = ds.get("train"); val = ds.get("val"); test = ds.get("test", None); names = ds.get("names")
        if not (train and val and names):
            raise SystemExit("When 'data' is not provided, you must define dataset.train, dataset.val, and dataset.names.")
        train_p = ensure_images_path(Path(train))
        val_p   = ensure_images_path(Path(val))
        test_p  = ensure_images_path(Path(test)) if test else None
        for tag, path in (("train", train_p), ("val", val_p)):
            if not Path(path).exists():
                raise SystemExit(f"'{tag}' path does not exist: {path}")
        if test_p and not Path(test_p).exists():
            print(f"[warn] 'test' path does not exist: {test_p} (continuing without test)")
            test_p = None
        names_csv = ",".join(names) if isinstance(names, (list, tuple)) else str(names)
        data_yaml = out_dir / "data_autogen.yaml"
        write_data_yaml(data_yaml, str(train_p), str(val_p), str(test_p) if test_p else None, names_csv)
        print(f"[cfg] Wrote {data_yaml}")

    # Eval settings
    imgsz   = int(cfg.get("imgsz", 1024))
    conf    = float(cfg.get("conf", 0.001))
    nms_iou = float(cfg.get("nms_iou", 0.5))
    device  = str(cfg.get("device", ""))      # "0" for GPU 0, "" for auto/CPU
    split   = str(cfg.get("split", "val"))     # val or test
    tta     = bool(cfg.get("tta", False))

    print("\n=== Evaluation Configuration ===")
    print(f"weights : {weights}")
    print(f"data    : {data_yaml}")
    print(f"out_dir : {out_dir}")
    print(f"imgsz   : {imgsz}")
    print(f"conf    : {conf}")
    print(f"nms_iou : {nms_iou}")
    print(f"device  : {device!r}")
    print(f"split   : {split}")
    print(f"tta     : {tta}")
    print("================================\n")

    # Run eval
    save_dir = run_eval(weights, data_yaml, out_dir, imgsz, conf, nms_iou, device, split, tta)
    print(f"Eval artifacts → {save_dir}")

    # Optional sweep
    if bool(cfg.get("sweep", False)):
        conf_list = list(map(float, cfg.get("conf_list", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])))
        iou_list  = list(map(float, cfg.get("iou_list",  [0.4, 0.5, 0.6, 0.7])))
        print(f"[SWEEP] conf_list={conf_list} | iou_list={iou_list}")
        out_csv = sweep_thresholds(weights, data_yaml, save_dir, imgsz, device, split, conf_list, iou_list)
        print(f"Sweep CSV → {out_csv}")


if __name__ == "__main__":
    main()
