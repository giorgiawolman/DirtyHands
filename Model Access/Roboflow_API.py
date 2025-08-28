# gh_export_instances_workflow.py
# Calls a Roboflow WORKFLOW via Serverless Hosted API (V2),
# then exports polygons to DXF + a GH-friendly CSV.
# Modes: "live" (workflow), "mock" (no network), "fixture" (load JSON).

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Literal, Optional, Any
import random

import numpy as np
import cv2

# Optional: DXF export (install once if you want DXF: pip install ezdxf)
try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

# -------------------------------
# CONFIG — choose test mode
# -------------------------------
TEST_MODE: Literal["live", "mock", "fixture"] = "live"
# "live"    -> call Roboflow workflow normally
# "mock"    -> generate fake predictions (no network needed)
# "fixture" -> load predictions from a saved JSON file below

FIXTURE_JSON = r"C:\path\to\saved_prediction.json"  # used if TEST_MODE=="fixture"

# ---- Roboflow WORKFLOW access (values you provided) ----
API_KEY = "Sp3ArhygLcCOlxn6UV4P"
WORKSPACE_NAME = "forest-cover-changes"
WORKFLOW_ID = "workflow"
USE_CACHE = True  # cache workflow definition 15 minutes on Roboflow

# Input image (single file or folder)
INPUT_PATH = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\ForestDetectionImages\HighRes TrueColor 2018 2m Graz - Kopie.png"
OUTPUT_DIR = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\GrasshopperOutput"

# If you later want physical units, you can pass pixel size here (unused for export-only).
PIXEL_SIZE_MM = None
# -------------------------------

# --- Roboflow client (LIVE mode uses V2 endpoint) ---
from inference_sdk import InferenceHTTPClient

def make_live_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

# ---------- MOCK / FIXTURE HELPERS ----------
def make_mock_predictions(w: int, h: int) -> Dict:
    """Return a Roboflow-like predictions dict with 2 polygons."""
    random.seed(42)
    def poly(cx, cy, r, n=16):
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            rx = r * (0.7 + 0.6 * random.random())
            ry = r * (0.7 + 0.6 * random.random())
            x = max(0, min(w - 1, cx + rx * math.cos(ang)))
            y = max(0, min(h - 1, cy + ry * math.sin(ang)))
            pts.append({"x": float(x), "y": float(y)})
        return pts

    return {
        "predictions": [
            {
                "class": "object_a",
                "confidence": 0.93,
                "x": w * 0.30, "y": h * 0.40, "width": 120, "height": 100,
                "points": poly(w * 0.30, h * 0.40, min(w, h) * 0.12)
            },
            {
                "class": "object_b",
                "confidence": 0.88,
                "x": w * 0.65, "y": h * 0.55, "width": 140, "height": 130,
                "points": poly(w * 0.65, h * 0.55, min(w, h) * 0.15)
            }
        ],
        "image": {"width": int(w), "height": int(h)}
    }

def load_fixture_predictions(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- inference dispatcher + prediction extraction ---
def call_inference(image_path: str) -> Dict:
    """Returns a dict with at least a 'predictions' list + optional 'image' size."""
    if TEST_MODE == "mock":
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return make_mock_predictions(640, 480)
        h, w = img_bgr.shape[:2]
        return make_mock_predictions(w, h)

    if TEST_MODE == "fixture":
        return load_fixture_predictions(FIXTURE_JSON)

    # live mode (workflow)
    client = make_live_client()
    res = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        use_cache=USE_CACHE
    )

    # Try to normalize various possible workflow outputs into {"predictions":[...], "image":{...}}
    normalized = normalize_workflow_result(res)
    return normalized

def normalize_workflow_result(res: Any) -> Dict:
    """
    Tries to find a list of instance-segmentation predictions with 'points' in a workflow result.
    Returns {"predictions": [...], "image": {...}}.
    """
    # Fast path: already in expected format
    if isinstance(res, dict) and "predictions" in res and isinstance(res["predictions"], list):
        return res

    # Common workflow shapes: nestings under 'results', 'steps', etc.
    # We'll scan the dict recursively for a list of dicts that look like predictions with 'points'.
    preds = find_predictions_with_points(res)
    out = {"predictions": preds}
    # Try to attach image size if present anywhere
    w, h = find_image_size(res)
    if w and h:
        out["image"] = {"width": int(w), "height": int(h)}
    return out

def find_predictions_with_points(obj: Any) -> List[Dict]:
    found = []
    def rec(o):
        nonlocal found
        if isinstance(o, dict):
            # If this dict looks like ONE prediction with points
            if "points" in o and isinstance(o["points"], list):
                # Wrap as list if we accidentally hit a single prediction
                found.append(o)
            # If we see a list under 'predictions', extend
            if "predictions" in o and isinstance(o["predictions"], list):
                if o["predictions"] and isinstance(o["predictions"][0], dict) and "points" in o["predictions"][0]:
                    found.extend(o["predictions"])
            for v in o.values():
                rec(v)
        elif isinstance(o, list):
            for it in o:
                rec(it)
    rec(obj)

    # If we accidentally appended single predictions individually, make sure structure is a flat list of dicts
    # and filter to unique objects by id if present
    if not found:
        return []
    # Keep only dicts that have 'points' (and are polygons)
    found = [p for p in found if isinstance(p, dict) and isinstance(p.get("points"), list)]
    return found

def find_image_size(obj: Any) -> (Optional[int], Optional[int]):
    w = h = None
    def rec(o):
        nonlocal w, h
        if w and h:
            return
        if isinstance(o, dict):
            if "image" in o and isinstance(o["image"], dict):
                iw, ih = o["image"].get("width"), o["image"].get("height")
                if iw and ih:
                    w, h = iw, ih
                    return
            # sometimes under 'metadata' or similar
            for k, v in o.items():
                rec(v)
        elif isinstance(o, list):
            for it in o:
                rec(it)
    rec(obj)
    return w, h

# -------------------------------
# EXPORTS FOR GRASSHOPPER
# -------------------------------
def ensure_dirs(base_out: str):
    Path(base_out).mkdir(parents=True, exist_ok=True)
    Path(base_out, "dxf").mkdir(parents=True, exist_ok=True)
    Path(base_out, "json").mkdir(parents=True, exist_ok=True)

def export_dxf_for_gh(preds: List[Dict], dxf_path: Path):
    """
    Writes a DXF with:
      - One closed LWPOLYLINE per instance
      - Layer = class
      - Color mapped by confidence (optional)
      - Tiny text label 'id:<i>' at centroid (for joining in GH)
    """
    if not HAS_EZDXF:
        print("  (DXF skipped – ezdxf not installed)")
        return

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    def color_from_conf(c):
        # map 0..1 → 1..200 (ACI)
        return max(1, min(200, int(round(float(c) * 200))))

    for i, pred in enumerate(preds):
        pts = pred.get("points", [])
        if not pts:
            continue
        cls = pred.get("class", "object")
        conf = float(pred.get("confidence", 0.0))

        poly_xy = np.array([[p["x"], p["y"]] for p in pts], dtype=float)
        poly_list = [(float(x), float(y)) for x, y in poly_xy]
        if poly_list[0] != poly_list[-1]:
            poly_list.append(poly_list[0])

        # ensure layer exists
        if cls not in doc.layers:
            doc.layers.add(cls)

        # add polyline
        msp.add_lwpolyline(
            poly_list,
            dxfattribs={"layer": cls, "color": color_from_conf(conf)}
        )

        # add centroid text label; use insert attribute for maximum compatibility
        cx = float(np.mean(poly_xy[:, 0]))
        cy = float(np.mean(poly_xy[:, 1]))
        text = msp.add_text(
            f"id:{i}",
            dxfattribs={"height": 3.0, "layer": cls, "color": color_from_conf(conf)}
        )
        # place the text (older ezdxf versions don’t have set_pos)
        try:
            text.set_pos((cx, cy))
        except AttributeError:
            text.dxf.insert = (cx, cy)

    doc.saveas(str(dxf_path))

def rows_from_preds(image_name: str, preds: List[Dict]) -> List[Dict]:
    rows = []
    for i, p in enumerate(preds):
        pts = p.get("points", [])
        if not pts:
            continue
        rows.append({
            "image": image_name,
            "instance_id": i,                       # matches DXF id:<i>
            "class": p.get("class", "object"),
            "confidence": float(p.get("confidence", 0.0)),
            "num_vertices": len(pts),
            "centroid_x": sum(pt["x"] for pt in pts) / len(pts),
            "centroid_y": sum(pt["y"] for pt in pts) / len(pts),
        })
    return rows

# -------------------------------
# MAIN
# -------------------------------
def main():
    ensure_dirs(OUTPUT_DIR)

    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    # Process single image or folder
    if in_path.is_file():
        images = [in_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        images = [p for p in sorted(in_path.glob("*")) if p.suffix.lower() in exts]

    if not images:
        print("No images to process.")
        return

    # Prepare CSV once
    csv_path = Path(OUTPUT_DIR, "gh_metrics.csv")
    write_header = not csv_path.exists()

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        result = call_inference(str(img_path))
        preds = result.get("predictions", [])

        # Save raw JSON for reference/debug
        json_path = Path(OUTPUT_DIR, "json", f"{img_path.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # DXF export
        dxf_path = Path(OUTPUT_DIR, "dxf", f"{img_path.stem}.dxf")
        export_dxf_for_gh(preds, dxf_path)
        if HAS_EZDXF:
            print(f"  ✓ DXF: {dxf_path.name}")

        # Append CSV rows
        rows = rows_from_preds(img_path.name, preds)
        if rows:
            import csv
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "image", "instance_id", "class", "confidence",
                    "num_vertices", "centroid_x", "centroid_y"
                ])
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerows(rows)
            print(f"  ✓ CSV rows appended: {len(rows)} → {csv_path.name}")
        else:
            print("  ! No predictions returned for this image")

    print("\nDone.")

if __name__ == "__main__":
    main()
