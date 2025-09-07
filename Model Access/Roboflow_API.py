# gh_export_instances_workflow.py
# Run a Roboflow Workflow (serverless V2), optionally merge overlaps,
# then export polygons to DXF + a CSV for Grasshopper.

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import math
import random

import numpy as np
import cv2

# Optional: DXF (pip install ezdxf)
try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

# Optional: polygon ops (pip install shapely)
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

# Roboflow client
from inference_sdk import InferenceHTTPClient


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — EDIT THESE
# ──────────────────────────────────────────────────────────────────────────────

# Mode
TEST_MODE: Literal["live", "mock", "fixture"] = "live"
FIXTURE_JSON = r"C:\path\to\saved_prediction.json"  # used if TEST_MODE=="fixture"

# Roboflow
API_KEY        = "Sp3ArhygLcCOlxn6UV4P"
WORKSPACE_NAME = "forest-cover-changes"
WORKFLOW_ID    = "workflow"
USE_CACHE      = True

# IO
INPUT_PATH  = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\Images\ToBePredicted\leibintz2018 satelite.png"
OUTPUT_DIR  = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\GrasshopperOutput"

# Canvas/Image size handling
ORIGINAL_IMAGE_SIZE: Tuple[int, int] = (1280, 1280)   # (width, height)
FORCE_RESCALE_FROM_JSON_SIZE: bool = False           # set True if JSON reports a different canvas

# Coordinates
FLIP_Y_FOR_CAD: bool = True                         # y' = H - y

# DXF
EXPORT_ID_TEXT_IN_DXF: bool = False                 # set True to place tiny id labels

# Merge overlaps (requires shapely): "none" | "per_class" | "all"
MERGE_OVERLAPS: Literal["none", "per_class", "all"] = "per_class"
MERGE_BUFFER_EPS: float = 0.1                        # small cleanup buffer; 0.0 to disable

# ──────────────────────────────────────────────────────────────────────────────
# CLIENT / IO
# ──────────────────────────────────────────────────────────────────────────────

def rf_client() -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)

def load_fixture(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_workflow(image_path: str) -> Dict:
    if TEST_MODE == "fixture":
        return load_fixture(FIXTURE_JSON)

    if TEST_MODE == "mock":
        img = cv2.imread(image_path)
        h, w = (480, 640) if img is None else img.shape[:2]
        return make_mock(w, h)

    # live
    client = rf_client()
    res = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        use_cache=USE_CACHE
    )
    return normalize_result(res)

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTIONS — FIND, SCALE, FLIP, MERGE
# ──────────────────────────────────────────────────────────────────────────────

def normalize_result(res: Any) -> Dict:
    if isinstance(res, dict) and isinstance(res.get("predictions"), list):
        return res
    preds = find_polys(res)
    out = {"predictions": preds}
    w, h = find_image_size(res)
    if w and h:
        out["image"] = {"width": int(w), "height": int(h)}
    return out

def find_polys(obj: Any) -> List[Dict]:
    found: List[Dict] = []
    def rec(o: Any):
        if isinstance(o, dict):
            if isinstance(o.get("points"), list):
                found.append(o)
            if isinstance(o.get("predictions"), list):
                lst = o["predictions"]
                if lst and isinstance(lst[0], dict) and isinstance(lst[0].get("points"), list):
                    found.extend(lst)
            for v in o.values(): rec(v)
        elif isinstance(o, list):
            for it in o: rec(it)
    rec(obj)
    return [p for p in found if isinstance(p.get("points"), list)]

def find_image_size(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    w = h = None
    def rec(o: Any):
        nonlocal w, h
        if w and h: return
        if isinstance(o, dict):
            if isinstance(o.get("image"), dict):
                iw = o["image"].get("width"); ih = o["image"].get("height")
                if iw is not None and ih is not None:
                    w, h = iw, ih; return
            for v in o.values(): rec(v)
        elif isinstance(o, list):
            for it in o: rec(it)
    rec(obj)
    return w, h

def bbox_max(preds: List[Dict]) -> Tuple[float, float]:
    mx = my = 0.0
    for p in preds:
        for pt in p.get("points", []):
            x = float(pt.get("x", 0.0)); y = float(pt.get("y", 0.0))
            mx = max(mx, x); my = max(my, y)
    return mx, my

def scale_to_image(preds: List[Dict],
                   target_wh: Tuple[int, int],
                   json_wh: Optional[Tuple[int, int]],
                   force_from_json: bool) -> List[Dict]:
    if not preds:
        return preds
    W, H = target_wh
    xmax, ymax = bbox_max(preds)
    eps = 1e-6

    # normalized → scale
    if xmax <= 1.0 + eps and ymax <= 1.0 + eps:
        sx, sy = float(W), float(H)
        out = []
        for p in preds:
            q = dict(p)
            q["points"] = [{"x": float(pt["x"]) * sx, "y": float(pt["y"]) * sy} for pt in p["points"]]
            out.append(q)
        print(f"  • scaled normalized coords → {W}x{H}")
        return out

    # canvas → target
    if force_from_json and json_wh and json_wh != (W, H):
        jW, jH = json_wh
        if jW and jH:
            sx, sy = float(W) / float(jW), float(H) / float(jH)
            out = []
            for p in preds:
                q = dict(p)
                q["points"] = [{"x": float(pt["x"]) * sx, "y": float(pt["y"]) * sy} for pt in p["points"]]
                out.append(q)
            print(f"  • rescaled {jW}x{jH} → {W}x{H} (sx={sx:.6f}, sy={sy:.6f})")
            return out

    print("  • using polygon coords as-is")
    return preds

def flip_y(preds: List[Dict], image_h: int) -> List[Dict]:
    if not preds: return preds
    H = float(image_h)
    out: List[Dict] = []
    for p in preds:
        q = dict(p)
        q["points"] = [{"x": float(pt["x"]), "y": H - float(pt["y"])} for pt in p["points"]]
        out.append(q)
    return out

# ── merging (Shapely) ─────────────────────────────────────────────────────────

def pred_to_poly(p: Dict) -> Optional[Polygon]:
    pts = p.get("points") or []
    if len(pts) < 3:
        return None
    ring = [(float(pt["x"]), float(pt["y"])) for pt in pts]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    try:
        poly = Polygon(ring).buffer(0)  # normalize geometry
        return None if poly.is_empty else poly
    except Exception:
        return None

def poly_to_preds(geom, cls: str, conf: float) -> List[Dict]:
    geoms = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    out: List[Dict] = []
    for g in geoms:
        ext = g.exterior
        if not ext: continue
        pts = [{"x": float(x), "y": float(y)} for x, y in ext.coords]
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            out.append({"class": cls, "confidence": float(conf), "points": pts})
    return out

def merge_overlaps(preds: List[Dict], mode: str) -> List[Dict]:
    if mode == "none" or not preds or not HAS_SHAPELY:
        if mode != "none" and not HAS_SHAPELY:
            print("  ! shapely not installed → skipping merge")
        return preds

    def merge_list(items: List[Dict]) -> List[Dict]:
        geoms = []
        confs = []
        for it in items:
            g = pred_to_poly(it)
            if g is None: continue
            if MERGE_BUFFER_EPS > 0:
                g = g.buffer(MERGE_BUFFER_EPS)
            geoms.append(g)
            confs.append(float(it.get("confidence", 0.0)))
        if not geoms: return []
        merged = unary_union(geoms)
        if MERGE_BUFFER_EPS > 0:
            merged = merged.buffer(-MERGE_BUFFER_EPS).buffer(0)
        return poly_to_preds(merged, items[0].get("class", "object"), max(confs) if confs else 0.0)

    if mode == "per_class":
        by_cls: Dict[str, List[Dict]] = {}
        for p in preds:
            by_cls.setdefault(p.get("class", "object"), []).append(p)
        out: List[Dict] = []
        for cls, items in by_cls.items():
            out.extend(merge_list(items))
        print(f"  • merged overlaps per class → {len(out)} polys")
        return out

    if mode == "all":
        out = merge_list(preds)
        print(f"  • merged overlaps (global) → {len(out)} polys")
        return out

    return preds

# ──────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(base_out: str):
    Path(base_out).mkdir(parents=True, exist_ok=True)
    for sub in ("dxf", "json"):
        Path(base_out, sub).mkdir(parents=True, exist_ok=True)

def export_dxf(preds: List[Dict], dxf_path: Path):
    if not HAS_EZDXF:
        print("  (DXF skipped — ezdxf not installed)")
        return

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    def aci(c):
        try: return max(1, min(200, int(round(float(c) * 200))))
        except Exception: return 1

    for i, p in enumerate(preds):
        pts = p.get("points") or []
        if len(pts) < 2:
            continue
        layer = p.get("class", "object")
        color = aci(p.get("confidence", 0.0))
        poly = [(float(pt["x"]), float(pt["y"])) for pt in pts]
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        if layer not in doc.layers:
            doc.layers.add(layer)
        msp.add_lwpolyline(poly, dxfattribs={"layer": layer, "color": color})

        if EXPORT_ID_TEXT_IN_DXF:
            body = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
            cx = float(np.mean([x for x, _ in body])); cy = float(np.mean([y for _, y in body]))
            text = msp.add_text(f"id:{i}", dxfattribs={"height": 3.0, "layer": layer, "color": color})
            try: text.set_pos((cx, cy))
            except AttributeError: text.dxf.insert = (cx, cy)

    doc.saveas(str(dxf_path))

def rows_for_csv(image_name: str, preds: List[Dict]) -> List[Dict]:
    rows = []
    for i, p in enumerate(preds):
        pts = p.get("points") or []
        if not pts: continue
        rows.append({
            "image": image_name,
            "instance_id": i,
            "class": p.get("class", "object"),
            "confidence": float(p.get("confidence", 0.0)),
            "num_vertices": len(pts),
            "centroid_x": sum(pt["x"] for pt in pts) / len(pts),
            "centroid_y": sum(pt["y"] for pt in pts) / len(pts),
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# MOCK (for TEST_MODE="mock")
# ──────────────────────────────────────────────────────────────────────────────

def make_mock(w: int, h: int) -> Dict:
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
            {"class": "a", "confidence": 0.9, "points": poly(w*0.30, h*0.40, min(w, h)*0.12)},
            {"class": "b", "confidence": 0.8, "points": poly(w*0.65, h*0.55, min(w, h)*0.15)},
        ],
        "image": {"width": int(w), "height": int(h)}
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs(OUTPUT_DIR)

    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    images = [in_path] if in_path.is_file() else [
        p for p in sorted(in_path.glob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}
    ]
    if not images:
        print("No images to process.")
        return

    csv_path = Path(OUTPUT_DIR, "gh_metrics.csv")
    write_header = not csv_path.exists()

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        result = call_workflow(str(img_path))

        preds: List[Dict] = result.get("predictions", []) or []
        js_img = result.get("image") if isinstance(result.get("image"), dict) else {}
        json_wh = None
        if isinstance(js_img.get("width"), (int, float)) and isinstance(js_img.get("height"), (int, float)):
            json_wh = (int(js_img["width"]), int(js_img["height"]))

        # scale / flip
        preds = scale_to_image(preds, ORIGINAL_IMAGE_SIZE, json_wh, FORCE_RESCALE_FROM_JSON_SIZE)
        if FLIP_Y_FOR_CAD:
            preds = flip_y(preds, ORIGINAL_IMAGE_SIZE[1])
            print(f"  • y-flip → H={ORIGINAL_IMAGE_SIZE[1]}")

        # merge overlaps
        if MERGE_OVERLAPS != "none":
            preds_before = len(preds)
            preds = merge_overlaps(preds, MERGE_OVERLAPS)
            print(f"  • merge: {preds_before} → {len(preds)}")

        # save processed JSON
        out_json = Path(OUTPUT_DIR, "json", f"{img_path.stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "predictions": preds,
                "image": {"width": ORIGINAL_IMAGE_SIZE[0], "height": ORIGINAL_IMAGE_SIZE[1]},
                "y_flipped": FLIP_Y_FOR_CAD,
                "merged": MERGE_OVERLAPS,
                "merge_buffer_eps": MERGE_BUFFER_EPS
            }, f, indent=2)
        print(f"  • saved JSON → {out_json.name}")

        # DXF
        dxf_path = Path(OUTPUT_DIR, "dxf", f"{img_path.stem}.dxf")
        export_dxf(preds, dxf_path)
        if HAS_EZDXF:
            print(f"  ✓ DXF → {dxf_path.name}")

        # CSV
        rows = rows_for_csv(img_path.name, preds)
        if rows:
            import csv
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "image","instance_id","class","confidence","num_vertices","centroid_x","centroid_y"
                ])
                if write_header:
                    w.writeheader(); write_header = False
                w.writerows(rows)
            print(f"  ✓ CSV rows: {len(rows)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
