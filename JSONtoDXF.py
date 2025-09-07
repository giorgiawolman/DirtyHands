# JSONtoDXF.py
# Convert Roboflow-like JSON predictions (with polygon "points") to DXF.
# - JSON_INPUT can be a file OR a folder of .json files
# - DXF_OUTPUT can be a file path (ending .dxf) OR a folder

import json, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import ezdxf
except Exception:
    raise RuntimeError("Please install ezdxf first: pip install ezdxf")

# -------------------------------
# CONFIG: set your paths + display controls
# -------------------------------
JSON_INPUT  = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\JSON files\inputs\graz2018_predictions.json"  # file OR folder
DXF_OUTPUT  = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\JSON files\outputs"                           # file(.dxf) OR folder

SCALE              = 10.0   # multiply all coordinates to make pixels visible
FLIP_Y             = True   # flip Y using image.height (image origin top-left -> CAD bottom-left)
DRAW_IMAGE_FRAME   = True   # draw 0,0 → (width,height) rectangle (scaled/flipped)
TEXT_HEIGHT        = 2.5
DEFAULT_COLOR_ACI  = 1      # red
# -------------------------------


def _try_json_loads(text: str) -> List[Dict[str, Any]]:
    """Accepts:
       - single object
       - array of objects
       - JSONL (one object per line)
       - concatenated objects: {...}, {...} (wrapped into [ ... ])"""
    text = text.strip()
    if not text:
        return []

    # JSONL heuristic
    if "\n" in text and re.search(r"^\s*\{", text, flags=re.M) and re.search(r"\}\s*\n\s*\{", text):
        objs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    objs.append(obj)
            except json.JSONDecodeError:
                pass
        return objs

    # concatenated objects without array wrapper
    if re.search(r"\}\s*,\s*\{", text) and not text.lstrip().startswith("["):
        wrapped = f"[{text}]"
        try:
            data = json.loads(wrapped)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass

    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]
    return []


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    return _try_json_loads(path.read_text(encoding="utf-8"))


def centroid_xy(points_xy: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(points_xy[:, 0])), float(np.mean(points_xy[:, 1]))


def ensure_layer(doc, name: str):
    if name not in doc.layers:
        doc.layers.add(name)
    layer = doc.layers.get(name)
    layer.on()
    layer.unlock()
    layer.defreeze()


def transform_points(points: List[Dict[str, float]], img_h: float | None) -> np.ndarray:
    arr = np.array([[float(p["x"]), float(p["y"])] for p in points if "x" in p and "y" in p], dtype=float)
    if arr.size == 0:
        return arr
    if FLIP_Y and img_h and np.isfinite(img_h):
        arr[:, 1] = (img_h - arr[:, 1])
    if SCALE and SCALE != 1.0:
        arr *= SCALE
    return arr


def add_frame(msp, doc, width: float, height: float, layer_name="__frame__"):
    ensure_layer(doc, layer_name)
    # frame points after scale; FLIP_Y handled implicitly by transform elsewhere
    w, h = width * SCALE, height * SCALE
    pts = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
    msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name, "color": 7})


def add_prediction_poly(msp, doc, pred: Dict[str, Any], idx: int, img_h: float | None) -> bool:
    pts = pred.get("points") or []
    if not isinstance(pts, list) or len(pts) < 3:
        return False

    arr = transform_points(pts, img_h)
    if arr.shape[0] < 3:
        return False

    cls = f"class_{pred.get('class_id')}" if "class_id" in pred else pred.get("class", "object")
    ensure_layer(doc, cls)

    poly = [(float(x), float(y)) for x, y in arr]
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    if "confidence" in pred:
        conf = float(pred.get("confidence", 0.0))
        color = max(1, min(200, int(round(conf * 200))))
    else:
        color = DEFAULT_COLOR_ACI

    msp.add_lwpolyline(poly, dxfattribs={"layer": cls, "color": color})

    cx, cy = centroid_xy(arr)
    text = msp.add_text(f"id:{idx}", dxfattribs={"height": TEXT_HEIGHT, "layer": cls, "color": color})
    try:
        text.set_pos((cx, cy))
    except AttributeError:
        text.dxf.insert = (cx, cy)
    return True


def json_to_dxf_file(json_path: Path, dxf_path: Path):
    dxf_path.parent.mkdir(parents=True, exist_ok=True)
    recs = load_json_records(json_path)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    total_polys = 0
    total_frames = 0

    for rec in recs:
        img = rec.get("image") or {}
        img_w = img.get("width")
        img_h = img.get("height")

        # optional frame (only if width/height present)
        if DRAW_IMAGE_FRAME and img_w and img_h:
            try:
                add_frame(msp, doc, float(img_w), float(img_h))
                total_frames += 1
            except Exception:
                pass

        preds = rec.get("predictions") or []
        if not isinstance(preds, list):
            continue
        for p in preds:
            if add_prediction_poly(msp, doc, p, idx=total_polys, img_h=float(img_h) if img_h else None):
                total_polys += 1

    doc.saveas(str(dxf_path))
    print(f"[OK] DXF written: {dxf_path}")
    print(f"     Polygons: {total_polys}, Frames: {total_frames}")


def main():
    json_in = Path(JSON_INPUT)
    dxf_out = Path(DXF_OUTPUT)

    if json_in.is_dir():
        json_files = sorted([p for p in json_in.glob("*.json") if p.is_file()])
        if not json_files:
            print(f"No .json files found in: {json_in}")
            return

        # DXF_OUTPUT as folder or file?
        dxf_dir = dxf_out if dxf_out.suffix.lower() != ".dxf" else dxf_out.parent
        dxf_dir.mkdir(parents=True, exist_ok=True)

        for jf in json_files:
            out_file = dxf_dir / f"{jf.stem}.dxf"
            json_to_dxf_file(jf, out_file)

    else:
        if not json_in.exists():
            print(f"Input JSON not found: {json_in}")
            return

        if dxf_out.suffix.lower() != ".dxf":
            dxf_out.mkdir(parents=True, exist_ok=True)
            out_file = dxf_out / f"{json_in.stem}.dxf"
        else:
            out_file = dxf_out

        json_to_dxf_file(json_in, out_file)


if __name__ == "__main__":
    main()
