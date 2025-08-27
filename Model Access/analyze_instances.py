# analyze_instances.py
# Draw polygons from Roboflow instance segmentation and compute geometry per object.

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union

from inference_sdk import InferenceHTTPClient

# -------------------------------
# CONFIG — EDIT THESE
# -------------------------------
API_KEY = "YOUR_API_KEY"  # or set via .env and read with os.getenv
USE_WORKFLOW = False       # True if you deployed a Workflow, False for a single model
WORKFLOW_URL = "https://serverless.roboflow.com/workflows/YOUR-WORKFLOW-ID"
MODEL_ID = "football-pitch-segmentation/1"  # e.g. "your-project/3"

# Input can be a single image or a folder
INPUT_PATH = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\images_in"  # file OR folder

OUTPUT_DIR = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\out"
# If you know pixel size (e.g., 0.5 mm/px), set this to convert to mm²/mm:
PIXEL_SIZE_MM = None  # e.g., 0.5  -> areas in mm^2, perimeters in mm

# -------------------------------
# END CONFIG
# -------------------------------

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

def call_inference(image_path: str) -> Dict:
    if USE_WORKFLOW:
        return CLIENT.run_workflow(image=image_path, workflow_url=WORKFLOW_URL)
    else:
        return CLIENT.infer(image_path, model_id=MODEL_ID)

def ensure_dirs(base_out: str):
    Path(base_out).mkdir(parents=True, exist_ok=True)
    Path(base_out, "annotated").mkdir(parents=True, exist_ok=True)
    Path(base_out, "json").mkdir(parents=True, exist_ok=True)
    Path(base_out, "geojson").mkdir(parents=True, exist_ok=True)

def to_int_pts(points: List[Dict[str, float]]) -> np.ndarray:
    # Roboflow gives [{"x":..., "y":...}, ...]
    arr = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    return np.round(arr).astype(np.int32)

def polygon_metrics(points_xy: np.ndarray) -> Optional[Dict[str, float]]:
    # points_xy: Nx2 int (x, y)
    if points_xy.shape[0] < 3:
        return None

    # Shapely expects (x, y)
    poly = Polygon(points_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix self-intersections if any
    if poly.is_empty or not poly.is_valid:
        return None

    area_px = float(poly.area)
    perimeter_px = float(poly.length)

    # Convex hull for solidity
    hull = poly.convex_hull
    hull_area = float(hull.area) if not hull.is_empty else np.nan
    solidity = (area_px / hull_area) if hull_area and hull_area > 0 else np.nan

    # Extent: area / bbox area
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = float((maxx - minx) * (maxy - miny)) if (maxx > minx and maxy > miny) else np.nan
    extent = (area_px / bbox_area) if bbox_area and bbox_area > 0 else np.nan

    # Circularity: 4πA / P^2
    circularity = (4.0 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan

    # Centroid
    cx, cy = poly.centroid.x, poly.centroid.y

    # Orientation + major/minor axis via PCA on boundary points
    pts = points_xy.astype(np.float64)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # sorted ascending
    # principal axis is eigenvector with largest eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Orientation angle in degrees of major axis (x-axis reference)
    vx, vy = eigvecs[:, 0]
    orientation_rad = np.arctan2(vy, vx)
    orientation_deg = float(np.degrees(orientation_rad))

    # Approximate major/minor axis lengths from eigenvalues (std dev * 4 ~ diameter)
    # This is a rough shape descriptor; for strict ellipse fit, use fitEllipse on contour.
    major_axis = float(4.0 * np.sqrt(eigvals[0])) if eigvals[0] > 0 else np.nan
    minor_axis = float(4.0 * np.sqrt(eigvals[1])) if eigvals[1] > 0 else np.nan

    metrics = {
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "bbox_x": float(minx),
        "bbox_y": float(miny),
        "bbox_w": float(maxx - minx),
        "bbox_h": float(maxy - miny),
        "orientation_deg": orientation_deg,
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "num_vertices": int(points_xy.shape[0]),
    }

    # Convert to physical units if pixel size is known
    if PIXEL_SIZE_MM is not None:
        s = PIXEL_SIZE_MM
        metrics.update({
            "area_mm2": area_px * (s ** 2),
            "perimeter_mm": perimeter_px * s,
            "major_axis_mm": major_axis * s if not np.isnan(major_axis) else np.nan,
            "minor_axis_mm": minor_axis * s if not np.isnan(minor_axis) else np.nan,
        })

    return metrics

def draw_and_measure(img_bgr: np.ndarray, predictions: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    out_rows = []

    for i, pred in enumerate(predictions):
        label = pred.get("class", "object")
        conf = float(pred.get("confidence", np.nan))

        points = pred.get("points", [])
        if not points:
            # Some models may return 'mask' formats; here we expect 'points'
            continue

        pts = to_int_pts(points)
        # Clamp to image bounds to avoid draw errors
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = pts.reshape((-1, 1, 2))  # for cv2

        # Filled polygon with transparency
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        # Border
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        # Metrics
        metrics = polygon_metrics(pts.reshape(-1, 2))
        if metrics is None:
            continue

        row = {
            "instance_id": i,
            "label": label,
            "confidence": conf,
            **metrics
        }
        out_rows.append(row)

        # Put label near centroid
        text = f"{label} {conf:.2f}"
        cx, cy = int(metrics["centroid_x"]), int(metrics["centroid_y"])
        cv2.putText(overlay, text, (cx, max(0, cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (cx, max(0, cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)

    # Blend overlay for transparency
    annotated = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    return annotated, out_rows

def save_geojson(out_path: Path, image_name: str, rows: List[Dict]):
    # Minimal GeoJSON FeatureCollection with polygons in image coordinates
    features = []
    for r in rows:
        # Rebuild a rectangle Polygon from bbox? Better to store the polygon,
        # but we didn't keep raw polygon points here. If you want full polygon
        # coordinates per instance, store them during the loop above.
        # For now, we export bbox as Polygon to keep a valid GeoJSON.
        x, y = r["bbox_x"], r["bbox_y"]
        w, h = r["bbox_w"], r["bbox_h"]
        poly = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
            [x, y]
        ]
        features.append({
            "type": "Feature",
            "properties": {k: v for k, v in r.items() if k not in ["bbox_x","bbox_y","bbox_w","bbox_h"]},
            "geometry": {"type": "Polygon", "coordinates": [poly]}
        })
    fc = {"type": "FeatureCollection", "name": image_name, "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

def iter_images(path: str):
    p = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    if p.is_file():
        yield p
    else:
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in exts:
                yield f

def main():
    ensure_dirs(OUTPUT_DIR)
    all_rows = []

    for img_path in iter_images(INPUT_PATH):
        print(f"Processing: {img_path}")
        # Call inference
        result = call_inference(str(img_path))

        # Get predictions
        preds = result.get("predictions", [])
        # Sometimes workflows return nested structures; if needed, adjust here.

        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ! Could not read image, skipping: {img_path}")
            continue

        annotated, rows = draw_and_measure(img_bgr, preds)

        # Save annotated image
        out_img = Path(OUTPUT_DIR, "annotated", f"{img_path.stem}_annotated.png")
        cv2.imwrite(str(out_img), annotated)

        # Save raw JSON
        out_json = Path(OUTPUT_DIR, "json", f"{img_path.stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Save GeoJSON (bbox-based placeholder; see note in function)
        out_geo = Path(OUTPUT_DIR, "geojson", f"{img_path.stem}.geojson")
        save_geojson(out_geo, img_path.name, rows)

        # Accumulate rows for CSV
        for r in rows:
            r_with_file = {"image": str(img_path.name), **r}
            all_rows.append(r_with_file)

        print(f"  ✓ Saved: {out_img.name}, {out_json.name}, {out_geo.name} ({len(rows)} objects)")

    # Save CSV for all images
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = Path(OUTPUT_DIR, "measurements.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSummary saved: {out_csv}")
    else:
        print("\nNo predictions found.")

if __name__ == "__main__":
    main()
