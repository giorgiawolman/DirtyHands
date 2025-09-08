What this does

    Evaluates YOLO segmentation weights (.pt) on your dataset, auto-creates a data_autogen.yaml
    from your folder layout, and writes plots + metrics:

    PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
    confusion_matrix.png, confusion_matrix_normalized.png
    metrics_summary.json, sweep_results.csv (if enabled)

Files added

    run_eval_from_config.py
    yolo_seg_eval.py
    eval_config.yaml (you create/edit this)

Extra requirements (in addition to those above)

    pip install ultralytics>=8.2,<9 opencv-python>=4.8 numpy>=1.23 pandas>=2 matplotlib>=3.7 pyyaml>=6
    (GPU, CUDA 12.1 example)  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    Optional for COCO JSON export:  pip install faster-coco-eval>=1.6.7
    If faster-coco-eval is not installed, evaluation still works (we skip save_json).

Dataset layout expected

    Each split contains:  images/  and  labels/
    Point config paths to the images/ folders (labels are auto-found).

Config file (eval_config.yaml)

    weights: 'C:/path/to/weights.pt'

    dataset:
      train: 'C:/path/to/train/images'
      val:   'C:/path/to/valid/images'
      test:  'C:/path/to/test/images'    # optional
      names: ['forest','forestfragmentation']

    out_dir: 'C:/path/to/output'
    imgsz: 1024
    conf: 0.001
    nms_iou: 0.5
    device: '0'          # GPU id, or '' for auto/CPU
    split: 'val'
    tta: false

    sweep: true
    conf_list: [0.12, 0.16, 0.2, 0.24, 0.28, 0.32]
    iou_list:  [0.45, 0.5, 0.55, 0.6, 0.65]

    Notes:
    • On Windows, use forward slashes or single-quoted backslashes in YAML.
    • If VRAM is tight (≈4 GB), try: imgsz: 768, tta: false, sweep: false (quick smoke test).
    • The script writes data_autogen.yaml into out_dir.

Run evaluation

    python run_eval_from_config.py --cfg "C:/path/to/eval_config.yaml"

Alternate (no config file)

    python yolo_seg_eval.py ^
      --weights "C:/.../weights.pt" ^
      --train   "C:/.../train/images" ^
      --val     "C:/.../valid/images" ^
      --test    "C:/.../test/images" ^
      --names   "forest,forestfragmentation" ^
      --out_dir "C:/.../output" ^
      --imgsz 1024 --conf 0.001 --nms_iou 0.5 --split val --sweep

Outputs

    OUTPUT_DIR/
    eval_YYYYMMDD_HHMMSS/
      PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
      confusion_matrix.png, confusion_matrix_normalized.png
      metrics_summary.json
      sweep_results.csv
    data_autogen.yaml   (auto-generated)

Common issues

    YAML path error at "\U":
        Use single quotes or forward slashes in paths.

    FileNotFoundError (weights):
        Verify the exact file path; Windows may hide extensions.

    faster_coco_eval not installed:
        Either pip install faster-coco-eval>=1.6.7 or keep the default (evaluation runs without COCO JSON).

    "augment not supported, reverting":
        Set tta: false.

Recommended eval settings (based on current results)

    conf ≈ 0.24 (near F1 peak)
    nms_iou ≈ 0.40–0.60
    imgsz 1280 if VRAM allows (better for thin “fragmentation”), else 1024/768
    tta false

Deployment tip (optional)

    Apply per-class thresholds and class-agnostic NMS during inference to reduce duplicates:

    forest: 0.30
    forestfragmentation: 0.20
    Use agnostic_nms=True and iou 0.4–0.6 for cleaner outputs.
