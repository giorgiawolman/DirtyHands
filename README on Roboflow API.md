WHAT THIS DOES

    Runs a Roboflow Workflow (Serverless V2) on one or more images, fixes coordinates (optional Y-flip),
    optionally merges overlapping polygons, then exports:

    DXF (one closed polyline per polygon; layer = class; color = confidence)

    CSV (compact per-polygon table for Grasshopper)

    JSON (the processed polygons you exported)

REQUIREMENTS

    Python 3.9+ recommended.

    Install required packages:
    pip install inference-sdk opencv-python numpy

    Optional (recommended):
    pip install ezdxf (for DXF export)
    pip install shapely (for polygon merge/union)

    Conda users:
    conda install -c conda-forge shapely ezdxf
    pip install inference-sdk opencv-python numpy

QUICK START

    Open gh_export_instances_workflow.py

    In the CONFIG — EDIT THESE block, set:

    INPUT_PATH : file or folder with images

    OUTPUT_DIR : output folder (script creates dxf/, json/, plus gh_metrics.csv)

    API_KEY, WORKSPACE_NAME, WORKFLOW_ID (from your Roboflow project)

    Set image size and coordinate behavior:

    ORIGINAL_IMAGE_SIZE = (WIDTH, HEIGHT) e.g. (800, 1100)
    If predictions are normalized (0..1), they are scaled to this size.

    FORCE_RESCALE_FROM_JSON_SIZE = False
    Set True only if your workflow outputs polygons on a smaller canvas
    (e.g. 512x512) and you want them re-mapped to ORIGINAL_IMAGE_SIZE.

    FLIP_Y_FOR_CAD = True
    Converts image coordinate Y (down) to CAD Y (up) using your image height.

    Optional output behavior:

    EXPORT_ID_TEXT_IN_DXF = False
    Set True to stamp tiny id labels for debugging.

    MERGE_OVERLAPS = "per_class"
    Options: "none", "per_class", "all". Requires shapely.
    "per_class": dissolve overlaps within each class (recommended).
    "all": dissolve everything together.

    MERGE_BUFFER_EPS = 0.1
    Tiny cleaning buffer (pixels) for robust unions. Use 0.0 to disable.

    Run:
    python gh_export_instances_workflow.py

    If INPUT_PATH is a file, only that image is processed.
    If INPUT_PATH is a folder, all supported image files are processed.

OUTPUTS

    OUTPUT_DIR/
    dxf/
    <image-name>.dxf (closed LWPOLYLINEs; layer=class; color by confidence)
    json/
    <image-name>.json (processed polygons after scale/flip/merge)
    gh_metrics.csv (per-polygon table for Grasshopper)

    CSV columns:
    image, instance_id, class, confidence, num_vertices, centroid_x, centroid_y

    TYPICAL SETUPS

    A) UI polygons look correct but exported positions are off:

    ORIGINAL_IMAGE_SIZE = your true image pixel size (e.g. 800x1100)

    FORCE_RESCALE_FROM_JSON_SIZE = False (unless you know there’s a canvas)

    FLIP_Y_FOR_CAD = True

    B) Workflow slices tiles or uses a fixed 512x512 canvas:

    FORCE_RESCALE_FROM_JSON_SIZE = True

    ORIGINAL_IMAGE_SIZE = your true image pixel size

    C) Remove overlaps for clean Rhino geometry:

    MERGE_OVERLAPS = "per_class"

    If you see slivers, increase MERGE_BUFFER_EPS slightly (e.g. 0.2)

TROUBLESHOOTING

    Fewer polygons than expected or none:

    Check API_KEY, WORKSPACE_NAME, WORKFLOW_ID.

    Ensure the final polygon output block (and stitching, if used) is enabled in your workflow.

    Switch TEST_MODE to "fixture" and point FIXTURE_JSON to a known-good JSON
    to confirm the export path works.

    Polygons offset or scaled wrong:

    Ensure ORIGINAL_IMAGE_SIZE matches your real image pixels.

    If the workflow uses a preview canvas (e.g. 512x512), set FORCE_RESCALE_FROM_JSON_SIZE = True.

    Keep FLIP_Y_FOR_CAD = True for CAD-friendly orientation.

    DXF not written:

    Install ezdxf, or check write permissions of OUTPUT_DIR.

    Merge does nothing:

    Install shapely.

    Increase MERGE_BUFFER_EPS slightly to help clean invalid or sliver geometries.

    Notes:

    DPI does not matter here; use pixel dimensions.

    DXF LWPOLYLINEs don’t directly encode holes. If you need inner holes as separate polylines, you can extend the script to export interiors.

MINIMAL CONFIG EXAMPLE

    Edit these lines in the script:

    INPUT_PATH = r"C:\data\images\graz2018 satelite.png"
    OUTPUT_DIR = r"C:\data\out"

    ORIGINAL_IMAGE_SIZE = (800, 1100)
    FLIP_Y_FOR_CAD = True
    MERGE_OVERLAPS = "per_class" # or "none"
    EXPORT_ID_TEXT_IN_DXF = False # keep DXF clean

    Run:

    python gh_export_instances_workflow.py

MODE SETTINGS (optional)

    TEST_MODE = "live" | "mock" | "fixture"

    "live" : calls Roboflow

    "mock" : generates fake polygons (no network)

    "fixture" : loads from FIXTURE_JSON to test export steps

CREDITS

    Roboflow Inference SDK for workflow calls

    ezdxf for DXF writing

    Shapely for polygon unions/merging

    OpenCV + NumPy for image/array utilities