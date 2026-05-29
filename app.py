import argparse
import os

import neuroglancer
import numpy as np
import uvicorn
from cloudvolume import Bbox, CloudVolume
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# --- Global Configurations & State ---
app = FastAPI(title="Neuroglancer ROI Extractor API")
viewer = None
cv = None
CUBE_RADIUS = 20
OUTPUT_DIR = "./extracted_rois"
TARGET_TOKEN = "b8966e6d2e89245f71543638d237d5eceda58550"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Pydantic Models for API ---
class DatasetLoadRequest(BaseModel):
    url: str


class ExtractionRequest(BaseModel):
    x: int
    y: int
    z: int


# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Neuroglancer ROI Extractor API is running."}


# --- Core Logic ---
def extract_and_save_roi(x: int, y: int, z: int):
    """Core logic for data extraction, saving to disk, and layer injection."""
    global cv, viewer
    if cv is None:
        raise ValueError("Please load a dataset first via the /load endpoint.")

    print(f"[EXECUTING] Extracting volumetric data at X:{x}, Y:{y}, Z:{z}...")

    # Calculate precise bounding coordinates
    x_min, x_max = x - CUBE_RADIUS, x + CUBE_RADIUS
    y_min, y_max = y - CUBE_RADIUS, y + CUBE_RADIUS
    z_min, z_max = z - CUBE_RADIUS, z + CUBE_RADIUS

    # Create explicit Bounding Box
    request_bbox = Bbox((x_min, y_min, z_min), (x_max, y_max, z_max))

    # Download raw chunk data via cloud-volume
    chunk = cv[request_bbox]

    # Reformat array for Neuroglancer LocalVolume (requires z, y, x structure)
    chunk_data = np.transpose(np.squeeze(chunk), (2, 1, 0))

    # --- SAVE OUTPUT TO DISK ---
    filename = f"roi_X{x}_Y{y}_Z{z}.npy"
    filepath = os.path.join(OUTPUT_DIR, filename)
    np.save(filepath, chunk_data)
    print(f"[SAVED] ROI data successfully saved to: {filepath}")

    # --- UPDATE NEUROGLANCER VIEWER ---
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units=["nm", "nm", "nm"], scales=[8, 8, 8]
    )

    local_vol = neuroglancer.LocalVolume(
        data=chunk_data, dimensions=dimensions, voxel_offset=(x_min, y_min, z_min)
    )

    with viewer.txn() as s:
        s.layers["Extracted_ROI"] = neuroglancer.ImageLayer(
            source=local_vol, volume_rendering=True
        )
        if "base_image" in s.layers:
            s.layers["base_image"].volume_rendering = False

        s.position = [x, y, z]
        s.projectionScale = 100

    print("[SUCCESS] Data mapped to 'Extracted_ROI' in Viewer.")


def on_key_action(action_state):
    """Intercepts hardware event and routes coordinates."""
    pos = action_state.mouse_voxel_coordinates
    if pos is not None:
        try:
            extract_and_save_roi(int(pos[0]), int(pos[1]), int(pos[2]))
        except Exception as e:
            print(f"[ERROR] {e}")


# --- FastAPI Endpoints ---
@app.post("/load")
def load_dataset(req: DatasetLoadRequest):
    """Initializes CloudVolume and updates Neuroglancer based on the provided URL."""
    global cv, viewer
    dataset_url = req.url.strip()

    # Determine sources for CloudVolume and Neuroglancer
    if dataset_url.startswith(("zarr://", "precomputed://", "boss://")):
        cv_source = dataset_url
        ng_source = dataset_url
    elif ".zarr" in dataset_url:
        # If it's a web URL for zarr, Neuroglancer needs the zarr:// prefix,
        # but CloudVolume sometimes prefers it without if using fsspec/http.
        cv_source = (
            dataset_url if dataset_url.startswith("http") else f"zarr://{dataset_url}"
        )
        ng_source = (
            f"zarr://{dataset_url}"
            if not dataset_url.startswith("zarr://")
            else dataset_url
        )
    else:
        cv_source = dataset_url
        ng_source = (
            f"precomputed://{dataset_url}"
            if not dataset_url.startswith("http")
            else dataset_url
        )

    print(f"[LOADING] CV Source: {cv_source}")
    print(f"[LOADING] NG Source: {ng_source}")

    try:
        # mip=0 is default, fill_missing=True helps with sparse data
        cv = CloudVolume(cv_source, mip=0, fill_missing=True, progress=False)

        with viewer.txn() as s:
            s.layers["base_image"] = neuroglancer.ImageLayer(
                source=ng_source, volume_rendering=True
            )
            s.layout = "4panel"

        return {
            "status": "success",
            "message": f"Dataset loaded. CV: {cv_source}, NG: {ng_source}",
        }
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CloudVolume Error: {str(e)}")


@app.post("/extract")
def trigger_extraction(req: ExtractionRequest):
    """Allows triggering an extraction programmatically via REST API."""
    try:
        extract_and_save_roi(req.x, req.y, req.z)
        return {"status": "success", "message": f"ROI saved to {OUTPUT_DIR}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- CLI & Server Initialization ---
def setup_neuroglancer(ng_port, ng_token):
    global viewer
    neuroglancer.set_server_bind_address(bind_address="0.0.0.0", bind_port=ng_port)
    viewer = neuroglancer.Viewer(token=ng_token)

    viewer.actions.add("extract_roi_on_key", on_key_action)
    with viewer.config_state.txn() as config:
        config.input_event_bindings.viewer["keyx"] = "extract_roi_on_key"

    print(
        f"\n[SYSTEM READY] Neuroglancer Viewer Bound: http://localhost:{ng_port}/v/{ng_token}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuroglancer ROI Extractor API")
    parser.add_argument(
        "--api-port", type=int, default=8000, help="Port for the FastAPI server"
    )
    parser.add_argument(
        "--ng-port", type=int, default=5000, help="Port for the Neuroglancer viewer"
    )
    parser.add_argument(
        "--ng-token",
        type=str,
        default=TARGET_TOKEN,
        help="Token for Neuroglancer",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./extracted",
        help="Directory to save output arrays",
    )

    args = parser.parse_args()

    # Update global output directory based on CLI args
    OUTPUT_DIR = args.out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize the Neuroglancer viewer
    setup_neuroglancer(args.ng_port, args.ng_token)

    print(f"[INFO] Saving extracted ROIs to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"[INFO] Starting FastAPI server on port {args.api_port}...\n")

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.api_port)
