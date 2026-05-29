# Neuroglancer ROI Extractor

A robust API and visualization toolset for extracting 3D volumetric Regions of Interest (ROIs) from large-scale neuroimaging datasets. This project integrates **FastAPI**, **Neuroglancer**, and **CloudVolume** to provide a seamless workflow from dataset navigation to local data extraction and inspection.

---

## Purpose & Overview

The primary goal of this tool is to simplify the process of "snipping" small volumetric cubes (default 40x40x40 voxels) from massive remote datasets (Zarr, Precomputed, Boss) for local analysis, machine learning training, or detailed inspection.

### Key Features:
*   **Interactive Extraction:** Navigate the dataset in your browser and press the `x` key to instantly save a 3D cube centered at your mouse cursor.
*   **REST API Control:** Programmatically trigger extractions via POST requests.
*   **Live Preview:** Extracted ROIs are automatically injected back into the Neuroglancer viewer as a separate high-resolution layer.
*   **Offline Inspection:** A dedicated Jupyter notebook viewer for slice-by-slice analysis of saved `.npy` files.

---

## Process Workflow

```mermaid
graph TD
    A[Start app.py] --> B[POST /load {url}]
    B --> C[Initialize CloudVolume & Neuroglancer]
    C --> D
    D -- "POST /extract {x,y,z}" --> E
    E --> F[Calculate Bounding Box]
    F --> G[Download Chunk via CloudVolume]
    G --> H[Save roi_*.npy to Disk]
    H --> I[Inject 'Extracted_ROI' Layer into Viewer]
    I --> J[Open viewer.ipynb]
    J --> K[Interactive Slice Inspection]
```

---

## File Explanations

| File | Description |
| :--- | :--- |
| `app.py` | The core FastAPI server. It manages the Neuroglancer session, handles dataset loading, and executes the extraction logic. |
| `viewer.ipynb` | An interactive Jupyter notebook using `matplotlib` and `ipywidgets` to view the saved `.npy` files slice-by-slice. |
| `roi_X_Y_Z.npy` | 3D NumPy arrays containing the extracted volumetric data. The filename indicates the center voxel coordinates. |
| `extracted/` | Default output directory for all saved ROI files. |

---

## Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
pip install fastapi uvicorn neuroglancer cloud-volume numpy matplotlib ipywidgets
```

---

## Getting Started

### 1. Start the API Server
Run the server to initialize the Neuroglancer viewer and the REST endpoints.

```bash
python app.py --api-port 8000 --ng-port 10226 --out-dir ./extracted
```

### 2. Load a Dataset
Use `curl` to connect the tool to a remote volumetric dataset.

```bash
curl -X POST "http://localhost:8000/load" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://storage.googleapis.com/neuroglancer-public-data/flyem_fib-25/image"}'
```

### 3. Extract an ROI
You can trigger an extraction programmatically:

```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{"x": 1000, "y": 2000, "z": 3000}'
```

---

## Visualizing Results

After extracting several ROIs, use the provided notebook to inspect them:

1.  Open `viewer.ipynb` in VS Code or Jupyter Lab.
2.  Update the `roi_path` variable in the second cell to point to one of your saved `.npy` files.
3.  Run the cells to use the **Z-Slice Slider** for interactive navigation.

---

## Configuration
You can modify the following constants in `app.py` to change behavior:
*   `CUBE_RADIUS`: Set to `20` by default (results in a 40px cube). Increase this for larger extractions.
*   `scales`: In the `extract_and_save_roi` function, ensure the `scales=[8, 8, 8]` matches your dataset's voxel resolution for accurate coordinate mapping.

---

## Developer

**Tahmeed Ahmad**  
GitHub: [@syedtahmeed12](https://github.com/syedtahmeed12)
