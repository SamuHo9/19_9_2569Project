# ICP.py Line-by-Line Code Explanation

This document contains the complete source code and a detailed line-by-line explanation of the `ICP.py` group-wise alignment pipeline, designed to help developers and users understand the underlying algorithm and structure.

---

## Section 1: Library Imports and SlicerSALT Bootstrap

- **Input:** None
- **Output:** Ensures the script runs inside SlicerSALT

```python
import os
import sys

def _bootstrap_slicer():
    try:
        import slicer
        return
    except ImportError:
        pass

    slicer_exe = os.environ.get(
        "SLICER_EXE",
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
    )
    if not os.path.exists(slicer_exe):
        print(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
        print("        Set env var SLICER_EXE or edit the default path in ICP.py")
        sys.exit(1)

    import subprocess
    script_path = os.path.abspath(__file__)
    cmd = [slicer_exe, "--no-main-window", "--no-splash",
           "--python-script", script_path] + sys.argv[1:]
    print(f"[INFO] No 'slicer' module in this Python. Re-launching via SlicerSALT:")
    print(f"       {slicer_exe}")
    sys.exit(subprocess.call(cmd))

_bootstrap_slicer()
```

### Line-by-line explanation:
- **Lines 1-2**: Import `os` and `sys` modules for file and system management.
- **Lines 4-9**: Declare the `_bootstrap_slicer()` function. Tries to import `slicer`. If successful, the script is running inside Slicer and returns. If an `ImportError` occurs, it passes and proceeds.
- **Lines 11-14**: Fetch the `SlicerSALT.exe` path from the `SLICER_EXE` environment variable. Falls back to a default path if not set.
- **Lines 15-18**: Check if the `SlicerSALT.exe` file exists. If not, print an error and exit the program.
- **Lines 20-21**: Import `subprocess` and get the absolute path of this script (`ICP.py`).
- **Lines 22-26**: Create the command (`cmd`) to launch SlicerSALT in headless mode without a UI, instructing it to run this script. Print an alert and exit the current Python process.
- **Lines 28**: Call the `_bootstrap_slicer()` function immediately to enforce the check.

---

## Section 2: Library Imports and Path Configuration

- **Input:** None
- **Output:** Prepares libraries and defines the script directory

```python
import vtk
import numpy as np
import glob
import argparse
import slicer
from datetime import datetime
from vtk.util.numpy_support import vtk_to_numpy

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    for arg in sys.argv:
        if arg.endswith(".py") and os.path.isfile(arg):
            SCRIPT_DIR = os.path.dirname(os.path.abspath(arg))
            break
```

### Line-by-line explanation:
- **Lines 30-36**: Import required libraries: `vtk` for 3D processing, `numpy` for math, `glob` for file searching, `argparse` for parameters, and `datetime` for timestamps.
- **Lines 38-45**: Use try-except to determine the script's path. If `__file__` is undefined, it searches through `sys.argv` to set the `SCRIPT_DIR` variable as the working base directory.

---

## Section 3: Logging System

- **Input:** `msg` (String message)
- **Output:** Prints the message to the console and saves it to `icp_debug_log.txt`

```python
DEBUG_LOG = os.path.join(SCRIPT_DIR, "icp_debug_log.txt")

def sprint(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [ICP-LOG] {msg}"
    print(formatted_msg)
    sys.stdout.flush()
    with open(DEBUG_LOG, "a") as f:
        f.write(formatted_msg + "\n")
```

### Line-by-line explanation:
- **Lines 47**: Define the log file path by combining `SCRIPT_DIR` with `icp_debug_log.txt`.
- **Lines 49**: Create the `sprint` (Slicer Print) function to print messages with timestamps.
- **Lines 50-51**: Get the current time, format it as HH:MM:SS, and prepend it to the message `msg`.
- **Lines 52-53**: Print the message and force flush the buffer (`sys.stdout.flush()`) to display it immediately without delay.
- **Lines 54-55**: Open the log file in append mode ('a') and write the message with a newline.

---

## Section 4: 3D Point Extraction and UI Folder Picker

- **Input:** `vtkPolyData` / Window `title` text
- **Output:** Numpy Array of points / Folder path

```python
def get_points_numpy(poly):
    if poly is None or poly.GetPoints() is None:
        return np.zeros((0, 3), dtype=np.float64)
    return vtk_to_numpy(poly.GetPoints().GetData())

def prompt_folder(title):
    import qt
    folder = qt.QFileDialog.getExistingDirectory(None, title)
    if not folder:
        return None
    return folder
```

### Line-by-line explanation:
- **Lines 57-60**: The `get_points_numpy` function extracts coordinates from a 3D model (poly) and converts them into a numpy array. If the model is empty, it returns a safe empty 0x3 array to prevent errors.
- **Lines 62-67**: The `prompt_folder` function opens a folder selection UI using `qt.QFileDialog`. Returns the path or `None` if canceled.

---

## Section 5: 3D Model Coordinate Transformation

- **Input:** `poly` (3D model), `matrix_np` (4x4 Transformation Matrix)
- **Output:** Transformed 3D model

```python
def apply_poly_transform(poly, matrix_np):
    transform = vtk.vtkTransform()
    matrix_vtk = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix_vtk.SetElement(r, c, float(matrix_np[r, c]))
    transform.SetMatrix(matrix_vtk)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(poly)
    transformer.SetTransform(transform)
    transformer.Update()
    return transformer.GetOutput()
```

### Line-by-line explanation:
- **Lines 69**: Declare `apply_poly_transform` receiving a model and a matrix.
- **Lines 70-71**: Create a new `vtkTransform` and an empty `vtkMatrix4x4`.
- **Lines 72-74**: Loop over 4x4 elements to copy values from the numpy array into the `vtkMatrix` (since VTK doesn't directly support numpy).
- **Lines 75**: Set the matrix into the Transform object.
- **Lines 76-79**: Use `vtkTransformPolyDataFilter` to apply the transform to the raw model and call `Update()` to execute.
- **Lines 80**: Retrieve and return the transformed model (`GetOutput()`).

---

## Section 6: Calculate Model Distance (ICP Distance)

- **Input:** Source model and Target model
- **Output:** Mean distance (Float)

```python
def icp_distance(source_poly, target_poly, n_sample=500):
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(target_poly)
    locator.BuildLocator()
    src_pts = get_points_numpy(source_poly)
    n_sample = min(len(src_pts), n_sample)
    idx = np.linspace(0, len(src_pts) - 1, n_sample, dtype=int)
    total_dist = 0.0
    for i in idx:
        pt = src_pts[i].tolist()
        closest_id = locator.FindClosestPoint(pt)
        cp = target_poly.GetPoint(closest_id)
        total_dist += np.sqrt(sum((src_pts[i][k] - cp[k]) ** 2 for k in range(3)))
    return total_dist / n_sample
```

### Line-by-line explanation:
- **Lines 82**: Declare `icp_distance` with a default of 500 sample points.
- **Lines 83-85**: Create a `vtkKdTreePointLocator` and build a spatial index for the target model to vastly speed up nearest-neighbor searches.
- **Lines 86**: Convert the source model into numpy coordinates.
- **Lines 87-88**: Ensure `n_sample` does not exceed the total number of points. Use `np.linspace` to sample indices evenly across the model.
- **Lines 89-91**: Initialize `total_dist` to 0 and loop through the sampled indices.
- **Lines 92**: Find the closest point on the target for the current source point using `FindClosestPoint`.
- **Lines 93-94**: Retrieve the XYZ coordinates and calculate the 3D Euclidean distance, adding it to `total_dist`.
- **Lines 95**: Return the average distance by dividing by `n_sample`.

---

## Section 7: Load NIfTI Image and Create 3D Surface Mesh

- **Input:** `filepath` of the 3D image (.nii/.nii.gz)
- **Output:** Surface mesh `vtkPolyData`

```python
def load_and_mesh_node(filepath, max_retries=2):
    normalized = os.path.abspath(filepath).replace("\\", "/")
    last_err = None
    for attempt in range(max_retries + 1):
        node = None
        try:
            node = slicer.util.loadLabelVolume(normalized)
            if not node:
                last_err = "loadLabelVolume returned None"
                continue
            img = node.GetImageData()
            if img is None or img.GetNumberOfPoints() == 0:
                last_err = "empty image data"
                slicer.mrmlScene.RemoveNode(node)
                continue
            dmc = vtk.vtkDiscreteMarchingCubes()
            dmc.SetInputData(img)
            dmc.GenerateValues(1, 1, 100)
            dmc.Update()
            poly = dmc.GetOutput()
            ijkToRas = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(ijkToRas)
            t = vtk.vtkTransform()
            t.SetMatrix(ijkToRas)
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetTransform(t)
            transformer.SetInputData(poly)
            transformer.Update()
            result_poly = transformer.GetOutput()
            slicer.mrmlScene.RemoveNode(node)
            return result_poly
        except Exception as e:
            last_err = str(e).split("\n")[0][:200]
            if node is not None:
                try: slicer.mrmlScene.RemoveNode(node)
                except Exception: pass
            if attempt < max_retries:
                try:
                    import gc; gc.collect()
                    import time; time.sleep(0.5)
                except Exception:
                    pass
    sprint(f"  !!! load_and_mesh_node failed after {max_retries+1} attempts: {last_err}")
    return None
```

### Line-by-line explanation:
- **Lines 97**: `load_and_mesh_node` function with `max_retries` to handle memory failures.
- **Lines 98**: Normalize the path string to use standard forward slashes.
- **Lines 99-100**: Prepare error tracking and start a loop for up to 3 attempts.
- **Lines 101-106**: Attempt to load the NIfTI file into Slicer as a `LabelVolume`. If it fails, record the error and retry.
- **Lines 107-111**: Extract image data (`GetImageData`). If empty (no shape present), remove the node and skip.
- **Lines 112-116**: Use `vtkDiscreteMarchingCubes` to generate polygons (surfaces) around voxels with a label of 1. Update and extract the model.
- **Lines 117-120**: Retrieve the IJKToRAS matrix, which maps array coordinates to physical world space.
- **Lines 121-125**: Apply this physical transform to the raw Marching Cubes model to position it correctly in world coordinates.
- **Lines 126-127**: Remove the image volume node to free up RAM and return the mesh.
- **Lines 128-132**: Catch any exceptions, truncate the error message, and clean up leftover nodes.
- **Lines 133-138**: If retries remain, force garbage collection (`gc.collect()`) and sleep for 0.5 seconds before retrying.
- **Lines 139-140**: If all retries fail, print a warning and return `None`.

---

## Section 8: Preliminary PCA Alignment

- **Input:** Raw `poly` model
- **Output:** Aligned model and Accumulation Matrix

```python
def principal_axis_align(poly):
    pts = get_points_numpy(poly)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    z_ax = Vt[0]
    x_ax = Vt[1]
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= np.linalg.norm(x_ax)
    R = np.eye(3)
    R[:, 0] = x_ax
    R[:, 1] = y_ax
    R[:, 2] = z_ax
    T_rot = np.eye(4)
    T_rot[:3, :3] = R.T
    T_cent = np.eye(4)
    T_cent[:3, 3] = -centroid
    T_combined = T_rot @ T_cent
    return apply_poly_transform(poly, T_combined), T_combined
```

### Line-by-line explanation:
- **Lines 142-143**: `principal_axis_align` function converts the model to points.
- **Lines 144-145**: Calculate the centroid (mean of all points) and subtract it from all points to center them at (0,0,0).
- **Lines 146**: Use `np.linalg.svd` (Singular Value Decomposition) to find the principal axes of variance (length, width, depth).
- **Lines 147**: Assign the 1st principal component (longest axis) as the Z-axis.
- **Lines 148**: Assign the 2nd principal component as the X-axis.
- **Lines 149-152**: Use cross products to compute the Y-axis perpendicular to Z and X, normalizing axes to length 1.
- **Lines 153-157**: Assemble the XYZ axes as columns of a 3x3 Rotation Matrix, transposing it into a 4x4 matrix (`T_rot`).
- **Lines 158-160**: Create a Translation Matrix to shift the centroid back.
- **Lines 161**: Combine Rotation and Translation matrices into `T_combined`.
- **Lines 162**: Return the transformed model and the combined matrix.

---

## Section 9: Orientation Correction (Pole Flip)

- **Input:** PCA-aligned model and reference model (optional)
- **Output:** Orientation-corrected model and flip status

```python
_FLIP_CANDIDATES = [
    np.eye(4),
    np.diag([1.0, -1.0, -1.0, 1.0]),
    np.diag([-1.0, 1.0, -1.0, 1.0]),
    np.diag([-1.0, -1.0, 1.0, 1.0]),
]
_FLIP_NAMES = ["identity", "rotX180", "rotY180", "rotZ180"]

def pole_flip_correction(poly, reference_poly=None):
    if reference_poly is None:
        pts = get_points_numpy(poly)
        z_mid = (pts[:, 2].min() + pts[:, 2].max()) / 2.0
        pts_pos = pts[pts[:, 2] > z_mid]
        pts_neg = pts[pts[:, 2] <= z_mid]
        fat_pos = (np.std(pts_pos[:, 0]) + np.std(pts_pos[:, 1])) if len(pts_pos) > 0 else 0.0
        fat_neg = (np.std(pts_neg[:, 0]) + np.std(pts_neg[:, 1])) if len(pts_neg) > 0 else 0.0
        if fat_neg > fat_pos:
            idx = 2
        else:
            idx = 0
        sprint(f"    Pole check (no ref): fat_neg={fat_neg:.4f}, fat_pos={fat_pos:.4f} -> {_FLIP_NAMES[idx]}")
        T = _FLIP_CANDIDATES[idx].copy()
        return apply_poly_transform(poly, T), T, (idx != 0)

    best_idx = 0
    best_dist = float("inf")
    dists = []
    for idx, T in enumerate(_FLIP_CANDIDATES):
        poly_test = apply_poly_transform(poly, T)
        d = icp_distance(poly_test, reference_poly)
        dists.append(d)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    sprint(f"    Pole check: dists=[{', '.join(f'{d:.5f}' for d in dists)}] -> {_FLIP_NAMES[best_idx]}")
    T = _FLIP_CANDIDATES[best_idx].copy()
    return apply_poly_transform(poly, T), T, (best_idx != 0)
```

### Line-by-line explanation:
- **Lines 164-170**: Define `_FLIP_CANDIDATES`, containing 4 rotations: Identity, X(180), Y(180), Z(180) degrees.
- **Lines 172-174**: `pole_flip_correction` function. If no reference model exists, it relies on anatomical assumptions (hippocampus head is larger than tail).
- **Lines 175-177**: Split the model into top and bottom halves along the Z-axis.
- **Lines 178-183**: Calculate the 'fatness' (Standard Deviation) of the top (`fat_pos`) and bottom (`fat_neg`). If the bottom is thicker, the model is flipped, and Y-axis rotation (index 2) is chosen.
- **Lines 184-186**: Print the reasoning, apply the selected flip matrix, and return the corrected model and status.
- **Lines 188-197**: If a reference model is provided, loop through all 4 flip candidates, applying each and calculating the `icp_distance` to the reference. The rotation with the smallest distance is chosen as correct.
- **Lines 198-200**: Print the distance results, apply the winning matrix, and return it.

---

## Section 10: Generate Mean Template (Mean Poly)

- **Input:** List of aligned models (`meshes`)
- **Output:** Average template model

```python
def compute_mean_poly(meshes):
    counts = [m.GetNumberOfPoints() for m in meshes]
    sprint(f"    compute_mean_poly: NN-matching, point counts {min(counts)}-{max(counts)}, N={len(meshes)}")

    ref_pts = get_points_numpy(meshes[0])
    mean_points = np.zeros_like(ref_pts)

    for m in meshes:
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(m)
        locator.BuildLocator()
        pts = get_points_numpy(m)
        for j, pt in enumerate(ref_pts):
            closest_id = locator.FindClosestPoint(pt.tolist())
            mean_points[j] += pts[closest_id]

    mean_points /= len(meshes)

    mean_poly = vtk.vtkPolyData()
    mean_poly.DeepCopy(meshes[0])
    vtk_pts = mean_poly.GetPoints()
    for j in range(len(mean_points)):
        vtk_pts.SetPoint(j, mean_points[j].tolist())

    return mean_poly
```

### Line-by-line explanation:
- **Lines 202-204**: `compute_mean_poly` function prints the number of input meshes.
- **Lines 206-207**: Use the points from the first model as the reference structure (`ref_pts`) and initialize a zero array `mean_points`.
- **Lines 209-216**: Loop through each model: Build a KdTree, find the closest point on the target mesh for each reference point, and accumulate the XYZ coordinates into `mean_points`.
- **Lines 218**: Divide `mean_points` by the number of meshes to compute the average position for each point.
- **Lines 220-224**: DeepCopy the first model's structure to preserve polygon connectivity, then overwrite its points with the calculated `mean_points`.
- **Lines 226**: Return the mean model.

---

## Section 11: Iterative Closest Point (ICP) Algorithm

- **Input:** Source model, Target model, tolerance settings
- **Output:** Final 4x4 alignment matrix

```python
EVAL_PAIRWISE_STEPS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]

def run_vtk_icp(source_poly, target_poly, return_history=False, max_iter=100, tolerance=0.0001, landmarks=200):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(max_iter)
    icp.SetMaximumMeanDistance(tolerance)
    icp.SetMaximumNumberOfLandmarks(landmarks)
    icp.CheckMeanDistanceOn()
    icp.Update()
    matrix = icp.GetMatrix()
    res = np.eye(4)
    for r in range(4):
        for c in range(4):
            res[r, c] = matrix.GetElement(r, c)

    if not return_history:
        return res

    history = []
    for k in EVAL_PAIRWISE_STEPS:
        icp_k = vtk.vtkIterativeClosestPointTransform()
        icp_k.SetSource(source_poly)
        icp_k.SetTarget(target_poly)
        icp_k.GetLandmarkTransform().SetModeToRigidBody()
        icp_k.SetMaximumNumberOfIterations(k)
        icp_k.SetMaximumMeanDistance(tolerance)
        icp_k.SetMaximumNumberOfLandmarks(landmarks)
        icp_k.CheckMeanDistanceOn()
        icp_k.Update()
        history.append(float(icp_k.GetMeanDistance()))

    return res, history
```

### Line-by-line explanation:
- **Lines 228**: `EVAL_PAIRWISE_STEPS` defines the iteration checkpoints used for logging convergence history.
- **Lines 230-239**: `run_vtk_icp` initializes `vtkIterativeClosestPointTransform`. Sets mode to `RigidBody` (no scaling/warping). Configures iteration limits and tolerance, then calls `Update()` to execute.
- **Lines 240-244**: Retrieve the resulting 4x4 matrix and convert it to a numpy array.
- **Lines 246-247**: If `return_history` is False, return the matrix and exit.
- **Lines 249-262**: If True, rerun ICP repeatedly, increasing max iterations according to `EVAL_PAIRWISE_STEPS` (1, 2, 3.. 30) to record the mean distance at each step. Return the matrix and the history array.

---

## Section 12: NIfTI Volume Resampling and Export

- **Input:** File list, T_matrices, Output directory
- **Output:** Resampled `.nii.gz` files

```python
def export_aligned_nifti(file_list, T_matrices, output_dir,
                         spacing_mm=0.5, n_voxels=128, interpolation="NearestNeighbor"):
    out_vol_dir = os.path.join(output_dir, "aligned_nifti")
    os.makedirs(out_vol_dir, exist_ok=True)
    N = len(file_list)
    half = (n_voxels / 2.0) * spacing_mm

    for i, f in enumerate(file_list):
        basename = os.path.basename(f).split('.')[0]
        sprint(f"Saving [{i+1}/{N}]: {basename} (box={2*half:.1f}mm, spacing={spacing_mm}mm)")
        node = slicer.util.loadLabelVolume(f)
        T = T_matrices[i]

        t_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        v_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                v_mat.SetElement(r, c, float(T[r, c]))
        t_node.SetMatrixTransformToParent(v_mat)

        ref_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ref")
        img = vtk.vtkImageData()
        img.SetDimensions(n_voxels, n_voxels, n_voxels)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        ref_node.SetAndObserveImageData(img)
        ref_node.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
        ref_node.SetOrigin(-half, -half, -half)

        params = {
            "inputVolume": node.GetID(),
            "referenceVolume": ref_node.GetID(),
            "outputVolume": ref_node.GetID(),
            "transformationFile": t_node.GetID(),
            "interpolationMode": interpolation
        }
        slicer.cli.run(
            slicer.modules.resamplescalarvectordwivolume,
            None, params, wait_for_completion=True
        )
        out_path = os.path.join(out_vol_dir, f"{basename}_aligned.nii.gz")
        slicer.util.saveNode(ref_node, out_path)

        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(t_node)
        slicer.mrmlScene.RemoveNode(ref_node)
```

### Line-by-line explanation:
- **Lines 264-269**: `export_aligned_nifti` creates an `aligned_nifti` folder. Calculates volume boundaries (`half`) based on `n_voxels` (128) and `spacing_mm`.
- **Lines 271-276**: Loop through original files, reload the volume (`loadLabelVolume`), and fetch its corresponding result matrix.
- **Lines 277-282**: Create a `vtkMRMLLinearTransformNode` and populate it with the 4x4 matrix.
- **Lines 284-290**: Create an empty reference image (`vtkMRMLScalarVolumeNode`) with 128x128x128 dimensions, setting spacing and centering the origin.
- **Lines 292-297**: Configure parameters linking the input volume, reference volume, and transform node. Sets interpolation to `NearestNeighbor` to avoid blurry labels.
- **Lines 298-302**: Run Slicer's `resamplescalarvectordwivolume` CLI module to resample the transformed input volume into the reference space.
- **Lines 303-308**: Save the resulting reference node as a `.nii.gz` file. Clean up nodes to free memory.

---

## Section 13: CLI Arguments and File Management

- **Input:** Command-line execution arguments
- **Output:** Parsed variables and valid file list

```python
OUTPUT_SPACING = 0.02
OUTPUT_VOXELS = 128
MAX_GW_ITERATIONS = 20
GW_TOLERANCE = 0.00005
PAIRWISE_ITERATIONS = 100
PAIRWISE_TOLERANCE = 0.0001
PAIRWISE_LANDMARKS = 200
INTERPOLATION_MODE = "NearestNeighbor"

def parse_args():
    parser = argparse.ArgumentParser(description="Group-wise rigid ICP alignment for NIfTI labels.")
    parser.add_argument("--input_dir", default=None, help="Input directory containing NIfTI labels")
    parser.add_argument("--output_dir", default=None, help="Output directory for aligned files")
    parser.add_argument("--output_spacing", type=float, default=OUTPUT_SPACING, help="Output voxel spacing in mm/unit")
    parser.add_argument("--output_voxels", type=int, default=OUTPUT_VOXELS, help="Output volume dimension size")
    parser.add_argument("--max_iterations", type=int, default=MAX_GW_ITERATIONS, help="Max groupwise ICP iterations")
    parser.add_argument("--tolerance", type=float, default=GW_TOLERANCE, help="Groupwise convergence tolerance")
    parser.add_argument("--pairwise_iterations", type=int, default=PAIRWISE_ITERATIONS, help="Pairwise ICP max iterations")
    parser.add_argument("--pairwise_tolerance", type=float, default=PAIRWISE_TOLERANCE, help="Pairwise ICP tolerance")
    parser.add_argument("--pairwise_landmarks", type=int, default=PAIRWISE_LANDMARKS, help="Pairwise ICP landmarks count")
    parser.add_argument("--interpolation", type=str, default=INTERPOLATION_MODE, help="Resampling interpolation mode")
    args, _ = parser.parse_known_args()
    return args

def find_input_files(input_dir):
    extensions = ["*.nii.gz", "*.nii", "*.hdr", "*.nrrd"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    file_list = sorted(list(set(file_list)))

    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        file_list = label_files
        sprint(f"Prioritizing {len(file_list)} label files.")
    return file_list

def clean_previous_outputs(output_dir):
    for old_file in ["icp_convergence_history.json", "icp_convergence.png"]:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
                sprint(f"  Cleaned up old output file: {old_file}")
            except Exception:
                pass
```

### Line-by-line explanation:
- **Lines 310-317**: Declare constant defaults (e.g., 128 voxels, 20 max iterations, 0.0001 tolerance).
- **Lines 319-332**: `parse_args()` sets up the argument parser to allow overriding defaults via command line.
- **Lines 334-345**: `find_input_files` scans recursively for image files (.nii.gz, .hdr, etc.). Filters the list to prioritize files containing 'label' in the name.
- **Lines 347-355**: `clean_previous_outputs` deletes old convergence reports (.json/.png) before a new run.

---

## Section 14: Pipeline Step A - Load and Mesh

- **Input:** List of NIfTI files
- **Output:** List of 3D surface meshes

```python
def step_load_and_mesh_all(file_list):
    sprint("Step A: Loading and meshing label volumes...")
    meshes = []
    valid_files = []
    for i, f in enumerate(file_list):
        sprint(f"  Loading [{i+1}/{len(file_list)}]: {os.path.basename(f)}")
        p = load_and_mesh_node(f)
        if p and p.GetNumberOfPoints() > 0:
            meshes.append(p)
            valid_files.append(f)
        else:
            sprint(f"  WARNING: Skipping {os.path.basename(f)} (no points)")
    return meshes, valid_files
```

### Line-by-line explanation:
- **Lines 357-362**: `step_load_and_mesh_all` prepares empty lists and loops over files.
- **Lines 363-366**: Calls `load_and_mesh_node`. If valid points are found, stores the mesh and filename.
- **Lines 367-369**: Skips corrupted or empty files and returns the valid meshes.

---

## Section 15: Pipeline Step 1 - Normalization

- **Input:** Loaded meshes
- **Output:** Scaled meshes and initial transform matrices

```python
def step1_per_mesh_normalize(meshes):
    sprint("Step 1: Per-mesh normalization (center + scale to |coord| <= 1)...")
    N = len(meshes)
    aligned_meshes = []
    T_initial = []
    for i in range(N):
        pts = get_points_numpy(meshes[i])
        centroid = pts.mean(axis=0)
        T_cent = np.eye(4)
        T_cent[:3, 3] = -centroid
        poly_centered = apply_poly_transform(meshes[i], T_cent)

        b = poly_centered.GetBounds()
        max_abs = max(abs(b[0]), abs(b[1]), abs(b[2]),
                      abs(b[3]), abs(b[4]), abs(b[5]))
        s = 1.0 / max_abs if max_abs > 0 else 1.0

        T_scale = np.eye(4)
        T_scale[0, 0] = T_scale[1, 1] = T_scale[2, 2] = s

        T_combined = T_scale @ T_cent
        aligned_meshes.append(apply_poly_transform(meshes[i], T_combined))
        T_initial.append(T_combined)
        if i < 3 or i == N - 1:
            sprint(f"  Mesh {i+1}: centroid={centroid}, scale={s:.6f}")
    return aligned_meshes, T_initial
```

### Line-by-line explanation:
- **Lines 371-375**: `step1_per_mesh_normalize` initializes lists and loops through meshes.
- **Lines 376-381**: Finds the centroid and creates a translation matrix to center the model at (0,0,0).
- **Lines 383-386**: Evaluates bounding box dimensions to find the absolute maximum coordinate. Calculates a scaling factor (`s`) to normalize model size between -1 and 1. (Prevents ICP failures on huge physical scales).
- **Lines 388-393**: Combines scale and translation matrices (`T_combined`), applies it to the model.
- **Lines 394-396**: Prints scale factors and returns scaled models along with `T_initial`.

---

## Section 16: Pipeline Step 2 - PCA Alignment

- **Input:** Normalized meshes
- **Output:** PCA Transformation matrices

```python
def step2_pca_alignment(aligned_meshes):
    sprint("Step 2: Principal Axis Alignment (PCA)...")
    N = len(aligned_meshes)
    T_pca_list = []
    for i in range(N):
        aligned_meshes[i], T_p = principal_axis_align(aligned_meshes[i])
        T_pca_list.append(T_p)
    return T_pca_list
```

### Line-by-line explanation:
- **Lines 398-400**: `step2_pca_alignment` initializes the output list.
- **Lines 401-405**: Loops through models, applies `principal_axis_align`, updates models, and stores PCA matrices in `T_pca_list`.

---

## Section 17: Pipeline Step 3 - Orientation Correction

- **Input:** Meshes, Initial Matrices, PCA Matrices
- **Output:** Updated Initial Matrices

```python
def step3_orientation_disambiguation(aligned_meshes, T_initial, T_pca_list):
    sprint("Step 3: Orientation disambiguation (proper rotations vs reference)...")
    N = len(aligned_meshes)
    T_flip = []
    aligned_meshes[0], tf0, flipped0 = pole_flip_correction(aligned_meshes[0], reference_poly=None)
    T_flip.append(tf0)
    sprint(f"  Mesh 1 (reference): {'REORIENTED' if flipped0 else 'OK'}")

    for i in range(1, N):
        aligned_meshes[i], tfi, flippedi = pole_flip_correction(
            aligned_meshes[i], reference_poly=aligned_meshes[0]
        )
        T_flip.append(tfi)
        sprint(f"  Mesh {i+1}: {'REORIENTED' if flippedi else 'OK'}")

    for i in range(N):
        T_initial[i] = T_flip[i] @ T_pca_list[i] @ T_initial[i]
    return T_initial
```

### Line-by-line explanation:
- **Lines 407-410**: `step3_orientation_disambiguation` processes the first model (index 0) using unreferenced `pole_flip_correction` to establish a global orientation standard.
- **Lines 412-420**: Loops through remaining models, using model 0 as the reference in `pole_flip_correction` to ensure all face the same direction.
- **Lines 422-424**: Multiplies Flip * PCA * Initial matrices together to finalize `T_initial`.

---

## Section 18: Pipeline Step 4 - Groupwise ICP

- **Input:** Meshes, Filenames, Algorithm arguments
- **Output:** Group ICP matrices and convergence history

```python
def step4_groupwise_icp(aligned_meshes, file_list, args):
    N = len(aligned_meshes)
    max_gw_iter = args.max_iterations
    gw_tolerance = args.tolerance
    sprint(f"Step 4: Groupwise ICP (rigid, max {max_gw_iter} rounds, tolerance={gw_tolerance})...")
    
    T_icp = [np.eye(4) for _ in range(N)]
    prev_mean_dist = float("inf")

    import time
    gw_start_time = time.time()

    gw_history = {
        "rounds": [],
        "elapsed_times_sec": [],
        "mean_distances": [],
        "dist_changes": [],
        "subject_names": [os.path.basename(f) for f in file_list],
        "subject_distances": {os.path.basename(f): [] for f in file_list}
    }

    for gw_iter in range(max_gw_iter):
        sprint(f"  [Groupwise Round {gw_iter+1}/{max_gw_iter}]")

        sprint(f"  [Round {gw_iter+1}] Pre-mean orientation re-check...")
        for i in range(1, N):
            aligned_meshes[i], tfi, flippedi = pole_flip_correction(
                aligned_meshes[i], reference_poly=aligned_meshes[0]
            )
            if flippedi:
                sprint(f"    NOTE: Mesh {i+1} reoriented before mean update")
                T_icp[i] = tfi @ T_icp[i]

        ref_mean = compute_mean_poly(aligned_meshes)

        pairwise_histories = []
        subj_pw_dict = {}
        for i in range(N):
            dT, p_hist = run_vtk_icp(aligned_meshes[i], ref_mean, return_history=True,
                                     max_iter=args.pairwise_iterations, tolerance=args.pairwise_tolerance,
                                     landmarks=args.pairwise_landmarks)
            aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], dT)
            T_icp[i] = dT @ T_icp[i]
            pairwise_histories.append(p_hist)
            bname = os.path.basename(file_list[i])
            subj_pw_dict[bname] = [float(v) for v in p_hist]

        if gw_iter == 0:
            avg_pairwise = np.mean(pairwise_histories, axis=0).tolist()
            gw_history["pairwise_iterations"] = EVAL_PAIRWISE_STEPS
            gw_history["mean_pairwise_distances"] = [float(v) for v in avg_pairwise]
            gw_history["subject_pairwise_distances"] = subj_pw_dict

        subj_dists = [icp_distance(aligned_meshes[i], ref_mean) for i in range(N)]
        current_mean_dist = sum(subj_dists) / N
        sprint(f"  [Round {gw_iter+1}] Mean ICP dist to template: {current_mean_dist:.6f}")

        dist_change = abs(prev_mean_dist - current_mean_dist) if prev_mean_dist != float("inf") else 0.0
        t_elapsed = round(time.time() - gw_start_time, 2)
        gw_history["rounds"].append(gw_iter + 1)
        gw_history["elapsed_times_sec"].append(t_elapsed)
        gw_history["mean_distances"].append(float(current_mean_dist))
        gw_history["dist_changes"].append(float(dist_change))
        for i, f in enumerate(file_list):
            bname = os.path.basename(f)
            gw_history["subject_distances"][bname].append(float(subj_dists[i]))

        if gw_iter > 0 and dist_change <= gw_tolerance:
            sprint(f"  --> Groupwise ICP CONVERGED at round {gw_iter+1} (change: {dist_change:.6f} <= {gw_tolerance})")
            break
        prev_mean_dist = current_mean_dist

    return T_icp, gw_history
```

### Line-by-line explanation:
- **Lines 426-433**: `step4_groupwise_icp` defines settings and initializes identity matrices `T_icp` for all models.
- **Lines 435-445**: Starts a timer and prepares a dictionary `gw_history` to log frame-by-frame data.
- **Lines 447-458**: Begins the groupwise loop (`max_gw_iter`). First, re-checks orientation against the reference to correct any drifting from previous loops.
- **Lines 459**: Generates the current iteration's mean template via `compute_mean_poly`.
- **Lines 461-471**: Pairwise loop: Matches each model to the mean template (`run_vtk_icp`). Updates the model, accumulates the transform in `T_icp`, and records pairwise histories.
- **Lines 473-477**: If it's the first loop, calculate the average pairwise convergence graph.
- **Lines 479-482**: Measures the distance from each model to the mean template, calculating the group's current mean distance.
- **Lines 483-491**: Calculates the difference (`dist_change`) from the previous loop. Logs all metrics to the history dictionary.
- **Lines 493-495**: Convergence check: If `dist_change` is less than `gw_tolerance`, the models are fully aligned. Breaks the loop early.
- **Lines 496-498**: Updates `prev_mean_dist` and returns the final `T_icp` matrices and history.

---

## Section 19: Pipeline Step 5 - Physical Scale Restoration

- **Input:** Final meshes, T_initial, T_icp
- **Output:** Final combined transformation matrices (`T_matrices`)

```python
def step5_global_bounding_box_normalize(aligned_meshes, T_initial, T_icp):
    sprint("Step 5: Global bounding-box normalization (preserving relative physical sizes)...")
    N = len(aligned_meshes)
    
    physical_aligned_meshes = []
    T_scale_back_list = []
    for i in range(N):
        s = T_initial[i][0, 0]
        T_scale_back = np.eye(4)
        T_scale_back[0, 0] = T_scale_back[1, 1] = T_scale_back[2, 2] = 1.0 / s
        T_scale_back_list.append(T_scale_back)
        m_phys = apply_poly_transform(aligned_meshes[i], T_scale_back)
        physical_aligned_meshes.append(m_phys)

    g_min = np.array([float('inf')] * 3)
    g_max = np.array([float('-inf')] * 3)
    for m in physical_aligned_meshes:
        b = m.GetBounds()
        g_min[0] = min(g_min[0], b[0]); g_max[0] = max(g_max[0], b[1])
        g_min[1] = min(g_min[1], b[2]); g_max[1] = max(g_max[1], b[3])
        g_min[2] = min(g_min[2], b[4]); g_max[2] = max(g_max[2], b[5])
        
    sprint(f"  Physical Union bounds: "
           f"X[{g_min[0]:+.4f},{g_max[0]:+.4f}]  "
           f"Y[{g_min[1]:+.4f},{g_max[1]:+.4f}]  "
           f"Z[{g_min[2]:+.4f},{g_max[2]:+.4f}]")

    g_center = (g_min + g_max) / 2.0
    g_half = (g_max - g_min) / 2.0
    max_half = float(g_half.max())
    global_scale = 1.0 / max_half if max_half > 0 else 1.0

    T_g_cent = np.eye(4)
    T_g_cent[:3, 3] = -g_center
    T_g_scale = np.eye(4)
    T_g_scale[0, 0] = T_g_scale[1, 1] = T_g_scale[2, 2] = global_scale
    T_global = T_g_scale @ T_g_cent
    sprint(f"  Global center = ({g_center[0]:+.4f}, {g_center[1]:+.4f}, {g_center[2]:+.4f})")
    sprint(f"  Global scale  = {global_scale:.6f}  "
           f"(max physical half-extent {max_half:.4f} -> 1.0)")

    for i in range(N):
        aligned_meshes[i] = apply_poly_transform(physical_aligned_meshes[i], T_global)

    v_min = np.array([float('inf')] * 3)
    v_max = np.array([float('-inf')] * 3)
    for m in aligned_meshes:
        b = m.GetBounds()
        v_min[0] = min(v_min[0], b[0]); v_max[0] = max(v_max[0], b[1])
        v_min[1] = min(v_min[1], b[2]); v_max[1] = max(v_max[1], b[3])
        v_min[2] = min(v_min[2], b[4]); v_max[2] = max(v_max[2], b[5])
    sprint(f"  Union bounds (after) : "
           f"X[{v_min[0]:+.4f},{v_max[0]:+.4f}]  "
           f"Y[{v_min[1]:+.4f},{v_max[1]:+.4f}]  "
           f"Z[{v_min[2]:+.4f},{v_max[2]:+.4f}]")

    T_matrices = [T_global @ T_scale_back_list[i] @ T_icp[i] @ T_initial[i] for i in range(N)]
    return T_matrices
```

### Line-by-line explanation:
- **Lines 500-512**: `step5_global_bounding_box_normalize` retrieves the initial scaling factor and creates a scale-back matrix (1.0 / s) to restore models to their real-world physical size.
- **Lines 514-525**: Computes a unified global bounding box that encapsulates all physical models.
- **Lines 527-539**: Calculates the global center and max half-width to generate a centering matrix (`T_global`). Shifts the entire group to (0,0,0) for optimal AI training input layout.
- **Lines 541-542**: Applies `T_global` to all meshes.
- **Lines 544-554**: Re-computes the bounding box to verify successful centering.
- **Lines 556-557**: Multiplies all transformations (Global * ScaleBack * ICP * Initial) into a definitive `T_matrices` list for each subject.

---

## Section 20: Pipeline Step 6 - Save Outputs

- **Input:** Final data
- **Output:** Saved files

```python
def step6_save_outputs(output_dir, file_list, aligned_meshes, T_matrices, gw_history, args):
    sprint("Step 6: Saving results...")
    N = len(file_list)
    np.save(os.path.join(output_dir, "T_matrices.npy"), np.array(T_matrices))
    sprint(f"  Saved T_matrices.npy ({N} matrices)")

    import json
    history_json_path = os.path.join(output_dir, "icp_convergence_history.json")
    with open(history_json_path, "w") as f:
        json.dump(gw_history, f, indent=2)
    sprint("  Saved icp_convergence_history.json")

    mean_poly = compute_mean_poly(aligned_meshes)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(os.path.join(output_dir, "mean_shape.ply"))
    writer.SetInputData(mean_poly)
    writer.Write()
    sprint("  Saved mean_shape.ply")

    export_aligned_nifti(file_list, T_matrices, output_dir,
                         spacing_mm=args.output_spacing,
                         n_voxels=args.output_voxels,
                         interpolation=args.interpolation)

    sprint(f"  All {N} aligned NIfTI saved to: {os.path.join(output_dir, 'aligned_nifti')}")
    sprint("--- ICP.py FINISHED ---")
```

### Line-by-line explanation:
- **Lines 559-563**: `step6_save_outputs` saves `T_matrices` to an `.npy` file.
- **Lines 565-569**: Dumps `gw_history` to a JSON file.
- **Lines 571-576**: Computes the final Mean Poly and exports it as `mean_shape.ply` using `vtkPLYWriter`.
- **Lines 578-584**: Calls `export_aligned_nifti` to resample and save the 3D volumes.

---

## Section 21: Main Function and Error Handler

- **Input:** None
- **Output:** Pipeline execution

```python
def main():
    sprint("--- ICP.py STARTING (rigid groupwise ICP) ---")
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir:
        sprint("No --input_dir given. Opening folder picker...")
        input_dir = prompt_folder("Select input folder containing NIfTI labels")
        if not input_dir:
            sprint("ERROR: No folder selected. Exiting.")
            return

    output_dir = args.output_dir
    if not output_dir:
        basename = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(SCRIPT_DIR, f"output_{basename}")
        sprint(f"No --output_dir given. Using default: {output_dir}")

    input_dir = input_dir.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    sprint(f"Input  dir: {input_dir}")
    sprint(f"Output dir: {output_dir}")

    clean_previous_outputs(output_dir)

    file_list = find_input_files(input_dir)
    sprint(f"Total files to process: {len(file_list)}")
    if not file_list:
        sprint("ERROR: No files found in input_dir.")
        return

    meshes, valid_files = step_load_and_mesh_all(file_list)
    N = len(meshes)
    if N == 0:
        sprint("ERROR: No valid label meshes found.")
        return
    file_list = valid_files
    sprint(f"  Loaded {N} meshes successfully.")

    aligned_meshes, T_initial = step1_per_mesh_normalize(meshes)
    T_pca_list = step2_pca_alignment(aligned_meshes)
    T_initial = step3_orientation_disambiguation(aligned_meshes, T_initial, T_pca_list)
    T_icp, gw_history = step4_groupwise_icp(aligned_meshes, file_list, args)
    T_matrices = step5_global_bounding_box_normalize(aligned_meshes, T_initial, T_icp)
    step6_save_outputs(output_dir, file_list, aligned_meshes, T_matrices, gw_history, args)

if __name__ == "__main__":
    import sys, os
    exit_code = 0
    with open(DEBUG_LOG, "w") as f:
        f.write(f"--- LOG START: {datetime.now()} ---\n")
    try:
        main()
    except Exception as e:
        import traceback
        sprint(f"FATAL ERROR: {str(e)}")
        with open(DEBUG_LOG, "a") as f:
            f.write(traceback.format_exc())
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        try:
            import time, threading
            def _force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            threading.Thread(target=_force_kill, daemon=True).start()
        except Exception:
            os._exit(exit_code)
```

### Line-by-line explanation:
- **Lines 586-591**: `main()` function checks if an input directory is provided.
- **Lines 592-596**: If not, opens a UI picker (`prompt_folder`). Exits if canceled.
- **Lines 598-608**: Auto-generates an output directory based on the input name, normalizes paths, and creates the folder.
- **Lines 610-616**: Cleans old files, searches for input files, and aborts if empty.
- **Lines 618-624**: Runs Step A, validates mesh counts.
- **Lines 626-631**: Sequentially executes Steps 1 through 6.
- **Lines 633-637**: `if __name__ == '__main__':` block initializes the log file with a startup timestamp.
- **Lines 638-645**: Uses `try` to execute `main()`. If an `Exception` occurs, it intercepts it, logs the full `traceback`, and sets `exit_code = 1`.
- **Lines 646-651**: The `finally` block ensures buffers are flushed, then attempts a clean exit via `slicer.util.exit`.
- **Lines 652-664**: Due to a known issue where Slicer headless sometimes hangs on exit, a daemon Thread is spawned to forcefully kill the process (`os._exit`) after 1.5 seconds if standard exit fails.

---

