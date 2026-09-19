import os
import sys
import glob
import csv
import argparse
import numpy as np
import vtk
from sklearn.cross_decomposition import PLSRegression

def prompt_folder(title):
    try:
        import qt
        folder = qt.QFileDialog.getExistingDirectory(None, title)
        return folder if folder else None
    except ImportError:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
        root.destroy()
        return folder if folder else None

def load_vtk_points(filepath):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    if not points:
        return None, None
        
    num_points = points.GetNumberOfPoints()
    coords = []
    for i in range(num_points):
        coords.append(list(points.GetPoint(i)))
        
    return np.array(coords), polydata

def save_vtk_points(coords, template_polydata, filepath):
    new_poly = vtk.vtkPolyData()
    new_poly.DeepCopy(template_polydata)
    
    points = vtk.vtkPoints()
    for pt in coords:
        points.InsertNextPoint(pt)
        
    new_poly.SetPoints(points)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(new_poly)
    writer.Write()

def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
    return "Unknown", -1

def main():
    print("="*60)
    print("--- PLS-DA Local Interpolation Augmentation (XYZ Subdivision) ---")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--num_per_pair", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)

    num_per_pair = args.num_per_pair
    if num_per_pair is None:
        while True:
            try:
                user_input = input("Enter the number of children to generate per parent pair (e.g. 8): ")
                num_per_pair = int(user_input)
                if num_per_pair <= 0:
                    print("Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input.")

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        print("No --output_dir given. Opening folder picker...")
        chosen = prompt_folder("Select output folder (containing spharm_results/)")
        if not chosen:
            print("ERROR: No folder selected. Exiting.")
            return
        if os.path.basename(chosen.rstrip("\\/")) == "spharm_results":
            output_root = os.path.abspath(os.path.dirname(chosen))
        else:
            output_root = os.path.abspath(chosen)

    if os.path.isdir(os.path.join(output_root, "spharm_results")):
        data_dir = os.path.join(output_root, "spharm_results")
    else:
        data_dir = output_root

    # Find _SPHARM.vtk files
    all_vtk_files = sorted(glob.glob(os.path.join(data_dir, "*_SPHARM.vtk")))
    # Exclude _ellalign, _grid, etc.
    vtk_files = [f for f in all_vtk_files if not any(s in os.path.basename(f) for s in ("_ellalign", "_grid", "_realigned", "_procalign"))]

    if not vtk_files:
        print(f"ERROR: No Subdivision meshes (*_SPHARM.vtk) found in: {data_dir}")
        return

    subject_names = []
    xyz_vectors = []
    groups = []
    classes = []
    template_poly = None
    expected_pts = None

    # First pass to find expected number of points and template
    for fpath in vtk_files:
        coords, poly = load_vtk_points(fpath)
        if coords is not None and len(coords) > 0:
            expected_pts = len(coords)
            template_poly = poly
            break

    if expected_pts is None:
        print("ERROR: No valid VTK files found.")
        return

    print(f"Found {len(vtk_files)} valid VTK files. Expected points per mesh: {expected_pts}")

    # Second pass
    for fpath in vtk_files:
        basename = os.path.basename(fpath)
        coords, _ = load_vtk_points(fpath)
        
        if coords is None or len(coords) != expected_pts:
            continue
            
        flat_coords = coords.flatten()
        xyz_vectors.append(flat_coords)
        
        subj_name = basename.replace("_SPHARM.vtk", "")
        subject_names.append(subj_name)
        g_name, cls = classify_subject(basename)
        if cls < 0:
            raise ValueError(f'Unknown label: {basename}; refusing to synthesize a Healthy label')
        groups.append(g_name)
        classes.append(cls)

    xyz_vectors = np.array(xyz_vectors)
    N, D = xyz_vectors.shape
    print(f"Loaded {N} subjects. XYZ vector size (Flattened): {D}")

    binary_labels = np.array([1 if cls == 1 else 0 for cls in classes])
    Y = np.zeros((N, 2))
    for i, label in enumerate(binary_labels):
        Y[i, label] = 1.0

    n_comp = min(args.n_components, N - 1)
    print(f"Fitting PLS-DA model with {n_comp} components on XYZ coordinates...")
    pls = PLSRegression(n_components=n_comp, scale=True)
    X_scores, _ = pls.fit_transform(xyz_vectors, Y)

    aug_dir = os.path.join(output_root, "plsda_interpolated_surfaces")
    os.makedirs(aug_dir, exist_ok=True)

    metadata_csv_path = os.path.join(aug_dir, "interpolated_metadata.csv")
    features_csv_path = os.path.join(aug_dir, "augmented_xyz_features.csv")
    metadata_rows = []
    features_rows = []

    pool = list(range(N))

    def form_pairs(pool):
        formed = []
        temp_pool = list(pool)
        while len(temp_pool) >= 2:
            idx_A = np.random.choice(temp_pool)
            temp_pool.remove(idx_A)
            
            distances = [np.linalg.norm(X_scores[idx_A] - X_scores[cand]) for cand in temp_pool]
            min_idx = np.argmin(distances)
            idx_B = temp_pool[min_idx]
            temp_pool.remove(idx_B)
            
            formed.append((idx_A, idx_B))
        return formed

    # Never interpolate opposite target classes and assign an arbitrary label.
    all_pairs = []
    for label in (0, 1):
        all_pairs.extend(form_pairs([i for i in pool if binary_labels[i] == label]))
    total_pairs = len(all_pairs)
    num_augmented = total_pairs * num_per_pair
    print(f"Generating {num_per_pair} children per pair. Total output: {num_augmented} meshes.")

    global_idx = 1
    for pair_idx, (idx_A, idx_B) in enumerate(all_pairs):
        alphas = np.linspace(0.1, 0.9, num_per_pair)
        if num_per_pair > 1:
            jitter = np.random.uniform(-0.02, 0.02, num_per_pair)
            alphas = np.clip(alphas + jitter, 0.05, 0.95)

        for child_idx, alpha in enumerate(alphas):
            score_A = X_scores[idx_A]
            score_B = X_scores[idx_B]
            score_new = (1 - alpha) * score_A + alpha * score_B

            if alpha < 0.5:
                assigned_class = binary_labels[idx_A]
                assigned_group = groups[idx_A]
                assigned_parent = subject_names[idx_A]
            else:
                assigned_class = binary_labels[idx_B]
                assigned_group = groups[idx_B]
                assigned_parent = subject_names[idx_B]

            dir_name = "Diseased" if assigned_class == 1 else "Healthy"
            
            parent_name = subject_names[idx_A]
            parent_name_upper = parent_name.upper()
            is_left = parent_name.startswith("left_") or parent_name.startswith("lh_") or "_lh" in parent_name.lower() or "left" in parent_name.lower()
            
            if parent_name.startswith("lh_") or "_lh_" in parent_name.lower():
                side_str = "lh_"
            elif parent_name.startswith("rh_") or "_rh_" in parent_name.lower():
                side_str = "rh_"
            else:
                side_str = "left_" if is_left else "right_"
            
            if assigned_class == 0:
                class_prefix = "Normal" if "NORMAL" in parent_name_upper else "Healthy"
            else:
                if "LEFT-TLE" in parent_name_upper or "RIGHT-TLE" in parent_name_upper:
                    class_prefix = "Left-TLE" if is_left else "Right-TLE"
                else:
                    class_prefix = "TLE"

            # Reconstruct XYZ features (1 x D)
            flat_recon = pls.inverse_transform(score_new.reshape(1, -1))[0]
            coords_recon = flat_recon.reshape(expected_pts, 3)

            filename = f"{side_str}{class_prefix}_interp_{global_idx:04d}_SPHARM.vtk"
            filepath = os.path.join(aug_dir, filename)
            
            save_vtk_points(coords_recon, template_poly, filepath)

            # Log metadata row
            metadata_rows.append([
                filename,
                dir_name,
                subject_names[idx_A],
                groups[idx_A],
                subject_names[idx_B],
                groups[idx_B],
                f"{alpha:.4f}",
                assigned_parent,
                assigned_group
            ])
            
            # Log feature row for CSV
            features_rows.append([filename, assigned_group, assigned_class, dir_name] + flat_recon.tolist())

            if global_idx % 20 == 0 or global_idx == num_augmented:
                print(f"  Progress: {global_idx}/{num_augmented} meshes generated")
            global_idx += 1

    with open(metadata_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "DirectoryClass", "Parent_A_Subject", "Parent_A_Group",
            "Parent_B_Subject", "Parent_B_Group", "Alpha", "Closest_Parent", "Assigned_Group"
        ])
        writer.writerows(metadata_rows)
        
    with open(features_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Subject", "Group", "Class", "BinaryClass"] + [f"f_{i}" for i in range(D)]
        writer.writerow(header)
        writer.writerows(features_rows)

    print("\n" + "="*60)
    print("Interpolation Augmentation Complete (Subdivision XYZ)!")
    print(f"Metadata file saved to:     {metadata_csv_path}")
    print(f"Features file saved to:     {features_csv_path}")
    print(f"Augmented meshes saved to:  {os.path.abspath(aug_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()
