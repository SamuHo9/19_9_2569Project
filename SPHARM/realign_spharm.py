import os
import sys
import glob
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def popup_select_directory(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title=title, initialdir=os.getcwd())
    root.destroy()
    return path

def load_polydata(filepath):
    r = vtk.vtkPolyDataReader()
    r.SetFileName(filepath)
    r.Update()
    return r.GetOutput()

def write_polydata(poly, filepath):
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(filepath)
    w.SetInputData(poly)
    w.SetFileTypeToBinary()
    w.Write()

def points_to_numpy(poly):
    if poly is None or poly.GetPoints() is None:
        return np.zeros((0, 3), dtype=np.float64)
    return vtk_to_numpy(poly.GetPoints().GetData())

def replace_points(poly, new_pts):
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    vtk_pts = out.GetPoints()
    for i, p in enumerate(new_pts):
        vtk_pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
    return out

def kabsch_proper(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t

LABEL_PERMS = [
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
]

def best_kabsch_with_flips(lm, mean_lm):
    best = None
    for perm in LABEL_PERMS:
        lm_p = lm[list(perm)]
        R, t = kabsch_proper(lm_p, mean_lm)
        aligned = (R @ lm_p.T).T + t
        residual = float(np.linalg.norm(aligned - mean_lm, axis=1).sum())
        if best is None or residual < best[3]:
            best = (R, t, perm, residual)
    return best

def gpa_landmarks(all_landmarks, max_iter=20, tol=1e-6):
    N = len(all_landmarks)
    K = all_landmarks[0].shape[0]

    mean_lm = all_landmarks[0] - all_landmarks[0].mean(axis=0)

    history = []
    perms = [(0, 1, 2, 3)] * N
    Rs = [np.eye(3)] * N
    ts = [np.zeros(3)] * N

    for it in range(max_iter):
        aligned = []
        new_perms = []
        for i, lm in enumerate(all_landmarks):
            R, t, perm, _ = best_kabsch_with_flips(lm, mean_lm)
            lm_p = lm[list(perm)]
            aligned_lm = (R @ lm_p.T).T + t
            aligned.append(aligned_lm)
            Rs[i] = R
            ts[i] = t
            new_perms.append(perm)

        new_mean = np.mean(aligned, axis=0)
        diff = float(np.linalg.norm(new_mean - mean_lm))
        history.append(diff)
        perms = new_perms
        if diff < tol:
            mean_lm = new_mean
            break
        mean_lm = new_mean

    return Rs, ts, mean_lm, perms, history

def find_anatomical_landmarks(pts):
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    long_axis = Vt[0]
    long_axis /= np.linalg.norm(long_axis)

    proj = pts_c @ long_axis

    p90 = np.percentile(proj, 90)
    p10 = np.percentile(proj, 10)
    top_pts = pts_c[proj > p90]
    bot_pts = pts_c[proj < p10]

    def spread_perpendicular(subset, axis):
        perp = subset - np.outer(subset @ axis, axis)
        return float(np.std(perp, axis=0).sum())

    top_spread = spread_perpendicular(top_pts, long_axis) if len(top_pts) > 0 else 0.0
    bot_spread = spread_perpendicular(bot_pts, long_axis) if len(bot_pts) > 0 else 0.0

    if top_spread >= bot_spread:
        head_idx = int(np.argmax(proj))
        tail_idx = int(np.argmin(proj))
    else:
        head_idx = int(np.argmin(proj))
        tail_idx = int(np.argmax(proj))

    middle_mask = (proj > np.percentile(proj, 25)) & (proj < np.percentile(proj, 75))
    middle_pts = pts_c[middle_mask]
    if len(middle_pts) == 0:
        middle_pts = pts_c

    middle_perp = middle_pts - np.outer(middle_pts @ long_axis, long_axis)
    curl_axis = middle_perp.mean(axis=0)
    norm = np.linalg.norm(curl_axis)
    if norm > 1e-9:
        curl_axis = curl_axis / norm
    else:
        pc2 = Vt[1]
        curl_axis = pc2 - (pc2 @ long_axis) * long_axis
        curl_axis /= np.linalg.norm(curl_axis)

    proj_curl = pts_c @ curl_axis
    lateral_idx = int(np.argmax(proj_curl))
    medial_idx = int(np.argmin(proj_curl))

    return head_idx, tail_idx, lateral_idx, medial_idx

CANONICAL_4PTS = np.array([
    [0.0, 0.0, +1.0],
    [0.0, 0.0, -1.0],
    [+1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
])

def main():
    print("=" * 72)
    print("--- 4-Point Anatomical Template Re-alignment ---")
    print("=" * 72)

    parser = argparse.ArgumentParser()
    parser.add_argument("--spharm_dir", default=None,
                        help="Path to spharm_results folder")
    parser.add_argument("--template", default=None,
                        help="Path to official reference template VTK")
    args = parser.parse_args()

    folder = args.spharm_dir
    if not folder:
        print("\nSelect 'spharm_results' folder...")
        folder = popup_select_directory("Select spharm_results folder")
        if not folder:
            print("Canceled.")
            return

    if not os.path.isdir(folder):
        print(f"[ERROR] Not a folder: {folder}")
        sys.exit(1)

    spharm_sub = os.path.join(folder, "spharm_results")
    if os.path.isdir(spharm_sub) and glob.glob(os.path.join(spharm_sub, "*.vtk")):
        folder = spharm_sub

    old_aligned_files = glob.glob(os.path.join(folder, "*_SPHARM_realigned.vtk"))
    if old_aligned_files:
        print(f"Cleaning up {len(old_aligned_files)} old realigned files...")
        for f in old_aligned_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"  Failed to delete {os.path.basename(f)}: {e}")

    candidate_files = sorted(glob.glob(os.path.join(folder, "*_SPHARM_procalign.vtk")))
    source = "_SPHARM_procalign.vtk"

    if not candidate_files:
        all_spharm = sorted(glob.glob(os.path.join(folder, "*_SPHARM.vtk")))
        candidate_files = [f for f in all_spharm
                           if not any(s in os.path.basename(f)
                                      for s in ("_ellalign", "_grid", "_realigned", "_procalign"))]
        source = "_SPHARM.vtk"

    if not candidate_files:
        candidate_files = sorted(glob.glob(os.path.join(folder, "*_SPHARM_ellalign.vtk")))
        source = "_SPHARM_ellalign.vtk"

    if not candidate_files:
        all_vtk = sorted(glob.glob(os.path.join(folder, "*.vtk")))
        candidate_files = [f for f in all_vtk
                           if not any(s in os.path.basename(f)
                                      for s in ("_realigned.vtk",))]
        source = ".vtk"

    if not candidate_files:
        print(f"[ERROR] No SPHARM .vtk found in {folder}")
        sys.exit(1)

    filtered_files = {}
    for f in candidate_files:
        basename = os.path.basename(f)
        name = basename
        for suffix in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        if name.endswith("_aligned"):
            subj_key = name[:-len("_aligned")]
            is_aligned = True
        else:
            subj_key = name
            is_aligned = False
            
        if subj_key not in filtered_files:
            filtered_files[subj_key] = (f, is_aligned)
        else:
            stored_file, stored_is_aligned = filtered_files[subj_key]
            if is_aligned and not stored_is_aligned:
                filtered_files[subj_key] = (f, True)
                
    files = sorted([f for f, _ in filtered_files.values()])

    print(f"\nSource: {source}")
    print(f"Found {len(files)} meshes\n")

    subjects = []
    skipped = 0
    for f in files:
        poly = load_polydata(f)
        if poly is None or poly.GetNumberOfPoints() < 10:
            print(f"  SKIP: {os.path.basename(f)} (empty)")
            skipped += 1
            continue
        pts = points_to_numpy(poly)
        subjects.append({
            "file": f,
            "poly": poly,
            "pts": pts,
        })

    if not subjects:
        raise RuntimeError("No valid subjects to align")

    # Resolve official template mesh instead of picking first subject
    template_file = args.template
    if not template_file:
        folder_lower = os.path.abspath(folder).lower()
        side = None
        if "left" in folder_lower or "lh" in folder_lower:
            side = "left"
        elif "right" in folder_lower or "rh" in folder_lower:
            side = "right"
        else:
            for s in subjects:
                fname = os.path.basename(s["file"]).lower()
                if fname.startswith("lh_") or "left" in fname:
                    side = "left"
                    break
                elif fname.startswith("rh_") or "right" in fname:
                    side = "right"
                    break

        if side:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.abspath(os.path.join(script_dir, "..", "Templates", "SPHARM", f"template_spharm_{side}.vtk"))
            if os.path.isfile(cand):
                template_file = cand

    if template_file and os.path.isfile(template_file):
        print(f"Loading official reference template: {template_file}")
        tmpl_poly = load_polydata(template_file)
        ref_pts = points_to_numpy(tmpl_poly)
    else:
        raise FileNotFoundError("Reference template not found; pass --template")

    h_idx, t_idx, y_idx, g_idx = 470, 276, 0, 272
    if ref_pts.shape[0] <= max(h_idx, t_idx, y_idx, g_idx):
        print(f"[WARNING] SPHARM template mesh has {ref_pts.shape[0]} points (expected >= 1002 points for subdiv level 10).")
        raise ValueError("Template has insufficient vertices for the configured landmarks")

    print(f"Phase 1: Aligning official reference template to canonical orientation...")
    lm_ref = ref_pts[[h_idx, t_idx, y_idx, g_idx]]
    size_ref = float(np.linalg.norm(lm_ref - lm_ref.mean(axis=0), axis=1).mean())
    R_ref, t_ref = kabsch_proper(lm_ref, CANONICAL_4PTS * size_ref)
    aligned_ref_pts = (R_ref @ ref_pts.T).T + t_ref

    print(f"Phase 2: Re-aligning all subjects to canonical template using full-topology Kabsch alignment...")
    print(f"\n{'Subject':<48} {'resid':>10}")
    print("-" * 60)

    final_landmarks = []
    for s in subjects:
        pts = s["pts"]
        name = os.path.basename(s["file"])
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            name = name.replace(suf, "")

        if pts.shape[0] != ref_pts.shape[0]:
            print(f"  SKIP: {name} (point count {pts.shape[0]} != template {ref_pts.shape[0]})")
            continue

        R, t = kabsch_proper(pts, aligned_ref_pts)
        new_pts = (R @ pts.T).T + t

        new_lm = new_pts[[h_idx, t_idx, y_idx, g_idx]]
        target_ref_lm = aligned_ref_pts[[h_idx, t_idx, y_idx, g_idx]]
        residual = float(np.linalg.norm(new_lm - target_ref_lm, axis=1).mean())
        final_landmarks.append(new_lm)

        out_poly = replace_points(s["poly"], new_pts)
        base = s["file"]
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        
        out_path_realigned = base + "_SPHARM_realigned.vtk"
        write_polydata(out_poly, out_path_realigned)

        name = os.path.basename(s["file"])
        for suf in ("_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            name = name.replace(suf, "")
        print(f"  {name:<46} {residual:>10.5f}")

    if len(final_landmarks) > 0:
        final_landmarks = np.array(final_landmarks)
        print()
        print("=" * 72)
        print(f"Done. Re-aligned {len(final_landmarks)} subjects using 4-point anatomical alignment.")
        print(f"\nLandmark clustering quality after realignment:")
        for k, name in enumerate(["HEAD (RED)", "TAIL (BLUE)", "LAT (GREEN)", "MED (YELLOW)"]):
            cluster_pts = final_landmarks[:, k, :]
            spread = float(np.linalg.norm(cluster_pts - cluster_pts.mean(axis=0),
                                          axis=1).mean())
            print(f"    {name}: mean distance from cluster center = {spread:.4f}")
        print(f"\nOutput saved as *_SPHARM_realigned.vtk in {folder}")
        print("=" * 72)
    else:
        raise RuntimeError("No subjects were realigned; check point counts/subdivision levels")
    if skipped or len(final_landmarks) != len(files):
        raise RuntimeError(f"Realignment incomplete: {len(final_landmarks)}/{len(files)} succeeded")

if __name__ == "__main__":
    main()
