import os
import glob
import json
import time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def main():
    print("[INFO] Starting SPHARM vs ICP 3D Comparison Data Extraction...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    icp_output_dir = os.path.join(repo_root, "ICP", "output_left_hippocampus")
    aligned_dir = os.path.join(icp_output_dir, "aligned_meshes")
    spharm_dir = os.path.join(icp_output_dir, "spharm_results")
    mean_poly_path = os.path.join(icp_output_dir, "mean_shape.ply")

    if not os.path.exists(aligned_dir) or not os.path.exists(spharm_dir):
        print(f"[ERROR] Required directories not found: {aligned_dir} or {spharm_dir}")
        return

    # Find matching pairs
    spharm_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM.vtk")))
    valid_pairs = []
    for sf in spharm_files:
        bn = os.path.basename(sf).replace("_SPHARM.vtk", "")
        icp_f = os.path.join(aligned_dir, f"{bn}.vtk")
        if os.path.exists(icp_f):
            valid_pairs.append((bn, icp_f, sf))

    N = len(valid_pairs)
    print(f"[INFO] Found {N} completed SPHARM vs ICP matching subject pairs.")

    # 1. Process Mean Template if available
    mean_template_data = None
    if os.path.exists(mean_poly_path):
        print("[INFO] Processing Mean Shape Template...")
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mean_poly_path)
        ply_reader.Update()
        m_poly = ply_reader.GetOutput()
        
        deci = vtk.vtkQuadricDecimation()
        deci.SetInputData(m_poly)
        deci.SetTargetReduction(0.70)
        deci.Update()
        m_poly_dec = deci.GetOutput()
        
        m_pts = vtk_to_numpy(m_poly_dec.GetPoints().GetData())
        m_faces = []
        cells = m_poly_dec.GetPolys()
        id_list = vtk.vtkIdList()
        cells.InitTraversal()
        while cells.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:
                m_faces.extend([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
                
        mean_template_data = {
            "vertices": [round(float(v), 4) for v in m_pts.flatten()],
            "faces": m_faces,
            "bounds": [round(float(v), 4) for v in m_poly.GetBounds()]
        }

    # 2. Process each subject pair
    subjects = []
    mean_errors = []
    max_errors = []
    t0 = time.time()

    for i, (bname, icp_f, spharm_f) in enumerate(valid_pairs):
        clean_name = bname.replace("_aligned", "")
        group = "Healthy" if "Healthy" in clean_name or "healthy" in clean_name else "TLE"
        
        parts = clean_name.split("_")
        subj_id = clean_name
        for p in parts:
            if p.startswith("sub-"):
                subj_id = p
                break
        short_name = f"{subj_id} ({group})"

        # Load SPHARM mesh
        r_spharm = vtk.vtkPolyDataReader()
        r_spharm.SetFileName(spharm_f)
        r_spharm.Update()
        poly_spharm = r_spharm.GetOutput()

        # Load ICP mesh
        r_icp = vtk.vtkPolyDataReader()
        r_icp.SetFileName(icp_f)
        r_icp.Update()
        poly_icp = r_icp.GetOutput()

        # Decimate ICP mesh for web rendering (~400-500 points)
        deci = vtk.vtkQuadricDecimation()
        deci.SetInputData(poly_icp)
        deci.SetTargetReduction(0.85)
        deci.Update()
        poly_icp_dec = deci.GetOutput()

        # Compute point-to-surface distance (SPHARM error vs original ICP ground truth)
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(poly_icp)
        locator.BuildLocator()

        pts_spharm = vtk_to_numpy(poly_spharm.GetPoints().GetData())
        n_spharm = len(pts_spharm)
        dists = np.zeros(n_spharm, dtype=np.float32)

        for j in range(n_spharm):
            pt = pts_spharm[j].tolist()
            cid = locator.FindClosestPoint(pt)
            cp = poly_icp.GetPoint(cid)
            dists[j] = np.linalg.norm(pts_spharm[j] - np.array(cp))

        mean_err = float(dists.mean())
        max_err = float(dists.max())
        p95_err = float(np.percentile(dists, 95))
        mean_errors.append(mean_err)
        max_errors.append(max_err)

        # Extract SPHARM faces
        faces_spharm = []
        cells_s = poly_spharm.GetPolys()
        id_list = vtk.vtkIdList()
        cells_s.InitTraversal()
        while cells_s.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:
                faces_spharm.extend([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])

        # Extract ICP faces & vertices
        pts_icp = vtk_to_numpy(poly_icp_dec.GetPoints().GetData())
        faces_icp = []
        cells_i = poly_icp_dec.GetPolys()
        cells_i.InitTraversal()
        while cells_i.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:
                faces_icp.extend([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])

        c_spharm = pts_spharm.mean(axis=0).tolist()
        b_spharm = [float(v) for v in poly_spharm.GetBounds()]
        b_icp = [float(v) for v in poly_icp.GetBounds()]

        # Normalize dists for color map (0.0 to 1.0, where 0.05 is max heat)
        dists_norm = np.clip(dists / 0.05, 0.0, 1.0)

        subjects.append({
            "id": i,
            "name": clean_name,
            "short_name": short_name,
            "group": group,
            "centroid": [round(v, 4) for v in c_spharm],
            "b_spharm": [round(v, 4) for v in b_spharm],
            "b_icp": [round(v, 4) for v in b_icp],
            "v_spharm": [round(float(v), 4) for v in pts_spharm.flatten()],
            "f_spharm": faces_spharm,
            "v_icp": [round(float(v), 4) for v in pts_icp.flatten()],
            "f_icp": faces_icp,
            "errors": [round(float(v), 3) for v in dists_norm.tolist()],
            "mean_err": round(mean_err, 4),
            "max_err": round(max_err, 4),
            "p95_err": round(p95_err, 4)
        })

        if (i + 1) % 25 == 0 or (i + 1) == N:
            print(f"  Processed [{i+1}/{N}] SPHARM pairs... ({time.time()-t0:.1f}s)")

    stats = {
        "total_pairs": N,
        "healthy_count": sum(1 for s in subjects if s["group"] == "Healthy"),
        "tle_count": sum(1 for s in subjects if s["group"] == "TLE"),
        "overall_mean_error": round(float(np.mean(mean_errors)), 4),
        "overall_max_error": round(float(np.mean(max_errors)), 4)
    }

    print(f"\n[INFO] SPHARM vs ICP Extraction Finished!")
    print(f"  Overall Average Surface Error: {stats['overall_mean_error']} (normalized space, ~1 voxel)")
    print(f"  Processed {N} subjects.")

    payload = {
        "stats": stats,
        "mean_template": mean_template_data,
        "subjects": subjects
    }

    out_js_path = os.path.join(os.path.dirname(__file__), "spharm_comparison_data.js")
    print(f"[INFO] Saving payload to: {out_js_path}")
    with open(out_js_path, "w", encoding="utf-8") as f:
        f.write("window.SPHARM_COMPARISON_DATA = ")
        json.dump(payload, f)
        f.write(";\n")

    size_mb = os.path.getsize(out_js_path) / (1024 * 1024)
    print(f"[SUCCESS] Wrote spharm_comparison_data.js ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
