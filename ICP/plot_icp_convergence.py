import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="Plot ICP convergence time-series")
    parser.add_argument("--output_dir", type=str, help="Output directory containing icp_convergence_history.json")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    args, _ = parser.parse_known_args()
    return args


def load_convergence_data(json_path):
    if not os.path.exists(json_path):
        print(f"[ERROR] Convergence log not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read convergence log: {e}")
        return None

    rounds = data.get("rounds", [])
    mean_dists = data.get("mean_distances", [])
    if not rounds or not mean_dists:
        print("[ERROR] Convergence log is empty.")
        return None

    return data


def generate_convergence_plot(data, output_plot):
    rounds = data.get("rounds", [])
    mean_dists = data.get("mean_distances", [])
    subject_dists = data.get("subject_distances", {})

    pw_iters = data.get("pairwise_iterations", [])
    pw_dists = data.get("mean_pairwise_distances", [])
    subj_pw_dists = data.get("subject_pairwise_distances", {})

    has_pairwise = bool(pw_iters) and bool(pw_dists)
    target_dict = subj_pw_dists if (has_pairwise and subj_pw_dists) else subject_dists
    target_x = pw_iters if (has_pairwise and subj_pw_dists) else rounds
    target_xlabel = "VTK Pairwise Iteration" if (has_pairwise and subj_pw_dists) else "Groupwise Round"

    subjs = [item for item in target_dict.items() if len(item[1]) == len(target_x)]
    num_subjs = len(subjs)
    mean_overlay = pw_dists if (has_pairwise and subj_pw_dists) else mean_dists

    fig, ax = plt.subplots(figsize=(7.5, 6))

    if subjs:
        subj_matrix = np.array([dists for _, dists in subjs])
        std_arr = np.std(subj_matrix, axis=0, ddof=1)
        mean_arr = np.array(mean_overlay) if len(mean_overlay) == len(target_x) else np.mean(subj_matrix, axis=0)

        ci_lower = np.maximum(0, mean_arr - 1.96 * std_arr)
        ci_upper = mean_arr + 1.96 * std_arr

        ax.fill_between(target_x, ci_lower, ci_upper, color='royalblue', alpha=0.22, label='95% Population Band (Mean ± 1.96 SD)')

    ax.plot(target_x, mean_overlay, color='black', linestyle='--', marker='o', linewidth=3.2, markersize=6, label='Group Mean Distance')
    ax.set_xlabel(target_xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Pairwise ICP Distance to Template', fontsize=12, fontweight='bold')
    ax.set_title(f'ICP Convergence Across VTK Iterations ({num_subjs} Meshes)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(target_x)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close(fig)
    print(f"[OK] Time-series convergence plot saved to: {output_plot}")


def show_plot_window(output_plot):
    try:
        print("[INFO] Displaying convergence plot window...")
        plt.show()
    except Exception as e:
        print(f"[NOTE] Could not open matplotlib window: {e}")
        try:
            if sys.platform == "win32":
                os.startfile(output_plot)
        except Exception:
            pass


def main():
    args = parse_args()

    output_root = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(SCRIPT_DIR, "output")
    json_path = os.path.join(output_root, "icp_convergence_history.json")
    output_plot = os.path.join(output_root, "icp_convergence.png")

    print(f"--- Plotting ICP Convergence Time-Series for {os.path.basename(output_root)} ---")

    data = load_convergence_data(json_path)
    if data is None:
        return

    generate_convergence_plot(data, output_plot)

    if args.show:
        show_plot_window(output_plot)


if __name__ == "__main__":
    main()
