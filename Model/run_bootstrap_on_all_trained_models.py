import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)

from evaluation import compute_bootstrap_stats

def run_all():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    datasets = [
        "Ds005602", 
        "All_Augment_tain", 
        "Ds004469",
        "Ds004469Train_Ds005602test",
        "Ds005602Train_Ds004469test"
    ]
    sides = ["left", "right"]
    
    models_config = [
        ("SVM", "SVM", "test_predictions.npz"),
        ("MLP", "MLP", "test_predictions.npz"),
        ("ResNet (Standard)", "ResNet", "test_predictions.npz"),
        ("ResNet (AutoEncoder)", "ResNet", "test_predictions_ae.npz"),
        ("MobileNet", "MobileNet", "test_predictions.npz"),
        ("SqueezeNet", "SqueezeNet", "test_predictions.npz"),
        ("PointNet", "PointNet", "test_predictions.npz"),
    ]
    
    results = []
    
    for ds in datasets:
        print(f"\n========================================================")
        print(f"  BOOTSTRAP 1,000 RUNS: DATASET = {ds}")
        print(f"========================================================")
        for side in sides:
            side_label = side.upper()
            print(f"\n--- Side: {side_label} ---")
            for m_name, subfolder, npz_name in models_config:
                npz_path = os.path.join(base_dir, ds, side, subfolder, "results", npz_name)
                if not os.path.exists(npz_path):
                    continue
                
                data = np.load(npz_path)
                y_true = data['y_test'] if 'y_test' in data else data['y_true']
                y_prob = data['y_prob']
                
                if 'y_pred' not in data:
                    raise ValueError(f'Missing saved decisions: {npz_path}')
                stats = compute_bootstrap_stats(y_true, y_prob, n_boot=1000, y_pred=data['y_pred'])
                stats.update({
                    "Dataset": ds,
                    "Side": side.capitalize(),
                    "Model": m_name,
                    "Test_N": len(y_true),
                    "Protocol": "legacy_predictions_unverified_cohort",
                    "Pos_N": int(np.sum(y_true == 1)),
                    "Neg_N": int(np.sum(y_true == 0))
                })
                results.append(stats)
                print(f"  [{m_name:<20}] Mean Acc: {stats['Mean_Acc']:<7} {stats['Acc_SD']:<9} (95% CI: {stats['Acc_95CI']:<20}) | Sens: {stats['Mean_Sens']:<7} | Spec: {stats['Mean_Spec']:<7} | F1: {stats['Mean_F1']:<7} | AUC: {stats['Mean_AUC']}")

    df_res = pd.DataFrame(results)
    
    # Save CSVs
    out_csv = os.path.join(base_dir, "bootstrap_1000_summary_corrected_legacy.csv")
    df_res.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\nSaved Bootstrap summary to: {out_csv}")
    
    # Also save to Model_Evaluation_and_Benchmarks
    bench_csv = os.path.abspath(os.path.join(base_dir, "..", "Model_Evaluation_and_Benchmarks", "results_csv", "bootstrap_1000_summary_corrected_legacy.csv"))
    os.makedirs(os.path.dirname(bench_csv), exist_ok=True)
    df_res.to_csv(bench_csv, index=False, encoding='utf-8')
    print(f"Saved to dedicated benchmark folder: {bench_csv}")

if __name__ == "__main__":
    run_all()
