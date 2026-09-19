"""
================================================================================
Organize Model Performance Plots for Presentation / Academic Submission
================================================================================
Copies and systematically organizes evaluation plots from Model directories
into Model_Results_Excel/07_Detailed_Model_Plots/ for presentation to advisors.

Plot types included:
- Confusion Matrix (Normalized & Raw)
- ROC Curves (Standard, CI band, individual iterations)
- PLS-DA Components Comparison
- Training Loss Curves
================================================================================
"""

import os
import shutil
import glob

def organize_plots():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_root = os.path.join(repo_root, "Model")
    target_base = os.path.join(repo_root, "Model_Results_Excel", "07_Detailed_Model_Plots")
    os.makedirs(target_base, exist_ok=True)
    
    datasets = [
        "Ds005602",
        "All_Augment_tain",
        "Ds004469",
        "Ds005602Train_Ds004469test",
        "Ds004469Train_Ds005602test",
        "combind_methode"
    ]
    
    sides = ["left", "right"]
    
    models = ["MLP", "MobileNet", "PointNet", "ResNet", "SqueezeNet", "SVM"]
    
    copied_count = 0
    records = []
    
    print("=" * 70)
    print("ORGANIZING MODEL EVALUATION PLOTS INTO Model_Results_Excel")
    print(f"Destination: {target_base}")
    print("=" * 70)
    
    for ds in datasets:
        for side in sides:
            for m in models:
                source_plots_dir = os.path.join(model_root, ds, side, m, "plots")
                if not os.path.exists(source_plots_dir):
                    continue
                    
                # Standard destination
                dest_dir = os.path.join(target_base, ds, side, m)
                os.makedirs(dest_dir, exist_ok=True)
                
                files = os.listdir(source_plots_dir)
                for f in files:
                    if not f.endswith(".png"):
                        continue
                        
                    src_f = os.path.join(source_plots_dir, f)
                    
                    # Special handling for ResNet+AE
                    if m == "ResNet" and "_ae" in f:
                        dest_ae = os.path.join(target_base, ds, side, "ResNet+AE")
                        os.makedirs(dest_ae, exist_ok=True)
                        shutil.copy2(src_f, os.path.join(dest_ae, f))
                        copied_count += 1
                        records.append((ds, side, "ResNet+AE", f, os.path.join(dest_ae, f)))
                    else:
                        shutil.copy2(src_f, os.path.join(dest_dir, f))
                        copied_count += 1
                        records.append((ds, side, m, f, os.path.join(dest_dir, f)))
                        
                # If ResNet has pls_components_comparison.png, copy to ResNet+AE as well
                if m == "ResNet":
                    pls_src = os.path.join(source_plots_dir, "pls_components_comparison.png")
                    dest_ae = os.path.join(target_base, ds, side, "ResNet+AE")
                    if os.path.exists(pls_src) and os.path.exists(dest_ae):
                        dest_pls = os.path.join(dest_ae, "pls_components_comparison.png")
                        if not os.path.exists(dest_pls):
                            shutil.copy2(pls_src, dest_pls)
                            copied_count += 1

    print(f"\nSuccessfully copied {copied_count} plot files into 07_Detailed_Model_Plots!")
    
    # Generate a comprehensive README markdown index
    readme_path = os.path.join(target_base, "README_PLOTS_INDEX.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Plots Index (ดัชนีรวมภาพกราฟประสิทธิภาพโมเดล)\n\n")
        f.write("โฟลเดอร์นี้รวบรวมไฟล์รูปภาพกราฟผลการประเมินประสิทธิภาพของแต่ละโมเดล (Confusion Matrix, ROC Curve, PLS Comparison, Loss Curve) ")
        f.write("แยกตาม Dataset, ด้านของสมอง (Left/Right) และโครงสร้างโมเดล เพื่อความสะดวกในการเตรียมนำเสนออาจารย์ที่ปรึกษาครับ\n\n")
        f.write("## โครงสร้างโฟลเดอร์ (Folder Hierarchy)\n")
        f.write("```\n")
        f.write("07_Detailed_Model_Plots/\n")
        f.write("├── Ds005602/                     # Dataset หลัก (Primary Clinical Dataset)\n")
        f.write("│   ├── left/ & right/            # แยก Left / Right Hippocampus\n")
        f.write("│   │   ├── MLP/\n")
        f.write("│   │   ├── MobileNet/\n")
        f.write("│   │   ├── PointNet/\n")
        f.write("│   │   ├── ResNet/               # โมเดลหลักที่ใช้ทำ Grad-CAM Heatmap\n")
        f.write("│   │   ├── ResNet+AE/            # ResNet พร้อม AutoEncoder Reconstruction\n")
        f.write("│   │   ├── SqueezeNet/\n")
        f.write("│   │   └── SVM/\n")
        f.write("├── All_Augment_tain/             # Dataset ที่ผ่านการ Augmentation (PLS-DA Balanced)\n")
        f.write("├── Ds004469/                     # Dataset ภายนอก (External Validation Cohort)\n")
        f.write("└── README_PLOTS_INDEX.md\n")
        f.write("```\n\n")
        
        f.write("## สรุปชนิดรูปภาพในแต่ละโมเดล\n")
        f.write("1. **`confusion_matrix.png`**: ตาราง Confusion Matrix แสดง True Positive, False Positive, Precision/Recall\n")
        f.write("2. **`roc_curve.png`**: กราฟ Receiver Operating Characteristic มาตรฐานพร้อมค่า AUC\n")
        f.write("3. **`roc_curve_ci.png`**: กราฟ ROC พร้อมแถบความเชื่อมั่น (Confidence Interval Band)\n")
        f.write("4. **`roc_curve_lines.png`**: กราฟเส้น ROC ของการสุ่มประเมินผลในแต่ละรอบ\n")
        f.write("5. **`pls_components_comparison.png`**: กราฟเปรียบเทียบค่าคะแนน PLS-DA Components\n")
        f.write("6. **`*_loss_curve.png`**: กราฟ Convergence ของ Loss ระหว่าง Train vs Validation (สำหรับโมเดล Deep Learning)\n\n")

    print(f"Created Index README at: {readme_path}")

if __name__ == "__main__":
    organize_plots()
