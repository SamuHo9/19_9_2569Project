# Model Performance Plots Index (ดัชนีรวมภาพกราฟประสิทธิภาพโมเดล)

โฟลเดอร์นี้รวบรวมไฟล์รูปภาพกราฟผลการประเมินประสิทธิภาพของแต่ละโมเดล (Confusion Matrix, ROC Curve, PLS Comparison, Loss Curve) แยกตาม Dataset, ด้านของสมอง (Left/Right) และโครงสร้างโมเดล เพื่อความสะดวกในการเตรียมนำเสนออาจารย์ที่ปรึกษาครับ

## โครงสร้างโฟลเดอร์ (Folder Hierarchy)
```
07_Detailed_Model_Plots/
├── Ds005602/                     # Dataset หลัก (Primary Clinical Dataset)
│   ├── left/ & right/            # แยก Left / Right Hippocampus
│   │   ├── MLP/
│   │   ├── MobileNet/
│   │   ├── PointNet/
│   │   ├── ResNet/               # โมเดลหลักที่ใช้ทำ Grad-CAM Heatmap
│   │   ├── ResNet+AE/            # ResNet พร้อม AutoEncoder Reconstruction
│   │   ├── SqueezeNet/
│   │   └── SVM/
├── All_Augment_tain/             # Dataset ที่ผ่านการ Augmentation (PLS-DA Balanced)
├── Ds004469/                     # Dataset ภายนอก (External Validation Cohort)
└── README_PLOTS_INDEX.md
```

## สรุปชนิดรูปภาพในแต่ละโมเดล
1. **`confusion_matrix.png`**: ตาราง Confusion Matrix แสดง True Positive, False Positive, Precision/Recall
2. **`roc_curve.png`**: กราฟ Receiver Operating Characteristic มาตรฐานพร้อมค่า AUC
3. **`roc_curve_ci.png`**: กราฟ ROC พร้อมแถบความเชื่อมั่น (Confidence Interval Band)
4. **`roc_curve_lines.png`**: กราฟเส้น ROC ของการสุ่มประเมินผลในแต่ละรอบ
5. **`pls_components_comparison.png`**: กราฟเปรียบเทียบค่าคะแนน PLS-DA Components
6. **`*_loss_curve.png`**: กราฟ Convergence ของ Loss ระหว่าง Train vs Validation (สำหรับโมเดล Deep Learning)

