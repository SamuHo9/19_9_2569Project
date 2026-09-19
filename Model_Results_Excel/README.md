# Model Results & Evaluation Directory (โครงสร้างผลการประเมินโมเดล)

โครงสร้างโฟลเดอร์ถูกจัดหมวดหมู่อย่างเป็นระเบียบตามประเภทของไฟล์ แยกไฟล์ Excel, ข้อมูล Bootstrap 1,000 รอบ, ตารางสรุป CSV, โค้ดสคริปต์ และข้อมูลตัวอย่างเก่าออกจากกันอย่างชัดเจน

---

## 📁 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
Model_Results_Excel/
│
├── 01_Excel_Workbooks/                   # 📊 ไฟล์ Excel และเอกสารรายงานทางวิชาการ
│   ├── Model_Evaluation_and_Selection_Report.docx   # 📄 เล่มรายงานวิชาการระเบียบวิธีตัดสินใจคัดเลือกโมเดลที่ดีที่สุด (Word)
│   ├── Model_Performance_and_Bootstrap_Master.xlsx  # ไฟล์ Master รวม 5 แผ่นงาน (Performance, Bootstrap, Mean±SD, P-Values, Legend)
│   └── Model_Performance_Summary.xlsx               # ตารางสรุป Class 0 vs Class 1 พร้อมไฮไลต์โมเดลที่ดีที่สุด
│
├── 02_Bootstrap_and_Statistical_Tests/   # 📈 ผลลัพธ์ Bootstrap 1,000 รอบ และการทดสอบนัยสำคัญทางสถิติ
│   ├── Combined_Bootstrap_Results.csv               # ผล Bootstrap 1,000 รอบ × 379 คอลัมน์ (TP, TN, FP, FN, Acc, Sens, Spec, F1, AUC)
│   ├── Model_Legend.txt                             # รายชื่อและดัชนีโมเดลทั้งหมด 42 โมเดล (Index 0 ถึง 41)
│   ├── Final_Summary_Mean_SD.csv                    # ค่า Mean ± SD สรุปรายโมเดลข้าม 1,000 รอบ
│   └── PValue_Significance_Summary.csv              # ผลการทดสอบ Paired t-test เทียบทุกคู่โมเดล (1,008 คู่)
│
├── 03_Performance_Summary_CSVs/          # 📋 ตารางสรุปผลเมทริกซ์ในรูปแบบ CSV
│   ├── model_performance_summary.csv                # ตารางสรุป 1 โมเดลต่อ 1 แถว รวมทั้ง 2 คลาส
│   └── model_performance_by_class.csv               # ตารางสรุปแยกแถว Class 0 (HC), Class 1 (TLE) และ Overall
│
├── 04_Scripts/                           # ⚙️ โค้ด Python สำหรับคำนวณและสร้างผลลัพธ์
│   ├── generate_newest_bootstrap_results.py         # สคริปต์หลักคำนวณ Bootstrap 1,000 รอบ, P-values และ Excel
│   ├── generate_model_excel.py                      # สคริปต์สร้างตารางสรุป Excel แบบ 2 คลาส
│   ├── generate_plsda_train_test_score_plots.py     # สคริปต์วาด PLS-DA Score Plots (แยกซ้าย-ขวา/Dataset/Markers)
│   └── organize_and_consolidate_gradcam_results.py  # สคริปต์รวบรวม Grad-CAM Attention และ Distance Mapping
│
├── 05_Bootstrap_Violin_Plots/            # 🎻 กราฟ Violin Plot & Box Plot พร้อมไฟล์ข้อมูลดิบ 1,000 รอบ (Raw Iterations CSV)
├── 06_ROC_Curves/                        # 📉 กราฟ ROC Curves และ Confidence Band
├── 07_Detailed_Model_Plots/              # 🔍 กราฟวิเคราะห์ประสิทธิภาพโมเดลรายชุดข้อมูล
├── 08_PLSDA_Class_Violin_Plots/          # 🎻 กราฟ Violin Plot พารามิเตอร์ PLS-DA แยกตามคลาส
├── 09_PLSDA_Score_Plots/                 # 🎯 กราฟ PLS-DA Train/Test Score Space ตามภาพไวท์บอร์ด
├── 10_GradCAM_and_Distance_Mapping/      # 🧠 กราฟ Grad-CAM Attention Profile และ Distance Mapping Deformation
└── 05_Legacy_Example_Data/               # 🗄️ ไฟล์ตัวอย่างเก่า (เฉพาะ Ds005602 และ AllAugment รุ่นก่อน)
```

---

## 🎯 ข้อมูลโมเดลทั้งหมดที่ประเมิน (Total 42 Models)

ครอบคลุม 3 ชุดข้อมูล $\times$ 2 ข้างฮิปโปแคมปัส (Left, Right) $\times$ 7 สถาปัตยกรรมโมเดล:
1. **Cohorts**:
   - `All_Augment_tain` (Left: 67 samples, Right: 68 samples)
   - `Ds005602` (Left: 56 samples, Right: 57 samples)
   - `Ds004469` (Left: 11 samples, Right: 11 samples)
2. **Models**:
   - `MLP`
   - `MobileNet`
   - `PointNet`
   - `ResNet (Standard)`
   - `ResNet (AutoEncoder)`
   - `SqueezeNet`
   - `SVM`

---

## 🚀 วิธีการรันเพื่อสร้างข้อมูลใหม่ (How to Run)

หากต้องการรันคำนวณและสร้างผลลัพธ์ใหม่ทั้งหมด:

```bash
uv run --with openpyxl --with pandas --with numpy --with scikit-learn --with scipy python Model_Results_Excel/04_Scripts/generate_newest_bootstrap_results.py
```
*(หมายเหตุ: ปิดไฟล์ Excel ก่อนรันเพื่อให้สคริปต์สามารถบันทึกไฟล์ทับได้อย่างสมบูรณ์)*
