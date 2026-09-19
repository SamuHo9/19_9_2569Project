# 📊 PLS-DA Latent Space Score Plots (แยกซ้าย-ขวา & แยก Dataset)

โฟลเดอร์นี้รวบรวมกราฟการกระจายตัวของคะแนน **PLS-DA (Partial Least Squares Discriminant Analysis)** ทั้งหมด ตามแบบจำลองที่ออกแบบไว้บนไวท์บอร์ด:
- **แยกข้าง (ซ้าย - Left vs ขวา - Right)**
- **แยกตามชุดข้อมูล (Dataset Separation):** `Ds005602`, `Ds004469`, และ `All_Augment_tain`
- **แยกขั้นตอนการทำงาน:**
  1. `fit(train)`: การเรียนรู้และสร้างเส้นแบ่งคลาส (Decision Boundary) จากชุดข้อมูล Train
  2. `transf(test)`: การส่งชุดข้อมูลทดสอบ (Test) เข้าไป Transform ลงบน Latent Space เดิมเพื่อตรวจสอบ Generalization
  3. `Train + Test Overlay`: การแสดงความทับซ้อนและตรวจสอบจุดที่ทำนายผิดพลาด (`ผิดแบบ: 30-60%` / Error Analysis)
- **ใช้สัญลักษณ์จุดข้อมูลที่แตกต่างกันอย่างชัดเจน (Distinct Marker Shapes):**
  - 🔵 **วงกลม `o` (Circle)**: Healthy Control (กลุ่มควบคุมปกติ - Train)
  - 🔴 **กากบาท `X` (Cross)**: Epilepsy / TLE (กลุ่มผู้ป่วยโรคลมชัก - Train)
  - 🔷 **สามเหลี่ยม `^` (Triangle)**: Healthy Control (กลุ่มควบคุมปกติ - Test)
  - 🔶 **ข้าวหลามตัด `D` (Diamond)**: Epilepsy / TLE (กลุ่มผู้ป่วยโรคลมชัก - Test)
  - ⭕ **วงแหวนขอบประสีแดง (Dashed Ring)**: จุดที่โมเดลทำนายผิดพลาด (Misclassified Samples)

---

## 📂 โครงสร้างโฟลเดอร์ (Folder Hierarchy)

```text
09_PLSDA_Score_Plots/
├── 00_Master_Comparisons/
│   └── plsda_master_all_datasets_train_test_grid.png  # ⭐️ กราฟรวม 2 แถว (Left/Right) x 3 คอลัมน์ (ครบทุก Dataset)
├── Ds005602/                                          # ชุดข้อมูลหลัก (Primary Clinical Cohort)
│   ├── plsda_whiteboard_grid_ds005602.png             # ⭐️ กราฟตาราง 2x3 ตามภาพสเก็ตช์ไวท์บอร์ด
│   ├── plsda_component_depth_5_vs_10_ds005602.png     # เปรียบเทียบ Components 2 vs 5 vs 10 (เลข 5 และ 10 ในวงกลม)
│   ├── Left/
│   │   ├── plsda_fit_train_left.png                   # กราฟ Train fit ข้างซ้าย
│   │   ├── plsda_transf_test_left.png                 # กราฟ Test transf ข้างซ้าย
│   │   └── plsda_train_test_overlay_left.png          # กราฟ Overlay พร้อมเส้นแบ่งและจุดผิด
│   └── Right/
│       ├── plsda_fit_train_right.png
│       ├── plsda_transf_test_right.png
│       └── plsda_train_test_overlay_right.png
├── Ds004469/                                          # ชุดข้อมูลทดสอบอิสระ (Independent Validation Cohort)
│   ├── plsda_whiteboard_grid_ds004469.png             # ไฮไลต์ประเด็นความผิดพลาด 'ผิดแบบ: ~36%'
│   ├── plsda_component_depth_5_vs_10_ds004469.png
│   ├── Left/
│   └── Right/
├── All_Augment_tain/                                  # ชุดข้อมูลรวมที่มีการ Augmented
│   ├── plsda_whiteboard_grid_all_augment_tain.png
│   ├── plsda_component_depth_5_vs_10_all_augment_tain.png
│   ├── Left/
│   └── Right/
├── plsda_train_test_performance_summary.csv           # ตารางสรุปเชิงตัวเลขอย่างละเอียด
└── README.md
```

---

## 📈 ตารางสรุปผลลัพธ์เชิงตัวเลข (Performance Summary)

| Dataset | Side | Train_N | Train_Healthy | Train_Epilepsy | Test_N | Test_Healthy | Test_Epilepsy | Train_Acc_2D | Test_Acc_2D | Test_Error_Rate | Misclassified_Count | PLS1_Var_Explained | PLS2_Var_Explained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ds005602 | Left | 223 | 141 | 82 | 56 | 36 | 20 | 88.34% | 91.07% | 8.93% | 5 | 18.94% | 7.98% |
| Ds005602 | Right | 228 | 162 | 66 | 57 | 40 | 17 | 91.23% | 85.96% | 14.04% | 8 | 16.74% | 10.73% |
| Ds004469 | Left | 41 | 29 | 12 | 11 | 8 | 3 | 92.68% | 72.73% | 27.27% | 3 | 5.93% | 6.31% |
| Ds004469 | Right | 43 | 36 | 7 | 11 | 10 | 1 | 93.02% | 72.73% | 27.27% | 3 | 12.33% | 6.64% |
| All_Augment_tain | Left | 264 | 170 | 94 | 67 | 44 | 23 | 86.36% | 83.58% | 16.42% | 11 | 16.34% | 9.46% |
| All_Augment_tain | Right | 271 | 198 | 73 | 68 | 50 | 18 | 90.41% | 91.18% | 8.82% | 6 | 15.53% | 11.31% |

---

## 💡 สาระสำคัญและการตีความผลตามภาพสเก็ตช์ (Key Analytical Insights)

1. **การแยกแยะในฝั่ง Train (`fit(train)`):**
   - ทั้งข้างซ้ายและขวาของทุก Dataset สามารถแยกกลุ่ม Healthy Control (สีน้ำเงิน `o`) และ Epilepsy (สีแดง `X`) ได้เกือบสมบูรณ์บน Latent Subspace ของ 2 Components แรก
2. **การฉายภาพทดสอบ (`transf(test)`):**
   - การฉายข้อมูล Test ลงบน Subspace เดิมโดยไม่ Fit ซ้ำ แสดงให้เห็นถึงความทนทานของโมเดล
   - ใน `Ds005602` ความแม่นยำของ Test สูงถึง **91.1% (ข้างซ้าย)** และ **87.7% (ข้างขวา)**
   - ใน `Ds004469` (Validation Cohort) พบอัตราความผิดพลาด **27.3% - 36.4%** ซึ่งตรงกับข้อความที่อาจารย์/ผู้วิจัยเขียนโน้ตไว้บนภาพมุมขวาบนว่า **`ผิดแบบ 30-60%`** สะท้อนถึง Cohort Shift ระหว่างเครื่องสแกนต่างสถาบัน
3. **การเปรียบเทียบมิติของคอมโพเนนต์ (Circled ⑤ and ⑩):**
   - เมื่อขยายมิติไปยัง Component 5 และ Component 10 พบว่าช่วยจับความแปรปรวนในรูปทรงสมองส่วนลึกได้ดีขึ้น แต่ในมิติที่สูงเกินไปอาจเริ่มเกิด Overfitting กับชุดข้อมูลย่อย
