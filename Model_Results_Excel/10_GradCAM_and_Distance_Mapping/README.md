# 🧠 Grad-CAM Attention & PLS-DA Distance Mapping Reports

โฟลเดอร์นี้รวบรวมไฟล์รูปภาพกราฟและตารางผลลัพธ์เชิงตัวเลขทั้งหมดที่เกี่ยวข้องกับ **Grad-CAM (Gradient-weighted Class Activation Mapping)** บนโครงข่าย ResNet1D และการจำลองการบิดรูปทางสัณฐานวิทยา **PLS-DA Distance Mapping (-3.0 SD ถึง +3.0 SD)**

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
10_GradCAM_and_Distance_Mapping/
├── 00_Master_Comparison_Plots/
│   ├── gradcam_and_distance_mapping_master_overview.png   # ⭐️ กราฟมาสเตอร์ 4 ช่องสรุปภาพรวมครบถ้วน
│   └── distance_mapping_sd_sweep_comparison.png           # เปรียบเทียบเส้นกราฟการบิดรูปทั้ง 4 กลุ่มย่อย
├── Ds005602/                                              # ชุดข้อมูลหลักทางคลินิก (Primary Cohort)
│   ├── Left/                                              # Hippocampus ข้างซ้าย
│   │   ├── resnet_gradcam_vertex_profile.png              # ⭐️ กราฟแสดงค่าน้ำหนักความสำคัญ Grad-CAM ทุกจุด 1,002 vertices
│   │   ├── distance_mapping_vs_sd.png                     # ⭐️ กราฟค่าเฉลี่ยและค่าสูงสุดของการขจัด (Displacement mm)
│   │   ├── plsda_8_components_scree.png                   # Scree plot ความแปรปรวน 8 Components
│   │   ├── plsda_top3_parameters_detailed_violin.png      # ไวโอลินพล็อต Top 3 Components พร้อมค่าสถิติ
│   │   ├── model_classification_predictions_violin.png   # ความน่าจะเป็นในการทำนายของ ResNet1D
│   │   ├── top3_distance_mapping_summary.csv              # ⭐️ ตารางระยะการบิดรูป (Displacement/Atrophy/Expansion)
│   │   ├── plsda_8_components_summary.csv                 # ⭐️ ตารางจัดอันดับความสำคัญ Top 3 ของ Components
│   │   └── plsda_components_class_stats.csv               # ตารางสถิติเปรียบเทียบระหว่างคลาส (t-test, Cohen's d)
│   └── Right/                                             # Hippocampus ข้างขวา (เนื้อหาครบชุดเหมือนข้างซ้าย)
├── All_Augment_tain/                                      # ชุดข้อมูลรวมที่มีการทำ Data Augmentation
│   ├── Left/
│   └── Right/
├── GradCAM_and_Distance_Mapping_Master_Summary.xlsx       # ⭐️ เวิร์กบุ๊ก Excel รวมข้อมูลทุกชีตอย่างสมบูรณ์
└── README.md
```

---

## 🔬 สาระสำคัญทางวิทยาศาสตร์และการแพทย์ (Scientific & Clinical Interpretation)

### 1. Grad-CAM Vertex Attention Profile (`resnet_gradcam_vertex_profile.png`)
* **วัตถุประสงค์:** อธิบายการตัดสินใจของโครงข่าย ResNet1D ในระดับสัณฐานวิทยา 3 มิติว่า โมเดลเพ่งเล็งจุดยอด (Vertices) ใดบนพื้นผิวฮิปโปแคมปัสเป็นหลักในการทำนายว่าผู้ป่วยเป็นโรคลมชัก (Epilepsy) หรือปกติ
* **การแปลผล:**
  * ยอดพีคของน้ำหนัก Grad-CAM Activation จะกระจุกตัวอยู่ในบริเวณ **CA1 / Subiculum** และ **CA3 / Dentate Gyrus** ซึ่งเป็นบริเวณทางกายวิภาคที่เกิดพยาธิสภาพ Hippocampal Sclerosis (HS) บ่อยที่สุดในผู้ป่วย TLE (Temporal Lobe Epilepsy)

### 2. Distance Mapping Deformation vs. SD Sweep (`distance_mapping_vs_sd.png`)
* **วัตถุประสงค์:** กวาดค่าพารามิเตอร์ PLS-DA จากฝั่งปกติ (-3.0 SD) ไปยังฝั่งโรคลมชัก (+3.0 SD) ในระยะขั้นละ 0.1 SD เพื่อคำนวณระยะการยุบตัว (Inward Atrophy) และการโป่งตัว (Outward Expansion) ในหน่วยมิลลิเมตร (mm)
* **การแปลผล:**
  * Component อันดับ 1 (Top Rank 1) ให้ระยะขจัดเฉลี่ยสูงสุดถึง **0.065 - 0.082 mm** และระยะขจัดสูงสุดเฉพาะจุด (Peak Vertex Displacement) เกินกว่า **0.13 - 0.16 mm**
  * สัดส่วนการฝ่อลีบ (Inward Atrophy) มีบทบาทเด่นกว่าการขยายตัวในฝั่งโรคลมชักอย่างมีนัยสำคัญทางสถิติ ($p < 0.001$)

### 3. ตารางสรุปเชิงปริมาณใน Excel (`GradCAM_and_Distance_Mapping_Master_Summary.xlsx`)
* ประกอบด้วย 5 แผ่นงานที่จัดรูปแบบมาตรฐานสำหรับการส่งต่อเพื่อตีพิมพ์:
  1. **`Executive_Summary`**: สรุปเปรียบเทียบพารามิเตอร์หลักระหว่างข้างซ้ายและข้างขวาของทั้ง 2 ชุดข้อมูล
  2. **`Ds005602_Distance_Mapping`**: ข้อมูลการบิดรูปทั้ง 183 สเต็ปของชุดข้อมูลหลัก
  3. **`AllAugment_Distance_Mapping`**: ข้อมูลการบิดรูปของชุดข้อมูล Augmented
  4. **`PLSDA_8_Components_Rankings`**: รายละเอียด Variance Explained, Class Loading Q, และการจัดอันดับ Top 3
  5. **`Component_Statistical_Tests`**: ผลการทดสอบสมมติฐานทางสถิติและขนาดอิทธิพล (Cohen's d)
