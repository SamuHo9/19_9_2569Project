# PLS-DA Parameter Comparison by Class (ก่อนนำไปทำ Distance Mapping)

โฟลเดอร์นี้รวบรวมกราฟ **Violin Plot ของผลลัพธ์ PLS-DA ในแต่ละ Parameter (Latent Component) เปรียบเทียบระหว่างกลุ่ม Healthy Control vs Epilepsy (TLE)** ก่อนที่จะนำค่าไปกวาดพารามิเตอร์ทำ **Distance Mapping (-3.0 SD ถึง +3.0 SD) บน Grad-CAM 3D Mesh**

---

## 📂 โครงสร้างโฟลเดอร์ (Folder Structure)

```text
08_PLSDA_Class_Violin_Plots/
├── Ds005602/                         # ชุดข้อมูลหลัก (Primary Clinical Cohort)
│   ├── right/                        # Hippocampus ข้างขวา (Focus หลักของงานวิจัย)
│   │   ├── plsda_top3_parameters_detailed_violin.png   # ⭐️ รูปหลัก: Top 3 Parameters พร้อมค่าสถิติ p-value & Cohen's d
│   │   ├── plsda_all_8_components_violin.png           # รูปเปรียบเทียบครบทั้ง 8 Components
│   │   ├── plsda_components_class_stats.csv            # ตารางสถิติเชิงปริมาณ (Mean, SD, p-value, Effect Size)
│   │   └── model_classification_predictions_violin.png # ความน่าจะเป็นที่โมเดลทำนายแยกตามคลาส
│   └── left/                         # Hippocampus ข้างซ้าย
├── All_Augment_tain/                 # ชุดข้อมูล Augmented (Balanced Cohort)
│   ├── right/
│   └── left/
└── README.md
```

---

## 📊 รายละเอียดของกราฟแต่ละภาพ

1. **`plsda_top3_parameters_detailed_violin.png` (แนะนำสำหรับนำเสนออาจารย์)**:
   - แสดงการกระจายตัวของ Latent Scores ของ **Top 3 PLS-DA Components** ที่ถูกคัดเลือกไปทำ Distance Mapping (เช่น PLS1, PLS3, PLS2)
   - ประกอบด้วย **Violin Plot + Boxplot + Data Points (Jittered Stripplot)** ครบถ้วน
   - มีการทดสอบนัยสำคัญทางสถิติ ($t$-test & Mann-Whitney $U$ test) แสดงค่า $p$-value
   - แสดงขนาดอิทธิพล **Effect Size (Cohen's $d$)** เพื่อยืนยันว่าพารามิเตอร์แยกกลุ่มโรคออกจากกลุ่มควบคุมได้อย่างชัดเจนจริงก่อนนำไปกวาดเป็น Morphological Distance Map

2. **`plsda_all_8_components_violin.png`**:
   - กราฟ Split-Violin Plot เปรียบเทียบทั้ง 8 PLS-DA Components พร้อมกันในภาพเดียว
   - ไฮไลต์แถบสีชมพูแดงเฉพาะ Top 3 Components ที่ถูกเลือก
   - แสดงให้เห็นชัดเจนว่าเหตุใดบาง Component (เช่น PLS1, PLS3) จึงถูกเลือก ในขณะที่บาง Component (เช่น PLS5) มีความทับซ้อนกันระหว่างคลาส

3. **`plsda_components_class_stats.csv`**:
   - ตารางสรุปค่าเฉลี่ย, ส่วนเบี่ยงเบนมาตรฐาน (SD), ค่า $t$-statistic, $p$-value และ Cohen's $d$ ของทุก Component อย่างละเอียด
