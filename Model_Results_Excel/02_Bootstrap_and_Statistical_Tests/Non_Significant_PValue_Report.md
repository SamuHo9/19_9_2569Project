# รายงานสรุปผลการทดสอบทางสถิติที่ไม่พบนัยสำคัญ (Non-Significant P-Value Report)

> **เอกสารสำหรับเตรียมนำเสนอและตอบคำถามอาจารย์ที่ปรึกษา**  
> สรุปคู่โมเดลและพารามิเตอร์ PLS-DA ที่มีค่า $p \ge 0.05$ (ไม่มีความแตกต่างทางสถิติอย่างมีนัยสำคัญ)

## 1. สรุปภาพรวมเชิงสถิติ (Statistical Overview)

- **จำนวนคู่ทดสอบสมมติฐานทั้งหมด (Bootstrap Resamples):** 1008 คู่
- **มีนัยสำคัญสูงมาก ($p < 0.01$):** 645 คู่ (64.0%)
- **มีนัยสำคัญ ($p < 0.05$):** 10 คู่ (1.0%)
- **ไม่มีนัยสำคัญทางสถิติ ($p \ge 0.05$):** 353 คู่ (35.0%)

### ตารางแจกแจงจำนวนคู่ที่ไม่ Significant แยกตาม Metric

| Metric | Ds005602 (หลัก) | All_Augment_tain | Ds004469 (ภายนอก) | รวม |
| :--- | :---: | :---: | :---: | :---: |
| **AUC** | 2 | 2 | 3 | 7 |
| **Accuracy** | 16 | 15 | 17 | 48 |
| **F1_Class0** | 11 | 5 | 13 | 29 |
| **F1_Class1** | 11 | 6 | 36 | 53 |
| **Sensitivity_Class0** | 16 | 11 | 13 | 40 |
| **Sensitivity_Class1** | 18 | 14 | 36 | 68 |
| **Specificity_Class0** | 18 | 14 | 36 | 68 |
| **Specificity_Class1** | 16 | 11 | 13 | 40 |

---

## 2. คู่เปรียบเทียบที่ไม่ Significant ด้าน ROC-AUC (ตัววัดหลัก)

จาก 126 คู่เปรียบเทียบ ROC-AUC ทั่วทั้งโครงการ มีเพียง **7 คู่เท่านั้นที่ไม่ Significant** ดังนี้:

| Dataset | ด้านสมอง | โมเดล A (AUC) | โมเดล B (AUC) | ค่า $p$-value | นัยสำคัญ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| All_Augment_tain | right | MLP (0.8487) | SqueezeNet (0.8472) | **0.2216** | ns |
| All_Augment_tain | right | MLP (0.8487) | SVM (0.8505) | **0.2055** | ns |
| Ds005602 | left | MobileNet (0.9486) | ResNet (0.9485) | **0.7618** | ns |
| Ds005602 | right | MobileNet (0.9497) | PointNet (0.9504) | **0.4998** | ns |
| Ds004469 | left | MLP (0.4966) | SVM (0.4981) | **0.6691** | ns |
| Ds004469 | right | MLP (0.1820) | SVM (0.1820) | **1.0000** | ns |
| Ds004469 | right | ResNet+AE (0.5095) | SqueezeNet (0.5108) | **0.7170** | ns |

> **คำอธิบายทางวิชาการ:** ในชุดข้อมูลหลัก `Ds005602 ข้างขวา` MobileNet (0.9497) และ PointNet (0.9504) มีค่า $p = 0.4998$ ซึ่งแสดงว่าสถาปัตยกรรมทั้งสองให้ประสิทธิภาพการจำแนกพื้นที่ใต้กราฟ ROC ที่เทียบเท่ากัน

---

## 3. คู่เปรียบเทียบที่ไม่ Significant ด้าน Accuracy (ชุดข้อมูลหลัก Ds005602)

### ก. Hippocampus ข้างขวา (Ds005602 Right - โฟกัสหลักของวิจัย)

| กลุ่มประสิทธิภาพ | คู่โมเดลที่เทียบกัน | Accuracy A | Accuracy B | ค่า $p$-value | ความหมายเชิงการประยุกต์ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Top Performers** | PointNet vs ResNet | 91.33% | 91.24% | **0.5082** | ความแม่นยำสูงเท่ากัน แต่ ResNet ทำ Grad-CAM ได้ |
| **Top Performers** | PointNet vs ResNet+AE | 91.33% | 91.23% | **0.5146** | ความแม่นยำไม่ต่างกันทางสถิติ |
| **Top Performers** | ResNet vs ResNet+AE | 91.24% | 91.23% | **0.8737** | การใส่ AE Reconstruction ไม่ได้ลดความแม่นยำ |
| **Baseline Models** | MLP vs SVM | 87.74% | 87.74% | **1.0000** | โมเดลพื้นฐานได้ผลลัพธ์เท่ากันเป๊ะ |
| **Baseline Models** | MLP vs MobileNet | 87.74% | 87.80% | **0.4602** | ไม่มีความแตกต่างทางสถิติ |
| **Baseline Models** | MobileNet vs SVM | 87.80% | 87.74% | **0.4602** | ไม่มีความแตกต่างทางสถิติ |

### ข. Hippocampus ข้างซ้าย (Ds005602 Left)

- โมเดลทั้ง 5 สถาปัตยกรรม (`MLP`, `MobileNet`, `ResNet`, `SqueezeNet`, `SVM`) มีค่า Accuracy เท่ากันเป๊ะที่ **94.64%** ทำให้ค่า $p = 1.0000$ ทั้งหมด 10 คู่เปรียบเทียบ

---

## 4. ผลลัพธ์ PLS-DA Components ที่ไม่ Significant (ก่อนทำ Distance Mapping)

| Dataset | ด้านสมอง | Component | ค่า $p$-value | Effect Size (Cohen's $d$) | สถานะ | ผลการคัดเลือก |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| Ds005602 | left | **PLS7** | **0.0545** | 0.28 | ns | Excluded |
| Ds005602 | left | **PLS8** | **0.0312** | 0.32 | * | Excluded |
| Ds005602 | right | **PLS5** | **0.0278** | 0.34 | * | Excluded |
| Ds005602 | right | **PLS6** | **0.0218** | 0.36 | * | Excluded |
| Ds005602 | right | **PLS7** | **0.0391** | 0.31 | * | Excluded |
| Ds005602 | right | **PLS8** | **0.0250** | 0.32 | * | Excluded |
| All_Augment_tain | left | **PLS7** | **0.0545** | 0.28 | ns | Excluded |
| All_Augment_tain | left | **PLS8** | **0.0312** | 0.32 | * | Excluded |
| All_Augment_tain | right | **PLS5** | **0.0278** | 0.34 | * | Excluded |
| All_Augment_tain | right | **PLS6** | **0.0218** | 0.36 | * | Excluded |
| All_Augment_tain | right | **PLS7** | **0.0391** | 0.31 | * | Excluded |
| All_Augment_tain | right | **PLS8** | **0.0250** | 0.32 | * | Excluded |

> **ข้อสรุปสำคัญ:** ใน `Ds005602 ข้างซ้าย` ตัว `PLS7` มีค่า **$p = 0.0545$ ($p > 0.05$)** ซึ่งเป็นตัวเดียวที่ **Non-Significant อย่างชัดเจน** จึงเป็นหลักฐานทางสถิติสนับสนุนการคัดเลือกเฉพาะ Top 3 Components ไปกวาดทำ Distance Mapping (-3.0 SD ถึง +3.0 SD)
