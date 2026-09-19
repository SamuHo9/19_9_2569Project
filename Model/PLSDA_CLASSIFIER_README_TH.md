# PLSDA classifier baseline

ไฟล์ `leakage_free_plsda_classifier.py` เป็นโมเดล PLS-DA โดยตรง แยกจาก
`leakage_free_plsda_training.py` ที่ใช้ PLS-DA เพื่อสร้างข้อมูล augmentation
ให้โมเดลอื่น

คำสั่งรันทั้ง 3 cohort และซ้าย/ขวา:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\Model\run_leakage_free_plsda_classifier.ps1 `
  --cohort all --side all --folds 10 --seed 42 --n_components 8 `
  --output_root .\Model\plsda_classifier_runs_10fold
```

หลักการประเมิน:

- ในแต่ละ fold จะ fit `StandardScaler` และ PLS-DA จาก training subjects เท่านั้น
- validation ใช้สำหรับคำนวณ out-of-fold score และไม่ถูกใช้ fit PLS-DA
- หลังจบ CV จะ fitโมเดล PLS-DA เพียงตัวเดียวจาก training ทั้งชุด
- test ถูกประเมินด้วยโมเดลตัวเดียวนี้ จึงไม่มีการเฉลี่ย fold models และไม่มี ensemble
- `n_components=8` เป็นค่าที่ระบุในคำสั่งและถูกบันทึกไว้ใน `run_manifest.json`

ผลลัพธ์อยู่ใน `Model\plsda_classifier_runs_10fold` ได้แก่ `model.pkl`,
`oof_predictions.csv`, `test_predictions.csv`, `metrics.json`,
`run_manifest.json` และ summary รวม

โมเดลนี้ใช้ PLS-DA เป็นตัวจำแนกโดยตรง จึงควรรายงานแยกจาก protocol ที่ใช้
PLS-DA เพื่อ augmentation และไม่ควรนำผลสอง protocol มาปนกันเป็นโมเดลเดียว
