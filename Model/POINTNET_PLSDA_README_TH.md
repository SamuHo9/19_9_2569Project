# PointNet แบบ PLS-DA

ไฟล์ `leakage_free_pointnet_plsda_training.py` เป็น PointNet รุ่นที่แยกจาก
raw XYZ baseline (`leakage_free_pointnet_training.py`) อย่างชัดเจน

กระบวนการของรุ่นนี้คือ

1. โหลด XYZ ต้นฉบับจาก `Output_Dataset` และ normalize แยกต่อเมช
2. แบ่งข้อมูลตามผู้ป่วยเป็น 5 folds
3. fit StandardScaler และ PLS-DA ภายใน training fold เท่านั้น
4. จับคู่เมชคลาสเดียวกันใน PLS score space และ reconstruct เมช XYZ สังเคราะห์
5. normalize เมชสังเคราะห์อีกครั้ง แล้ว balance จำนวนคลาสด้วย `children_per_pair=8`
6. ฝึก PointNet ด้วยเมชต้นฉบับและเมชสังเคราะห์ของ training fold
7. ใช้ validation/test ที่เป็นเมชต้นฉบับเท่านั้น

จึงไม่มี PLS model หรือ synthetic row ที่ fit จาก validation/test และไม่มีการลด
อินพุต PointNet เหลือคะแนน PLS; PointNet ยังรับ XYZ ขนาด `3 x 1002` เหมือนเดิม

## รันจาก PowerShell

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\Model\run_leakage_free_pointnet_plsda.ps1 `
  --cohort all --side all --folds 5 --epochs 40 --patience 8 `
  --seed 42 --device cuda --children_per_pair 8 --n_components 8
```

## ผลรันล่าสุด

- สรุป: `Model/pointnet_plsda_runs/FINAL_SUMMARY.csv`
- audit: `Model/pointnet_plsda_runs/FINAL_AUDIT.json`
- model weights ราย fold: อยู่ในโฟลเดอร์ run ของแต่ละ cohort/side ใน `fold_models.pt`
- comparison รวม: `Model/MODEL_COMPARISON_SUMMARY.csv`

ผลเฉลี่ย 6 cohort/side ของรอบล่าสุด: accuracy `0.8356`, balanced accuracy
`0.6620`, macro-F1 `0.6528`, ROC-AUC `0.8143`.

ค่า Ds004469 มี positive ใน test เพียง 1–2 รายต่อข้าง จึงต้องอ่าน sensitivity
และ balanced accuracy ของชุดนี้อย่างระมัดระวัง และควรมี external validation ก่อน
นำไปใช้กับผู้ป่วยจริง
