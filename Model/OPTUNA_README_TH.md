# Optuna tuning protocol

ตัวรัน `optuna_leakage_free_all.py` แยก study ต่อ `protocol × cohort × side × model`
และใช้ grouped 10-fold OOF balanced accuracy เป็น objective ค่า test ไม่ถูกอ่าน
ระหว่าง Optuna

รันครบทุก protocol ด้วยคำสั่ง:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\Model\run_optuna_leakage_free_all.ps1 `
  --protocol all --cohort all --side all --model all `
  --folds 10 --n_trials 10 --tune_epochs 20 --tune_patience 5 `
  --final_epochs 40 --eval_seeds 42,123,2026 --device cuda `
  --children_per_pair 8 --noise_scale 0.02 --pls_components 8 `
  --output_root .\Model\optuna_runs_10fold_all
```

ช่วงค่าที่ค้นหา:

- SVM: `C` แบบ log จาก `1e-2..1e2`; `gamma` เป็น `scale`, `auto` หรือค่าตัวเลข log `1e-5..1e-1`
- neural coefficient models: `lr 1e-4..3e-3`, `weight_decay 1e-6..1e-2`,
  `batch_size 8/16/32`, `dropout 0.10..0.50`
- ResNetAE: เพิ่ม `recon_weight 0.01..0.20`
- PointNet: `lr`, `weight_decay`, `batch_size 8/16/32`, `dropout 0.10..0.60`
- direct PLS-DA: `n_components` จาก `2,4,6,8,12,16,24`

Optuna เลือกค่าโดยใช้ OOF เท่านั้น หลังล็อกค่าแล้วจะ fit final model เดี่ยวจาก train
ทั้งหมดและทดสอบแยกด้วย seed `42,123,2026` ไม่มีการเฉลี่ย fold models หรือรวมซ้าย/ขวา

รายงานหลักอยู่ใน `optuna_runs_10fold_all`:

- `OPTUNA_TUNING_SUMMARY.csv` ค่าที่ชนะและ OOF score ต่อ study
- `OPTUNA_FINAL_SUMMARY.csv` ผล test ทุกโมเดลและทุก seed
- `OPTUNA_WINNERS_BY_COHORT_SIDE.csv` ผู้ชนะภายในแต่ละ protocol/cohort/side
- `OPTUNA_WINNER_TEST_REPORT.csv` ค่าเฉลี่ยและ SD ของผู้ชนะจาก 3 seeds
- `OPTUNA_AUDIT.csv` จำนวน trial, fold และ trial ที่ fail/prune
