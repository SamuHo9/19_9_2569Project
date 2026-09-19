# คู่มือโฟลเดอร์ของระบบ

เอกสารนี้อธิบายว่าแต่ละโฟลเดอร์ในโปรเจกต์ใช้ทำอะไร อยู่ในขั้นตอนไหน และควรใช้เป็นผลลัพธ์หลัก
หรือเป็นข้อมูลอ้างอิงเท่านั้น โดยยึด pipeline ปัจจุบันใน `RERUN_GUIDE_TH.md`

## แผนผังการไหลของข้อมูล

```text
FastSurfer / MRI
    -> ICP
    -> Templates + fixed reference
    -> SPHARM
    -> SPHARM/split_data
    -> Data_Processing
    -> Model/All_coef* และ Model/Output_Dataset
    -> Model/optuna_runs_10fold_all
    -> DesktopApp inference
```

## 1. โฟลเดอร์ระดับโปรเจกต์

| โฟลเดอร์ | หน้าที่ | สถานะ |
|---|---|---|
| `FastSurfer` | source และโมดูล segmentation สำหรับแปลง MRI เป็น brain/hippocampus masks | ใช้ใน preprocessing |
| `ICP` | สร้าง mesh, align แต่ละ subject กับ fixed reference และเก็บสถานะ ICP/SPHARM ที่เกี่ยวข้อง | ใช้ใน pipeline |
| `Templates` | fixed ICP และ SPHARM reference แยกซ้าย/ขวา | ต้องเก็บไว้ |
| `SPHARM` | รัน SPHARM-PDM, realignment, resampling, ตรวจ output และสร้าง split | ใช้ใน pipeline |
| `Data_Processing` | แปลง SPHARM เป็น coefficient/XYZ features, สร้าง manifest และ dataset | ใช้ใน pipeline |
| `Model` | data contract, model runners, training runtimes, datasets และผลประเมิน | ใช้ใน pipeline |
| `DesktopApp` | application สำหรับ import, แสดง mesh และ inference | ใช้สำหรับใช้งานปลายทาง |
| `Visualize` | ตัวดู mesh, ROC, bootstrap, PLS-DA และ plot สำหรับวิเคราะห์/นำเสนอ | เครื่องมือเสริม |
| `Model_Results_Excel` | workbook, CSV และ script สำหรับจัดรูปผลวิจัย | รายงาน/วิเคราะห์เสริม |
| `Model_Evaluation_and_Benchmarks` | ผล benchmark รุ่นก่อน เช่น bootstrap summary | ผลอ้างอิง ไม่ใช่ผล Optuna หลัก |
| `tests` | regression tests ของ contract, pipeline status, reference และ predictor schema | ใช้ตรวจโค้ด |
| `build` | ไฟล์ build/package ของ application หรือ dependency | ไม่ใช่ input ของ model |

## 2. `ICP`

### `ICP\output_left_hippocampus`

ผลการประมวลผลฝั่งซ้าย ประกอบด้วย `aligned_nifti`, `aligned_meshes`, `spharm_results`,
`T_matrices.npy`, `mean_shape.ply`, `icp_status.json` และ log/status ของ SPHARM

- `aligned_nifti`: mask หลัง ICP
- `aligned_meshes`: mesh หลัง align
- `spharm_results`: coefficient, VTK และ grid จาก SPHARM ของฝั่งซ้าย
- `T_matrices.npy`: transformation matrix ของ ICP
- `icp_convergence_history.json`: ประวัติ convergence
- `spharm_verification*.json`: ผลตรวจความครบถ้วนของ SPHARM

### `ICP\output_right_hippocampus`

ทำหน้าที่เหมือนฝั่งซ้าย แต่ใช้ fixed reference และ input ของ hippocampus ฝั่งขวาแยกกัน
ไม่ควรนำไฟล์ซ้ายมาใส่รวมกับโฟลเดอร์นี้

### `ICP\log`

พื้นที่เก็บ log จากการรัน ICP รุ่นต่าง ๆ ใช้ตรวจปัญหา ไม่ใช่ dataset สำหรับ training

### ไฟล์สำคัญใน `ICP`

- `ICP.py`: runner หลักของ ICP
- `reference_contract.py`: ตรวจ metadata และ physical scale ของ reference
- `plot_icp_convergence.py`: plot convergence
- `run_icp_only.bat`: เรียก ICP แยกจาก pipeline เต็ม

## 3. `Templates`

### `Templates\ICP`

- `template_mean_left.ply`, `template_mean_left.vtk`: fixed reference ฝั่งซ้าย
- `template_mean_right.ply`, `template_mean_right.vtk`: fixed reference ฝั่งขวา

### `Templates\SPHARM`

- `template_spharm_left.vtk`, `template_spharm_left.coef`: reference ฝั่งซ้าย
- `template_spharm_right.vtk`, `template_spharm_right.coef`: reference ฝั่งขวา

reference เหล่านี้ทำให้ rerun ใช้ coordinate frame เดิม ไม่ขึ้นกับลำดับ subject ที่ป้อนเข้ามา
ห้ามลบทิ้งหรือสร้างใหม่โดยไม่บันทึก manifest เพราะจะทำให้ผล ICP/SPHARM เปรียบเทียบกับรอบเดิมไม่ได้

## 4. `SPHARM`

### ไฟล์คำสั่งหลัก

- `run_spharm.bat`: entry point ที่ควรใช้จาก terminal; บังคับใช้ SlicerSALT/PythonSlicer
- `run_spharm_parallel.py`: รัน SPHARM แบบแบ่ง worker
- `run_spharm_batch.py`: batch processing ภายใน SlicerSALT
- `resample_spharm_grid.py`: สร้าง/resample grid mesh
- `realign_spharm.py`: จัด orientation/landmark ของ SPHARM mesh
- `check_spharm_environment.py`: ตรวจ runtime
- `verify_bilateral_outputs.py`: ตรวจ output ซ้าย/ขวา

### `SPHARM\split_data`

โฟลเดอร์ split หลักที่ตรวจสอบแล้ว มี:

- `ALL_Left`, `ALL_Right`
- `Ds004469_Left`, `Ds004469_Right`
- `Ds005602_Left`, `Ds005602_Right`
- `ALL_Left_file_status.csv`, `ALL_Right_file_status.csv`
- `current_split_manifest.json`
- `current_split_completeness.json`
- `dataset_subsets_manifest.json`
- `dataset_train_test_manifest.json`

status tables เป็นแหล่งอ้างอิงว่าไฟล์ใดเป็นปกติ/ป่วยและอยู่ cohort ใด ส่วน manifest ใช้ตรวจว่า
copy และ split ครบ ไม่ได้ใช้เป็น model weights

### `SPHARM\split_data_current`

พื้นที่สำหรับ split แบบ current ที่ script รองรับ ปัจจุบันไม่มี output หลักที่ใช้ใน Optuna
หากจะสร้าง split ใหม่ให้ใช้ `split_current_dataset.py` และตรวจ manifest ก่อนนำไป feature extraction

### `SPHARM\split_data_train_test_current`

พื้นที่สำหรับ materialize train/test folders จาก current split ปัจจุบันไม่มีผล final ของ Optuna
ใช้เฉพาะเมื่อจำเป็นต้องสร้างโครงสร้าง folder สำหรับงาน downstream รุ่นที่ต้องการไฟล์แยกโฟลเดอร์

### `SPHARM\output_left_new`

พื้นที่ output/repair จากรอบก่อน ใช้เป็นข้อมูลอ้างอิงหรือเปรียบเทียบเท่านั้น เว้นแต่มี manifest ระบุ
ชัดเจนว่าเป็น input ของรอบปัจจุบัน ไม่ควรใช้แทน `ICP\output_*\spharm_results` โดยอัตโนมัติ

## 5. `Data_Processing`

- `extract_ml_features.py`: สร้าง feature dataset จาก mesh/XYZ สำหรับงาน ML
- `extract_ml_features_coef.py`: สร้าง SPHARM coefficient features
- `export_ml_dataset.py`: export ตาราง ML พร้อม metadata
- `build_all_coef_datasets.py`: สร้าง `All_coef`, `All_coef_Ds004469` และ `All_coef_Ds005602`
- `feature_metadata.py`: เขียน/อ่าน feature manifest และ label metadata
- `augment_plsda_balanced.py`: สร้าง balanced PLS-DA dataset รุ่น exploratory/legacy
- `augment_plsda_interpolation.py`: interpolation augmentation รุ่นเก่า/เสริม

สองไฟล์ augmentation ถูกเก็บไว้เพื่ออ้างอิงและสร้าง dataset exploratory แต่ไม่ควรรันก่อนแบ่ง fold
สำหรับ final leakage-free Optuna เพราะ augmentation หลักต้องเกิดภายใน training fold

## 6. `Model` — dataset และ input

### `Model\All_coef`

cohort coefficient raw ชุดรวมปัจจุบัน แยก `left` และ `right` ใช้เป็น input หลักของ `coef_raw`,
`coef_plsda` และ direct PLS-DA โดยมี manifest กำกับว่าไม่มี PLS-DA global และไม่มี synthetic test row

### `Model\All_coef_Ds004469`

cohort coefficient ของข้อมูล Ds004469 ชุดปัจจุบัน แยกซ้าย/ขวา ใช้ใน Optuna เป็น cohort หนึ่ง
ไม่ใช่โฟลเดอร์ผลเก่าของ `Model\Ds004469`

### `Model\All_coef_Ds005602`

cohort coefficient ของข้อมูล Ds005602 ชุดปัจจุบัน แยกซ้าย/ขวา ใช้ใน Optuna เป็น cohort หนึ่ง

### `Model\Output_Dataset` / `Model\output_dataset`

โฟลเดอร์ feature export ที่มี coefficient, XYZ, mesh edges, train/test และ balanced files
ใน Windows อาจเห็นการสะกดตัวพิมพ์ต่างกัน แต่ต้องตรวจ path จริงก่อนใช้งาน; code หลักอ้าง `Output_Dataset`

- `*_train_*`: input train ของ runner ที่อ่านจาก Output_Dataset
- `*_test_*`: untouched test ที่ใช้ประเมิน
- `*_balanced_*`: balanced/exploratory input สำหรับ protocol รุ่นเก่าหรือการตรวจเทียบ
- `.json`: manifest ของ feature contract และ provenance

### `Model\All_Augment_tain`, `Model\Ds004469`, `Model\Ds005602`

เป็น dataset/ผลจาก training pipeline รุ่นก่อน มี CSV, NPZ, plot และผล model เก่าอยู่ จึงเก็บข้อมูลไว้
เพื่ออ้างอิง แต่ training entry point รุ่นเก่าถูกลบแล้ว และโฟลเดอร์เหล่านี้ไม่ใช่ source หลักของ final Optuna

### `Model\Legacy_PLS_Models_READONLY`

พื้นที่อ้างอิง source ของ PLS model รุ่นเก่า ปัจจุบันไม่มี active entry point แล้ว ใช้ดูประวัติเท่านั้น

### `Model\Legacy_PointNet_READONLY`

พื้นที่อ้างอิง PointNet รุ่นเก่า ปัจจุบันไม่มี active entry point แล้ว ไม่ใช้สร้างผล final ใหม่

## 7. `Model` — code และ runtime

### code ที่ใช้จริง

- `data_contract.py`: schema, provenance, patient grouping, overlap/hash checks
- `leakage_free_coef_training.py`: coefficient raw runner และ model implementations
- `leakage_free_plsda_training.py`: fold-local PLS-DA augmentation
- `leakage_free_pointnet_training.py`: raw XYZ PointNet
- `leakage_free_pointnet_plsda_training.py`: PointNet PLS-DA
- `leakage_free_plsda_classifier.py`: direct PLS-DA classifier
- `optuna_leakage_free_all.py`: orchestrate tuning, winner selection และ final runs
- `run_optuna_leakage_free_all.ps1`: ตั้งค่า runtime และเรียก Optuna จาก PowerShell
- `verify_leakage_free_pipeline.py`: preflight dataset และ model interface
- `audit_*.py`: ตรวจ summary/output ของ runner รุ่น leakage-free

### `Model\_training_site`

Python package runtime ที่ใช้กับงาน model เช่น pandas, sklearn, matplotlib, Optuna และ package ที่เกี่ยวข้อง
ใช้เมื่อ SlicerSALT Python หลักไม่มี dependency ของ training ครบ

### `Model\_training_cuda_site`

runtime ของ PyTorch CUDA (`torch 2.8.0+cu128`) สำหรับ neural model และ PointNet ที่รันบน GPU
ห้ามลบ เพราะ runner และผลที่ได้อาศัย runtime นี้

### `Model\_mplconfig`

พื้นที่ config/cache ของ matplotlib เพื่อไม่ให้ matplotlib เขียนไฟล์ลงโฟลเดอร์ระบบหรือ user profile
ไม่ใช่ dataset และสร้างใหม่ได้ แต่เก็บไว้ช่วยให้ rerun เหมือนเดิม

## 8. `Model` — ผลการ train/evaluation รุ่น leakage-free

โฟลเดอร์ต่อไปนี้เป็นผลจาก runner ก่อนรอบ Optuna เต็ม ใช้เปรียบเทียบและตรวจ contract:

- `leakage_free_runs`: coefficient raw/no-PLS baseline
- `leakage_free_plsda_runs`: coefficient fold-local PLS-DA
- `pointnet_runs`: raw XYZ PointNet baseline
- `pointnet_plsda_runs`: PointNet fold-local PLS-DA
- `plsda_classifier_runs_10fold`: direct PLS-DA seed 42
- `plsda_classifier_runs_10fold_seed123`: direct PLS-DA seed 123
- `plsda_classifier_runs_10fold_seed2026`: direct PLS-DA seed 2026

ในแต่ละโฟลเดอร์มักมี `summary_*.csv`, `FINAL_SUMMARY.csv`, `FINAL_AUDIT.json`, model artifact,
metrics, manifest และ test predictions ใช้ตรวจย้อนหลังได้ แต่ผลรวมสำหรับรายงานล่าสุดควรอ้างจาก Optuna

## 9. `Model\optuna_runs_10fold_all` — ผลหลักปัจจุบัน

นี่คือโฟลเดอร์ผลลัพธ์หลักของการเปรียบเทียบปัจจุบัน

### `studies`

เก็บผล Optuna แยกตาม protocol/cohort/side/model รวม 90 studies และ trial records ของแต่ละ study

### `final`

เก็บ final single model ที่เลือกจาก OOF แล้ว แยก protocol/cohort/side/model/seed
แต่ละ run มี `run_manifest.json`, metrics, model artifact และ `test_predictions.csv`

### CSV ระดับราก

- `OPTUNA_TUNING_SUMMARY.csv`: ผล tuning และ OOF
- `OPTUNA_FINAL_SUMMARY.csv`: ผล test ของทุก final seed
- `OPTUNA_WINNERS_BY_COHORT_SIDE.csv`: ผู้ชนะภายใน protocol
- `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv`: ผู้ชนะสุดท้าย 6 cohort-side
- `OPTUNA_WINNER_TEST_REPORT.csv`: mean/SD จาก seed 42, 123, 2026
- `OPTUNA_AUDIT.csv`: ตรวจ trial, fold และ failure
- `RUN_MANIFEST.json`: กติกาของ run ใหญ่ เช่น folds, seeds และ test policy

เมื่อเขียนผลวิจัยให้ใช้ `OPTUNA_GLOBAL_WINNERS_BY_COHORT_SIDE.csv` และอ้างรายละเอียดจาก
`OPTUNA_WINNER_TEST_REPORT.csv` ไม่ใช้ไฟล์ smoke หรือ summary รุ่นเก่า

## 10. `DesktopApp`

- `DesktopApp\main.py`: entry point ของ application
- `DesktopApp\LeftUI`: workflow ฝั่งซ้าย เช่น import, FastSurfer, ICP, SPHARM และผลลัพธ์
- `DesktopApp\RightUI`: viewer และ workflow ฝั่งขวา
- `DesktopApp\models\predictor.py`: เตรียม feature และเรียก model inference
- `DesktopApp\app_history.json`: ประวัติ/สถานะการใช้งาน app

DesktopApp ไม่ได้ใช้แทนการ train; ใช้ model artifact และ feature contract ที่สร้างจาก pipeline
เพื่อทำนายข้อมูลใหม่และแสดง mesh/result ให้ผู้ใช้

## 11. `Visualize`

### `Visualize\3D_Mesh_Viewers`

ดู aligned mesh, mean shape, SPHARM/ICP comparison, PLS-DA surfaces และ Grad-CAM output

### `Visualize\Data_Plots`

สร้าง ROC, violin, bootstrap และ plot จาก CSV/summary เพื่อวิเคราะห์และนำเสนอผล

โฟลเดอร์นี้เป็นเครื่องมือแสดงผล ไม่ได้เป็น input ให้ model และไม่ควรนำ plot กลับเข้า training

## 12. `Model_Results_Excel`

โฟลเดอร์นี้ใช้จัดผลสำหรับรายงานวิจัย:

- `01_Excel_Workbooks`: workbook ที่สร้างแล้ว
- `02_Bootstrap_and_Statistical_Tests`: bootstrap และ statistical outputs
- `03_Performance_Summary_CSVs`: ตาราง performance
- `04_Scripts`: script สร้างรายงาน/ตาราง/plot
- `05_Bootstrap_Violin_Plots`: violin ของ bootstrap
- `06_ROC_Curves`: ROC curves
- `07_Detailed_Model_Plots`: plot แยก model
- `08_PLSDA_Class_Violin_Plots`: plot class ของ PLS-DA
- `09_PLSDA_Score_Plots`: PLS-DA score plots
- `10_GradCAM_and_Distance_Mapping`: ผลวิเคราะห์เชิงรูปร่าง/attention รุ่นก่อน

เป็น output/reporting layer ไม่ใช่ source dataset หลัก และไม่ควรใช้ไฟล์ในนี้แทน
`OPTUNA_FINAL_SUMMARY.csv` เมื่อทำตัวเลข final ใหม่

## 13. `Model_Evaluation_and_Benchmarks`

`results_csv\bootstrap_1000_summary.csv` เป็น benchmark/report รุ่นก่อน ใช้ตรวจเทียบประวัติได้
แต่ผล final ที่แก้ leakage แล้วให้ยึด `Model\optuna_runs_10fold_all`

## 14. `tests`

- `test_pipeline_regressions.py`: ตรวจ data contract, label, patient split, reference, pipeline status,
  predictor schema และ evaluation utilities
- `smoke_legacy_svm.py`: smoke test รุ่นเก่าที่เก็บไว้เป็น reference ไม่ใช่ training command ปัจจุบัน

คำสั่งรัน:

```powershell
$Project = "C:\Users\IHCK\Desktop\17-9-2569\Hippocampal-Shape-Analysis-for-Epilepsy-Detection"
$Model = Join-Path $Project "Model"
$env:PYTHONPATH = "$Project;$Model;$(Join-Path $Model '_training_site')"
$PythonSlicer = "C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe"
& $PythonSlicer -W ignore::DeprecationWarning -m unittest discover -s (Join-Path $Project "tests") -v
```

ผลล่าสุด: 26 tests ผ่าน (`OK`)

## 15. ควรใช้โฟลเดอร์ไหนในแต่ละงาน

| งาน | โฟลเดอร์หลัก |
|---|---|
| ตรวจ MRI/segmentation | `FastSurfer`, `run_pipeline.py`, `ICP` |
| ตรวจ ICP | `ICP\output_left_hippocampus`, `ICP\output_right_hippocampus`, `Templates\ICP` |
| รัน SPHARM | `SPHARM`, `Templates\SPHARM`, `ICP\output_*\aligned_nifti` |
| ตรวจ split | `SPHARM\split_data` และ manifest ในโฟลเดอร์เดียวกัน |
| สร้าง coefficient dataset | `Data_Processing`, `Model\All_coef*` |
| สร้าง raw XYZ input | `Model\Output_Dataset` |
| รัน model เปรียบเทียบ | `Model\run_optuna_leakage_free_all.ps1` |
| ตรวจ tuning/final | `Model\optuna_runs_10fold_all` |
| ทำ inference | `DesktopApp`, model artifact และ feature contract |
| ทำกราฟ/รายงาน | `Visualize`, `Model_Results_Excel` |
| ตรวจ code | `tests` และ `Model\verify_leakage_free_pipeline.py` |

## 16. นโยบายการเก็บไฟล์

ควรเก็บ:

- `Templates`
- raw/current datasets และ manifest
- `_training_site`, `_training_cuda_site`
- `Model\optuna_runs_10fold_all`
- source code ที่ระบุใน `RERUN_GUIDE_TH.md`
- report และ audit ที่ใช้ในวิจัย

ไม่ควรนำมาเป็นผล final โดยอัตโนมัติ:

- `Model\All_Augment_tain`, `Model\Ds004469`, `Model\Ds005602`
- `Model\Legacy_*_READONLY`
- `SPHARM\output_left_new`
- `Model_Evaluation_and_Benchmarks\results_csv`
- smoke output หรือ summary ที่ไม่ได้อยู่ใน `optuna_runs_10fold_all`

ก่อนลบโฟลเดอร์ใด ๆ ให้ตรวจว่าไม่มี manifest, dataset หรือ output ที่ต้องใช้สำหรับการ rerun
และอย่าลบ runtime หรือ fixed reference โดยไม่สร้างทดแทนพร้อมตรวจผลใหม่

