# รายละเอียดการปรับปรุงระบบและผลตรวจสอบ

วันที่: 18 กันยายน 2026

ปรับโค้ดเพื่อให้การประมวลผล การแบ่งข้อมูล และการวัดผลตรวจสอบได้มากขึ้น พร้อมทดสอบส่วนข้อมูลและ SVM จริงแล้ว งานนี้ยังไม่ใช่การรับรองความถูกต้องของการวินิจฉัย และยังไม่ได้รัน MRI → Desktop ครบทั้งระบบบนภาพใหม่หลังแก้ไข

## ระบบทำอะไร และ preprocessing อยู่ที่ไหน

ระบบศึกษาความสัมพันธ์ระหว่างรูปร่างฮิปโปแคมปัสกับ label กลุ่มโรคลมชักชนิด temporal lobe epilepsy (TLE) โดยแยกประมวลผลซ้าย/ขวา:

```text
MRI
 → image preprocessing/conform ภาพตาม FastSurfer
 → FastSurfer segmentation
 → extraction: แยก mask hippocampus ซ้าย/ขวา
 → mask preprocessing: closing/fill holes ตามโหมด
 → เตรียม mesh และ ICP: จัดตำแหน่ง/แนว/สเกลเทียบ reference
 → label preprocessing ก่อน SPHARM และตรวจคุณภาพ
 → SPHARM: แทนรูปร่างด้วย spherical harmonic coefficients/จุดสามมิติ
 → features + ตรวจ label/schema/ประวัติ preprocessing
 → แบ่งผู้ป่วย train/validation/test
 → fit scaler/PLS และ augmentation เฉพาะ train fold ใน baseline ใหม่
 → training / model selection / evaluation
 → บรรจุโมเดลพร้อม preprocessing ที่ตรงกัน
 → Desktop inference และบันทึกผล
```

Preprocessing จึงมีทั้งระดับภาพ, mask, geometry และ feature การ fit ขั้นตอนที่เรียนรู้จากข้อมูลต้องไม่ใช้ validation/test ของการประเมินนั้น ตัวเลือก groupwise ICP เดิมยังมีไว้สำรวจข้อมูล แต่ไม่ใช่เส้นทางยืนยันการทำนายผู้ป่วยใหม่

## รายการแก้ไข

### 1. MRI/extraction: ไม่รายงานสำเร็จเมื่อข้อมูลไม่ครบ

ไฟล์: `run_pipeline.py`

- เพิ่ม `numpy` ที่ขาด ซึ่งเดิมทำให้เส้นทางบันทึก paired MRI ล้มเหลว
- สร้าง NIfTI จาก array กับ affine โดยไม่ส่ง header ของ MGZ ที่ไม่เข้ากัน
- ส่งสถานะล้มเหลวกลับเมื่อ paired MRI ที่จำเป็นบันทึกไม่ได้ และจบคำสั่งด้วย exit code ที่สะท้อนผลจริง
- การข้ามงานที่ทำแล้วตรวจว่ามี paired MRI ด้วยในโหมดที่ต้องใช้
- โหมดข้าม FastSurfer ไม่บังคับตรวจ checkpoint ของขั้นตอนที่ไม่ได้เรียก
- แยก temporary directory ต่อรอบและปฏิเสธชื่อ subject ซ้ำจากคนละ input เพื่อป้องกันการทับกัน

### 2. ICP: เพิ่ม reference คงที่สำหรับข้อมูลใหม่

ไฟล์: `ICP/ICP.py`, `ICP/reference_contract.py`, `DesktopApp/LeftUI/icp_panel.py`

- รองรับ `--reference_template` จริง เดิมคำสั่งจาก GUI ไม่ได้ทำให้ alignment ใช้ reference คงที่ตามที่คาด
- เพิ่ม `--fit_reference` เพื่อสร้าง reference จาก training masks และบันทึก physical scale, grid, hash ของ template และแหล่งข้อมูลลงไฟล์ `.json` คู่กัน
- fixed-reference mode ใช้สเกลและ template ที่บันทึกไว้ ไม่คำนวณใหม่จากผู้ป่วยอื่นใน batch
- ตรวจ template ถูกเปลี่ยน, metadata หาย, grid ไม่เหมาะสม และรูปร่างหลุดกรอบ
- เพิ่ม margin ตอนกำหนดกรอบ และเปลี่ยนการ export ที่ผิดพลาด/ไม่ผ่านเกณฑ์ QC ให้หยุดพร้อมสถานะไม่สำเร็จ
- แก้ชื่อพารามิเตอร์ GUI ให้ตรงกับ CLI; กำหนด reference แยกข้างผ่าน `HIPPO_ICP_REFERENCE_LEFT` และ `HIPPO_ICP_REFERENCE_RIGHT`

**ผลต่อการใช้งาน:** template เก่าที่ไม่มี metadata จะใช้ fixed-reference mode ไม่ได้ ต้องสร้าง reference ใหม่จาก train เท่านั้น แล้วประมวลผลทั้ง train และ test ด้วย reference เดียวกัน การเปลี่ยน geometry นี้ต้องสร้าง features และฝึกโมเดลใหม่ ห้ามเติม metadata เดา ๆ ให้ weights เก่า

### 3. SPHARM: ตรวจผลจริงและที่มาของผลลัพธ์

ไฟล์: `SPHARM/run_spharm_batch.py`, `SPHARM/realign_spharm.py`, `DesktopApp/LeftUI/spharm_panel.py`

- ตรวจสถานะ CLI แบบแยกตามโมดูล: ขั้นเตรียมข้อมูลต้องเป็น `Completed` ส่วน GenParaMesh/ParaToSPHARMMesh ของ SlicerSALT อาจรายงาน `Completed with errors` หลังเขียนผลสำเร็จ จึงยอมรับได้เฉพาะเมื่อ mesh มีจุดจริง, VTK มี geometry, coefficient และ grid ผ่านการตรวจครบ
- การประมวลผลแต่ละ subject ส่งคืนผลสำเร็จ/ล้มเหลวจริง และ batch หยุดด้วยข้อผิดพลาดเมื่อมีงานล้มเหลว
- บันทึก `spharm_status_shardN.json` และ `_processing.json` เพื่อผูกผลกับ input, template, config และ geometry
- status รุ่นปัจจุบันบันทึกโหมด, `iter/subdiv/degree`, input/output directory, required suffixes และ SHA-256 ของ template เพื่อแยกผลคนละมาตรฐานออกจากกันได้
- แก้จุดที่ `run_batch_spharm()` เดิมละเลยค่า `False` จาก `process_single_subject()` ทำให้ mesh ว่างยังถูกพิมพ์ว่า completed; ตอนนี้ตรวจ `Completed` เท่านั้น ตรวจ ParaToSPHARMMesh/grid และคืน exit code ไม่สำเร็จเมื่อมี subject ตก
- เพิ่ม `SPHARM/verify_spharm_outputs.py` สำหรับตรวจ input ทุกตัวกับ `.coef`, `.vtk`, `_grid.vtk` และ `_ellalign.coef` จาก Terminal
- ทำคำสั่งซ้าย/ขวาให้ใช้มาตรฐานเดียวกันผ่าน `SPHARM/run_spharm.bat left|right` และ wrapper `run_spharm_parallel_left.bat`/`run_spharm_parallel_right.bat`: production `iter=1000`, `subdiv=10`, `degree=12`, grid 4.5°×4.5°, Slicer flags และ validation ชุดเดียวกัน โดยเลือก template ซ้าย/ขวาตาม hemisphere
- เพิ่ม Windows error mode เพื่อไม่ให้ native access violation ของ `ParaToSPHARMMeshCLP.exe` เปิด dialog ค้างทั้ง batch และกำหนด wrapper มาตรฐานเป็น 5 workers เพื่อให้ซ้าย/ขวาใช้ความขนานเท่ากัน โดย subject ที่มี topology ผิดปกติยังถูกบันทึกเป็น failed รายตัว
- ข้ามผลเดิมได้เมื่อ fingerprint ตรงและผลที่จำเป็นมีครบ ลดการใช้ไฟล์ค้างจากการตั้งค่ารอบก่อน
- ตรวจ shard, input ว่าง, reference ที่หาย และ dependency ของ preprocessing
- แก้ GUI ให้ส่ง `--template` และไม่ส่ง flags ที่ realignment ไม่รองรับ พร้อมตรวจ exit code

### 4. Feature/label: ไม่แปลงข้อมูลไม่ทราบกลุ่มเป็น Healthy

ไฟล์: `Data_Processing/extract_ml_features_coef.py`, `Data_Processing/feature_metadata.py`, สคริปต์ augmentation สองไฟล์

- Unknown ใช้ `BinaryClass=-1` และตัวโหลดฝึกจะปฏิเสธ แทนการตกเป็น class 0
- เพิ่ม `--labels_csv` โดยต้องมี `Subject,BinaryClass`; เพิ่ม `Class,Group` ได้ ตรวจค่าซ้ำ/ความสอดคล้องของ label
- เมื่อใช้ label CSV ต้องมี label ครบตาม Subject ที่ extractor สร้าง ไม่เดา label ที่ขาดจากชื่อไฟล์
- ตรวจทั้ง batch ก่อนเปิด CSV: coefficients ต้องมีจำนวนตรงกัน, ไม่ว่าง, ไม่มี NaN/Inf และใช้ preprocessing contract เดียวกัน
- เขียน sidecar ของ feature CSV พร้อม hash และแหล่ง label
- บังคับให้ coefficient ทุกตัวมี `<subject>_processing.json` ที่มี `feature_contract`; ถ้า provenance หายหรือไม่ใช่ object จะหยุดทันที ไม่ปล่อยให้ไฟล์เก่าถูกตีความว่าใช้ fixed reference
- augmentation แบบ interpolation ไม่ผสม parent ข้าม binary class; ปฏิเสธ Unknown

**นิยามที่ยังคงเดิม:** class 0 อาจรวม contralateral ตามชุดข้อมูลเดิม จึงไม่ควรแปลว่าทุกตัวอย่างใน class 0 เป็นคนสุขภาพดี การเปลี่ยนนิยามโรคต้องยืนยันกับข้อมูลกำกับและออกแบบการทดลองใหม่

### 5. การแบ่งข้อมูล: ทุกสถาปัตยกรรมใช้ผู้ป่วยชุดเดียวกัน

ไฟล์หลัก: `Model/data_contract.py`; เชื่อมสคริปต์ฝึกเดิม **74 entry points**

- โหลดข้อมูลต้นฉบับจาก split ที่กำหนดร่วมกัน แทนแต่ละโมเดลเลือก CSV ต่างกัน
- ตรวจตัวตนผู้ป่วย/label ของ coefficient และ XYZ ให้ตรงกัน จึงเปรียบเทียบ SVM กับ PointNet ได้บน test cohort เดียวกัน
- ตรวจชื่อซ้ำ, train/test overlap, feature ซ้ำข้าม split, schema, finite values และ synthetic rows
- สคริปต์ฝึกเดิมใช้ original subjects เท่านั้น ไม่ใช้ข้อมูลที่ augment ทั้งชุดมาก่อน CV
- บันทึก subject IDs, protocol และ input identity/hash ลงผลรอบใหม่
- ทุกครั้งสร้างโฟลเดอร์ใหม่ใน `Model/validated_runs/` ชื่อโฟลเดอร์นี้หมายถึง workflow ที่เพิ่มการตรวจ ไม่ใช่ clinical validation

จำนวน train/test ที่ตรวจจริงสำหรับแต่ละข้าง:

| Cohort | ซ้าย | ขวา |
|---|---:|---:|
| Ds005602 | 223 / 56 | 228 / 57 |
| Ds004469 | 41 / 11 | 43 / 11 |
| All_Augment_tain และ combind_methode | 264 / 67 | 271 / 68 |
| Ds005602Train_Ds004469test | 279 / 52 | 285 / 54 |
| Ds004469Train_Ds005602test | 52 / 279 | 54 / 285 |

### 6. เพิ่ม baseline แบบ nested CV และ augmentation ภายใน fold

ไฟล์: `Model/validated_model.py`, `Model/train_validated_baseline.py`

- outer folds วัดผลกับผู้ป่วยที่ไม่ได้ใช้เลือก hyperparameters ของ fold นั้น
- inner folds เลือกจำนวน PLS components และค่า C
- scaler, PLS และ synthetic interpolation fit จาก training fold เท่านั้น; validation/test เป็นข้อมูลต้นฉบับ
- บันทึก out-of-fold predictions, test predictions, โมเดล และรายละเอียด fold/ผู้ป่วย/ค่าที่เลือก
- ปรับเพดาน PLS และ batch size ของสคริปต์เดิมให้รองรับชุดต้นฉบับที่เล็กลง; PointNet แบบ combined ใช้ loader และ stratification ร่วมกัน

**ขอบเขต:** ยังไม่ได้แปลงทั้ง 74 สคริปต์เป็น nested CV ค่า CV ของสคริปต์เดิมที่ใช้เลือกโมเดลยังเป็น selection CV ไม่ใช่ค่าประมาณอิสระแบบ nested CV

### 7. Evaluation: AUC และคำทำนายต้องตรงกับโมเดลจริง

ไฟล์: `Model/evaluation.py`, `Model/run_bootstrap_on_all_trained_models.py`, `Model/compare_runs.py`

- ใช้ `roc_auc_score` แทน double argsort ที่จัดการคะแนนเสมอผิด; ทดสอบกรณีทุกคะแนนเท่ากันได้ AUC 0.5
- bootstrap ใช้ saved `y_pred`; ไม่สร้างคำตอบใหม่จาก threshold 0.5 ซึ่งอาจต่างจาก `SVC.predict`
- ปฏิเสธ label/probability ผิดรูปแบบ และนับ bootstrap รอบที่ข้ามเพราะเหลือ class เดียว
- ใช้ RNG เฉพาะส่วนเพื่อทำซ้ำได้โดยไม่เปลี่ยน random state ของระบบอื่น
- `compare_runs.py` ตรวจ subject IDs, labels และ protocol ก่อนเปรียบเทียบ แม้ลำดับแถวต่างกัน; ห้ามเขียนทับไฟล์ output ที่มีแล้ว
- สรุป legacy bootstrap ใช้ชื่อใหม่พร้อมระบุว่า provenance เก่ายังไม่ผ่านการยืนยัน

### 8. รายงานเดิม: หยุดการสร้างตารางที่อ้างผลเก่าอย่างไม่ถูกต้อง

ไฟล์: `Model_Results_Excel/04_Scripts/generate_model_excel.py`, `generate_newest_bootstrap_results.py`

- ปิด entry point สองตัวนี้ด้วยข้อความอธิบายและชี้ไป `Model/compare_runs.py` เพราะอ่านผลเก่าที่ไม่ยืนยัน test cohort และอาจทับรายงานเดิม
- เอา t-test บน bootstrap replicates ออก ไม่แทนด้วย p-value ที่แต่งขึ้น
- ผลเปรียบเทียบใหม่เป็น CSV จาก predictions ที่ระบุตัวผู้ป่วยได้ ยังไม่ได้สร้าง workbook รูปแบบเดิมทดแทน
- ไฟล์ Excel/Word/ภาพทางสถิติที่มีอยู่ก่อนยังเป็นหลักฐานการทดลองเก่า การเปิดไฟล์ได้ไม่ทำให้ข้อสรุปเดิมได้รับการยืนยันใหม่

### 9. Desktop inference: ตรวจลำดับฟีเจอร์และรุ่น preprocessing

ไฟล์: `DesktopApp/models/predictor.py`, `DesktopApp/LeftUI/result_panel.py`

- รักษาชื่อคอลัมน์ DataFrame/Series แล้วเรียงตาม scaler/feature schema ที่ฝึกไว้
- ปฏิเสธคอลัมน์ซ้ำ/ขาด/เกิน, รูปร่าง input ผิด, input ว่าง และ NaN/Inf
- coefficient ที่มี preprocessing contract ใหม่ต้องตรงกับ `model_manifest.json` ของโมเดล และต้องเป็น fixed-reference mode
- ไม่รายงานสำเร็จเมื่อทำนายทั้งหมดล้มเหลวหรือบันทึกผลไม่ได้

**ยังไม่ได้เปลี่ยน weights ของ Desktop:** baseline SVM ใหม่ไม่ใช่ checkpoint ResNet ทดแทนโดยอัตโนมัติ เส้นทาง legacy ที่ไม่มี contract ยังไม่ได้รับการยืนยันทาง geometry การใช้งานรุ่นใหม่ต้องฝึก/บรรจุโมเดลที่ตรงกันก่อน

### 10. คำสั่งรันและ dependencies

ไฟล์: `run_everything.bat`, `setup.bat`, `Model/train_all_6_datasets.py`, `Model/run_dataset.ps1`, launchers ของ dataset และ requirements

- หยุดเมื่อขั้นตอนก่อนหน้าส่ง exit code ล้มเหลว แทนการไปต่อแล้วแจ้งสำเร็จ
- ฝึกครบรายการ 6 datasets ตามชื่อ runner, จัดเส้นทาง combined PointNet ให้ถูก และสร้าง log/summary แยกแต่ละรอบ
- เพิ่ม dependency ที่ขาด พร้อม `requirements-training.txt` และ `requirements-desktop.txt`
- README ระบุ Python 3.10+ ตามความต้องการ FastSurfer และเชื่อมรายงานนี้

## ผลทดสอบที่ทำจริง

- ใช้ PythonSlicer 3.9.10 ที่มีอยู่ในเครื่องรัน regression tests และ sklearn; ไม่ได้ติดตั้ง dependencies เพิ่ม
- รายละเอียดผลปัจจุบัน: `regression_results.txt`, `regression_summary.json`
- ผลตรวจล่าสุด: **26 tests ผ่านทั้งหมด**, syntax ของ Python **156 ไฟล์ไม่พบข้อผิดพลาด** (ไม่นับ source ภายนอก FastSurfer/backup/build ตามตัวกรองใน `verify_changes.py`)
- ทดสอบ schema/predictor, unknown labels, reference metadata/hash, failure propagation, bootstrap, augmentation ภายใน fit, cohort ทั้ง 12 คู่ dataset/side และสคริปต์ฝึก 74 ตัว
- ส่วนที่ต้องใช้ Slicer/VTK/MRI/Qt/torch บางรายการใช้ AST extraction และ mocks เพื่อทดสอบตรรกะ จึงไม่ใช่ end-to-end test
- ตรวจ syntax ของ source Python และ parser ของ PowerShell launchers
- ฝึก nested SVM จริงและบันทึกผลใน `Model/validated_runs/Ds004469/left/nested_svm/20260918_172830_yrqfu93d/`
- ทดสอบส่วนโมเดลจาก SVM entry point เดิมจริง โดยแทนเฉพาะ plotting ด้วย stub เพราะไม่มี matplotlib; ผลใน `Model/validated_runs/Ds004469/left/SVM/20260918_173547_train_svm_pls_l1wj1hch/`
- การเรียก legacy SVM ทั้งไฟล์โดยตรงยังติด matplotlib ที่ขาด มีหลักฐานใน `legacy_svm_smoke.txt`; ผลทดสอบแยกส่วนอยู่ใน `legacy_svm_model_smoke.txt`
- เปรียบเทียบ predictions สองรอบสำเร็จหลังตรวจ identities: `Model/validated_runs/smoke_comparison.csv`

การตรวจ artifact รอบมาตรฐานล่าสุดพบว่า ICP มี `icp_status.json` สำเร็จ 381 รายต่อข้าง แต่เป็น `exploratory_groupwise` และ `reference_sha256=null` จึงยังไม่ใช่ fixed training reference. SPHARM ใช้คำสั่งและสัญญาเดียวกันทั้งสองข้าง (`iter=1000`, `subdiv=10`, `degree=12`, grid 4.5°×4.5°, template แยกตาม hemisphere, validation ชุดเดียวกัน) และมีรายงานรวมใน `bilateral_spharm_verification.json` โดยเปิด deep VTK geometry check แล้ว ผลคือซ้าย **373/381** และขวา **377/381** ครบ artifact ที่จำเป็นและไม่มี VTK geometry ที่เสียใน 750 ผลลัพธ์ที่ตรวจได้ รวมยังตก 12 ราย (ซ้าย 8 รายจาก `GenParaMesh` ได้ mesh ว่าง และขวา 4 รายที่ยังสร้างผลไม่ครบ). ตัวอย่าง input จับคู่ตาม subject ได้ **381/381**; label ซ้าย/ขวาต่างกัน 246 ราย ซึ่งเป็นข้อมูลที่ต้องยืนยันกับ label manifest ว่าหมายถึงสถานะ hippocampus รายข้าง ห้ามสกัด feature จาก subject ที่ไม่ผ่าน verifier; Current SPHARM artifacts also lack provenance sidecars, so feature extraction must wait for a rerun that writes a fixed-reference contract.

ผล nested baseline: Ds004469 ซ้าย, train 41, test 11, outer CV 3 folds, augmentation 1 ลูกต่อ training subject:

| Metric | Nested out-of-fold | Test |
|---|---:|---:|
| Accuracy | 73.17% | 63.64% |
| Sensitivity ของ class 1 | 16.67% | 0.00% |
| Specificity ของ class 0 | 96.55% | 87.50% |
| Macro F1 | 0.5512 | 0.3889 |
| ROC AUC | 0.5431 | 0.4583 |

ใน test มี class 1 เพียง 3 ราย และรอบนี้ทำนายผิดทั้ง 3 ราย ส่วน class 0 ถูก 7/8 ราย **ผลนี้ยังไม่สนับสนุนการใช้งานวินิจฉัย** และไม่ควรเหมารวมว่าเป็นผลของทุกสถาปัตยกรรม ผลรอบนี้ใช้ historical features ซึ่งยังพิสูจน์ไม่ได้ว่า geometry เดิมเป็นอิสระจาก test ดังนั้นเป็นการตรวจ workflow ฝึก/ประเมินที่แก้ ไม่ใช่ผลยืนยันระบบภาพรุ่นใหม่

## วิธีรันซ้ำและย้ายไป preprocessing รุ่นใหม่

จาก root repository ใน environment ที่ติดตั้ง dependency ครบ:

```powershell
python verify_changes.py
python SPHARM/verify_spharm_outputs.py --input_dir ICP/output_left_hippocampus/aligned_nifti --output_dir ICP/output_left_hippocampus
python SPHARM/verify_spharm_outputs.py --input_dir ICP/output_right_hippocampus/aligned_nifti --output_dir ICP/output_right_hippocampus
python SPHARM/verify_bilateral_outputs.py --left_input_dir ICP/output_left_hippocampus/aligned_nifti --left_output_dir ICP/output_left_hippocampus --right_input_dir ICP/output_right_hippocampus/aligned_nifti --right_output_dir ICP/output_right_hippocampus --report bilateral_spharm_verification.json --deep
python Model/train_validated_baseline.py --cohort Ds004469 --side left --outer-folds 3 --children 1 --components 2 5
python Model/compare_runs.py path/to/run_A/test_predictions.npz path/to/run_B/test_predictions.npz --output comparison_new.csv
```

baseline อ่าน canonical CSV ของ repository ตาม `Model/data_contract.py` ไม่ใช่รับ CSV ใดก็ได้ผ่านคำสั่งนี้ อย่านำผลของ historical features ไปอ้างว่าได้ผ่าน preprocessing รุ่นใหม่แล้ว

ลำดับ migration ที่ยังต้องรันกับข้อมูลจริง:

1. ยืนยัน subject IDs, label และ train/test split ก่อนสร้าง template; ถ้ามีหลาย scan ต่อคนต้องแยกตามคน ไม่ให้ข้าม fold
2. สร้าง ICP reference แยกข้างจาก training masks เท่านั้นด้วย `--fit_reference`; เก็บ `mean_shape.ply` และ `.json` คู่กัน
3. รัน ICP ทั้ง train และ test ด้วย `--reference_template` ชี้คู่ไฟล์นั้น โดยใช้ output directory ใหม่; ห้ามใช้ output ของ groupwise fit ปะปนกับ fixed-reference output
4. ใช้ SPHARM template ที่มาจาก train และ freeze ค่า preprocessing/degree/grid ให้ตรงกัน ตรวจ segmentation และ mesh ด้วยข้อมูลจริง
5. สกัด features ใหม่พร้อม label CSV และ metadata; จัด canonical splits ให้ตรงกับ loader พร้อมเก็บชุดเก่าแยกไว้ก่อน
6. ฝึกและประเมินใหม่ด้วย fold-local preprocessing; หากจะอ้าง CV ที่ครอบคลุมการเรียนรู้ geometry ต้องสร้าง reference ภายในแต่ละ training fold ด้วย baseline ปัจจุบันยังไม่ orchestration ขั้นภาพภายใน folds
7. ฝึกโมเดลที่จะใช้ใน Desktop จริง บันทึก weights/scaler/PLS/schema/manifest ที่ตรงกัน และทดสอบ round-trip ว่า training กับ Desktop ให้ผลเหมือนกัน
8. ประเมินบนผู้ป่วยและแหล่งข้อมูลภายนอกที่ไม่ได้ใช้เลือกโมเดล พร้อมตรวจนิยาม target และ calibration ก่อนพิจารณาการใช้งานกับคน

ตัวอย่าง ICP CLI ผ่าน SlicerSALT (เปลี่ยน placeholder เป็น directory จริง; ยังไม่ได้รันคำสั่งตัวอย่างนี้):

```powershell
& 'C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe' --no-main-window --python-script ICP/ICP.py --input_dir TRAIN_LEFT_MASKS --output_dir NEW_LEFT_REFERENCE --fit_reference
& 'C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe' --no-main-window --python-script ICP/ICP.py --input_dir LEFT_MASKS_TO_TRANSFORM --output_dir NEW_LEFT_FIXED --reference_template NEW_LEFT_REFERENCE/mean_shape.ply
```

## ข้อจำกัดและการเก็บงานเดิม

- ยังไม่ได้ทดสอบ FastSurfer inference, MRI export และ Desktop GUI ครบวงจรจริงหลังแก้; ส่วน ICP/SPHARM CLI รันกับข้อมูลจริงแล้ว แต่ยังมี SPHARM 12 รายที่ต้องแก้ segmentation/topology หรือทำ manual QC ก่อนนำไปสร้าง feature
- environment ที่ใช้ทดสอบไม่มี torch, nibabel, PyQt6, matplotlib และ VTK import ติด DLL จึงยังไม่ได้ฝึก neural models ทั้งหมดหรือแทน weights ในแอป
- ไม่มีการยืนยันด้าน sensitivity/calibration/external validation สำหรับโมเดลใช้งานจริงจากงานแก้โค้ดนี้
- เก็บสำเนา source/config ก่อนแก้ใน `maintenance_backup_20260918/`; ไม่ได้ reset การแก้ที่มีมาก่อน และผลทดลองใหม่อยู่ใน directory ใหม่
- สคริปต์ `maintenance_*.py` เป็นบันทึกการย้ายโค้ดครั้งนี้ ไม่ใช่คำสั่งใช้งานประจำและไม่ควรรันซ้ำ
- รายงานตรวจครั้งแรกอยู่ใน `AUDIT_REPORT_TH.md`; `audit_results.json` เป็นผลก่อนแก้ ส่วน `regression_summary.json` เป็นการตรวจ source ปัจจุบัน
