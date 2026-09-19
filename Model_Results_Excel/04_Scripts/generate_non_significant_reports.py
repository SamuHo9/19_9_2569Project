"""
================================================================================
Generate Non-Significant P-Value Reports & Excel for Presentation
================================================================================
Extracts all non-significant pairs (p >= 0.05) from PValue_Significance_Summary.csv
and PLS-DA component analysis, and generates:
1. Non_Significant_Pairs_Summary.csv
2. Non_Significant_PValue_Report.md
3. Non_Significant_Statistical_Tests.xlsx (Formatted professional Excel workbook)
================================================================================
"""

import os
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def generate_reports():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(repo_root, "Model_Results_Excel")
    stats_dir = os.path.join(results_dir, "02_Bootstrap_and_Statistical_Tests")
    excel_dir = os.path.join(results_dir, "01_Excel_Workbooks")
    plsda_dir = os.path.join(results_dir, "08_PLSDA_Class_Violin_Plots")
    
    csv_path = os.path.join(stats_dir, "PValue_Significance_Summary.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Cannot find {csv_path}")
        return
        
    df_all = pd.read_csv(csv_path)
    df_ns = df_all[df_all["Significant"] == "No"].copy()
    
    # -------------------------------------------------------------------------
    # 1. Save CSV of all non-significant pairs
    # -------------------------------------------------------------------------
    ns_csv_path = os.path.join(stats_dir, "Non_Significant_Pairs_Summary.csv")
    df_ns.to_csv(ns_csv_path, index=False)
    print(f"Saved: {ns_csv_path} ({len(df_ns)} rows)")
    
    # -------------------------------------------------------------------------
    # 2. Extract PLS-DA Non-significant components
    # -------------------------------------------------------------------------
    plsda_ns_records = []
    for ds in ["Ds005602", "All_Augment_tain"]:
        for side in ["left", "right"]:
            p_csv = os.path.join(plsda_dir, ds, side, "plsda_components_class_stats.csv")
            if os.path.exists(p_csv):
                p_df = pd.read_csv(p_csv)
                # Components with p >= 0.05 or Significance == 'ns' or p > 0.04
                for _, r in p_df.iterrows():
                    is_ns = (r["P_Value"] >= 0.05) or (r["Significance"] == "ns")
                    plsda_ns_records.append({
                        "Dataset": ds,
                        "Side": side,
                        "Component": r["Component"],
                        "Is_Top3": r["Is_Top3"],
                        "Top3_Rank": r["Top3_Rank"] if r["Is_Top3"] else "Excluded",
                        "P_Value": r["P_Value"],
                        "Cohens_d": r["Cohens_d"],
                        "Significance": r["Significance"],
                        "Status": "Non-Significant (ns)" if is_ns else "Significant"
                    })
    df_plsda = pd.DataFrame(plsda_ns_records)
    
    # -------------------------------------------------------------------------
    # 3. Create Markdown Analysis Report
    # -------------------------------------------------------------------------
    md_path = os.path.join(stats_dir, "Non_Significant_PValue_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# รายงานสรุปผลการทดสอบทางสถิติที่ไม่พบนัยสำคัญ (Non-Significant P-Value Report)\n\n")
        f.write("> **เอกสารสำหรับเตรียมนำเสนอและตอบคำถามอาจารย์ที่ปรึกษา**  \n")
        f.write("> สรุปคู่โมเดลและพารามิเตอร์ PLS-DA ที่มีค่า $p \\ge 0.05$ (ไม่มีความแตกต่างทางสถิติอย่างมีนัยสำคัญ)\n\n")
        
        f.write("## 1. สรุปภาพรวมเชิงสถิติ (Statistical Overview)\n\n")
        total_tests = len(df_all)
        n_p01 = len(df_all[df_all["Significant"] == "Yes (p<0.01)"])
        n_p05 = len(df_all[df_all["Significant"] == "Yes (p<0.05)"])
        n_ns = len(df_ns)
        
        f.write(f"- **จำนวนคู่ทดสอบสมมติฐานทั้งหมด (Bootstrap Resamples):** {total_tests} คู่\n")
        f.write(f"- **มีนัยสำคัญสูงมาก ($p < 0.01$):** {n_p01} คู่ ({n_p01/total_tests*100:.1f}%)\n")
        f.write(f"- **มีนัยสำคัญ ($p < 0.05$):** {n_p05} คู่ ({n_p05/total_tests*100:.1f}%)\n")
        f.write(f"- **ไม่มีนัยสำคัญทางสถิติ ($p \\ge 0.05$):** {n_ns} คู่ ({n_ns/total_tests*100:.1f}%)\n\n")
        
        f.write("### ตารางแจกแจงจำนวนคู่ที่ไม่ Significant แยกตาม Metric\n\n")
        f.write("| Metric | Ds005602 (หลัก) | All_Augment_tain | Ds004469 (ภายนอก) | รวม |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m in ["AUC", "Accuracy", "F1_Class0", "F1_Class1", "Sensitivity_Class0", "Sensitivity_Class1", "Specificity_Class0", "Specificity_Class1"]:
            sub_m = df_ns[df_ns["Metric"] == m]
            c_ds = len(sub_m[sub_m["Dataset"] == "Ds005602"])
            c_aug = len(sub_m[sub_m["Dataset"] == "All_Augment_tain"])
            c_44 = len(sub_m[sub_m["Dataset"] == "Ds004469"])
            f.write(f"| **{m}** | {c_ds} | {c_aug} | {c_44} | {len(sub_m)} |\n")
        f.write("\n---\n\n")
        
        f.write("## 2. คู่เปรียบเทียบที่ไม่ Significant ด้าน ROC-AUC (ตัววัดหลัก)\n\n")
        f.write("จาก 126 คู่เปรียบเทียบ ROC-AUC ทั่วทั้งโครงการ มีเพียง **7 คู่เท่านั้นที่ไม่ Significant** ดังนี้:\n\n")
        f.write("| Dataset | ด้านสมอง | โมเดล A (AUC) | โมเดล B (AUC) | ค่า $p$-value | นัยสำคัญ |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: | :---: |\n")
        auc_ns = df_ns[df_ns["Metric"] == "AUC"]
        for _, r in auc_ns.iterrows():
            f.write(f"| {r['Dataset']} | {r['Side']} | {r['Model_A']} ({r['Mean_Value_A']:.4f}) | {r['Model_B']} ({r['Mean_Value_B']:.4f}) | **{r['P-Value']:.4f}** | ns |\n")
        f.write("\n")
        f.write("> **คำอธิบายทางวิชาการ:** ในชุดข้อมูลหลัก `Ds005602 ข้างขวา` MobileNet (0.9497) และ PointNet (0.9504) มีค่า $p = 0.4998$ ซึ่งแสดงว่าสถาปัตยกรรมทั้งสองให้ประสิทธิภาพการจำแนกพื้นที่ใต้กราฟ ROC ที่เทียบเท่ากัน\n\n")
        f.write("---\n\n")

        f.write("## 3. คู่เปรียบเทียบที่ไม่ Significant ด้าน Accuracy (ชุดข้อมูลหลัก Ds005602)\n\n")
        f.write("### ก. Hippocampus ข้างขวา (Ds005602 Right - โฟกัสหลักของวิจัย)\n\n")
        f.write("| กลุ่มประสิทธิภาพ | คู่โมเดลที่เทียบกัน | Accuracy A | Accuracy B | ค่า $p$-value | ความหมายเชิงการประยุกต์ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")
        f.write("| **Top Performers** | PointNet vs ResNet | 91.33% | 91.24% | **0.5082** | ความแม่นยำสูงเท่ากัน แต่ ResNet ทำ Grad-CAM ได้ |\n")
        f.write("| **Top Performers** | PointNet vs ResNet+AE | 91.33% | 91.23% | **0.5146** | ความแม่นยำไม่ต่างกันทางสถิติ |\n")
        f.write("| **Top Performers** | ResNet vs ResNet+AE | 91.24% | 91.23% | **0.8737** | การใส่ AE Reconstruction ไม่ได้ลดความแม่นยำ |\n")
        f.write("| **Baseline Models** | MLP vs SVM | 87.74% | 87.74% | **1.0000** | โมเดลพื้นฐานได้ผลลัพธ์เท่ากันเป๊ะ |\n")
        f.write("| **Baseline Models** | MLP vs MobileNet | 87.74% | 87.80% | **0.4602** | ไม่มีความแตกต่างทางสถิติ |\n")
        f.write("| **Baseline Models** | MobileNet vs SVM | 87.80% | 87.74% | **0.4602** | ไม่มีความแตกต่างทางสถิติ |\n\n")

        f.write("### ข. Hippocampus ข้างซ้าย (Ds005602 Left)\n\n")
        f.write("- โมเดลทั้ง 5 สถาปัตยกรรม (`MLP`, `MobileNet`, `ResNet`, `SqueezeNet`, `SVM`) มีค่า Accuracy เท่ากันเป๊ะที่ **94.64%** ทำให้ค่า $p = 1.0000$ ทั้งหมด 10 คู่เปรียบเทียบ\n\n")
        f.write("---\n\n")

        f.write("## 4. ผลลัพธ์ PLS-DA Components ที่ไม่ Significant (ก่อนทำ Distance Mapping)\n\n")
        f.write("| Dataset | ด้านสมอง | Component | ค่า $p$-value | Effect Size (Cohen's $d$) | สถานะ | ผลการคัดเลือก |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |\n")
        for _, r in df_plsda.iterrows():
            if r["Status"] == "Non-Significant (ns)" or r["P_Value"] > 0.02:
                f.write(f"| {r['Dataset']} | {r['Side']} | **{r['Component']}** | **{r['P_Value']:.4f}** | {r['Cohens_d']:.2f} | {r['Significance']} | {r['Top3_Rank']} |\n")
        f.write("\n")
        f.write("> **ข้อสรุปสำคัญ:** ใน `Ds005602 ข้างซ้าย` ตัว `PLS7` มีค่า **$p = 0.0545$ ($p > 0.05$)** ซึ่งเป็นตัวเดียวที่ **Non-Significant อย่างชัดเจน** จึงเป็นหลักฐานทางสถิติสนับสนุนการคัดเลือกเฉพาะ Top 3 Components ไปกวาดทำ Distance Mapping (-3.0 SD ถึง +3.0 SD)\n")

    print(f"Saved: {md_path}")
    
    # -------------------------------------------------------------------------
    # 4. Generate Professional Multi-Tab Excel Workbook
    # -------------------------------------------------------------------------
    excel_path = os.path.join(excel_dir, "Non_Significant_Statistical_Tests.xlsx")
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    
    # Styling helpers
    navy_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    dark_gray_fill = PatternFill(start_color="404040", end_color="404040", fill_type="solid")
    light_gray_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    accent_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Arial", size=14, bold=True, color="1F497D")
    sub_font = Font(name="Arial", size=10, italic=True, color="595959")
    data_font = Font(name="Arial", size=10)
    bold_data_font = Font(name="Arial", size=10, bold=True)
    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )
    
    # --- Sheet 1: Executive Summary ---
    ws1 = wb.create_sheet(title="Executive Summary")
    ws1.views.sheetView[0].showGridLines = True
    ws1["A1"] = "NON-SIGNIFICANT STATISTICAL TESTS SUMMARY (สรุปผลการทดสอบที่ไม่พบนัยสำคัญ)"
    ws1["A1"].font = title_font
    ws1["A2"] = "Summary of model pairs and PLS-DA parameters with p-value >= 0.05 (ns)"
    ws1["A2"].font = sub_font
    
    headers1 = ["Category", "Metric / Feature", "Key Finding (Non-Significant Pairs)", "P-Value Range", "Academic Interpretation & Implication"]
    for c_i, h in enumerate(headers1, 1):
        cell = ws1.cell(row=4, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
    summary_rows = [
        ("Model Performance", "ROC-AUC", "Only 7 of 126 pairs are non-significant (e.g. MobileNet vs PointNet)", "0.2055 - 1.0000", "Models are statistically distinct across 94.4% of comparisons. In top tiers, models achieve comparable discrimination."),
        ("Model Performance", "Accuracy (Ds005602 Right)", "PointNet (91.33%) vs ResNet (91.24%) vs ResNet+AE (91.23%)", "0.5082 - 0.8737", "Top 3 models achieve equivalent high accuracy. ResNet is selected for clinical deployment due to Grad-CAM explainability."),
        ("Model Performance", "Accuracy (Ds005602 Right Baseline)", "MLP (87.74%) vs MobileNet (87.80%) vs SVM (87.74%)", "0.4602 - 1.0000", "Baseline traditional models cluster around 87.7% without significant divergence."),
        ("Model Performance", "Accuracy (Ds005602 Left)", "MLP, MobileNet, ResNet, SqueezeNet, SVM all tied at 94.64%", "p = 1.0000", "Left hippocampus shows ceiling effect with identical accuracy across 5 architectures."),
        ("Morphological Shape", "PLS-DA Component 7 (Left)", "PLS7 score between Healthy vs TLE (p = 0.0545, ns)", "p = 0.0545", "Scientifically justifies excluding PLS7 and keeping only Top 3 components for Distance Mapping."),
        ("Morphological Shape", "PLS-DA Component 5 & 7 (Right)", "PLS7 (p = 0.039) and PLS5 (p = 0.028) have lowest effect sizes (d = 0.30)", "0.028 - 0.039", "Demonstrates why PLS1 & PLS3 (p < 1e-5, d > 0.68) were selected over weaker components.")
    ]
    
    for r_i, r_data in enumerate(summary_rows, 5):
        for c_i, val in enumerate(r_data, 1):
            cell = ws1.cell(row=r_i, column=c_i, value=val)
            cell.font = bold_data_font if c_i <= 2 else data_font
            cell.border = border_thin
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if r_i % 2 == 1:
                cell.fill = light_gray_fill
                
    # --- Sheet 2: ROC-AUC Non-Significant Pairs ---
    ws2 = wb.create_sheet(title="ROC-AUC (ns pairs)")
    ws2.views.sheetView[0].showGridLines = True
    ws2["A1"] = "Non-Significant Model Comparisons: ROC-AUC (p >= 0.05)"
    ws2["A1"].font = title_font
    
    headers2 = ["Dataset", "Side", "Model A", "Mean AUC A", "Model B", "Mean AUC B", "P-Value", "Significance", "Interpretation"]
    for c_i, h in enumerate(headers2, 1):
        cell = ws2.cell(row=3, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for r_i, r in enumerate(auc_ns.itertuples(), 4):
        ws2.cell(row=r_i, column=1, value=r.Dataset).font = data_font
        ws2.cell(row=r_i, column=2, value=r.Side).font = data_font
        ws2.cell(row=r_i, column=3, value=r.Model_A).font = bold_data_font
        ws2.cell(row=r_i, column=4, value=r.Mean_Value_A).font = data_font
        ws2.cell(row=r_i, column=5, value=r.Model_B).font = bold_data_font
        ws2.cell(row=r_i, column=6, value=r.Mean_Value_B).font = data_font
        cell_p = ws2.cell(row=r_i, column=7, value=r._8) # P-Value
        cell_p.font = bold_data_font
        cell_p.number_format = "0.0000"
        ws2.cell(row=r_i, column=8, value="No (ns)").font = data_font
        interp = "Comparable discrimination capacity (no statistical difference)"
        ws2.cell(row=r_i, column=9, value=interp).font = data_font
        for c in range(1, 10):
            ws2.cell(row=r_i, column=c).border = border_thin
            
    # --- Sheet 3: Accuracy Non-Significant Pairs (Ds005602) ---
    ws3 = wb.create_sheet(title="Accuracy Ds005602 (ns pairs)")
    ws3.views.sheetView[0].showGridLines = True
    ws3["A1"] = "Non-Significant Model Comparisons: Accuracy in Ds005602 Primary Dataset"
    ws3["A1"].font = title_font
    
    acc_ns_ds = df_ns[(df_ns["Metric"] == "Accuracy") & (df_ns["Dataset"] == "Ds005602")]
    headers3 = ["Dataset", "Side", "Model A", "Accuracy A (%)", "Model B", "Accuracy B (%)", "P-Value", "Significance"]
    for c_i, h in enumerate(headers3, 1):
        cell = ws3.cell(row=3, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for r_i, r in enumerate(acc_ns_ds.itertuples(), 4):
        ws3.cell(row=r_i, column=1, value=r.Dataset).font = data_font
        ws3.cell(row=r_i, column=2, value=r.Side).font = data_font
        ws3.cell(row=r_i, column=3, value=r.Model_A).font = bold_data_font
        ws3.cell(row=r_i, column=4, value=r.Mean_Value_A * 100).number_format = "0.00"
        ws3.cell(row=r_i, column=5, value=r.Model_B).font = bold_data_font
        ws3.cell(row=r_i, column=6, value=r.Mean_Value_B * 100).number_format = "0.00"
        cell_p = ws3.cell(row=r_i, column=7, value=r._8)
        cell_p.font = bold_data_font
        cell_p.number_format = "0.0000"
        ws3.cell(row=r_i, column=8, value="No (ns)").font = data_font
        for c in range(1, 9):
            ws3.cell(row=r_i, column=c).border = border_thin
            
    # --- Sheet 4: PLS-DA Components ---
    ws4 = wb.create_sheet(title="PLS-DA Components Stats")
    ws4.views.sheetView[0].showGridLines = True
    ws4["A1"] = "PLS-DA Component Latent Scores: Statistical Tests by Class"
    ws4["A1"].font = title_font
    
    headers4 = ["Dataset", "Side", "Component", "Is Top 3", "Top 3 Rank", "P-Value", "Cohen's d (Effect Size)", "Significance", "Status"]
    for c_i, h in enumerate(headers4, 1):
        cell = ws4.cell(row=3, column=c_i, value=h)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for r_i, r in enumerate(df_plsda.itertuples(), 4):
        ws4.cell(row=r_i, column=1, value=r.Dataset).font = data_font
        ws4.cell(row=r_i, column=2, value=r.Side).font = data_font
        ws4.cell(row=r_i, column=3, value=r.Component).font = bold_data_font
        ws4.cell(row=r_i, column=4, value="Yes" if r.Is_Top3 else "No").font = data_font
        ws4.cell(row=r_i, column=5, value=str(r.Top3_Rank)).font = data_font
        cell_p = ws4.cell(row=r_i, column=6, value=r.P_Value)
        cell_p.font = bold_data_font
        cell_p.number_format = "0.0000" if r.P_Value >= 0.0001 else "0.00E+00"
        ws4.cell(row=r_i, column=7, value=r.Cohens_d).number_format = "0.00"
        ws4.cell(row=r_i, column=8, value=r.Significance).font = bold_data_font
        cell_st = ws4.cell(row=r_i, column=9, value=r.Status)
        cell_st.font = bold_data_font
        if r.Status == "Non-Significant (ns)":
            cell_st.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
            cell_p.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
        for c in range(1, 10):
            ws4.cell(row=r_i, column=c).border = border_thin
            
    # Auto-fit columns for all sheets
    for ws in [ws1, ws2, ws3, ws4]:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
    wb.save(excel_path)
    print(f"Saved Excel Workbook: {excel_path}")

if __name__ == "__main__":
    generate_reports()
