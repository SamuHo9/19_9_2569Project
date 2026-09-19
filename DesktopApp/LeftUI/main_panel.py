import os
import sys
import json
import glob
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
    QLineEdit, QGroupBox, QMenu, QTabBar
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QColor, QFont

class MainPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str)
    signal_diagnostic_info = pyqtSignal(str)

    def __init__(self, import_panel=None, fastsurfer_panel=None, icp_panel=None, spharm_panel=None, result_panel=None, parent=None):
        super().__init__(parent)
        self.import_panel = import_panel
        self.fastsurfer_panel = fastsurfer_panel
        self.icp_panel = icp_panel
        self.spharm_panel = spharm_panel
        self.result_panel = result_panel
        self.is_running = False
        self.all_pipeline_results = []

        self.load_history()
        self.setup_ui()
        self.connect_pipeline_signals()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_stage_preview()
        self.populate_main_table()

    def load_history(self):
        self.history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app_history.json")
        self.recent_dirs = []
        self.last_input_dir = None
        self.last_output_dir = None
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recent_dirs = data.get("recent_directories", [])
                    self.last_input_dir = data.get("last_input_directory", None)
                    self.last_output_dir = data.get("last_output_directory", None)
            except Exception:
                pass

    def save_history(self):
        data = {
            "recent_directories": self.recent_dirs,
            "last_input_directory": self.last_input_dir,
            "last_output_directory": self.last_output_dir
        }
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def add_to_history(self, directory):
        if not directory or not os.path.isdir(directory):
            return
        if directory in self.recent_dirs:
            self.recent_dirs.remove(directory)
        self.recent_dirs.insert(0, directory)
        self.recent_dirs = self.recent_dirs[:10]
        self.last_input_dir = directory
        self.save_history()

    @staticmethod
    def find_valid_mri_files(directory):
        """
        Scans for valid MRI images only (.nii.gz, .nii, .mgz).
        Excludes non-MRI files, masks, segmentations, and 3D meshes.
        """
        if not directory or not os.path.isdir(directory):
            return []

        mri_extensions = (".nii.gz", ".nii", ".mgz")
        skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk"]

        files = []
        for ext in mri_extensions:
            files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

        seen = set()
        valid_mris = []
        for f in files:
            norm = os.path.normpath(f)
            if norm in seen:
                continue
            seen.add(norm)

            fname = os.path.basename(norm).lower()
            if any(keyword in fname for keyword in skip_keywords):
                continue

            valid_mris.append(norm)

        return valid_mris

    @staticmethod
    def check_existing_stages(output_dir):
        """
        Robustly checks which pipeline stages already have generated results in output_dir.
        Returns: (has_fastsurfer: bool, has_icp: bool, has_spharm: bool, has_result: bool)
        """
        if not output_dir or not os.path.isdir(output_dir):
            return False, False, False, False

        # 1. Check FastSurfer results
        fs_dir = os.path.join(output_dir, "fastsurfer")
        fs_files = []
        if os.path.isdir(fs_dir):
            for ext in ("*.nii*", "*.vtk"):
                fs_files.extend(glob.glob(os.path.join(fs_dir, ext)))
                fs_files.extend(glob.glob(os.path.join(fs_dir, "**", ext), recursive=True))
        has_fastsurfer = len(fs_files) > 0

        # 2. Check ICP results
        icp_dir = os.path.join(output_dir, "output_ICP")
        icp_files = []
        if os.path.isdir(icp_dir):
            for ext in ("*.vtk", "*.ply", "*.nii*"):
                icp_files.extend(glob.glob(os.path.join(icp_dir, ext)))
                icp_files.extend(glob.glob(os.path.join(icp_dir, "**", ext), recursive=True))
            # Filter out reference templates and mean shapes
            icp_files = [f for f in icp_files if "mean_shape" not in os.path.basename(f).lower() and not os.path.basename(f).lower().startswith("template_")]
        has_icp = len(icp_files) > 0

        # 3. Check SPHARM results
        spharm_dir = os.path.join(output_dir, "output_SPHARM")
        sph_files = []
        if os.path.isdir(spharm_dir):
            for ext in ("*SPHARM*.vtk", "*.vtk"):
                sph_files.extend(glob.glob(os.path.join(spharm_dir, ext)))
                sph_files.extend(glob.glob(os.path.join(spharm_dir, "**", ext), recursive=True))
            sph_files = [
                f for f in sph_files 
                if "mean_shape" not in os.path.basename(f).lower() 
                and not os.path.basename(f).lower().startswith("template_") 
                and not any(aux in os.path.basename(f).lower() for aux in ("_para.", "_surf.", "medialaxis", "_grid."))
            ]
        has_spharm = len(sph_files) > 0

        # 4. Check Result panel results (ResNet prediction & 3D Grad-CAM)
        res_dir = os.path.join(output_dir, "output_Result")
        summary_json = os.path.join(res_dir, "evaluation_summary.json")
        summary_csv = os.path.join(res_dir, "predictions_summary.csv")
        has_result = os.path.isfile(summary_json) or os.path.isfile(summary_csv)

        return has_fastsurfer, has_icp, has_spharm, has_result

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Directory & File Selection Properties
        config_group = QGroupBox("Pipeline Input & Output Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        cg_layout = QVBoxLayout(config_group)
        cg_layout.setContentsMargins(10, 20, 10, 10)
        cg_layout.setSpacing(8)

        # Row 1: Choose Data Directory (MRI only) & Open File History
        data_btn_row = QHBoxLayout()
        data_btn_row.setSpacing(6)

        dir_select_btn = QPushButton("📁 Choose MRI Data Directory")
        dir_select_btn.setToolTip("Select directory containing raw MRI scans (*.nii.gz, *.nii, *.mgz)")
        dir_select_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
        """)
        dir_select_btn.clicked.connect(self.select_directory)
        data_btn_row.addWidget(dir_select_btn, stretch=3)

        self.history_btn = QPushButton("🕒 Open File History")
        self.history_btn.setToolTip("View and select from previously chosen directories")
        self.history_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 8px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
        """)
        self.history_btn.clicked.connect(self.show_history_menu)
        data_btn_row.addWidget(self.history_btn, stretch=2)

        cg_layout.addLayout(data_btn_row)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("📂 Choose Output Directory")
        out_dir_btn.setToolTip("Select base output directory. Subfolders (fastsurfer, output_ICP, output_SPHARM) will be organized automatically.")
        out_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
        """)
        out_dir_btn.clicked.connect(self.select_out_directory)
        cg_layout.addWidget(out_dir_btn)

        # Row 3: Display Input & Output Paths
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("MRI Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("Select folder with MRI scans (.nii.gz, .nii, .mgz)...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output Folder:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("Select base output directory...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        cg_layout.addLayout(path_row)
        main_layout.addWidget(config_group)

        # 2. Main Panel Pipeline Results & Subjects Table
        table_group = QGroupBox("Pipeline Subjects & Output Status")
        table_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        tg_layout = QVBoxLayout(table_group)
        tg_layout.setContentsMargins(10, 20, 10, 10)
        tg_layout.setSpacing(6)

        # Tab Bar for Hemisphere Filtering
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All Meshes (0)")
        self.tab_bar.addTab("Left (0)")
        self.tab_bar.addTab("Right (0)")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setStyleSheet("""
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 4px 10px;
                margin-right: 2px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #ced6e0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4fa3e3, stop:1 #2980b9);
                color: white;
                border: 1px solid #1f618d;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
            }
        """)
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        tg_layout.addWidget(self.tab_bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Subject", "Side", "Diagnosis", "Probability"
        ])
        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 4px;
                font-size: 11px;
                gridline-color: #f1f2f6;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 4px;
                border: 1px solid #dcdde1;
            }
        """)
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        tg_layout.addWidget(self.table)
        main_layout.addWidget(table_group)

        # 3. Pipeline Stages & Execution Controls
        workflow_group = QGroupBox("Pipeline Workflow Execution")
        workflow_group.setStyleSheet("""
            QGroupBox {
                margin-top: 15px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        wg_layout = QVBoxLayout(workflow_group)
        wg_layout.setContentsMargins(10, 18, 10, 10)
        wg_layout.setSpacing(8)

        # Progress Stages Display
        stages_layout = QVBoxLayout()
        stages_layout.setSpacing(4)

        self.stage1_lbl = QLabel("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Pending")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage1_lbl)

        self.stage2_lbl = QLabel("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage2_lbl)

        self.stage3_lbl = QLabel("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage3_lbl)

        self.stage4_lbl = QLabel("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
        self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        stages_layout.addWidget(self.stage4_lbl)

        wg_layout.addLayout(stages_layout)

        # Status notification label
        self.status_lbl = QLabel("Ready to run full pipeline.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("""
            color: #2c3e50;
            background-color: #eaf2f8;
            border: 1px solid #d4e6f1;
            font-size: 11px;
            padding: 6px 8px;
            border-radius: 4px;
            font-weight: 500;
        """)
        wg_layout.addWidget(self.status_lbl)

        # Run Button
        self.run_btn = QPushButton("▶ Run Full Pipeline (FastSurfer → ICP → SPHARM → Result)")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #219a52);
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 10px 15px;
                border: 1px solid #1e8449;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60);
                border: 1px solid #196f3d;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e8449, stop:1 #145a32);
                border: 1px solid #145a32;
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
                border: 1px solid #95a5a6;
            }
        """)
        self.run_btn.clicked.connect(self.run_full_pipeline)
        wg_layout.addWidget(self.run_btn)

        main_layout.addWidget(workflow_group)

        # Initial check for stage status preview & populate table
        self.update_stage_preview()
        self.populate_main_table()

    def connect_pipeline_signals(self):
        if self.fastsurfer_panel:
            self.fastsurfer_panel.signal_fastsurfer_finished.connect(self.on_fastsurfer_finished_step)
        if self.icp_panel:
            self.icp_panel.signal_icp_finished.connect(self.on_icp_finished_step)
        if self.spharm_panel:
            self.spharm_panel.signal_spharm_finished.connect(self.on_spharm_finished_step)
        if self.result_panel and hasattr(self.result_panel, 'signal_batch_prediction_finished'):
            self.result_panel.signal_batch_prediction_finished.connect(self.on_result_finished_step)

    def set_directories(self, in_dir, out_dir):
        if in_dir:
            self.folder_input.setText(in_dir)
            self.last_input_dir = in_dir
        if out_dir:
            self.out_folder_input.setText(out_dir)
            self.last_output_dir = out_dir
            self.sync_panel_output_folders(out_dir)
        self.update_stage_preview()
        self.populate_main_table(in_dir, out_dir)

    def sync_panel_output_folders(self, out_dir):
        """Pre-configures individual subfolders matching each panel's standard structure and refreshes tables."""
        if not out_dir:
            return
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")
        res_dir = os.path.join(out_dir, "output_Result")

        if self.fastsurfer_panel and hasattr(self.fastsurfer_panel, 'fs_dir_input'):
            self.fastsurfer_panel.fs_dir_input.setText(fs_dir)
            if os.path.isdir(fs_dir):
                self.fastsurfer_panel.populate_results_table()
        if self.icp_panel and hasattr(self.icp_panel, 'icp_dir_input'):
            self.icp_panel.icp_dir_input.setText(icp_dir)
            if os.path.isdir(icp_dir):
                self.icp_panel.populate_results_table()
        if self.spharm_panel and hasattr(self.spharm_panel, 'spharm_dir_input'):
            self.spharm_panel.spharm_dir_input.setText(spharm_dir)
            if os.path.isdir(spharm_dir):
                self.spharm_panel.populate_results_table()
        if self.result_panel:
            if hasattr(self.result_panel, 'spharm_dir_input'):
                self.result_panel.spharm_dir_input.setText(spharm_dir)
            if hasattr(self.result_panel, 'result_dir_input'):
                self.result_panel.result_dir_input.setText(res_dir)
            if os.path.isdir(res_dir) and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()

    def update_stage_preview(self):
        """Checks output directory and previews which stages exist and which will be run."""
        out_dir = self.out_folder_input.text().strip()
        if not out_dir or not os.path.isdir(out_dir):
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Pending")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.status_lbl.setText("Ready: Select Input MRI Directory & Output Directory to begin.")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            return

        has_fs, has_icp, has_spharm, has_result = self.check_existing_stages(out_dir)

        # Stage 1 label
        if has_fs:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Results exist (Will skip)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
        else:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏸ Missing (Will run first)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")

        # Stage 2 label
        if has_icp:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Results exist (Will skip)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
        elif has_fs:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Missing (Will run next)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Stage 3 label
        if has_spharm:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Results exist (Will skip)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
        elif has_fs and has_icp:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Missing (Will run next)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Stage 4 label
        if has_result:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Results exist (Will skip)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
        elif has_fs and has_icp and has_spharm:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Missing (Will run next)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Status text summary
        if has_fs and has_icp and has_spharm and has_result:
            self.status_lbl.setText("✓ Complete results exist in output folder. Running will refresh all tables without re-computing.")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and has_icp and has_spharm and not has_result:
            self.status_lbl.setText("FastSurfer, ICP & SPHARM results exist (Skipping 1-3). Running will execute ResNet Prediction & 3D Grad-CAM.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and has_icp and not has_spharm:
            self.status_lbl.setText("FastSurfer & ICP results exist (Skipping both). Running will execute SPHARM-PDM → ResNet Prediction.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and not has_icp:
            self.status_lbl.setText("FastSurfer results exist (Skipping FastSurfer). Running will execute ICP → SPHARM → ResNet Prediction.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        else:
            self.status_lbl.setText("Ready to run full pipeline (FastSurfer → ICP → SPHARM → Result).")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")

    @staticmethod
    def find_spharm_mesh(out_dir, subject, side):
        """Finds matching SPHARM .vtk mesh in output_SPHARM directory for given subject and side."""
        if not out_dir or not subject or not os.path.isdir(out_dir):
            return None
        side_key = "left" if str(side).lower() in ("left", "lh") else "right"
        search_dirs = [
            os.path.join(out_dir, "output_SPHARM", side_key, "spharm_results"),
            os.path.join(out_dir, "output_SPHARM", side_key),
            os.path.join(out_dir, "output_SPHARM", "spharm_results"),
            os.path.join(out_dir, "output_SPHARM"),
            os.path.join(out_dir, f"output_{side_key}_hippocampus", f"spharm_results_{side_key}"),
        ]
        candidates = []
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "**", f"*{subject}*.vtk"), recursive=True):
                fname = os.path.basename(f).lower()
                if any(aux in fname for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_", "mean_shape")):
                    continue
                # verify side
                f_norm = f.replace("\\", "/").lower()
                if side_key == "left" and not (fname.startswith("lh_") or "left" in fname or "/left/" in f_norm):
                    continue
                if side_key == "right" and not (fname.startswith("rh_") or "right" in fname or "/right/" in f_norm):
                    continue
                candidates.append(os.path.normpath(f))

        if not candidates:
            return None

        # Prioritize realigned > procalign > ellalign > SPHARM
        for suf in ("_SPHARM_realigned.vtk", "_realigned.vtk", "_SPHARM_procalign.vtk", "_procalign.vtk", "_SPHARM_ellalign.vtk", "_ellalign.vtk", "_SPHARM.vtk"):
            for c in candidates:
                if c.endswith(suf):
                    return c
        return candidates[0]

    def on_tab_changed(self, index):
        self.render_table_rows()

    def populate_main_table(self, in_dir=None, out_dir=None):
        """
        Loads paired Grad-CAM evaluation results and SPHARM 3D meshes into the Main Panel table.
        Keeps table empty on fresh startup until output directory is selected or pipeline finishes.
        """
        if not out_dir:
            out_dir = self.out_folder_input.text().strip()
        if not out_dir and getattr(self, 'import_panel', None):
            out_dir = self.import_panel.get_output_folder().strip()

        if not out_dir or not os.path.isdir(out_dir):
            self.all_pipeline_results = []
            self.table.setRowCount(0)
            self.tab_bar.blockSignals(True)
            self.tab_bar.setTabText(0, "All Meshes (0)")
            self.tab_bar.setTabText(1, "Left (0)")
            self.tab_bar.setTabText(2, "Right (0)")
            self.tab_bar.blockSignals(False)
            return

        json_file = os.path.join(out_dir, "output_Result", "evaluation_summary.json")

        records = []
        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item in data:
                    s_raw = item.get('subject', '')
                    s_clean = s_raw.strip()
                    if not s_clean:
                        continue
                    summary = item.get('summary', {})
                    lh_mesh = item.get('left_mesh')
                    rh_mesh = item.get('right_mesh')
                    left_prob = summary.get('left_prob')
                    right_prob = summary.get('right_prob')
                    overall_prob = summary.get('probability', 0.0)

                    # Left Hemisphere record: ONLY include if BOTH real SPHARM mesh and evaluated result exist
                    lh_spharm = self.find_spharm_mesh(out_dir, s_clean, "left")
                    if lh_spharm and left_prob is not None and lh_mesh and os.path.isfile(lh_mesh):
                        diag_str = "🚨 Epilepsy (TLE)" if left_prob > 0.5 else "✅ Normal (HC)"
                        prob_str = f"{left_prob * 100:.1f}%"
                        records.append({
                            "subject": s_clean,
                            "side": "left",
                            "side_str": "LH",
                            "diag": diag_str,
                            "prob": left_prob,
                            "prob_str": prob_str,
                            "gradcam_mesh": lh_mesh,
                            "spharm_mesh": lh_spharm
                        })

                    # Right Hemisphere record: ONLY include if BOTH real SPHARM mesh and evaluated result exist
                    rh_spharm = self.find_spharm_mesh(out_dir, s_clean, "right")
                    if rh_spharm and right_prob is not None and rh_mesh and os.path.isfile(rh_mesh):
                        diag_str = "🚨 Epilepsy (TLE)" if right_prob > 0.5 else "✅ Normal (HC)"
                        prob_str = f"{right_prob * 100:.1f}%"
                        records.append({
                            "subject": s_clean,
                            "side": "right",
                            "side_str": "RH",
                            "diag": diag_str,
                            "prob": right_prob,
                            "prob_str": prob_str,
                            "gradcam_mesh": rh_mesh,
                            "spharm_mesh": rh_spharm
                        })
            except Exception as e:
                self.signal_log_message.emit(f"[WARNING] Main Panel could not parse {json_file}: {e}")

        self.all_pipeline_results = records
        self.render_table_rows()

    def render_table_rows(self):
        tab_idx = self.tab_bar.currentIndex()

        total_lh = sum(1 for r in self.all_pipeline_results if r["side"] == "left")
        total_rh = sum(1 for r in self.all_pipeline_results if r["side"] == "right")

        self.tab_bar.blockSignals(True)
        self.tab_bar.setTabText(0, f"All Meshes ({len(self.all_pipeline_results)})")
        self.tab_bar.setTabText(1, f"Left ({total_lh})")
        self.tab_bar.setTabText(2, f"Right ({total_rh})")
        self.tab_bar.blockSignals(False)

        if tab_idx == 1:
            filtered = [r for r in self.all_pipeline_results if r["side"] == "left"]
        elif tab_idx == 2:
            filtered = [r for r in self.all_pipeline_results if r["side"] == "right"]
        else:
            filtered = list(self.all_pipeline_results)

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(filtered))

        for row, rec in enumerate(filtered):
            subj = rec["subject"]
            side_str = rec["side_str"]
            diag_str = rec["diag"]
            prob_str = rec["prob_str"]
            gradcam_p = rec["gradcam_mesh"]
            spharm_p = rec["spharm_mesh"]

            item_subj = QTableWidgetItem(subj)
            item_subj.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            item_subj.setData(Qt.ItemDataRole.UserRole, rec)
            item_subj.setToolTip(f"Subject: {subj}\nSide: {side_str}")

            item_side = QTableWidgetItem(side_str)
            item_side.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if side_str == "LH":
                item_side.setForeground(QColor("#2980b9"))
            else:
                item_side.setForeground(QColor("#d35400"))
            item_side.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            item_diag = QTableWidgetItem(diag_str)
            item_diag.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if "TLE" in diag_str or "Epilepsy" in diag_str:
                item_diag.setForeground(QColor("#c0392b"))
            elif "Normal" in diag_str or "HC" in diag_str:
                item_diag.setForeground(QColor("#27ae60"))
            else:
                item_diag.setForeground(QColor("#7f8c8d"))
            item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            item_prob = QTableWidgetItem(prob_str)
            item_prob.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table.setItem(row, 0, item_subj)
            self.table.setItem(row, 1, item_side)
            self.table.setItem(row, 2, item_diag)
            self.table.setItem(row, 3, item_prob)

        self.table.blockSignals(False)

        if self.table.rowCount() > 0:
            self.table.selectRow(0)
            self.on_table_row_selected()
        else:
            self.signal_mesh_selected.emit("", "all")
            self.signal_diagnostic_info.emit("")

    def on_table_row_selected(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        if not item:
            return
        rec = item.data(Qt.ItemDataRole.UserRole)
        if not rec or not isinstance(rec, dict):
            return

        spharm_path = rec.get("spharm_mesh", "")
        side = rec.get("side", "left")
        subj = rec.get("subject", "")
        diag = rec.get("diag", "")
        prob_str = rec.get("prob_str", "")
        gradcam = rec.get("gradcam_mesh", "")

        is_tle = "TLE" in diag or "Epilepsy" in diag
        diag_color = "#e74c3c" if is_tle else "#2ecc71"
        diag_badge = f'<span style="color: {diag_color}; font-weight: bold;">&#9679; {diag} ({prob_str})</span>'
        g_name = os.path.basename(gradcam) if gradcam and gradcam != "—" else "None"
        info_html = f'{diag_badge}  |  <span style="color: #bdc3c7;">Grad-CAM: {g_name}</span>'

        self.signal_diagnostic_info.emit(info_html)

        if spharm_path and os.path.isfile(spharm_path):
            self.signal_mesh_selected.emit(spharm_path, side)
            self.signal_log_message.emit(f"Main Panel displaying SPHARM mesh: {os.path.basename(spharm_path)} for {subj} [{side.upper()}] (Diagnosis: {diag}, Prob: {prob_str})")
        else:
            self.signal_mesh_selected.emit("", side)
            self.signal_log_message.emit(f"Main Panel: No SPHARM mesh found on disk for {subj} [{side.upper()}].")

    def select_directory(self):
        initial_dir = self.last_input_dir if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        parent_win = self.window() if self.window() else self
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Directory with MRI Scans", initial_dir)
        if folder:
            mri_files = self.find_valid_mri_files(folder)
            if not mri_files:
                self.signal_log_message.emit(f"[WARNING] No MRI scan files (*.nii.gz, *.nii, *.mgz) found in: {folder}")
                self.status_lbl.setText("⚠️ Warning: No valid MRI scan files (*.nii.gz, *.nii, *.mgz) found in selected directory.")
                self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
            else:
                self.status_lbl.setText(f"✓ Detected {len(mri_files)} MRI image(s). Ready to process.")
                self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px;")

            self.folder_input.setText(folder)
            self.add_to_history(folder)
            self.signal_log_message.emit(f"Main Panel selected MRI directory: {folder} (Found {len(mri_files)} MRI files)")

            if self.import_panel:
                self.import_panel.folder_input.setText(folder)
                self.import_panel.add_to_history(folder)
                self.import_panel.load_subjects_from_directory(folder)
                self.import_panel.signal_directories_changed.emit(folder, self.out_folder_input.text().strip())

            self.populate_main_table(folder, self.out_folder_input.text().strip())

    def select_out_directory(self):
        initial_dir = self.last_output_dir if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        parent_win = self.window() if self.window() else self
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            self.save_history()
            self.sync_panel_output_folders(folder)
            self.update_stage_preview()
            self.populate_main_table(self.folder_input.text().strip(), folder)
            self.signal_log_message.emit(f"Main Panel selected output directory: {folder}")

            if self.import_panel:
                self.import_panel.out_folder_input.setText(folder)
                self.import_panel.last_output_dir = folder
                self.import_panel.save_history()
                self.import_panel.signal_directories_changed.emit(self.folder_input.text().strip(), folder)

    def show_history_menu(self):
        if not self.recent_dirs:
            menu = QMenu(self)
            empty_action = menu.addAction("No recent directories")
            empty_action.setEnabled(False)
            menu.exec(self.history_btn.mapToGlobal(QPoint(0, self.history_btn.height())))
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #ffffff; border: 1px solid #ced6e0; border-radius: 4px; padding: 4px; }
            QMenu::item { padding: 6px 20px; font-size: 11px; color: #2c3e50; border-radius: 2px; }
            QMenu::item:selected { background-color: #3498db; color: #ffffff; }
        """)

        for d in self.recent_dirs:
            action = menu.addAction(d)
            action.triggered.connect(lambda checked, path=d: self.on_history_item_selected(path))

        menu.exec(self.history_btn.mapToGlobal(QPoint(0, self.history_btn.height())))

    def on_history_item_selected(self, path):
        if os.path.isdir(path):
            mri_files = self.find_valid_mri_files(path)
            self.folder_input.setText(path)
            self.add_to_history(path)
            self.signal_log_message.emit(f"Main Panel selected directory from history: {path} (Found {len(mri_files)} MRI files)")
            if not mri_files:
                self.status_lbl.setText("⚠️ Warning: No valid MRI scan files (*.nii.gz, *.nii, *.mgz) found.")
                self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
            else:
                self.status_lbl.setText(f"✓ Detected {len(mri_files)} MRI image(s). Ready to process.")
                self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px;")

            if self.import_panel:
                self.import_panel.folder_input.setText(path)
                self.import_panel.add_to_history(path)
                self.import_panel.load_subjects_from_directory(path)
                self.import_panel.signal_directories_changed.emit(path, self.out_folder_input.text().strip())

            self.populate_main_table(path, self.out_folder_input.text().strip())

    def run_full_pipeline(self):
        in_dir = self.folder_input.text().strip()
        out_dir = self.out_folder_input.text().strip()

        if not out_dir:
            self.signal_log_message.emit("[ERROR] Main Panel: Please select an Output Directory first.")
            self.status_lbl.setText("❌ Error: Output directory not specified.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            return

        # Create structured output folders exactly as each panel expects
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")
        res_dir = os.path.join(out_dir, "output_Result")

        os.makedirs(fs_dir, exist_ok=True)
        os.makedirs(icp_dir, exist_ok=True)
        os.makedirs(spharm_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        self.sync_panel_output_folders(out_dir)

        # Sync directories with ImportPanel
        if self.import_panel:
            if in_dir and os.path.isdir(in_dir):
                self.import_panel.folder_input.setText(in_dir)
                self.import_panel.load_subjects_from_directory(in_dir)
            self.import_panel.out_folder_input.setText(out_dir)
            self.import_panel.signal_directories_changed.emit(in_dir, out_dir)

        # Check existing results in output dir (all 4 stages)
        has_fs, has_icp, has_spharm, has_result = self.check_existing_stages(out_dir)

        # -------------------------------------------------------------
        # Condition A: Results exist for ALL 4 panels!
        # "ถ้ามีผลลัพของ panel ครบทุก panel แล้วให้ข้ามได้เลยไม่ต้องทำ และให้แสดงผลลัพที่ตารางด้วย"
        # -------------------------------------------------------------
        if has_fs and has_icp and has_spharm and has_result:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Skipped)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

            self.status_lbl.setText("🎉 All results already exist in output folder! Skipped execution and refreshed all tables.")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Complete results for all panels (FastSurfer, ICP, SPHARM, Result) already exist in output folder. Skipping execution and populating all tables.")

            # Populate all result tables across all panels immediately
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            if self.result_panel and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()
            self.populate_main_table(in_dir, out_dir)
            return

        # If FastSurfer is missing, raw MRI scans are strictly required
        if not has_fs:
            if not in_dir or not os.path.isdir(in_dir):
                self.signal_log_message.emit("[ERROR] Main Panel: FastSurfer results missing. Please select a valid MRI Data Directory first.")
                self.status_lbl.setText("❌ Error: FastSurfer results missing. Please select MRI data folder.")
                self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
                return

            mri_files = self.find_valid_mri_files(in_dir)
            if not mri_files:
                self.signal_log_message.emit(f"[ERROR] Main Panel: Selected directory '{in_dir}' contains no valid MRI files (*.nii.gz, *.nii, *.mgz).")
                self.status_lbl.setText("❌ Error: Selected folder contains no MRI scans. Please choose MRI data.")
                self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
                return

        self.is_running = True
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Running Pipeline...")

        # -------------------------------------------------------------
        # Condition B: FastSurfer, ICP & SPHARM exist, but Result is missing
        # -------------------------------------------------------------
        if has_fs and has_icp and has_spharm and not has_result:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()

            self.populate_main_table(in_dir, out_dir)
            self.run_result_step(out_dir)
            return

        # -------------------------------------------------------------
        # Condition C: FastSurfer & ICP exist, but SPHARM is missing
        # -------------------------------------------------------------
        if has_fs and has_icp and not has_spharm:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()
            if self.icp_panel:
                self.icp_panel.populate_results_table()

            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            if has_result:
                self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Will skip)")
                self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            else:
                self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
                self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

            self.status_lbl.setText("🚀 FastSurfer & ICP results exist (Skipped). Running SPHARM-PDM Processing...")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Existing FastSurfer and ICP results detected (Skipped). Starting SPHARM-PDM Processing...")

            self.populate_main_table(in_dir, out_dir)

            if self.spharm_panel:
                self.spharm_panel.update_run_button_state()
                self.spharm_panel.run_spharm_process()
            else:
                self.reset_run_state()
            return

        # -------------------------------------------------------------
        # Condition D: FastSurfer exists, but ICP is missing
        # -------------------------------------------------------------
        if has_fs and not has_icp:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Found existing results (Skipped)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.fastsurfer_panel:
                self.fastsurfer_panel.populate_results_table()

            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏳ In Progress...")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            if has_spharm:
                self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Will skip)")
                self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            else:
                self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
                self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            if has_result:
                self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Will skip)")
                self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            else:
                self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
                self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

            self.status_lbl.setText("🚀 FastSurfer results exist (Skipped). Running Groupwise ICP Registration...")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] Existing FastSurfer results detected (Skipping FastSurfer). Starting Groupwise ICP Registration...")

            self.populate_main_table(in_dir, out_dir)

            if self.icp_panel:
                self.icp_panel.update_run_button_state()
                self.icp_panel.run_icp_process()
            else:
                self.reset_run_state()
            return

        # -------------------------------------------------------------
        # Condition E: FastSurfer missing -> Run from Stage 1: FastSurfer
        # -------------------------------------------------------------
        self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ⏳ In Progress...")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏸ Pending")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏸ Pending")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏸ Pending")
        self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        mri_count = len(self.find_valid_mri_files(in_dir)) if in_dir else 0
        self.status_lbl.setText(f"🚀 Step 1/4: Running FastSurfer Segmentation on {mri_count} MRI scans...")
        self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        self.signal_log_message.emit(f">>> [MAIN PIPELINE] Starting FastSurfer Segmentation on {mri_count} MRI scans...")

        self.populate_main_table(in_dir, out_dir)

        if self.fastsurfer_panel:
            self.fastsurfer_panel.run_fastsurfer_process()
        else:
            self.signal_log_message.emit("[ERROR] FastSurfer panel not configured.")
            self.reset_run_state()

    def on_fastsurfer_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ❌ Failed")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.status_lbl.setText("❌ Pipeline stopped: FastSurfer Segmentation encountered an error.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.stage1_lbl.setText("  1️⃣ FastSurfer Hippocampal Segmentation:  ✓ Completed")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        # Populate FastSurfer results table
        if self.fastsurfer_panel:
            self.fastsurfer_panel.populate_results_table()
        self.populate_main_table()

        # Check if downstream stages already exist
        out_dir = self.out_folder_input.text().strip()
        _, has_icp, has_spharm, has_result = self.check_existing_stages(out_dir)

        if has_icp and has_spharm and has_result:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Skipped)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            if self.result_panel and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()
            self.on_result_finished_step(True)
            return
        elif has_icp and has_spharm and not has_result:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            self.run_result_step(out_dir)
            return
        elif has_icp and not has_spharm:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Found existing results (Skipped)")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.icp_panel:
                self.icp_panel.populate_results_table()
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
            self.status_lbl.setText("🚀 Step 3/4: Running SPHARM-PDM Processing (ICP skipped, results exist)...")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] ICP results already exist. Starting SPHARM-PDM Processing...")
            if self.spharm_panel:
                self.spharm_panel.update_run_button_state()
                self.spharm_panel.run_spharm_process()
            else:
                self.reset_run_state()
            return

        # Step 2: ICP Registration
        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ⏳ In Progress...")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.status_lbl.setText("🚀 Step 2/4: Running Groupwise ICP Registration...")
        self.signal_log_message.emit(">>> [MAIN PIPELINE] FastSurfer step completed. Starting Groupwise ICP Registration...")

        if self.icp_panel:
            self.icp_panel.update_run_button_state()
            self.icp_panel.run_icp_process()
        else:
            self.signal_log_message.emit("[ERROR] ICP panel not configured.")
            self.reset_run_state()

    def on_icp_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ❌ Failed")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.status_lbl.setText("❌ Pipeline stopped: Groupwise ICP Registration encountered an error.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.stage2_lbl.setText("  2️⃣ Groupwise ICP Mesh Registration:         ✓ Completed")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        # Populate ICP results table
        if self.icp_panel:
            self.icp_panel.populate_results_table()
        self.populate_main_table()

        # Check downstream stages
        out_dir = self.out_folder_input.text().strip()
        _, _, has_spharm, has_result = self.check_existing_stages(out_dir)

        if has_spharm and has_result:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Skipped)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            if self.result_panel and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()
            self.on_result_finished_step(True)
            return
        elif has_spharm and not has_result:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Found existing results (Skipped)")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.spharm_panel:
                self.spharm_panel.populate_results_table()
            self.run_result_step(out_dir)
            return

        # Step 3: SPHARM Processing
        self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⏳ In Progress...")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.status_lbl.setText("🚀 Step 3/4: Running SPHARM-PDM Processing...")
        self.signal_log_message.emit(">>> [MAIN PIPELINE] ICP step completed. Starting SPHARM-PDM Processing...")

        if self.spharm_panel:
            self.spharm_panel.update_run_button_state()
            self.spharm_panel.run_spharm_process()
        else:
            self.signal_log_message.emit("[ERROR] SPHARM panel not configured.")
            self.reset_run_state()

    def on_spharm_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ⚠️ Finished with warnings")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #d35400; font-weight: bold;")
            self.status_lbl.setText("⚠️ SPHARM-PDM finished with warnings or errors.")
            self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
        else:
            self.stage3_lbl.setText("  3️⃣ SPHARM-PDM Shape Analysis:            ✓ Completed")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        # Populate all previous stages tables
        in_dir = self.folder_input.text().strip()
        if self.import_panel and in_dir:
            self.import_panel.load_subjects_from_directory(in_dir)
        if self.fastsurfer_panel:
            self.fastsurfer_panel.populate_results_table()
        if self.icp_panel:
            self.icp_panel.populate_results_table()
        if self.spharm_panel:
            self.spharm_panel.populate_results_table()
        self.populate_main_table()

        # Check if Result stage already exists
        out_dir = self.out_folder_input.text().strip()
        _, _, _, has_result = self.check_existing_stages(out_dir)

        if has_result:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Found existing results (Skipped)")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.result_panel and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()
            self.on_result_finished_step(True)
            return

        # Proceed to Step 4: ResNet Epilepsy Prediction & 3D Grad-CAM
        self.run_result_step(out_dir)

    def run_result_step(self, out_dir):
        """Executes Step 4: Result Panel ResNet Batch Prediction & 3D Grad-CAM generation."""
        self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⏳ In Progress...")
        self.stage4_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.status_lbl.setText("🚀 Step 4/4: Running ResNet Epilepsy Prediction & 3D Grad-CAM...")
        self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        self.signal_log_message.emit(">>> [MAIN PIPELINE] SPHARM step completed. Starting ResNet Epilepsy Prediction & 3D Grad-CAM...")

        if self.result_panel and hasattr(self.result_panel, 'run_batch_prediction'):
            sph_d = os.path.join(out_dir, "output_SPHARM")
            res_d = os.path.join(out_dir, "output_Result")
            self.result_panel.spharm_dir_input.setText(sph_d)
            self.result_panel.result_dir_input.setText(res_d)
            self.result_panel.run_batch_prediction()
        else:
            self.signal_log_message.emit("[WARNING] Result panel not configured or run_batch_prediction unavailable.")
            self.on_result_finished_step(True)

    def on_result_finished_step(self, success):
        if not self.is_running:
            return

        if not success:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ⚠️ Finished with warnings")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #d35400; font-weight: bold;")
            self.status_lbl.setText("⚠️ ResNet Prediction & Grad-CAM finished with warnings or errors.")
            self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
        else:
            self.stage4_lbl.setText("  4️⃣ ResNet Epilepsy Prediction & 3D Grad-CAM: ✓ Completed")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.status_lbl.setText("🎉 Complete Pipeline (All 4 Stages) Finished Successfully!")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.signal_log_message.emit(">>> [MAIN PIPELINE] All 4 pipeline stages (FastSurfer, ICP, SPHARM, Result) completed successfully!")

        # Populate all results tables across all panels
        in_dir = self.folder_input.text().strip()
        if self.import_panel and in_dir:
            self.import_panel.load_subjects_from_directory(in_dir)
        if self.fastsurfer_panel:
            self.fastsurfer_panel.populate_results_table()
        if self.icp_panel:
            self.icp_panel.populate_results_table()
        if self.spharm_panel:
            self.spharm_panel.populate_results_table()
        if self.result_panel and hasattr(self.result_panel, 'load_existing_results'):
            self.result_panel.load_existing_results()

        self.populate_main_table()
        self.reset_run_state()

    def reset_run_state(self):
        self.is_running = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run Full Pipeline (FastSurfer → ICP → SPHARM → Result)")
