import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QGroupBox, QFormLayout, QComboBox, QSpinBox,
                             QHBoxLayout, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTabBar, QRadioButton, QButtonGroup, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QLocale

class ToggleTableWidget(QTableWidget):
    """QTableWidget supporting ExtendedSelection (Ctrl/Shift multi-select)
    and single-click toggle/deselect when clicking an already selected sole row."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                row = item.row()
                modifiers = event.modifiers()
                has_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

                selected_rows = list(set(it.row() for it in self.selectedItems()))

                if not has_ctrl and not has_shift:
                    if selected_rows == [row]:
                        self.clearSelection()
                        return
                    elif row in selected_rows and len(selected_rows) > 1:
                        self.clearSelection()
                        self.selectRow(row)
                        return

        super().mousePressEvent(event)

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def find_slicer_salt_exe():
    env_path = os.environ.get("SLICER_EXE")
    if env_path and os.path.isfile(env_path):
        return env_path
    
    root = get_project_root()
    local_candidates = [
        os.path.join(root, "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
    ]
    for c in local_candidates:
        if os.path.isfile(c):
            return c

    candidates = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
    if candidates:
        return candidates[0]

    candidates_x86 = glob.glob(r"C:\Program Files (x86)\SlicerSALT*\SlicerSALT.exe")
    if candidates_x86:
        return candidates_x86[0]

    for user_dir in glob.glob(r"C:\Users\*"):
        user_cands = [
            os.path.join(user_dir, "AppData", "Local", "NA-MIC", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
            os.path.join(user_dir, "AppData", "Local", "Programs", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
        ]
        for uc in user_cands:
            if os.path.isfile(uc):
                return uc

    default_salt = r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
    if os.path.isfile(default_salt):
        return default_salt

    return None

class SPHARMWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params=None):
        super().__init__()
        self.tasks = tasks # list of (side_name, in_dir, out_dir)
        self.adv_params = adv_params or {}

    def run(self):
        slicer_exe = find_slicer_salt_exe()
        if not slicer_exe:
            self.signal_log.emit("[ERROR] SlicerSALT.exe not found! Please check installation.")
            self.signal_finished.emit(False)
            return

        project_root = get_project_root()
        batch_script = os.path.join(project_root, "SPHARM", "run_spharm_batch.py")
        realign_script = os.path.join(project_root, "SPHARM", "realign_spharm.py")

        if not os.path.isfile(batch_script):
            self.signal_log.emit(f"[ERROR] run_spharm_batch.py not found at: {batch_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running Batch SPHARM for [{side_name.upper()} Hippocampus]")
            
            os.makedirs(out_dir, exist_ok=True)
            
            # Step 1: Batch SPHARM via SlicerSALT
            cmd = [
                slicer_exe,
                "--no-main-window",
                "--no-splash",
                "--python-script", batch_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--num_iterations", str(self.adv_params.get("num_iter", 1000)),
                "--subdiv_level", str(self.adv_params.get("subdiv", 10)),
                "--spharm_degree", str(self.adv_params.get("degree", self.adv_params.get("deg", 12)))
            ]
            
            # Use official template from Templates/SPHARM if present
            tmpl_file = os.path.join(project_root, "Templates", "SPHARM", f"template_spharm_{side_name.lower()}.vtk")
            coef_file = os.path.join(project_root, "Templates", "SPHARM", f"template_spharm_{side_name.lower()}.coef")
            if os.path.isfile(tmpl_file) and os.path.isfile(coef_file):
                cmd.extend(["--reference_template", tmpl_file])
                self.signal_log.emit(f"    Using Reference Template: {os.path.basename(tmpl_file)} and {os.path.basename(coef_file)}")
            else:
                self.signal_log.emit(f"    [INFO] Official template for {side_name} not found or incomplete (need both .vtk and .coef). Proceeding without --reference_template.")

            self.signal_log.emit(f"Executing: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                for line in proc.stdout:
                    self.signal_log.emit(line.strip())
                proc.wait()
                
                if proc.returncode != 0:
                    self.signal_log.emit(f"[ERROR] SPHARM batch failed for {side_name} with return code {proc.returncode}")
                    overall_success = False
                    continue
                else:
                    self.signal_log.emit(f"[SUCCESS] SPHARM processing completed for {side_name}.")
                    
                # Step 2: Post-process Procrustes Re-alignment (Self-alignment against cohort mean)
                if os.path.isfile(realign_script):
                    self.signal_log.emit(f">>> Running Procrustes Re-alignment for [{side_name.upper()}]...")
                    realign_target = os.path.join(out_dir, "spharm_results") if os.path.isdir(os.path.join(out_dir, "spharm_results")) else out_dir
                    realign_cmd = [
                        sys.executable,
                        realign_script,
                        "--spharm_dir", realign_target
                    ]
                    # Pass official template as fixed target if available
                    if os.path.isfile(tmpl_file):
                        realign_cmd.extend(["--template", tmpl_file])
                        self.signal_log.emit(f"    Aligning cohort to official template: {os.path.basename(tmpl_file)}")
                        
                    re_proc = subprocess.Popen(
                        realign_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    for line in re_proc.stdout:
                        self.signal_log.emit(line.strip())
                    re_proc.wait()
                    if re_proc.returncode != 0:
                        self.signal_log.emit(f"[ERROR] Realignment failed for {side_name}: exit {re_proc.returncode}")
                        overall_success = False
                        continue
                    self.signal_log.emit(f"[SUCCESS] Procrustes Re-alignment finished for {side_name}.")
                    
            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running SPHARM {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)

SpharmWorker = SPHARMWorker

class SpharmPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str) # filepath can be str or list of str
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_all_toggled = pyqtSignal(bool, list, str) # enabled, file_list, side_filter
    signal_side_changed = pyqtSignal(str) # "all", "lh", "rh"
    signal_spharm_completed = pyqtSignal()
    signal_spharm_finished = pyqtSignal(bool)

    def __init__(self, get_folder_func, get_output_folder_func=None, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.get_output_folder = get_output_folder_func
        self.all_files = []
        self.current_side_filter = "all"
        self.last_selected_row = None
        self.setup_ui()

    def setup_ui(self):
        spharm_layout = QVBoxLayout(self)
        spharm_layout.setContentsMargins(10, 10, 10, 10)
        spharm_layout.setSpacing(10)
        
        help_label = QLabel("SPHARM-PDM spherical parameterization and shape model generation for ICP-aligned hippocampus meshes.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        spharm_layout.addWidget(help_label)
        
        # 1. Directory Configuration (Mesh / ICP Import & Dedicated output_SPHARM)
        dir_group = QGroupBox("Directory Configuration (Mesh Import & Output)")
        dir_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
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
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setContentsMargins(10, 16, 10, 10)
        dir_layout.setSpacing(6)

        # Row A: Input ICP Aligned Meshes
        in_lbl = QLabel("📥 Input ICP Aligned Meshes (output_ICP or Custom ICP Folder):")
        in_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        dir_layout.addWidget(in_lbl)

        in_row = QHBoxLayout()
        self.mesh_input_dir = QLineEdit()
        self.mesh_input_dir.setPlaceholderText("Auto (.../output_ICP from pipeline) or Browse to import ICP folder...")
        self.mesh_input_dir.textChanged.connect(self.on_input_dir_changed)
        in_row.addWidget(self.mesh_input_dir)

        browse_in_btn = QPushButton("📁 Browse...")
        browse_in_btn.setToolTip("Import existing ICP output folder from disk (must contain aligned meshes)")
        browse_in_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
        """)
        browse_in_btn.clicked.connect(self.browse_input_directory)
        in_row.addWidget(browse_in_btn)

        reset_in_btn = QPushButton("🔄 Pipeline")
        reset_in_btn.setToolTip("Reset input back to current pipeline output_ICP")
        reset_in_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
        """)
        reset_in_btn.clicked.connect(self.reset_to_pipeline_input)
        in_row.addWidget(reset_in_btn)

        dir_layout.addLayout(in_row)

        # Row B: Output Directory (Dedicated output_SPHARM)
        out_lbl = QLabel("📤 Output Directory (Dedicated output_SPHARM):")
        out_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50; margin-top: 4px;")
        dir_layout.addWidget(out_lbl)

        out_row = QHBoxLayout()
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.setPlaceholderText("Auto (.../output_SPHARM)")
        self.spharm_dir_input.textChanged.connect(self.populate_results_table)
        out_row.addWidget(self.spharm_dir_input)

        browse_out_btn = QPushButton("📁 Browse...")
        browse_out_btn.setToolTip("Select custom destination for output_SPHARM")
        browse_out_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
        """)
        browse_out_btn.clicked.connect(self.browse_output_directory)
        out_row.addWidget(browse_out_btn)

        reload_btn = QPushButton("🔄 Reload")
        reload_btn.setToolTip("Scan output_SPHARM folder and reload results table")
        reload_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
        """)
        reload_btn.clicked.connect(self.populate_results_table)
        out_row.addWidget(reload_btn)

        dir_layout.addLayout(out_row)
        spharm_layout.addWidget(dir_group)

        # 2. Side Selection
        side_group = QGroupBox("Side Execution Option")
        side_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 8px;
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
        side_layout = QHBoxLayout(side_group)
        side_layout.setContentsMargins(10, 15, 10, 10)
        
        self.side_both_rb = QRadioButton("Both Sides (LH && RH)")
        self.side_both_rb.setChecked(True)
        self.side_lh_rb = QRadioButton("Left (LH) Only")
        self.side_rh_rb = QRadioButton("Right (RH) Only")

        self.side_btn_group = QButtonGroup(self)
        self.side_btn_group.addButton(self.side_both_rb, 0)
        self.side_btn_group.addButton(self.side_lh_rb, 1)
        self.side_btn_group.addButton(self.side_rh_rb, 2)

        side_layout.addWidget(self.side_both_rb)
        side_layout.addWidget(self.side_lh_rb)
        side_layout.addWidget(self.side_rh_rb)
        spharm_layout.addWidget(side_group)

        # 3. Main Action Button & Status Hint
        self.run_spharm_btn = QPushButton("▶ Run Batch SPHARM Processing")
        self.run_spharm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 9px 15px;
                border: 1px solid #ced6e0;
                border-radius: 5px;
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
            QPushButton:disabled {
                background: #f1f2f6;
                color: #a4b0be;
                border: 1px solid #dfe4ea;
            }
        """)
        self.run_spharm_btn.clicked.connect(self.run_spharm_process)
        spharm_layout.addWidget(self.run_spharm_btn)

        self.spharm_status_hint = QLabel("")
        self.spharm_status_hint.setWordWrap(True)
        self.spharm_status_hint.setStyleSheet("font-size: 11px; padding: 5px 8px; border-radius: 4px;")
        spharm_layout.addWidget(self.spharm_status_hint)

        # 4. Collapsible Advanced Parameters
        self.toggle_adv_btn = QPushButton("⚙️ Advanced Parameters ▾")
        self.toggle_adv_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #57606f;
                font-weight: bold;
                font-size: 11px;
                text-align: left;
                padding: 4px 6px;
                border: none;
            }
            QPushButton:hover {
                color: #2f3542;
            }
        """)
        self.toggle_adv_btn.clicked.connect(self.toggle_advanced_params)
        spharm_layout.addWidget(self.toggle_adv_btn)

        self.adv_container = QFrame()
        self.adv_container.setStyleSheet("""
            QFrame {
                border: 1px solid #e1e2e6;
                border-radius: 5px;
                background-color: #fafbfc;
            }
            QSpinBox, QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
                color: #34495e;
                font-size: 11px;
            }
            QLabel {
                font-size: 11px;
                color: #2c3e50;
            }
        """)
        adv_form = QFormLayout(self.adv_container)
        adv_form.setContentsMargins(10, 10, 10, 10)
        adv_form.setSpacing(6)

        self.spharm_mode_combo = QComboBox()
        self.spharm_mode_combo.addItems(["Production Mode (Default)", "Fast Test Mode", "Custom Mode"])
        self.spharm_mode_combo.currentIndexChanged.connect(self.on_spharm_mode_changed)

        self.spharm_iter_spin = QSpinBox()
        self.spharm_iter_spin.setRange(50, 5000)
        self.spharm_iter_spin.setValue(1000)
        self.spharm_iter_spin.setSingleStep(100)
        self.spharm_iter_spin.setEnabled(False)

        self.spharm_subdiv_spin = QSpinBox()
        self.spharm_subdiv_spin.setRange(1, 30)
        self.spharm_subdiv_spin.setValue(10)
        self.spharm_subdiv_spin.setEnabled(False)

        self.spharm_degree_spin = QSpinBox()
        self.spharm_degree_spin.setRange(1, 30)
        self.spharm_degree_spin.setValue(12)
        self.spharm_degree_spin.setEnabled(False)

        self.spharm_regen_cb = QCheckBox("Regenerate SPHARM Only (Reuse existing _surf.vtk and _para.vtk)")
        self.spharm_regen_cb.setChecked(False)

        adv_form.addRow("Preset Mode:", self.spharm_mode_combo)
        adv_form.addRow("GenParaMesh Iterations:", self.spharm_iter_spin)
        adv_form.addRow("Subdivision Level:", self.spharm_subdiv_spin)
        adv_form.addRow("SPHARM Degree:", self.spharm_degree_spin)
        adv_form.addRow("", self.spharm_regen_cb)

        self.adv_container.setVisible(False)
        spharm_layout.addWidget(self.adv_container)

        # 5. Results Table with Category Tabs
        res_group = QGroupBox("SPHARM Surface Meshes && Models")
        res_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #ffffff;
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
        res_layout = QVBoxLayout(res_group)
        res_layout.setContentsMargins(10, 16, 10, 10)
        res_layout.setSpacing(6)

        # Reference template overlay toggle checkbox and Overlay All Meshes checkbox
        template_bar = QHBoxLayout()
        template_bar.setContentsMargins(0, 0, 0, 2)
        template_bar.setSpacing(12)
        self.template_cb = QCheckBox("Show Reference Template")
        self.template_cb.setToolTip("Overlay standard reference template (mean shape) in 3D view")
        self.template_cb.setStyleSheet("""
            QCheckBox {
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
            }
            QCheckBox::indicator:checked {
                background: #f39c12;
                border: 1px solid #d68910;
            }
        """)
        self.template_cb.toggled.connect(self.signal_template_toggled.emit)
        template_bar.addWidget(self.template_cb)

        self.overlay_cb = QCheckBox("Overlay All Meshes")
        self.overlay_cb.setToolTip("Superimpose and view all SPHARM meshes together in 3D view")
        self.overlay_cb.setStyleSheet("""
            QCheckBox {
                color: #16a085;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
            }
            QCheckBox::indicator:checked {
                background: #1abc9c;
                border: 1px solid #16a085;
            }
        """)
        self.overlay_cb.toggled.connect(self.on_overlay_cb_toggled)
        template_bar.addWidget(self.overlay_cb)

        template_bar.addStretch()
        res_layout.addLayout(template_bar)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All")
        self.tab_bar.addTab("Left (LH)")
        self.tab_bar.addTab("Right (RH)")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setStyleSheet("""
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 6px 14px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #ced6e0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
                color: #2c3e50;
                border: 1px solid #b2bec3;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
            }
        """)
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        res_layout.addWidget(self.tab_bar)

        self.results_table = ToggleTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["SPHARM Mesh Name", "Side", "File Path"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #ecf0f1;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #dcdde1;
                font-size: 11px;
            }
        """)
        self.results_table.itemSelectionChanged.connect(self.on_mesh_selected)
        res_layout.addWidget(self.results_table)

        spharm_layout.addWidget(res_group)

        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    def toggle_advanced_params(self):
        should_show = self.adv_container.isHidden()
        self.adv_container.setVisible(should_show)
        self.toggle_adv_btn.setText("⚙️ Advanced Parameters ▴" if should_show else "⚙️ Advanced Parameters ▾")

    def on_spharm_mode_changed(self, index):
        if index == 0:  # Production
            self.spharm_iter_spin.setValue(1000)
            self.spharm_subdiv_spin.setValue(10)
            self.spharm_degree_spin.setValue(12)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 1:  # Fast Test
            self.spharm_iter_spin.setValue(200)
            self.spharm_subdiv_spin.setValue(5)
            self.spharm_degree_spin.setValue(6)
            self.spharm_iter_spin.setEnabled(False)
            self.spharm_subdiv_spin.setEnabled(False)
            self.spharm_degree_spin.setEnabled(False)
        elif index == 2:  # Custom
            self.spharm_iter_spin.setEnabled(True)
            self.spharm_subdiv_spin.setEnabled(True)
            self.spharm_degree_spin.setEnabled(True)

    def browse_input_directory(self):
        initial = self.mesh_input_dir.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Mesh Folder (FastSurfer or ICP Output)", initial)
        if folder:
            self.mesh_input_dir.setText(folder)
            self.spharm_dir_input.setText(self.get_default_output_dir(folder))
            self.update_run_button_state()
            self.populate_results_table()

    def reset_to_pipeline_input(self):
        self.mesh_input_dir.clear()
        self.spharm_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()
        self.signal_log_message.emit("[INFO] Reset SPHARM input to default pipeline output.")

    def on_input_dir_changed(self, text):
        if text.strip() and os.path.isdir(text.strip()):
            self.spharm_dir_input.setText(self.get_default_output_dir(text.strip()))
        elif not text.strip():
            self.spharm_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()

    def browse_output_directory(self):
        initial = self.spharm_dir_input.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Output Directory (output_SPHARM)", initial)
        if folder:
            self.spharm_dir_input.setText(folder)
            self.populate_results_table()
            self.update_run_button_state()

    def resolve_input_folders(self):
        """
        Resolves Left and Right input folders for SPHARM.
        CRITICAL: SPHARM cannot skip ICP. It MUST use meshes that have passed ICP alignment.
        Candidates:
        1. Custom input folder (must contain aligned_nifti or *_aligned.nii.gz).
        2. Pipeline output_ICP (output_ICP/left/aligned_nifti, output_ICP/right/aligned_nifti).
        3. Legacy pipeline icp folder (icp/left/aligned_nifti, icp/right/aligned_nifti).
        Returns: (lh_dir, rh_dir, base_folder, source_type_str)
        """
        custom_input = self.mesh_input_dir.text().strip()
        candidates = []
        if custom_input and os.path.isdir(custom_input):
            candidates.append(custom_input)
            
        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            candidates.append(os.path.join(out_base, "output_ICP"))
            candidates.append(os.path.join(out_base, "icp"))

        for base in candidates:
            # Check 0: If base itself is named aligned_nifti
            base_name = os.path.basename(base.rstrip(r'\/')).lower()
            parent = os.path.dirname(base.rstrip(r'\/'))
            if base_name == "aligned_nifti":
                grandparent = os.path.dirname(parent)
                if os.path.basename(parent).lower() in ("left", "lh"):
                    r_cand = os.path.join(grandparent, "right", "aligned_nifti")
                    if os.path.isdir(r_cand) and glob.glob(os.path.join(r_cand, "*.nii*")):
                        return base, r_cand, grandparent, "ICP Aligned"
                    return base, None, grandparent, "ICP Left Only"
                elif os.path.basename(parent).lower() in ("right", "rh"):
                    l_cand = os.path.join(grandparent, "left", "aligned_nifti")
                    if os.path.isdir(l_cand) and glob.glob(os.path.join(l_cand, "*.nii*")):
                        return l_cand, base, grandparent, "ICP Aligned"
                    return None, base, grandparent, "ICP Right Only"

            # Check 1: Standard output_ICP structure (left/aligned_nifti and right/aligned_nifti)
            lh_icp = os.path.join(base, "left", "aligned_nifti")
            rh_icp = os.path.join(base, "right", "aligned_nifti")
            if (os.path.isdir(lh_icp) and glob.glob(os.path.join(lh_icp, "*.nii*"))) or \
               (os.path.isdir(rh_icp) and glob.glob(os.path.join(rh_icp, "*.nii*"))):
                return lh_icp, rh_icp, base, "ICP Aligned"

            # Check 2: Direct left and right folders containing aligned_nifti or *_aligned.nii.gz
            lh_lr = os.path.join(base, "left")
            rh_lr = os.path.join(base, "right")
            if os.path.isdir(lh_lr) or os.path.isdir(rh_lr):
                lh_target = os.path.join(lh_lr, "aligned_nifti") if os.path.isdir(os.path.join(lh_lr, "aligned_nifti")) else lh_lr
                rh_target = os.path.join(rh_lr, "aligned_nifti") if os.path.isdir(os.path.join(rh_lr, "aligned_nifti")) else rh_lr
                lh_files = glob.glob(os.path.join(lh_target, "*.nii*"))
                rh_files = glob.glob(os.path.join(rh_target, "*.nii*"))
                # Require ICP alignment markers
                is_icp = ("aligned" in lh_target.lower() or "aligned" in rh_target.lower() or
                          "output_icp" in base.lower() or "icp" in base.lower() or
                          any("aligned" in f.lower() for f in (lh_files + rh_files)))
                if is_icp and (lh_files or rh_files):
                    return lh_target, rh_target, base, "ICP Aligned"

            # Check 3: Base folder directly contains *_aligned.nii.gz files
            aligned_files = glob.glob(os.path.join(base, "*aligned*.nii*"))
            if aligned_files:
                lh_files = [f for f in aligned_files if os.path.basename(f).startswith("lh_") or "_lh" in os.path.basename(f).lower() or "left" in os.path.basename(f).lower()]
                rh_files = [f for f in aligned_files if os.path.basename(f).startswith("rh_") or "_rh" in os.path.basename(f).lower() or "right" in os.path.basename(f).lower()]
                if lh_files and rh_files:
                    sub_lh = os.path.join(base, "left", "aligned_nifti")
                    sub_rh = os.path.join(base, "right", "aligned_nifti")
                    os.makedirs(sub_lh, exist_ok=True)
                    os.makedirs(sub_rh, exist_ok=True)
                    for f in lh_files:
                        dst = os.path.join(sub_lh, os.path.basename(f))
                        if not os.path.exists(dst):
                            try: os.link(f, dst)
                            except Exception:
                                import shutil; shutil.copy2(f, dst)
                    for f in rh_files:
                        dst = os.path.join(sub_rh, os.path.basename(f))
                        if not os.path.exists(dst):
                            try: os.link(f, dst)
                            except Exception:
                                import shutil; shutil.copy2(f, dst)
                    return sub_lh, sub_rh, base, "ICP Aligned"
                elif lh_files:
                    return base, None, os.path.dirname(base), "ICP Left Only"
                elif rh_files:
                    return None, base, os.path.dirname(base), "ICP Right Only"

        return None, None, None, "None"

    def get_default_output_dir(self, resolved_base=None):
        custom_input = self.mesh_input_dir.text().strip()
        if custom_input and os.path.isdir(custom_input):
            p_dir = os.path.dirname(custom_input.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_SPHARM")
            return os.path.join(custom_input, "output_SPHARM")

        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            return os.path.join(out_base, "output_SPHARM")
        elif resolved_base and os.path.isdir(resolved_base):
            p_dir = os.path.dirname(resolved_base.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_SPHARM")
            return os.path.join(resolved_base, "output_SPHARM")
        return ""

    def get_source_paths(self):
        lh, rh, base, _ = self.resolve_input_folders()
        out = self.get_default_output_dir(base)
        return lh, rh, out

    def update_run_button_state(self):
        lh_in, rh_in, resolved_base, src_type = self.resolve_input_folders()
        
        has_lh = bool(lh_in and os.path.isdir(lh_in) and glob.glob(os.path.join(lh_in, "*.nii*")))
        has_rh = bool(rh_in and os.path.isdir(rh_in) and glob.glob(os.path.join(rh_in, "*.nii*")))
        
        default_out = self.get_default_output_dir(resolved_base)
        custom_dir = self.spharm_dir_input.text().strip()
        if not custom_dir and default_out:
            self.spharm_dir_input.setText(default_out)
            
        target_out = custom_dir if custom_dir else default_out
        
        if not has_lh and not has_rh:
            self.run_spharm_btn.setEnabled(False)
            msg = "[LOCKED] SPHARM requires ICP-aligned meshes (output_ICP). Please run Groupwise ICP Registration first, or select an ICP output folder."
            self.run_spharm_btn.setToolTip(msg)
            self.spharm_status_hint.setText(msg)
            self.spharm_status_hint.setStyleSheet("""
                color: #c0392b; 
                background-color: #fdedec; 
                border: 1px solid #f5b7b1; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)
        else:
            self.run_spharm_btn.setEnabled(True)
            self.run_spharm_btn.setToolTip("Click to run Batch SPHARM Processing")
            lh_count = len(glob.glob(os.path.join(lh_in, "*.nii*"))) if has_lh else 0
            rh_count = len(glob.glob(os.path.join(rh_in, "*.nii*"))) if has_rh else 0
            src_label = f"Custom: {src_type}" if self.mesh_input_dir.text().strip() else f"Pipeline: {src_type}"
            msg = f"[OK] Ready ({src_label}): Detected {lh_count} Left & {rh_count} Right ICP-aligned subjects. Output: {target_out}"
            self.spharm_status_hint.setText(msg)
            self.spharm_status_hint.setStyleSheet("""
                color: #1e8449; 
                background-color: #eafaf1; 
                border: 1px solid #a9dfbf; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)

        self.populate_results_table()

    def run_spharm_process(self):
        lh_in, rh_in, resolved_base, _ = self.resolve_input_folders()
        target_base = self.spharm_dir_input.text().strip() or self.get_default_output_dir(resolved_base)
        
        if not target_base:
            self.signal_log_message.emit("[ERROR] Output directory is missing. Please configure Output Directory.")
            return

        os.makedirs(target_base, exist_ok=True)
        tasks = []
        mode = self.side_btn_group.checkedId()
        
        # 0: Both, 1: Left only, 2: Right only
        # Create output_SPHARM/left and output_SPHARM/right separately
        if mode in (0, 1):
            if lh_in and os.path.isdir(lh_in):
                out_left = os.path.join(target_base, "left")
                os.makedirs(out_left, exist_ok=True)
                tasks.append(("left", lh_in, out_left))
            else:
                self.signal_log_message.emit("[WARNING] Left hippocampus input folder not found.")
                
        if mode in (0, 2):
            if rh_in and os.path.isdir(rh_in):
                out_right = os.path.join(target_base, "right")
                os.makedirs(out_right, exist_ok=True)
                tasks.append(("right", rh_in, out_right))
            else:
                self.signal_log_message.emit("[WARNING] Right hippocampus input folder not found.")

        if not tasks:
            self.signal_log_message.emit("[ERROR] No valid hippocampus folders found to run SPHARM.")
            return

        adv_params = {
            "num_iter": self.spharm_iter_spin.value(),
            "subdiv": self.spharm_subdiv_spin.value(),
            "degree": self.spharm_degree_spin.value(),
            "regen_only": self.spharm_regen_cb.isChecked()
        }

        self.run_spharm_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(f">>> Initiating Batch SPHARM-PDM Pipeline (Output: {target_base})...")

        self.worker = SPHARMWorker(tasks, adv_params)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_spharm_finished)
        self.worker.start()

    def on_spharm_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> Batch SPHARM-PDM Pipeline completed successfully.")
            self.signal_spharm_completed.emit()
        else:
            self.signal_log_message.emit("[ERROR] SPHARM Pipeline completed with warnings or errors.")
        self.signal_spharm_finished.emit(success)
        self.populate_results_table()

    def on_tab_changed(self, index):
        if index == 1:
            self.current_side_filter = "lh"
        elif index == 2:
            self.current_side_filter = "rh"
        else:
            self.current_side_filter = "all"
        self.signal_side_changed.emit(self.current_side_filter)
        self.update_table_display()
        if self.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def set_template_visible(self, visible: bool):
        self.template_cb.blockSignals(True)
        self.template_cb.setChecked(visible)
        self.template_cb.blockSignals(False)

    def set_overlay_visible(self, visible: bool):
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(visible)
        self.overlay_cb.blockSignals(False)
        if visible:
            self.emit_overlay_meshes()

    def on_overlay_cb_toggled(self, checked: bool):
        if checked:
            self.emit_overlay_meshes()
        else:
            self.signal_overlay_all_toggled.emit(False, [], self.current_side_filter)

    def get_current_display_files(self):
        if self.current_side_filter == "lh":
            return [f for f in self.all_files if f["side_key"] == "lh"]
        elif self.current_side_filter == "rh":
            return [f for f in self.all_files if f["side_key"] == "rh"]
        return self.all_files

    def emit_overlay_meshes(self):
        display_files = self.get_current_display_files()
        filepaths = [f["filepath"] for f in display_files]
        self.signal_overlay_all_toggled.emit(True, filepaths, self.current_side_filter)

    def populate_results_table(self):
        target_base = self.spharm_dir_input.text().strip()
        if not target_base:
            _, _, default_spharm_out = self.get_source_paths()
            target_base = default_spharm_out
            
        self.all_files = []
        if target_base and os.path.isdir(target_base):
            search_dirs = [
                (os.path.join(target_base, "left", "spharm_results"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right", "spharm_results"), "rh", "Right (RH)"),
                (os.path.join(target_base, "spharm_results"), "all", "SPHARM"),
                (os.path.join(target_base, "output_left_hippocampus", "spharm_results_left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "output_right_hippocampus", "spharm_results_right"), "rh", "Right (RH)"),
                (os.path.join(target_base, "split_data", "ALL_Left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "split_data", "ALL_Right"), "rh", "Right (RH)"),
                (os.path.join(target_base, "left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right"), "rh", "Right (RH)"),
                (target_base, "all", "SPHARM"),
            ]
            
            seen = set()
            for directory, side_key, side_label in search_dirs:
                if os.path.isdir(directory):
                    all_vtk = glob.glob(os.path.join(directory, "*.vtk"))
                    # Group meshes by subject to pick the single best SPHARM surface mesh per subject
                    subject_map = {}
                    for vtk_file in all_vtk:
                        bn = os.path.basename(vtk_file)
                        bn_lower = bn.lower()
                        # Filter out templates, mean shapes, and intermediate/auxiliary files
                        if "mean_shape" in bn_lower or bn_lower.startswith("template_"):
                            continue
                        if any(aux in bn_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.")):
                            continue

                        # Extract base subject key
                        s_key = bn
                        for suf in ("_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk", ".vtk"):
                            if s_key.endswith(suf):
                                s_key = s_key[:-len(suf)]
                                break

                        if s_key not in subject_map:
                            subject_map[s_key] = []
                        subject_map[s_key].append(os.path.normpath(vtk_file))

                    for s_key, f_list in sorted(subject_map.items()):
                        # Priority: _realigned.vtk > _procalign.vtk > _ellalign.vtk > _SPHARM.vtk
                        best_mesh = None
                        for suf in ("_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                            cand = [f for f in f_list if f.endswith(suf)]
                            if cand:
                                best_mesh = cand[0]
                                break
                        if not best_mesh and f_list:
                            best_mesh = f_list[0]

                        norm_p = best_mesh
                        if norm_p not in seen:
                            seen.add(norm_p)
                            basename = os.path.basename(norm_p)

                            cur_key = side_key
                            cur_label = side_label
                            norm_lower = norm_p.lower()
                            if cur_key == "all":
                                if basename.startswith("lh_") or "left" in norm_lower or "_lh" in norm_lower or "\\left\\" in norm_lower or "/left/" in norm_lower:
                                    cur_key, cur_label = "lh", "Left (LH)"
                                elif basename.startswith("rh_") or "right" in norm_lower or "_rh" in norm_lower or "\\right\\" in norm_lower or "/right/" in norm_lower:
                                    cur_key, cur_label = "rh", "Right (RH)"

                            self.all_files.append({
                                "filename": basename,
                                "side": cur_label,
                                "side_key": cur_key,
                                "filepath": norm_p
                            })

        # Update tab counts
        all_count = len(self.all_files)
        lh_count = sum(1 for f in self.all_files if f["side_key"] == "lh")
        rh_count = sum(1 for f in self.all_files if f["side_key"] == "rh")
        self.tab_bar.setTabText(0, f"All ({all_count})")
        self.tab_bar.setTabText(1, f"Left ({lh_count})")
        self.tab_bar.setTabText(2, f"Right ({rh_count})")

        self.update_table_display()
        if self.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def update_table_display(self):
        display_files = self.get_current_display_files()

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(display_files))
        for i, item in enumerate(display_files):
            name_item = QTableWidgetItem(item["filename"])
            name_item.setData(Qt.ItemDataRole.UserRole, item["filepath"])
            name_item.setData(Qt.ItemDataRole.UserRole + 1, item["side_key"])
            self.results_table.setItem(i, 0, name_item)

            side_item = QTableWidgetItem(item["side"])
            side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if item["side_key"] == "lh":
                side_item.setForeground(Qt.GlobalColor.blue)
            elif item["side_key"] == "rh":
                side_item.setForeground(Qt.GlobalColor.darkYellow)
            self.results_table.setItem(i, 1, side_item)

            path_item = QTableWidgetItem(item["filepath"])
            path_item.setToolTip(item["filepath"])
            self.results_table.setItem(i, 2, path_item)

        self.results_table.blockSignals(False)
        self.results_table.clearSelection()

    def on_mesh_selected(self):
        selected_items = self.results_table.selectedItems()
        selected_rows = sorted(list(set(it.row() for it in selected_items)))

        if not selected_rows:
            # 0 items selected -> clear 3D mesh
            self.signal_mesh_selected.emit("", self.current_side_filter)
            return

        if len(selected_rows) == 1:
            row = selected_rows[0]
            name_item = self.results_table.item(row, 0)
            if name_item:
                filepath = name_item.data(Qt.ItemDataRole.UserRole)
                side_key = name_item.data(Qt.ItemDataRole.UserRole + 1) or self.current_side_filter
                if filepath:
                    self.signal_mesh_selected.emit(filepath, side_key)
            return

        # Multi-select (> 1 meshes selected via Ctrl / Shift)
        filepaths = []
        for row in selected_rows:
            name_item = self.results_table.item(row, 0)
            if name_item:
                fp = name_item.data(Qt.ItemDataRole.UserRole)
                if fp:
                    filepaths.append(fp)

        if filepaths:
            self.signal_mesh_selected.emit(filepaths, self.current_side_filter)
