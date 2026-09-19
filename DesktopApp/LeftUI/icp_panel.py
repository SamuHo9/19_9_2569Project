import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
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
    for cand in local_candidates:
        if os.path.isfile(cand):
            return cand
            
    candidates = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
    if candidates:
        return candidates[0]
    return "C:\\Program Files\\SlicerSALT 6.0.0\\SlicerSALT.exe"

class ICPWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params, parent=None):
        super().__init__(parent)
        self.tasks = tasks  # list of tuples: (side_name, input_dir, output_dir)
        self.adv_params = adv_params

    def run(self):
        project_root = get_project_root()
        icp_script = os.path.join(project_root, "ICP", "ICP.py")

        if not os.path.isfile(icp_script):
            self.signal_log.emit(f"[ERROR] ICP.py not found at: {icp_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running Batch ICP for [{side_name.upper()} Hippocampus]")
            self.signal_log.emit(f"    Input:  {in_dir}")
            self.signal_log.emit(f"    Output: {out_dir}")
            self.signal_log.emit(f"==================================================")
            
            os.makedirs(out_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                icp_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--output_spacing", str(self.adv_params.get("spacing", 0.02)),
                "--output_voxels", str(self.adv_params.get("voxels", 128)),
                "--max_iterations", str(self.adv_params.get("max_iter", 20)),
                "--tolerance", str(self.adv_params.get("tol", 0.00005)),
                "--pairwise_iterations", str(self.adv_params.get("pw_iter", 100)),
                "--pairwise_tolerance", str(self.adv_params.get("pw_tol", 0.0001)),
                "--pairwise_landmarks", str(self.adv_params.get("pw_landmarks", 200)),
                "--interpolation", str(self.adv_params.get("interp", "NearestNeighbor"))
            ]

            # Check for standard reference template in Templates/ICP
            tmpl_cand = os.path.join(project_root, "Templates", "ICP", f"template_mean_{side_name.lower()}.vtk")
            if not os.path.isfile(tmpl_cand):
                tmpl_cand = os.path.join(project_root, "Templates", "ICP", f"template_mean_{side_name.lower()}.ply")
            tmpl_cand = os.environ.get(f'HIPPO_ICP_REFERENCE_{side_name.upper()}', tmpl_cand)
            if os.path.isfile(tmpl_cand):
                cmd.extend(["--reference_template", tmpl_cand])
                self.signal_log.emit(f"    Using Reference Template: {os.path.basename(tmpl_cand)}")
            else:
                self.signal_log.emit(f'[ERROR] Missing fixed reference for {side_name}. Set HIPPO_ICP_REFERENCE_{side_name.upper()} to a training reference bundle.')
                overall_success = False
                continue

            try:
                kwargs = {}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                    
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **kwargs
                )
                
                for line in process.stdout:
                    clean = line.strip()
                    if clean:
                        self.signal_log.emit(clean)
                process.wait()
                
                if process.returncode != 0:
                    self.signal_log.emit(f"[ERROR] ICP process failed for {side_name} with return code {process.returncode}")
                    overall_success = False
                else:
                    self.signal_log.emit(f"[OK] ICP completed successfully for {side_name}.")
            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running ICP {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)

IcpWorker = ICPWorker

class IcpPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str) # filepath can be str or list of str
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_all_toggled = pyqtSignal(bool, list, str) # enabled, file_list, side_filter
    signal_side_changed = pyqtSignal(str) # "all", "lh", "rh"
    signal_icp_completed = pyqtSignal()
    signal_icp_finished = pyqtSignal(bool)

    def __init__(self, get_folder_func, get_output_folder_func=None, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.get_output_folder = get_output_folder_func
        self.all_files = []
        self.current_side_filter = "all"
        self.last_selected_row = None
        self.setup_ui()

    def setup_ui(self):
        icp_layout = QVBoxLayout(self)
        icp_layout.setContentsMargins(10, 10, 10, 10)
        icp_layout.setSpacing(10)
        
        help_label = QLabel("Groupwise rigid ICP alignment for Left and Right Hippocampus masks from FastSurfer.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        icp_layout.addWidget(help_label)
        
        # 1. Directory Configuration (Import Meshes & Dedicated output_ICP)
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

        # Row A: Input Meshes (FastSurfer / Segmentation Folder)
        in_lbl = QLabel("📥 Input Meshes (FastSurfer / Custom Mesh Folder):")
        in_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        dir_layout.addWidget(in_lbl)

        in_row = QHBoxLayout()
        self.mesh_input_dir = QLineEdit()
        self.mesh_input_dir.setPlaceholderText("Auto (FastSurfer output) or Browse to import meshes...")
        self.mesh_input_dir.textChanged.connect(self.on_input_dir_changed)
        in_row.addWidget(self.mesh_input_dir)

        browse_in_btn = QPushButton("📁 Browse...")
        browse_in_btn.setToolTip("Import existing FastSurfer mesh folder from disk (skip previous steps)")
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
        reset_in_btn.setToolTip("Reset input back to current Data Importer / FastSurfer pipeline output")
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

        # Row B: Output Directory (output_ICP)
        out_lbl = QLabel("📤 Output Directory (Dedicated output_ICP):")
        out_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50; margin-top: 4px;")
        dir_layout.addWidget(out_lbl)

        out_row = QHBoxLayout()
        self.icp_dir_input = QLineEdit()
        self.icp_dir_input.setPlaceholderText("Auto (.../output_ICP)")
        self.icp_dir_input.textChanged.connect(self.populate_results_table)
        out_row.addWidget(self.icp_dir_input)

        browse_out_btn = QPushButton("📁 Browse...")
        browse_out_btn.setToolTip("Select custom destination for output_ICP")
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
        reload_btn.setToolTip("Scan output_ICP folder and reload results table")
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
        icp_layout.addWidget(dir_group)

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
        icp_layout.addWidget(side_group)

        # 3. Main Action Button & Status Hint
        self.run_icp_btn = QPushButton("▶ Run Groupwise ICP Registration")
        self.run_icp_btn.setStyleSheet("""
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
        self.run_icp_btn.clicked.connect(self.run_icp_process)
        icp_layout.addWidget(self.run_icp_btn)

        self.icp_status_hint = QLabel("")
        self.icp_status_hint.setWordWrap(True)
        self.icp_status_hint.setStyleSheet("font-size: 11px; padding: 5px 8px; border-radius: 4px;")
        icp_layout.addWidget(self.icp_status_hint)

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
        icp_layout.addWidget(self.toggle_adv_btn)

        self.adv_container = QFrame()
        self.adv_container.setStyleSheet("""
            QFrame {
                border: 1px solid #e1e2e6;
                border-radius: 5px;
                background-color: #fafbfc;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
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

        self.icp_mode_combo = QComboBox()
        self.icp_mode_combo.addItems(["Production Mode (Default)", "Custom Mode"])
        self.icp_mode_combo.currentIndexChanged.connect(self.on_icp_mode_changed)

        self.icp_spacing_spin = QDoubleSpinBox()
        self.icp_spacing_spin.setLocale(QLocale.c())
        self.icp_spacing_spin.setRange(0.001, 10.0)
        self.icp_spacing_spin.setDecimals(3)
        self.icp_spacing_spin.setValue(0.02)
        self.icp_spacing_spin.setEnabled(False)

        self.icp_voxels_spin = QSpinBox()
        self.icp_voxels_spin.setLocale(QLocale.c())
        self.icp_voxels_spin.setRange(16, 1024)
        self.icp_voxels_spin.setValue(128)
        self.icp_voxels_spin.setEnabled(False)

        self.icp_max_iter_spin = QSpinBox()
        self.icp_max_iter_spin.setRange(1, 1000)
        self.icp_max_iter_spin.setValue(20)
        self.icp_max_iter_spin.setEnabled(False)

        self.icp_tol_spin = QDoubleSpinBox()
        self.icp_tol_spin.setLocale(QLocale.c())
        self.icp_tol_spin.setRange(0.000001, 1.0)
        self.icp_tol_spin.setDecimals(6)
        self.icp_tol_spin.setValue(0.00005)
        self.icp_tol_spin.setEnabled(False)

        self.icp_pw_iter_spin = QSpinBox()
        self.icp_pw_iter_spin.setRange(1, 1000)
        self.icp_pw_iter_spin.setValue(100)
        self.icp_pw_iter_spin.setEnabled(False)

        self.icp_pw_tol_spin = QDoubleSpinBox()
        self.icp_pw_tol_spin.setLocale(QLocale.c())
        self.icp_pw_tol_spin.setRange(0.000001, 1.0)
        self.icp_pw_tol_spin.setDecimals(6)
        self.icp_pw_tol_spin.setValue(0.0001)
        self.icp_pw_tol_spin.setEnabled(False)

        self.icp_pw_landmarks_spin = QSpinBox()
        self.icp_pw_landmarks_spin.setRange(10, 5000)
        self.icp_pw_landmarks_spin.setValue(200)
        self.icp_pw_landmarks_spin.setEnabled(False)

        self.icp_interp_combo = QComboBox()
        self.icp_interp_combo.addItems(["NearestNeighbor", "Linear", "BSpline"])
        self.icp_interp_combo.setEnabled(False)

        adv_form.addRow("Preset Mode:", self.icp_mode_combo)
        adv_form.addRow("Output Spacing:", self.icp_spacing_spin)
        adv_form.addRow("Output Voxels:", self.icp_voxels_spin)
        adv_form.addRow("Groupwise Max Iterations:", self.icp_max_iter_spin)
        adv_form.addRow("Groupwise Tolerance:", self.icp_tol_spin)
        adv_form.addRow("Pairwise Max Iterations:", self.icp_pw_iter_spin)
        adv_form.addRow("Pairwise Tolerance:", self.icp_pw_tol_spin)
        adv_form.addRow("Pairwise Landmarks:", self.icp_pw_landmarks_spin)
        adv_form.addRow("Interpolation Mode:", self.icp_interp_combo)

        self.adv_container.setVisible(False)
        icp_layout.addWidget(self.adv_container)

        # 5. Results Table with Category Tabs
        res_group = QGroupBox("ICP Aligned Meshes & Results")
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
        self.overlay_cb.setToolTip("Superimpose and view all aligned meshes together in 3D view")
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
        self.results_table.setHorizontalHeaderLabels(["Aligned Mesh Name", "Side", "File Path"])
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

        icp_layout.addWidget(res_group)

        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    def toggle_advanced_params(self):
        should_show = self.adv_container.isHidden()
        self.adv_container.setVisible(should_show)
        self.toggle_adv_btn.setText("⚙️ Advanced Parameters ▴" if should_show else "⚙️ Advanced Parameters ▾")

    def on_icp_mode_changed(self, index):
        is_custom = (index == 1)
        for widget in [self.icp_spacing_spin, self.icp_voxels_spin, self.icp_max_iter_spin,
                       self.icp_tol_spin, self.icp_pw_iter_spin, self.icp_pw_tol_spin,
                       self.icp_pw_landmarks_spin, self.icp_interp_combo]:
            widget.setEnabled(is_custom)
            
        if not is_custom:
            self.icp_spacing_spin.setValue(0.02)
            self.icp_voxels_spin.setValue(128)
            self.icp_max_iter_spin.setValue(20)
            self.icp_tol_spin.setValue(0.00005)
            self.icp_pw_iter_spin.setValue(100)
            self.icp_pw_tol_spin.setValue(0.0001)
            self.icp_pw_landmarks_spin.setValue(200)
            self.icp_interp_combo.setCurrentText("NearestNeighbor")

    def browse_input_directory(self):
        initial = self.mesh_input_dir.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Mesh Folder (FastSurfer Output or Custom)", initial)
        if folder:
            self.mesh_input_dir.setText(folder)
            self.icp_dir_input.setText(self.get_default_output_dir(folder))
            self.update_run_button_state()
            self.populate_results_table()

    def reset_to_pipeline_input(self):
        self.mesh_input_dir.clear()
        self.icp_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()
        self.signal_log_message.emit("[INFO] Reset ICP input to default pipeline output.")

    def on_input_dir_changed(self, text):
        if text.strip() and os.path.isdir(text.strip()):
            self.icp_dir_input.setText(self.get_default_output_dir(text.strip()))
        elif not text.strip():
            self.icp_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()

    def browse_output_directory(self):
        initial = self.icp_dir_input.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select ICP Output Directory (output_ICP)", initial)
        if folder:
            self.icp_dir_input.setText(folder)
            self.populate_results_table()
            self.update_run_button_state()

    def resolve_input_folders(self):
        """
        Resolves Left and Right input folders.
        Checks:
        1. Custom input folder (self.mesh_input_dir) if provided.
        2. Pipeline FastSurfer folder (from get_output_folder).
        """
        custom_input = self.mesh_input_dir.text().strip()
        candidates = []
        if custom_input and os.path.isdir(custom_input):
            candidates.append(custom_input)
            
        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            candidates.append(os.path.join(out_base, "fastsurfer"))
            candidates.append(out_base)

        for base in candidates:
            # Check 0: If base itself is named left_hippocampus or right_hippocampus
            base_name = os.path.basename(base.rstrip(r'\/')).lower()
            parent = os.path.dirname(base.rstrip(r'\/'))
            if base_name in ("left_hippocampus", "lh", "left"):
                for cand_r in ["right_hippocampus", "rh", "right"]:
                    r_path = os.path.join(parent, cand_r)
                    if os.path.isdir(r_path) and glob.glob(os.path.join(r_path, "*.nii*")):
                        return base, r_path, parent
                return base, None, parent
            elif base_name in ("right_hippocampus", "rh", "right"):
                for cand_l in ["left_hippocampus", "lh", "left"]:
                    l_path = os.path.join(parent, cand_l)
                    if os.path.isdir(l_path) and glob.glob(os.path.join(l_path, "*.nii*")):
                        return l_path, base, parent
                return None, base, parent

            # Check 1: subfolders left_hippocampus and right_hippocampus
            lh = os.path.join(base, "left_hippocampus")
            rh = os.path.join(base, "right_hippocampus")
            if (os.path.isdir(lh) and glob.glob(os.path.join(lh, "*.nii*"))) or \
               (os.path.isdir(rh) and glob.glob(os.path.join(rh, "*.nii*"))):
                return lh, rh, base
                
            # Check 2: subfolders fastsurfer/left_hippocampus and fastsurfer/right_hippocampus
            lh_fs = os.path.join(base, "fastsurfer", "left_hippocampus")
            rh_fs = os.path.join(base, "fastsurfer", "right_hippocampus")
            if (os.path.isdir(lh_fs) and glob.glob(os.path.join(lh_fs, "*.nii*"))) or \
               (os.path.isdir(rh_fs) and glob.glob(os.path.join(rh_fs, "*.nii*"))):
                return lh_fs, rh_fs, base

            # Check 3: subfolders left and right
            lh_lr = os.path.join(base, "left")
            rh_lr = os.path.join(base, "right")
            if (os.path.isdir(lh_lr) and glob.glob(os.path.join(lh_lr, "*.nii*"))) or \
               (os.path.isdir(rh_lr) and glob.glob(os.path.join(rh_lr, "*.nii*"))):
                return lh_lr, rh_lr, base

            # Check 4: base folder directly containing nii.gz files
            nii_files = glob.glob(os.path.join(base, "*.nii*"))
            if nii_files:
                lh_files = [f for f in nii_files if os.path.basename(f).startswith("lh_") or "_lh." in os.path.basename(f).lower() or "left" in os.path.basename(f).lower()]
                rh_files = [f for f in nii_files if os.path.basename(f).startswith("rh_") or "_rh." in os.path.basename(f).lower() or "right" in os.path.basename(f).lower()]
                if lh_files and rh_files:
                    sub_lh = os.path.join(base, "left_hippocampus")
                    sub_rh = os.path.join(base, "right_hippocampus")
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
                    return sub_lh, sub_rh, base
                elif lh_files:
                    return base, None, os.path.dirname(base)
                elif rh_files:
                    return None, base, os.path.dirname(base)
                else:
                    return base, base, base

        return None, None, None

    def get_default_output_dir(self, resolved_base=None):
        custom_input = self.mesh_input_dir.text().strip()
        if custom_input and os.path.isdir(custom_input):
            p_dir = os.path.dirname(custom_input.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_ICP")
            return os.path.join(custom_input, "output_ICP")

        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            return os.path.join(out_base, "output_ICP")
        elif resolved_base and os.path.isdir(resolved_base):
            p_dir = os.path.dirname(resolved_base.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_ICP")
            return os.path.join(resolved_base, "output_ICP")
        return ""

    def get_source_paths(self):
        return self.resolve_input_folders()

    def update_run_button_state(self):
        lh_dir, rh_dir, resolved_base = self.resolve_input_folders()
        
        has_lh = bool(lh_dir and os.path.isdir(lh_dir) and glob.glob(os.path.join(lh_dir, "*.nii*")))
        has_rh = bool(rh_dir and os.path.isdir(rh_dir) and glob.glob(os.path.join(rh_dir, "*.nii*")))
        
        default_out = self.get_default_output_dir(resolved_base)
        custom_dir = self.icp_dir_input.text().strip()
        if not custom_dir and default_out:
            self.icp_dir_input.setText(default_out)
            
        target_out = custom_dir if custom_dir else default_out
        
        if not has_lh and not has_rh:
            self.run_icp_btn.setEnabled(False)
            msg = "[LOCKED] No FastSurfer mesh outputs found. Please select a folder with meshes (Browse) or run FastSurfer first."
            self.run_icp_btn.setToolTip(msg)
            self.icp_status_hint.setText(msg)
            self.icp_status_hint.setStyleSheet("""
                color: #c0392b; 
                background-color: #fdedec; 
                border: 1px solid #f5b7b1; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)
        else:
            self.run_icp_btn.setEnabled(True)
            self.run_icp_btn.setToolTip("Click to run Groupwise ICP Registration")
            lh_count = len(glob.glob(os.path.join(lh_dir, "*.nii*"))) if has_lh else 0
            rh_count = len(glob.glob(os.path.join(rh_dir, "*.nii*"))) if has_rh else 0
            src_type = "Custom Folder" if self.mesh_input_dir.text().strip() else "Pipeline"
            msg = f"[OK] Ready ({src_type}): Detected {lh_count} Left & {rh_count} Right meshes. Results will be saved to: {target_out}"
            self.icp_status_hint.setText(msg)
            self.icp_status_hint.setStyleSheet("""
                color: #1e8449; 
                background-color: #eafaf1; 
                border: 1px solid #a9dfbf; 
                font-size: 11px; 
                padding: 6px 8px; 
                border-radius: 4px;
                font-weight: 500;
            """)

        self.populate_results_table()

    def run_icp_process(self):
        lh_dir, rh_dir, resolved_base = self.resolve_input_folders()
        target_base = self.icp_dir_input.text().strip() or self.get_default_output_dir(resolved_base)
        
        if not target_base:
            self.signal_log_message.emit("[ERROR] Output directory is missing. Please configure Output Directory.")
            return

        os.makedirs(target_base, exist_ok=True)
        tasks = []
        mode = self.side_btn_group.checkedId()
        
        # 0: Both, 1: Left only, 2: Right only
        # Create output_ICP/left and output_ICP/right separately
        if mode in (0, 1):
            if lh_dir and os.path.isdir(lh_dir):
                out_left = os.path.join(target_base, "left")
                os.makedirs(out_left, exist_ok=True)
                tasks.append(("left", lh_dir, out_left))
            else:
                self.signal_log_message.emit("[WARNING] Left hippocampus folder not found.")
                
        if mode in (0, 2):
            if rh_dir and os.path.isdir(rh_dir):
                out_right = os.path.join(target_base, "right")
                os.makedirs(out_right, exist_ok=True)
                tasks.append(("right", rh_dir, out_right))
            else:
                self.signal_log_message.emit("[WARNING] Right hippocampus folder not found.")

        if not tasks:
            self.signal_log_message.emit("[ERROR] No valid hippocampus folders found to run ICP.")
            return

        adv_params = {
            "spacing": self.icp_spacing_spin.value(),
            "voxels": self.icp_voxels_spin.value(),
            "max_iter": self.icp_max_iter_spin.value(),
            "tolerance": self.icp_tol_spin.value(),
            "pw_iter": self.icp_pw_iter_spin.value(),
            "pw_tol": self.icp_pw_tol_spin.value(),
            "pw_landmarks": self.icp_pw_landmarks_spin.value(),
            "interp": self.icp_interp_combo.currentText()
        }

        self.run_icp_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(f">>> Initiating Groupwise ICP Alignment Pipeline (Output: {target_base})...")

        self.worker = ICPWorker(tasks, adv_params)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_icp_finished)
        self.worker.start()

    def on_icp_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> Groupwise ICP Registration Pipeline completed successfully.")
            self.signal_icp_completed.emit()
        else:
            self.signal_log_message.emit("[ERROR] ICP Registration completed with warnings or errors.")
        self.signal_icp_finished.emit(success)
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

    def auto_extract_aligned_meshes(self, target_base):
        """Converts aligned_nifti/*.nii.gz to aligned_meshes/*.vtk if missing."""
        if not target_base or not os.path.isdir(target_base):
            return
        import vtk
        for side in ["left", "right"]:
            nii_dir = os.path.join(target_base, side, "aligned_nifti")
            out_mesh_dir = os.path.join(target_base, side, "aligned_meshes")
            if os.path.isdir(nii_dir):
                nii_files = glob.glob(os.path.join(nii_dir, "*.nii.gz"))
                if nii_files:
                    os.makedirs(out_mesh_dir, exist_ok=True)
                    for nf in nii_files:
                        bn = os.path.basename(nf)
                        for ext in [".nii.gz", ".nii", ".mgz"]:
                            if bn.endswith(ext):
                                bn = bn[:-len(ext)]
                                break
                        vtk_p = os.path.join(out_mesh_dir, f"{bn}.vtk")
                        if not os.path.isfile(vtk_p):
                            try:
                                r = vtk.vtkNIFTIImageReader()
                                r.SetFileName(nf)
                                r.Update()
                                mc = vtk.vtkDiscreteMarchingCubes()
                                mc.SetInputConnection(r.GetOutputPort())
                                mc.GenerateValues(1, 1, 100)
                                mc.Update()
                                qmat = r.GetQFormMatrix()
                                if not qmat:
                                    qmat = r.GetSFormMatrix()
                                t = vtk.vtkTransform()
                                if qmat:
                                    t.SetMatrix(qmat)
                                tf = vtk.vtkTransformPolyDataFilter()
                                tf.SetTransform(t)
                                tf.SetInputConnection(mc.GetOutputPort())
                                tf.Update()
                                w = vtk.vtkPolyDataWriter()
                                w.SetFileName(vtk_p)
                                w.SetInputData(tf.GetOutput())
                                w.Write()
                            except Exception as e:
                                print(f"[WARNING] Could not auto-convert {nf} to vtk: {e}")

    def populate_results_table(self):
        target_base = self.icp_dir_input.text().strip()
        if not target_base:
            _, _, default_icp_out = self.get_source_paths()
            target_base = default_icp_out
            
        self.all_files = []
        if target_base and os.path.isdir(target_base):
            # Check and auto-generate missing aligned .vtk meshes from aligned_nifti
            self.auto_extract_aligned_meshes(target_base)

            search_dirs = [
                (os.path.join(target_base, "left", "aligned_meshes"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right", "aligned_meshes"), "rh", "Right (RH)"),
                (os.path.join(target_base, "aligned_meshes"), "all", "Aligned"),
                (os.path.join(target_base, "output_left_hippocampus"), "lh", "Left (LH)"),
                (os.path.join(target_base, "output_right_hippocampus"), "rh", "Right (RH)"),
                (os.path.join(target_base, "left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right"), "rh", "Right (RH)"),
                (target_base, "all", "Aligned"),
            ]
            
            seen = set()
            for directory, side_key, side_label in search_dirs:
                if os.path.isdir(directory):
                    mesh_files = glob.glob(os.path.join(directory, "*.vtk")) + glob.glob(os.path.join(directory, "*.ply"))
                    for vtk_file in mesh_files:
                        norm_p = os.path.normpath(vtk_file)
                        if norm_p not in seen:
                            seen.add(norm_p)
                            basename = os.path.basename(norm_p)
                            
                            # Do NOT display reference templates / mean_shape in the results table
                            if "mean_shape" in basename.lower() or basename.lower().startswith("template_"):
                                continue

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
        selected_rows = sorted(list(set(index.row() for index in self.results_table.selectedIndexes())))
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
