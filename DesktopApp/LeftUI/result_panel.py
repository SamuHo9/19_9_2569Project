import os
import glob
import json
import csv
import shutil
import re
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QRadioButton, QButtonGroup, QProgressBar, QCheckBox, QSlider, QFrame,
    QScrollArea, QApplication, QMessageBox, QTabBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from models.predictor import HippocampalPredictor, parse_coef_file


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

    def keyPressEvent(self, event):
        # Prevent Left/Right arrow keys from scrolling columns; route to ResultPanel SD stepper!
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight,
                           Qt.Key.Key_Comma, Qt.Key.Key_Period, Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
            parent = self.parentWidget()
            while parent and not hasattr(parent, 'step_sd'):
                parent = parent.parentWidget()
            if parent and hasattr(parent, 'step_sd'):
                parent.keyPressEvent(event)
                event.accept()
                return
        super().keyPressEvent(event)


class ResultPanel(QWidget):
    """
    Result & Diagnostic Panel
    Inference-only evaluation using trained ResNet1D models and interactive
    3D Grad-CAM attention and deformation heatmap visualization.
    Batch-evaluates all meshes and saves structured outputs in output_Result.
    """
    signal_log_message = pyqtSignal(str)
    signal_gradcam_mesh_requested = pyqtSignal(str, str, str, str, str, float)
    signal_patient_overlay_requested = pyqtSignal(str, bool, float, str)
    signal_clear_gradcam_requested = pyqtSignal()
    signal_batch_prediction_finished = pyqtSignal(bool)

    def __init__(self, parent=None, get_input_folder=None, get_output_folder=None):
        super().__init__(parent)
        self.get_input_folder = get_input_folder
        self.get_output_folder = get_output_folder

        self.predictor = HippocampalPredictor()
        self.current_patient_data = None
        self.last_prediction_results = None
        self.all_evaluation_results = []
        self.current_tab_filter = "all"
        self.current_sd_step_idx = 30  # 30 corresponds to 0.0 SD (Mean)
        self.current_sd_value = 0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # Scroll area to handle smaller screens gracefully
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.horizontalScrollBar().setEnabled(False)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Diagnostic Model & Input Data
        # ---------------------------------------------------------------------
        header_group = QGroupBox("1. Diagnostic Model & Input Data")
        header_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        h_layout = QVBoxLayout(header_group)
        h_layout.setSpacing(6)

        # Path input
        path_row = QHBoxLayout()
        path_lbl = QLabel("SPHARM Dir:")
        path_lbl.setFixedWidth(68)
        path_lbl.setStyleSheet("font-size: 11px; color: #2c3e50; font-weight: bold;")
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.setPlaceholderText("Select directory with SPHARM .coef / .vtk results...")
        self.spharm_dir_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ced6e0;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
                background: #ffffff;
            }
            QLineEdit:focus { border: 1px solid #3498db; }
        """)

        browse_btn = QPushButton("📁 Browse")
        browse_btn.setFixedHeight(26)
        browse_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        browse_btn.clicked.connect(self.browse_spharm_dir)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.setStyleSheet(browse_btn.styleSheet())
        refresh_btn.clicked.connect(self.on_spharm_dir_changed)

        path_row.addWidget(path_lbl)
        path_row.addWidget(self.spharm_dir_input)
        path_row.addWidget(browse_btn)
        path_row.addWidget(refresh_btn)
        h_layout.addLayout(path_row)

        # Model architecture badge
        model_badge = QLabel("Architecture: <b>ResNet1D (1D-CNN) + PLS-DA</b> | Weights: <b>Trained & Frozen</b>")
        model_badge.setWordWrap(True)
        model_badge.setStyleSheet("font-size: 10px; color: #576574; background: #ffffff; padding: 4px 6px; border-radius: 3px; border: 1px solid #e9ecef;")
        h_layout.addWidget(model_badge)

        container_layout.addWidget(header_group)

        # ---------------------------------------------------------------------
        # 2. Execution & Output Directory (Dedicated output_Result)
        # ---------------------------------------------------------------------
        exec_group = QGroupBox("2. Batch Evaluation & Output Configuration")
        exec_group.setStyleSheet(header_group.styleSheet())
        e_layout = QVBoxLayout(exec_group)
        e_layout.setSpacing(6)

        # Output directory config
        out_row = QHBoxLayout()
        out_lbl = QLabel("Output Dir:")
        out_lbl.setFixedWidth(68)
        out_lbl.setStyleSheet("font-size: 11px; color: #2c3e50; font-weight: bold;")
        self.result_dir_input = QLineEdit()
        self.result_dir_input.setPlaceholderText("Auto (.../output_Result)")
        self.result_dir_input.setStyleSheet(self.spharm_dir_input.styleSheet())

        browse_out_btn = QPushButton("📁 Browse")
        browse_out_btn.setFixedHeight(26)
        browse_out_btn.setStyleSheet(browse_btn.styleSheet())
        browse_out_btn.clicked.connect(self.browse_result_dir)

        reload_res_btn = QPushButton("📂 Load")
        reload_res_btn.setFixedHeight(26)
        reload_res_btn.setToolTip("Scan output_Result folder and load existing evaluations")
        reload_res_btn.setStyleSheet(browse_btn.styleSheet())
        reload_res_btn.clicked.connect(self.load_existing_results)

        out_row.addWidget(out_lbl)
        out_row.addWidget(self.result_dir_input)
        out_row.addWidget(browse_out_btn)
        out_row.addWidget(reload_res_btn)
        e_layout.addLayout(out_row)

        # Prediction target options (Hemispheres)
        eval_opts_layout = QHBoxLayout()
        eval_lbl = QLabel("Target:")
        eval_lbl.setStyleSheet("font-size: 11px; color: #2c3e50; font-weight: bold;")
        eval_opts_layout.addWidget(eval_lbl)
        self.side_eval_group = QButtonGroup(self)
        self.rb_both = QRadioButton("Both")
        self.rb_left = QRadioButton("Left (LH)")
        self.rb_right = QRadioButton("Right (RH)")
        self.rb_both.setChecked(True)
        self.side_eval_group.addButton(self.rb_both, 0)
        self.side_eval_group.addButton(self.rb_left, 1)
        self.side_eval_group.addButton(self.rb_right, 2)
        eval_opts_layout.addWidget(self.rb_both)
        eval_opts_layout.addWidget(self.rb_left)
        eval_opts_layout.addWidget(self.rb_right)
        eval_opts_layout.addStretch()
        e_layout.addLayout(eval_opts_layout)

        # Run Batch Prediction Action Button
        self.predict_btn = QPushButton("⚡ Run ResNet Batch Prediction")
        self.predict_btn.setFixedHeight(32)
        self.predict_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a99d8, stop:1 #2471a3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2471a3, stop:1 #1b4f72);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #ecf0f1;
            }
        """)
        self.predict_btn.clicked.connect(self.run_batch_prediction)
        e_layout.addWidget(self.predict_btn)

        # Progress bar & Status hint
        self.batch_prog_bar = QProgressBar()
        self.batch_prog_bar.setRange(0, 100)
        self.batch_prog_bar.setValue(0)
        self.batch_prog_bar.setFixedHeight(14)
        self.batch_prog_bar.setTextVisible(True)
        self.batch_prog_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ced6e0;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
                font-size: 9px;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 2px;
            }
        """)
        self.batch_prog_bar.setVisible(False)
        e_layout.addWidget(self.batch_prog_bar)

        self.batch_status_hint = QLabel("")
        self.batch_status_hint.setWordWrap(True)
        self.batch_status_hint.setStyleSheet("font-size: 11px; padding: 2px 4px; border-radius: 3px;")
        e_layout.addWidget(self.batch_status_hint)

        container_layout.addWidget(exec_group)

        # ---------------------------------------------------------------------
        # 3. Evaluation Results & Meshes Table (With All / Left / Right Tabs)
        # ---------------------------------------------------------------------
        results_group = QGroupBox("3. Diagnostic Results & Evaluated Meshes")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        r_layout = QVBoxLayout(results_group)
        r_layout.setSpacing(6)

        # Tabs (All / Left / Right)
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All")
        self.tab_bar.addTab("Left (LH)")
        self.tab_bar.addTab("Right (RH)")
        self.tab_bar.setExpanding(False)
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
        r_layout.addWidget(self.tab_bar)

        # Results Table with 5 Columns: Subject, Side, Diagnosis, Prob, Mesh (.vtk)
        self.results_table = ToggleTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels([
            "Subject", "Side", "Diagnosis", "Prob", "Mesh (.vtk)"
        ])
        h_header = self.results_table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.results_table.setColumnWidth(0, 80)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(1, 38)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(2, 65)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(3, 48)
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        h_header.setStretchLastSection(True)
        self.results_table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_table.horizontalScrollBar().setEnabled(False)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #ecf0f1;
                font-size: 11px;
                background-color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 4px 2px;
                font-weight: bold;
                border: 1px solid #dcdde1;
                font-size: 11px;
            }
        """)
        self.results_table.setFixedHeight(145)
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        r_layout.addWidget(self.results_table)

        container_layout.addWidget(results_group)

        # ---------------------------------------------------------------------
        # 4. 3D Grad-CAM & Deformation Heatmap Controls
        # ---------------------------------------------------------------------
        cam_group = QGroupBox("4. 3D Grad-CAM & Atrophy Visualization")
        cam_group.setStyleSheet(header_group.styleSheet())
        c_layout = QVBoxLayout(cam_group)
        c_layout.setSpacing(6)

        # Active view hemisphere
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Side:"))
        self.rb_cam_left = QRadioButton("Left (LH)")
        self.rb_cam_right = QRadioButton("Right (RH)")
        self.rb_cam_left.setChecked(True)
        self.cam_side_group = QButtonGroup(self)
        self.cam_side_group.addButton(self.rb_cam_left, 0)
        self.cam_side_group.addButton(self.rb_cam_right, 1)
        self.cam_side_group.buttonClicked.connect(self.update_3d_view)
        side_row.addWidget(self.rb_cam_left)
        side_row.addWidget(self.rb_cam_right)
        side_row.addStretch()
        c_layout.addLayout(side_row)

        # Visualization mode options
        c_layout.addWidget(QLabel("Colormap Heatmap Mode:"))
        self.rb_dist = QRadioButton("🔵 Deformation Mag (mm)")
        self.rb_signed = QRadioButton("⚪ Signed Atrophy (Inward/Expansion)")
        self.rb_gradcam = QRadioButton("🔴 ResNet Grad-CAM Attention")
        self.rb_dist.setChecked(True)

        self.cam_mode_group = QButtonGroup(self)
        self.cam_mode_group.addButton(self.rb_dist, 0)
        self.cam_mode_group.addButton(self.rb_signed, 1)
        self.cam_mode_group.addButton(self.rb_gradcam, 2)
        self.cam_mode_group.buttonClicked.connect(self.update_3d_view)

        c_layout.addWidget(self.rb_dist)
        c_layout.addWidget(self.rb_signed)
        c_layout.addWidget(self.rb_gradcam)

        # Latent SD Trajectory & Atrophy Stepper Controls
        sd_box = QFrame()
        sd_box.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        sd_box_layout = QVBoxLayout(sd_box)
        sd_box_layout.setContentsMargins(4, 4, 4, 4)
        sd_box_layout.setSpacing(4)

        # Header Row with prominent SD indicator badge and quick Reset button
        sd_hdr_row = QHBoxLayout()
        sd_title = QLabel("SD Trajectory:")
        sd_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #2c3e50;")
        
        self.sd_reset_btn = QPushButton("🔄 Reset")
        self.sd_reset_btn.setFixedHeight(22)
        self.sd_reset_btn.setToolTip("Reset latent SD trajectory to 0.0 SD (Mean) [Hotkeys: 0, R, Space, Home]")
        self.sd_reset_btn.setStyleSheet("""
            QPushButton {
                background: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 6px;
            }
            QPushButton:hover {
                background: #ebf5fb;
                border-color: #2980b9;
                color: #2980b9;
            }
        """)
        self.sd_reset_btn.clicked.connect(lambda: self.set_sd_value(0.0))

        self.sd_val_badge = QLabel("0.0 SD (Mean)")
        self.sd_val_badge.setStyleSheet("""
            QLabel {
                background: #ebf5fb;
                color: #2980b9;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 3px;
                border: 1px solid #aed6f1;
            }
        """)
        sd_hdr_row.addWidget(sd_title)
        sd_hdr_row.addStretch()
        sd_hdr_row.addWidget(self.sd_reset_btn)
        sd_hdr_row.addWidget(self.sd_val_badge)
        sd_box_layout.addLayout(sd_hdr_row)

        # Stepper buttons + Continuous Slider Row
        slider_row = QHBoxLayout()
        self.sd_prev_btn = QPushButton("◀ -0.1 SD")
        self.sd_prev_btn.setFixedHeight(24)
        self.sd_prev_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 5px;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        self.sd_prev_btn.setToolTip("Step latent deformation backward by -0.1 SD (or press Left Arrow key)")
        self.sd_prev_btn.clicked.connect(lambda: self.step_sd(-0.1))

        self.sd_slider = QSlider(Qt.Orientation.Horizontal)
        self.sd_slider.setRange(0, 60)
        self.sd_slider.setValue(30)  # 30 = 0.0 SD (Mean)
        self.sd_slider.setTickInterval(10)
        self.sd_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sd_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sd_slider.valueChanged.connect(self.on_sd_slider_changed)

        self.sd_next_btn = QPushButton("+0.1 SD ▶")
        self.sd_next_btn.setFixedHeight(24)
        self.sd_next_btn.setStyleSheet(self.sd_prev_btn.styleSheet())
        self.sd_next_btn.setToolTip("Step latent deformation forward by +0.1 SD (or press Right Arrow key)")
        self.sd_next_btn.clicked.connect(lambda: self.step_sd(+0.1))

        slider_row.addWidget(self.sd_prev_btn)
        slider_row.addWidget(self.sd_slider)
        slider_row.addWidget(self.sd_next_btn)
        sd_box_layout.addLayout(slider_row)

        # Milestone quick buttons row
        milestone_row = QHBoxLayout()
        milestone_row.setSpacing(2)
        m_btn_style = """
            QPushButton {
                background: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 3px;
                font-size: 9px;
                font-weight: bold;
                padding: 2px 1px;
            }
            QPushButton:hover { background: #e2e8f0; }
        """
        btn_m3 = QPushButton("-3 SD")
        btn_m3.setToolTip("Severe Atrophy (-3.0 SD)")
        btn_m2 = QPushButton("-2 SD")
        btn_m2.setToolTip("Atrophy (-2.0 SD)")
        btn_m1 = QPushButton("-1 SD")
        btn_m1.setToolTip("Mild Atrophy (-1.0 SD)")
        btn_mean = QPushButton("Mean")
        btn_mean.setToolTip("Normative Mean (0.0 SD)")
        btn_p1 = QPushButton("+1 SD")
        btn_p1.setToolTip("Expansion (+1.0 SD)")
        btn_p2 = QPushButton("+2 SD")
        btn_p2.setToolTip("Expansion (+2.0 SD)")
        btn_p3 = QPushButton("+3 SD")
        btn_p3.setToolTip("Expansion (+3.0 SD)")

        for b in (btn_m3, btn_m2, btn_m1, btn_mean, btn_p1, btn_p2, btn_p3):
            b.setFixedHeight(22)
            b.setStyleSheet(m_btn_style)

        btn_m3.clicked.connect(lambda: self.set_sd_value(-3.0))
        btn_m2.clicked.connect(lambda: self.set_sd_value(-2.0))
        btn_m1.clicked.connect(lambda: self.set_sd_value(-1.0))
        btn_mean.clicked.connect(lambda: self.set_sd_value(0.0))
        btn_p1.clicked.connect(lambda: self.set_sd_value(1.0))
        btn_p2.clicked.connect(lambda: self.set_sd_value(2.0))
        btn_p3.clicked.connect(lambda: self.set_sd_value(3.0))

        milestone_row.addWidget(btn_m3)
        milestone_row.addWidget(btn_m2)
        milestone_row.addWidget(btn_m1)
        milestone_row.addWidget(btn_mean)
        milestone_row.addWidget(btn_p1)
        milestone_row.addWidget(btn_p2)
        milestone_row.addWidget(btn_p3)
        sd_box_layout.addLayout(milestone_row)

        # Keyboard helper hint
        hint_lbl = QLabel("⌨️ <i>Hotkeys: <b>← / →</b> Step SD  |  <b>0 / R / Space</b> Reset</i>")
        hint_lbl.setStyleSheet("font-size: 9px; color: #576574; font-style: italic;")
        sd_box_layout.addWidget(hint_lbl)

        c_layout.addWidget(sd_box)

        container_layout.addWidget(cam_group)
        container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # =========================================================================
    # Directory & Path Management
    # =========================================================================
    def get_default_output_dir(self, resolved_base=None):
        if self.get_output_folder:
            out_base = self.get_output_folder().strip()
            if out_base:
                return os.path.join(out_base, "output_Result")
        if resolved_base:
            return os.path.join(resolved_base, "output_Result")
        return ""

    def browse_spharm_dir(self):
        start_dir = self.get_output_folder() if self.get_output_folder else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Results Directory", start_dir)
        if folder:
            self.spharm_dir_input.setText(folder)
            self.on_spharm_dir_changed()

    def browse_result_dir(self):
        start_dir = self.result_dir_input.text().strip() or (self.get_output_folder() if self.get_output_folder else os.getcwd())
        folder = QFileDialog.getExistingDirectory(self, "Select Result Output Directory (output_Result)", start_dir)
        if folder:
            self.result_dir_input.setText(folder)
            self.load_existing_results()

    def on_spharm_dir_changed(self):
        subjs = self.discover_spharm_subjects()
        count = len(subjs)
        if count > 0:
            self.batch_status_hint.setText(f"Found {count} subject(s) in SPHARM directory ready for evaluation.")
            self.batch_status_hint.setStyleSheet("color: #27ae60; font-size: 11px;")
        else:
            self.batch_status_hint.setText("No SPHARM .coef / .vtk files found in current directory.")
            self.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")

    def on_panel_activated(self):
        """Called whenever user switches to Result Panel tab."""
        if not self.spharm_dir_input.text().strip():
            if self.get_output_folder:
                out_dir = self.get_output_folder().strip()
                if out_dir and os.path.isdir(out_dir):
                    spharm_cand = os.path.join(out_dir, "output_SPHARM")
                    if os.path.isdir(spharm_cand):
                        self.spharm_dir_input.setText(spharm_cand)
                    else:
                        self.spharm_dir_input.setText(out_dir)

        if not self.result_dir_input.text().strip():
            def_res = self.get_default_output_dir()
            if def_res:
                self.result_dir_input.setText(def_res)

        if self.spharm_dir_input.text().strip():
            self.on_spharm_dir_changed()
        if not self.all_evaluation_results and self.result_dir_input.text().strip():
            self.load_existing_results()

    # =========================================================================
    # SPHARM File Discovery & Deduplication
    # =========================================================================
    def discover_spharm_subjects(self):
        spharm_dir = self.spharm_dir_input.text().strip()
        search_dirs = []
        if spharm_dir and os.path.isdir(spharm_dir):
            search_dirs.append(spharm_dir)
            search_dirs.append(os.path.join(spharm_dir, "left"))
            search_dirs.append(os.path.join(spharm_dir, "right"))
            search_dirs.append(os.path.join(spharm_dir, "left", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "right", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "spharm_results"))

        # Fallback to output directory if provided
        if self.get_output_folder:
            out_d = self.get_output_folder().strip()
            if out_d and os.path.isdir(out_d) and out_d not in search_dirs:
                search_dirs.extend([
                    out_d,
                    os.path.join(out_d, "output_SPHARM"),
                    os.path.join(out_d, "output_left_hippocampus", "spharm_results_left"),
                    os.path.join(out_d, "output_right_hippocampus", "spharm_results_right")
                ])

        coef_map = {}
        vtk_map = {}

        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for coef_file in glob.glob(os.path.join(d, "**", "*_SPHARM.coef"), recursive=True):
                fname = os.path.basename(coef_file)
                fname_lower = fname.lower()
                if any(aux in fname_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_")):
                    continue
                side = "left" if (fname.startswith("lh_") or "left" in fname_lower) else ("right" if (fname.startswith("rh_") or "right" in fname_lower) else "unknown")
                if side not in ("left", "right"):
                    continue
                subj = self._extract_subject_id(fname)
                if not subj:
                    continue
                coef_map.setdefault(subj, {}).setdefault(side, []).append(os.path.normpath(coef_file))

            for vtk_file in glob.glob(os.path.join(d, "**", "*.vtk"), recursive=True):
                fname = os.path.basename(vtk_file)
                fname_lower = fname.lower()
                if any(aux in fname_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_", "mean_shape")):
                    continue
                side = "left" if (fname.startswith("lh_") or "left" in fname_lower) else ("right" if (fname.startswith("rh_") or "right" in fname_lower) else "unknown")
                if side not in ("left", "right"):
                    continue
                subj = self._extract_subject_id(fname)
                if not subj:
                    continue
                vtk_map.setdefault(subj, {}).setdefault(side, []).append(os.path.normpath(vtk_file))

        all_subj_keys = sorted(set(list(coef_map.keys()) + list(vtk_map.keys())))
        subjects = {}

        def pick_best_coef(c_list):
            if not c_list:
                return None
            for c in c_list:
                if c.endswith("_SPHARM.coef"):
                    return c
            return None

        def pick_best_vtk(v_list):
            if not v_list:
                return None
            for suf in ("_SPHARM_realigned.vtk", "_realigned.vtk", "_SPHARM_procalign.vtk", "_procalign.vtk", "_SPHARM_ellalign.vtk", "_ellalign.vtk", "_SPHARM.vtk"):
                cand = [v for v in v_list if v.endswith(suf)]
                if cand:
                    return cand[0]
            return v_list[0]

        for subj in all_subj_keys:
            lh_coef = pick_best_coef(coef_map.get(subj, {}).get("left", []))
            rh_coef = pick_best_coef(coef_map.get(subj, {}).get("right", []))
            lh_vtk = pick_best_vtk(vtk_map.get(subj, {}).get("left", []))
            rh_vtk = pick_best_vtk(vtk_map.get(subj, {}).get("right", []))

            # Only include subjects that have at least one valid SPHARM coef file
            if not lh_coef and not rh_coef:
                continue

            subjects[subj] = {
                "left_coef": lh_coef,
                "right_coef": rh_coef,
                "left_vtk": lh_vtk,
                "right_vtk": rh_vtk
            }

        return subjects

    def _extract_subject_id(self, filename: str) -> str:
        clean = os.path.splitext(filename)[0]
        # Strip side prefixes
        for p in ("lh_", "rh_", "left_", "right_"):
            if clean.lower().startswith(p):
                clean = clean[len(p):]
                break
        # Strip intermediate pipeline and SPHARM suffixes
        for pat in (
            "_SPHARM_realigned", "_SPHARM_procalign", "_SPHARM_ellalign",
            "_SPHARM_grid", "_SPHARMMedialAxis", "_MedialAxisScalars",
            "_MedialAxis", "_SPHARM", "_realigned", "_procalign",
            "_ellalign", "_surf", "_para", "_aligned"
        ):
            clean = clean.replace(pat, "")
        if clean.endswith("_lh") or clean.endswith("_rh"):
            clean = clean[:-3]
        clean = clean.replace("_hippocampus", "")
        return clean.strip("_")

    # =========================================================================
    # ResNet Batch Prediction Execution (All Meshes)
    # =========================================================================
    def run_batch_prediction(self):
        subjects = self.discover_spharm_subjects()
        if not subjects:
            self.signal_log_message.emit("[WARNING] No valid SPHARM .coef or .vtk files found in the specified directory.")
            self.batch_status_hint.setText("❌ No valid SPHARM files found in specified directory.")
            self.batch_status_hint.setStyleSheet("color: #c0392b; font-size: 11px;")
            self.signal_batch_prediction_finished.emit(False)
            return

        # Prepare output directory
        out_dir = self.result_dir_input.text().strip() or self.get_default_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        meshes_dir = os.path.join(out_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        eval_target = self.side_eval_group.checkedId()  # 0: both, 1: left, 2: right
        total_subjs = len(subjects)
        self.signal_log_message.emit(f">>> Running ResNet batch prediction for {total_subjs} subjects...")

        self.predict_btn.setEnabled(False)
        self.batch_prog_bar.setVisible(True)
        self.batch_prog_bar.setValue(0)
        self.batch_status_hint.setText(f"Evaluating {total_subjs} subjects across hippocampal meshes...")
        self.batch_status_hint.setStyleSheet("color: #2980b9; font-size: 11px; font-weight: bold;")

        evaluated_results = []
        csv_rows = []
        failed_predictions = 0
        save_failed = False

        for idx, (subj_name, data) in enumerate(sorted(subjects.items())):
            lh_coef = data.get("left_coef")
            rh_coef = data.get("right_coef")
            lh_vtk = data.get("left_vtk")
            rh_vtk = data.get("right_vtk")

            left_res = None
            right_res = None

            # Predict Left
            if eval_target in (0, 1) and lh_coef and os.path.isfile(lh_coef):
                try:
                    left_res = self.predictor.predict(lh_coef, side='left')
                except Exception as e:
                    self.signal_log_message.emit(f"[WARNING] Left predict failed for {subj_name}: {e}")
                    failed_predictions += 1

            # Predict Right
            if eval_target in (0, 2) and rh_coef and os.path.isfile(rh_coef):
                try:
                    right_res = self.predictor.predict(rh_coef, side='right')
                except Exception as e:
                    self.signal_log_message.emit(f"[WARNING] Right predict failed for {subj_name}: {e}")
                    failed_predictions += 1

            # Synthesize combined prediction
            if left_res and right_res:
                avg_prob = (left_res['probability'] + right_res['probability']) / 2.0
                is_epilepsy = avg_prob > 0.5
                higher_side = 'Left' if left_res['probability'] >= right_res['probability'] else 'Right'
                summary = {
                    'probability': avg_prob,
                    'is_epilepsy': is_epilepsy,
                    'label': "Temporal Lobe Epilepsy (TLE)" if is_epilepsy else "Healthy Control (HC)",
                    'primary_side': higher_side,
                    'left_prob': left_res['probability'],
                    'right_prob': right_res['probability']
                }
            elif left_res:
                prob = left_res['probability']
                summary = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Left",
                    'left_prob': prob,
                    'right_prob': None
                }
            elif right_res:
                prob = right_res['probability']
                summary = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Right",
                    'left_prob': None,
                    'right_prob': prob
                }
            else:
                # If neither side had an evaluated SPHARM output, skip completely
                continue

            prob = summary['probability']
            risk_level = "High Risk (Epilepsy-aligned)" if prob > 0.75 else ("Moderate Risk" if summary['is_epilepsy'] else "Low Risk (Healthy-aligned)")
            summary['risk_level'] = risk_level

            # Generate Grad-CAM colored meshes ONLY if the actual SPHARM coef exists and was evaluated
            out_lh_vtk = None
            out_rh_vtk = None

            if left_res and lh_coef and os.path.isfile(lh_coef):
                lh_milestone = "minus3SD" if left_res.get('probability', 0) > 0.75 else ("minus2SD" if left_res.get('probability', 0) > 0.5 else "Mean")
                src_lh_gradcam = self.predictor.get_gradcam_mesh_path(side="left", component="PLS1", milestone=lh_milestone, cohort="All_Augment_tain")
                if src_lh_gradcam and os.path.isfile(src_lh_gradcam):
                    out_lh_vtk = os.path.join(meshes_dir, f"{subj_name}_lh_gradcam.vtk")
                    try:
                        if not os.path.exists(out_lh_vtk) or os.path.getmtime(src_lh_gradcam) > os.path.getmtime(out_lh_vtk):
                            shutil.copy2(src_lh_gradcam, out_lh_vtk)
                    except Exception:
                        out_lh_vtk = src_lh_gradcam

            if right_res and rh_coef and os.path.isfile(rh_coef):
                rh_milestone = "minus3SD" if right_res.get('probability', 0) > 0.75 else ("minus2SD" if right_res.get('probability', 0) > 0.5 else "Mean")
                src_rh_gradcam = self.predictor.get_gradcam_mesh_path(side="right", component="PLS1", milestone=rh_milestone, cohort="All_Augment_tain")
                if src_rh_gradcam and os.path.isfile(src_rh_gradcam):
                    out_rh_vtk = os.path.join(meshes_dir, f"{subj_name}_rh_gradcam.vtk")
                    try:
                        if not os.path.exists(out_rh_vtk) or os.path.getmtime(src_rh_gradcam) > os.path.getmtime(out_rh_vtk):
                            shutil.copy2(src_rh_gradcam, out_rh_vtk)
                    except Exception:
                        out_rh_vtk = src_rh_gradcam

            item_record = {
                'subject_name': subj_name,
                'left_coef': lh_coef if left_res else None,
                'right_coef': rh_coef if right_res else None,
                'left_vtk': out_lh_vtk if left_res else None,
                'right_vtk': out_rh_vtk if right_res else None,
                'left_gradcam_mesh': out_lh_vtk if left_res else None,
                'right_gradcam_mesh': out_rh_vtk if right_res else None,
                'patient_left_mesh': lh_vtk if left_res else None,
                'patient_right_mesh': rh_vtk if right_res else None,
                'left_result': left_res,
                'right_result': right_res,
                'summary': summary,
                'output_left_mesh': out_lh_vtk if left_res else None,
                'output_right_mesh': out_rh_vtk if right_res else None
            }
            evaluated_results.append(item_record)

            csv_rows.append({
                'Subject': subj_name,
                'Diagnosis': summary['label'],
                'Epilepsy_Probability_Percent': f"{prob * 100:.2f}",
                'Left_Risk_Percent': f"{left_res['probability'] * 100:.2f}" if left_res else "N/A",
                'Right_Risk_Percent': f"{right_res['probability'] * 100:.2f}" if right_res else "N/A",
                'Suspected_Focus': summary['primary_side'],
                'Risk_Level': risk_level,
                'Left_Mesh': os.path.basename(out_lh_vtk) if out_lh_vtk else "None",
                'Right_Mesh': os.path.basename(out_rh_vtk) if out_rh_vtk else "None"
            })

            prog_pct = int(((idx + 1) / total_subjs) * 100)
            self.batch_prog_bar.setValue(prog_pct)
            QApplication.processEvents()

        # Save predictions_summary.csv in output_Result
        csv_file = os.path.join(out_dir, "predictions_summary.csv")
        try:
            fieldnames = [
                'Subject', 'Diagnosis', 'Epilepsy_Probability_Percent',
                'Left_Risk_Percent', 'Right_Risk_Percent', 'Suspected_Focus',
                'Risk_Level', 'Left_Mesh', 'Right_Mesh'
            ]
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            self.signal_log_message.emit(f"Saved CSV summary: {csv_file}")
        except Exception as e:
            self.signal_log_message.emit(f"[WARNING] Failed to save CSV summary: {e}")
            save_failed = True

        # Save evaluation_summary.json in output_Result
        json_file = os.path.join(out_dir, "evaluation_summary.json")
        try:
            json_data = []
            for r in evaluated_results:
                json_data.append({
                    'subject': r['subject_name'],
                    'summary': r['summary'],
                    'left_mesh': r['output_left_mesh'],
                    'right_mesh': r['output_right_mesh']
                })
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            self.signal_log_message.emit(f"Saved JSON report: {json_file}")
        except Exception as e:
            self.signal_log_message.emit(f"[WARNING] Failed to save JSON report: {e}")
            save_failed = True

        self.all_evaluation_results = evaluated_results
        self.populate_results_table()
        self.predict_btn.setEnabled(True)
        success = bool(evaluated_results) and not failed_predictions and not save_failed and len(evaluated_results) == total_subjs
        state = 'Completed' if success else 'Incomplete'
        self.batch_status_hint.setText(f"{state}: {len(evaluated_results)}/{total_subjs} subjects; {failed_predictions} side predictions failed.")
        self.batch_status_hint.setStyleSheet(f"color: {'#27ae60' if success else '#c0392b'}; font-size: 11px; font-weight: bold;")
        self.signal_log_message.emit(f"{state}: {len(evaluated_results)}/{total_subjs} evaluated. Output folder: {out_dir}")
        self.signal_batch_prediction_finished.emit(success)

    # =========================================================================
    # Results Table Loading & Tab Filtering
    # =========================================================================
    def load_existing_results(self):
        out_dir = self.result_dir_input.text().strip() or self.get_default_output_dir()
        if not os.path.isdir(out_dir):
            self.clear_view()
            self.batch_status_hint.setText(f"Folder not found: {os.path.basename(out_dir) if out_dir else 'None'}")
            self.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
            self.signal_log_message.emit(f"[INFO] Output folder '{out_dir}' does not exist. Cleared view.")
            return

        json_file = os.path.join(out_dir, "evaluation_summary.json")

        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                recovered_map = {}
                for item in data:
                    s_raw = item.get('subject', '')
                    s_clean = self._extract_subject_id(s_raw)
                    if not s_clean:
                        continue
                    summary = item.get('summary', {})
                    lh_mesh = item.get('left_mesh')
                    rh_mesh = item.get('right_mesh')

                    # Merge or prefer records with valid probability and non-empty meshes
                    if s_clean not in recovered_map or (summary.get('probability', 0.0) > 0 and not recovered_map[s_clean]['summary'].get('probability')):
                        recovered_map[s_clean] = {
                            'subject_name': s_clean,
                            'left_coef': None,
                            'right_coef': None,
                            'left_vtk': lh_mesh,
                            'right_vtk': rh_mesh,
                            'left_result': {'probability': summary.get('left_prob')} if summary.get('left_prob') is not None else None,
                            'right_result': {'probability': summary.get('right_prob')} if summary.get('right_prob') is not None else None,
                            'summary': summary,
                            'output_left_mesh': lh_mesh,
                            'output_right_mesh': rh_mesh
                        }
                    else:
                        existing = recovered_map[s_clean]
                        if not existing['left_vtk'] and lh_mesh:
                            existing['left_vtk'] = lh_mesh
                        if not existing['right_vtk'] and rh_mesh:
                            existing['right_vtk'] = rh_mesh

                recovered = list(recovered_map.values())
                if recovered:
                    self.all_evaluation_results = recovered
                    self.populate_results_table()
                    self.batch_status_hint.setText(f"Loaded {len(recovered)} evaluation results from {os.path.basename(out_dir)}")
                    self.batch_status_hint.setStyleSheet("color: #27ae60; font-size: 11px;")
                    return
            except Exception as e:
                self.signal_log_message.emit(f"[WARNING] Could not parse existing JSON report: {e}")

        # If folder is empty or JSON summary not found
        self.clear_view()
        self.batch_status_hint.setText("No evaluation results found in output directory.")
        self.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
        self.signal_log_message.emit(f"[INFO] No evaluation results found in '{out_dir}'. Cleared view.")

    def on_tab_changed(self, index):
        self.populate_results_table()

    def populate_results_table(self):
        tab_idx = self.tab_bar.currentIndex()  # 0: All, 1: Left (LH), 2: Right (RH)

        # Count total available meshes across all subjects with valid evaluated results
        total_lh = 0
        total_rh = 0
        for r in self.all_evaluation_results:
            if r.get('left_result') and (r.get('left_vtk') or r.get('left_gradcam_mesh')):
                total_lh += 1
            if r.get('right_result') and (r.get('right_vtk') or r.get('right_gradcam_mesh')):
                total_rh += 1

        self.tab_bar.blockSignals(True)
        self.tab_bar.setTabText(0, f"All ({total_lh + total_rh})")
        self.tab_bar.setTabText(1, f"Left ({total_lh})")
        self.tab_bar.setTabText(2, f"Right ({total_rh})")
        self.tab_bar.blockSignals(False)

        filtered = []
        for r in self.all_evaluation_results:
            has_left = bool(r.get('left_result') and (r.get('left_vtk') or r.get('left_gradcam_mesh')))
            has_right = bool(r.get('right_result') and (r.get('right_vtk') or r.get('right_gradcam_mesh')))
            if tab_idx == 1:
                if has_left:
                    filtered.append((r, "left"))
            elif tab_idx == 2:
                if has_right:
                    filtered.append((r, "right"))
            else:
                # Tab 0: Include both LH and RH meshes
                if has_left:
                    filtered.append((r, "left"))
                if has_right:
                    filtered.append((r, "right"))

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(0)
        self.results_table.setRowCount(len(filtered))

        for row, (record, view_side) in enumerate(filtered):
            subj_name = record['subject_name']
            summary = record.get('summary', {})
            prob = summary.get('probability', 0.0)

            if view_side == "left":
                lh_res = record.get('left_result')
                side_prob = lh_res['probability'] if (lh_res and lh_res.get('probability') is not None) else prob
                diag_str = "🚨 TLE" if side_prob > 0.5 else "✅ HC"
                prob_str = f"{side_prob * 100:.1f}%"
                side_str = "LH"
                side_full = "Left (LH)"
                mesh_path = record.get('left_vtk')
            else:
                rh_res = record.get('right_result')
                side_prob = rh_res['probability'] if (rh_res and rh_res.get('probability') is not None) else prob
                diag_str = "🚨 TLE" if side_prob > 0.5 else "✅ HC"
                prob_str = f"{side_prob * 100:.1f}%"
                side_str = "RH"
                side_full = "Right (RH)"
                mesh_path = record.get('right_vtk')

            mesh_str = os.path.basename(mesh_path) if mesh_path else "—"

            item_name = QTableWidgetItem(subj_name)
            item_name.setData(Qt.ItemDataRole.UserRole, (record, view_side))
            item_name.setToolTip(f"Subject: {subj_name}\nSide: {side_full}")

            item_side = QTableWidgetItem(side_str)
            item_side.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_side.setToolTip(side_full)

            item_diag = QTableWidgetItem(diag_str)
            item_diag.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_prob = QTableWidgetItem(prob_str)
            item_prob.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_mesh = QTableWidgetItem(mesh_str)
            item_mesh.setToolTip(mesh_path if mesh_path else "")

            if "TLE" in diag_str:
                item_diag.setForeground(QColor("#c0392b"))
                item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            else:
                item_diag.setForeground(QColor("#27ae60"))
                item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            self.results_table.setItem(row, 0, item_name)
            self.results_table.setItem(row, 1, item_side)
            self.results_table.setItem(row, 2, item_diag)
            self.results_table.setItem(row, 3, item_prob)
            self.results_table.setItem(row, 4, item_mesh)

        self.results_table.blockSignals(False)

        if self.results_table.rowCount() > 0:
            self.results_table.selectRow(0)

    def on_result_selected(self):
        selected = self.results_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        item = self.results_table.item(row, 0)
        if not item:
            return

        user_data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(user_data, tuple):
            record, view_side = user_data
        elif isinstance(user_data, dict):
            record, view_side = user_data, "left"
        else:
            return

        self.current_patient_data = record
        self.last_prediction_results = {
            'left': record.get('left_result'),
            'right': record.get('right_result'),
            'summary': record.get('summary')
        }

        self.display_prediction_results(self.last_prediction_results, selected_side=view_side)

        # Always initialize latent SD trajectory to 0.0 SD (Mean) matching view_gradcam_plsda_top3
        self.set_sd_value(0.0)
        self.signal_log_message.emit(f"Inspecting evaluation results for: {record['subject_name']} [{view_side.upper()}]")

    # =========================================================================
    def display_prediction_results(self, results, selected_side=None):
        summary = results.get('summary')
        if not summary:
            return

        lh_prob = summary.get('left_prob')
        rh_prob = summary.get('right_prob')

        # Prioritize selected_side if clicked from table; otherwise use higher risk side
        if selected_side in ("left", "right"):
            if selected_side == "right":
                self.rb_cam_right.setChecked(True)
            else:
                self.rb_cam_left.setChecked(True)
        elif lh_prob is not None and rh_prob is not None:
            if rh_prob > lh_prob:
                self.rb_cam_right.setChecked(True)
            else:
                self.rb_cam_left.setChecked(True)
        elif rh_prob is not None:
            self.rb_cam_right.setChecked(True)
        elif lh_prob is not None:
            self.rb_cam_left.setChecked(True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
            self.step_sd(-0.1)
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
            self.step_sd(+0.1)
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
            self.set_sd_value(0.0)
            event.accept()
            return
        super().keyPressEvent(event)

    def step_sd(self, delta: float):
        cur_val = self.current_sd_value
        new_val = max(-3.0, min(3.0, round(cur_val + delta, 1)))
        if abs(new_val - cur_val) > 0.01:
            self.set_sd_value(new_val)

    def set_sd_value(self, sd_val: float):
        sd_val = max(-3.0, min(3.0, round(sd_val, 1)))
        step_idx = int(round((sd_val + 3.0) * 10))
        self.sd_slider.blockSignals(True)
        self.sd_slider.setValue(step_idx)
        self.sd_slider.blockSignals(False)
        self.current_sd_step_idx = step_idx
        self.current_sd_value = sd_val
        self._update_sd_badge(sd_val, step_idx)
        self.update_3d_view()

    def on_sd_slider_changed(self, val: int):
        sd_val = round(-3.0 + val * 0.1, 1)
        self.current_sd_step_idx = val
        self.current_sd_value = sd_val
        self._update_sd_badge(sd_val, val)
        self.update_3d_view()

    def _update_sd_badge(self, sd_val: float, step_idx: int):
        if abs(sd_val) < 0.05:
            desc = "Mean"
            color = "#2980b9"
            bg = "#ebf5fb"
            border = "#aed6f1"
        elif sd_val <= -2.0:
            desc = "Severe Atrophy" if sd_val <= -2.5 else "Atrophy"
            color = "#c0392b"
            bg = "#fdf2f2"
            border = "#f5b7b1"
        elif sd_val < 0.0:
            desc = "Mild Atrophy"
            color = "#d35400"
            bg = "#fef5e7"
            border = "#fad7a0"
        else:
            desc = "Expansion"
            color = "#27ae60"
            bg = "#f2fbf6"
            border = "#abebc6"
        self.sd_val_badge.setText(f"{sd_val:+.1f} SD ({desc})")
        self.sd_val_badge.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {color};
                font-weight: bold;
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 3px;
                border: 1px solid {border};
            }}
        """)

    def find_step_mesh(self, side: str, sd_val: float, cohort: str = "All_Augment_tain", component: str = "PLS1"):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", cohort, side, component)
        steps_dir = os.path.join(base_dir, "steps")
        if os.path.isdir(steps_dir):
            files = glob.glob(os.path.join(steps_dir, "*.vtk"))
            best_f = None
            best_diff = float('inf')
            for f in files:
                m = re.search(r"val_([+-]?\d+\.?\d*)", os.path.basename(f))
                if m:
                    diff = abs(float(m.group(1)) - sd_val)
                    if diff < best_diff:
                        best_diff = diff
                        best_f = f
            if best_f and best_diff < 0.08:
                return best_f

        # Fallback to standard milestone VTK if steps folder is absent
        if sd_val <= -2.5:
            milestone = "minus3SD"
        elif sd_val <= -1.5:
            milestone = "minus2SD"
        elif sd_val <= -0.5:
            milestone = "minus1SD"
        elif sd_val < 0.5:
            milestone = "Mean"
        elif sd_val < 1.5:
            milestone = "plus1SD"
        elif sd_val < 2.5:
            milestone = "plus2SD"
        else:
            milestone = "plus3SD"

        milestone_path = os.path.join(base_dir, f"{component}_{milestone}.vtk")
        if os.path.isfile(milestone_path):
            return milestone_path
        return None

    # =========================================================================
    # 3D Grad-CAM & Heatmap Visualization
    # =========================================================================
    def update_3d_view(self):
        side = "left" if self.rb_cam_left.isChecked() else "right"

        # Determine Colormap and Scalar mode
        if self.rb_signed.isChecked():
            scalar_mode = "SignedDistance"
            lut_type = "signed_distance"
            mode_desc = "Signed Atrophy"
        elif self.rb_dist.isChecked():
            scalar_mode = "DistanceMapping"
            lut_type = "distance_mapping"
            mode_desc = "Deformation Mag (mm)"
        else:
            scalar_mode = "GradCAM_Importance"
            lut_type = "gradcam"
            mode_desc = "ResNet Grad-CAM"

        title = f"{side.upper()} {mode_desc} (SD = {self.current_sd_value:+.1f})"

        # Find the exact step mesh matching current SD value
        mesh_path = self.find_step_mesh(side, self.current_sd_value)

        if not mesh_path or not os.path.isfile(mesh_path):
            mesh_path = self.predictor.get_gradcam_mesh_path(
                side=side,
                component="PLS1",
                milestone="Mean",
                cohort="All_Augment_tain"
            )

        if not mesh_path or not os.path.isfile(mesh_path):
            self.signal_log_message.emit(f"[WARNING] 3D Grad-CAM mesh not found for {side}. Checking template...")
            mesh_path = self.predictor.get_template_mesh_path(side)

        if mesh_path and os.path.isfile(mesh_path):
            self.signal_gradcam_mesh_requested.emit(
                mesh_path, scalar_mode, lut_type, title, side, 1.0
            )

        # Handle Patient Mesh Overlay (Off by default)
        self.on_patient_overlay_toggled()

    def on_patient_overlay_toggled(self):
        side = "left" if self.rb_cam_left.isChecked() else "right"
        self.signal_patient_overlay_requested.emit("", False, 0.0, side)

    def clear_view(self):
        self.signal_clear_gradcam_requested.emit()
        self.all_evaluation_results = []
        self.current_patient_data = None
        self.last_prediction_results = None

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(0)
        self.results_table.blockSignals(False)

        self.tab_bar.blockSignals(True)
        self.tab_bar.setTabText(0, "All (0)")
        self.tab_bar.setTabText(1, "Left (0)")
        self.tab_bar.setTabText(2, "Right (0)")
        self.tab_bar.blockSignals(False)

        self.signal_log_message.emit("Diagnostic panel reset.")

    def copy_summary(self):
        if not self.last_prediction_results or not self.current_patient_data:
            return

        summary = self.last_prediction_results.get('summary', {})
        subj = self.current_patient_data.get('subject_name', 'Patient')
        diag = summary.get('label', 'N/A')
        prob = summary.get('probability', 0.0) * 100.0
        lh = summary.get('left_prob')
        rh = summary.get('right_prob')
        focus = summary.get('primary_side', 'N/A')

        text = (
            f"=== HIPPOCAMPAL SHAPE ANALYSIS - DIAGNOSTIC REPORT ===\n"
            f"Subject ID: {subj}\n"
            f"Model: ResNet1D Residual Neural Network + PLS-DA\n"
            f"Diagnosis: {diag}\n"
            f"Overall Epilepsy Probability: {prob:.2f}%\n"
            + (f"Left Hippocampus Risk: {lh*100:.2f}%\n" if lh is not None else "")
            + (f"Right Hippocampus Risk: {rh*100:.2f}%\n" if rh is not None else "")
            + f"Suspected Seizure Focus: {focus} Hemisphere\n"
            f"Methodology: SPHARM-PDM Point Distribution Parameterization\n"
            f"======================================================\n"
        )
        cb = QApplication.clipboard()
        if cb:
            cb.setText(text)
            self.signal_log_message.emit(f"Diagnostic summary copied to clipboard for {subj}.")
            QMessageBox.information(self, "Summary Copied", f"Diagnostic summary for '{subj}' has been copied to your clipboard.")
