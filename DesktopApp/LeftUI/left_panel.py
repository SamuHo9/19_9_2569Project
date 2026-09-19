import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedWidget
from PyQt6.QtCore import Qt, pyqtSignal

from .main_panel import MainPanel
from .import_panel import ImportPanel
from .fastsurfer_panel import FastsurferPanel
from .icp_panel import IcpPanel
from .spharm_panel import SpharmPanel
from .result_panel import ResultPanel

class AdaptiveStackedWidget(QStackedWidget):
    def sizeHint(self):
        cw = self.currentWidget()
        return cw.sizeHint() if cw else super().sizeHint()

    def minimumSizeHint(self):
        cw = self.currentWidget()
        return cw.minimumSizeHint() if cw else super().minimumSizeHint()

class LeftPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str) # filepath can be str or list of str
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_all_toggled = pyqtSignal(bool, list, str) # enabled, filepaths, side_filter
    signal_side_changed = pyqtSignal(str) # "all", "lh", "rh"
    signal_gradcam_mesh_requested = pyqtSignal(str, str, str, str, str, float) # mesh_path, scalar_mode, lut_type, title, side, opacity
    signal_patient_overlay_requested = pyqtSignal(str, bool, float, str) # mesh_path, visible, opacity, side
    signal_clear_gradcam_requested = pyqtSignal()
    signal_diagnostic_info = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        left_layout = QVBoxLayout(self)
        left_layout.setContentsMargins(12, 10, 12, 10)
        left_layout.setSpacing(10)

        # --- Logo / Header ---
        logo_label = QLabel("Shape Analysis Toolbox\nHippocampal Pipeline")
        font = logo_label.font()
        font.setPointSize(13)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #2c3e50; padding: 8px 10px; background-color: #ecf0f1; border-radius: 5px;")
        left_layout.addWidget(logo_label)

        # --- QStackedWidget (Module Switching) ---
        self.stacked_widget = AdaptiveStackedWidget()
        left_layout.addWidget(self.stacked_widget)

        # Create panels
        self.import_panel = ImportPanel()
        self.fastsurfer_panel = FastsurferPanel(self.import_panel.get_folder, self.import_panel.get_output_folder)
        self.icp_panel = IcpPanel(self.import_panel.get_folder, self.import_panel.get_output_folder)
        self.spharm_panel = SpharmPanel(self.import_panel.get_folder, self.import_panel.get_output_folder)
        self.result_panel = ResultPanel(get_input_folder=self.import_panel.get_folder, get_output_folder=self.import_panel.get_output_folder)
        self.main_panel = MainPanel(
            import_panel=self.import_panel,
            fastsurfer_panel=self.fastsurfer_panel,
            icp_panel=self.icp_panel,
            spharm_panel=self.spharm_panel,
            result_panel=self.result_panel
        )

        # Add to stacked widget (Main Panel first, Result Panel last)
        self.stacked_widget.addWidget(self.main_panel)
        self.stacked_widget.addWidget(self.import_panel)
        self.stacked_widget.addWidget(self.fastsurfer_panel)
        self.stacked_widget.addWidget(self.icp_panel)
        self.stacked_widget.addWidget(self.spharm_panel)
        self.stacked_widget.addWidget(self.result_panel)

        # Connect signals
        for panel in [self.main_panel, self.import_panel, self.fastsurfer_panel, self.icp_panel, self.spharm_panel, self.result_panel]:
            panel.signal_log_message.connect(self.signal_log_message)
            
        self.import_panel.signal_subject_selected.connect(self.signal_subject_selected)
        
        def on_directories_changed(in_dir, out_dir):
            self.main_panel.set_directories(in_dir, out_dir)
            self.fastsurfer_panel.update_run_button_state()
            self.icp_panel.update_run_button_state()
            self.spharm_panel.update_run_button_state()
            if hasattr(self.result_panel, 'on_spharm_dir_changed'):
                sph_d = os.path.join(out_dir, "output_SPHARM")
                res_d = os.path.join(out_dir, "output_Result")
                self.result_panel.spharm_dir_input.setText(sph_d)
                self.result_panel.result_dir_input.setText(res_d)
                self.result_panel.on_spharm_dir_changed()
            
        self.import_panel.signal_directories_changed.connect(on_directories_changed)

        # Pipeline sequence auto-progression:
        # FastSurfer completes -> ICP automatically ready to run using FastSurfer meshes
        self.fastsurfer_panel.signal_fastsurfer_completed.connect(self.icp_panel.update_run_button_state)
        # ICP completes -> SPHARM automatically ready to run using ICP aligned meshes
        self.icp_panel.signal_icp_completed.connect(self.spharm_panel.update_run_button_state)
        # SPHARM completes -> Result Panel ready to evaluate SPHARM output
        def on_spharm_finished():
            self.spharm_panel.update_run_button_state()
            out_d = self.import_panel.get_output_folder() if self.import_panel else ""
            if out_d:
                sph_d = os.path.join(out_d, "output_SPHARM")
                res_d = os.path.join(out_d, "output_Result")
                self.result_panel.spharm_dir_input.setText(sph_d)
                self.result_panel.result_dir_input.setText(res_d)
                self.result_panel.on_spharm_dir_changed()
        self.spharm_panel.signal_spharm_completed.connect(on_spharm_finished)

        # Forward mesh selection from Main Panel, FastSurfer, ICP, and SPHARM to right panel
        if hasattr(self.main_panel, 'signal_mesh_selected'):
            self.main_panel.signal_mesh_selected.connect(self.signal_mesh_selected)
        if hasattr(self.main_panel, 'signal_diagnostic_info'):
            self.main_panel.signal_diagnostic_info.connect(self.signal_diagnostic_info)
        self.fastsurfer_panel.signal_mesh_selected.connect(self.signal_mesh_selected)
        self.icp_panel.signal_mesh_selected.connect(self.signal_mesh_selected)
        self.spharm_panel.signal_mesh_selected.connect(self.signal_mesh_selected)

        # Forward reference template toggle signals from ICP and SPHARM panels
        self.icp_panel.signal_template_toggled.connect(self.signal_template_toggled)
        self.spharm_panel.signal_template_toggled.connect(self.signal_template_toggled)

        # Forward overlay all meshes toggle signals
        self.icp_panel.signal_overlay_all_toggled.connect(self.signal_overlay_all_toggled)
        self.spharm_panel.signal_overlay_all_toggled.connect(self.signal_overlay_all_toggled)

        # Forward side filter changed signals
        self.icp_panel.signal_side_changed.connect(self.signal_side_changed)
        self.spharm_panel.signal_side_changed.connect(self.signal_side_changed)

        # Forward Result Panel signals
        if hasattr(self.result_panel, 'signal_gradcam_mesh_requested'):
            self.result_panel.signal_gradcam_mesh_requested.connect(self.signal_gradcam_mesh_requested)
            self.result_panel.signal_patient_overlay_requested.connect(self.signal_patient_overlay_requested)
            self.result_panel.signal_clear_gradcam_requested.connect(self.signal_clear_gradcam_requested)
        if hasattr(self.result_panel, 'signal_mesh_selected'):
            self.result_panel.signal_mesh_selected.connect(self.signal_mesh_selected)

    def set_template_visible(self, visible: bool):
        self.icp_panel.set_template_visible(visible)
        self.spharm_panel.set_template_visible(visible)

    def set_overlay_visible(self, visible: bool):
        active_widget = self.stacked_widget.currentWidget()
        if hasattr(active_widget, 'set_overlay_visible'):
            active_widget.set_overlay_visible(visible)

    def get_current_module_panel(self):
        return self.stacked_widget.currentWidget()

    def switch_module(self, index):
        self.stacked_widget.setCurrentIndex(index)
        current = self.stacked_widget.currentWidget()
        if current == self.main_panel:
            self.main_panel.update_stage_preview()
            self.main_panel.populate_main_table()
        elif current == self.import_panel:
            in_dir = self.import_panel.get_folder().strip()
            if in_dir and os.path.isdir(in_dir):
                if getattr(self.import_panel, 'loaded_directory', None) != in_dir or self.import_panel.subjects_table.rowCount() == 0:
                    self.import_panel.load_subjects_from_directory(in_dir)
        elif current == self.fastsurfer_panel:
            out_dir = self.import_panel.get_output_folder().strip()
            if out_dir and os.path.isdir(out_dir):
                fs_dir = os.path.join(out_dir, "fastsurfer")
                if not self.fastsurfer_panel.fs_dir_input.text().strip() or not os.path.isdir(self.fastsurfer_panel.fs_dir_input.text().strip()):
                    self.fastsurfer_panel.fs_dir_input.setText(fs_dir)
            self.fastsurfer_panel.update_run_button_state()
            self.fastsurfer_panel.populate_results_table()
        elif current == self.icp_panel:
            out_dir = self.import_panel.get_output_folder().strip()
            if out_dir and os.path.isdir(out_dir):
                icp_dir = os.path.join(out_dir, "output_ICP")
                if not self.icp_panel.icp_dir_input.text().strip() or not os.path.isdir(self.icp_panel.icp_dir_input.text().strip()):
                    self.icp_panel.icp_dir_input.setText(icp_dir)
            self.icp_panel.update_run_button_state()
            self.icp_panel.populate_results_table()
        elif current == self.spharm_panel:
            out_dir = self.import_panel.get_output_folder().strip()
            if out_dir and os.path.isdir(out_dir):
                spharm_dir = os.path.join(out_dir, "output_SPHARM")
                if not self.spharm_panel.spharm_dir_input.text().strip() or not os.path.isdir(self.spharm_panel.spharm_dir_input.text().strip()):
                    self.spharm_panel.spharm_dir_input.setText(spharm_dir)
            self.spharm_panel.update_run_button_state()
            self.spharm_panel.populate_results_table()
        elif current == self.result_panel:
            out_dir = self.import_panel.get_output_folder().strip()
            if out_dir and os.path.isdir(out_dir):
                spharm_dir = os.path.join(out_dir, "output_SPHARM")
                if hasattr(self.result_panel, 'spharm_dir_input'):
                    if not self.result_panel.spharm_dir_input.text().strip() or not os.path.isdir(self.result_panel.spharm_dir_input.text().strip()):
                        self.result_panel.spharm_dir_input.setText(spharm_dir)
                res_dir = os.path.join(out_dir, "output_Result")
                if hasattr(self.result_panel, 'result_dir_input'):
                    if not self.result_panel.result_dir_input.text().strip():
                        self.result_panel.result_dir_input.setText(res_dir)
            if hasattr(self.result_panel, 'on_panel_activated'):
                self.result_panel.on_panel_activated()
        self.stacked_widget.updateGeometry()
        self.updateGeometry()
