import sys
import webbrowser
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
                             QLabel, QComboBox, QMessageBox, QTextEdit, QPushButton, 
                             QCheckBox, QApplication, QScrollArea, QFrame, QLineEdit, QPlainTextEdit)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QEvent

from LeftUI.left_panel import LeftPanel
from RightUI.right_panel import RightPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hippocampal Shape Analysis Pipeline (Slicer-style)")
        self.resize(1200, 800)

        self.create_menus_and_toolbar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setHandleWidth(8)
        self.v_splitter.setChildrenCollapsible(True)
        self.v_splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                background: #34495e;
                height: 8px;
                border-top: 1px solid #4a627a;
                border-bottom: 1px solid #243342;
            }
            QSplitter::handle:vertical:hover {
                background: #3498db;
            }
        """)
        main_layout.addWidget(self.v_splitter)

        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_splitter.setHandleWidth(6)
        self.h_splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                background: #dcdde1;
                width: 6px;
            }
            QSplitter::handle:horizontal:hover {
                background: #3498db;
            }
        """)

        self.left_panel = LeftPanel()
        self.right_panel = RightPanel(self)

        # Wrap LeftPanel in a scroll area so it doesn't get clipped and allows free vertical resizing
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.horizontalScrollBar().setEnabled(False)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_scroll.setWidget(self.left_panel)

        # Wrap RightPanel in a scroll area so the 4 viewports maintain their dimensions and don't squish
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_panel.setMinimumHeight(450)
        self.right_panel.setMinimumWidth(500)
        self.right_scroll.setWidget(self.right_panel)

        self.left_scroll.setMinimumWidth(380)
        self.h_splitter.addWidget(self.left_scroll)
        self.h_splitter.addWidget(self.right_scroll)
        self.h_splitter.setSizes([450, 830])
        self.saved_h_splitter_sizes = [450, 830]
        self.h_splitter.splitterMoved.connect(self.on_h_splitter_moved)
        
        self.v_splitter.addWidget(self.h_splitter)
        
        # Enhanced Execution Console Widget
        self.console_widget = QWidget()
        console_layout = QVBoxLayout(self.console_widget)
        console_layout.setContentsMargins(6, 6, 6, 6)
        console_layout.setSpacing(6)
        
        # Console Header Toolbar
        console_header = QHBoxLayout()
        console_header.setContentsMargins(2, 0, 2, 0)
        console_header.setSpacing(8)
        
        console_title = QLabel("🖥️ Terminal & Execution Console")
        console_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        console_header.addWidget(console_title)
        
        self.console_status_lbl = QLabel("Ready")
        self.console_status_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px; padding-left: 4px;")
        console_header.addWidget(self.console_status_lbl)
        
        console_header.addStretch()
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setStyleSheet("color: #34495e; font-size: 11px;")
        console_header.addWidget(self.auto_scroll_cb)
        
        copy_btn = QPushButton("📋 Copy All")
        copy_btn.setFixedHeight(24)
        copy_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover { background: #e4e7eb; }
        """)
        copy_btn.clicked.connect(self.copy_console_logs)
        console_header.addWidget(copy_btn)
        
        clear_btn = QPushButton("🧹 Clear")
        clear_btn.setFixedHeight(24)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #f1f2f6;
                color: #2f3542;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 2px 8px;
            }
            QPushButton:hover { background: #e4e7eb; color: #e74c3c; }
        """)
        clear_btn.clicked.connect(self.clear_console_logs)
        console_header.addWidget(clear_btn)
        
        self.expand_btn = QPushButton("⤢ Expand")
        self.expand_btn.setFixedHeight(24)
        self.expand_btn.setToolTip("Expand terminal to full screen covering workspace")
        self.expand_btn.setStyleSheet("""
            QPushButton {
                background: #34495e;
                color: white;
                border: 1px solid #2c3e50;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 10px;
            }
            QPushButton:hover { background: #415b76; }
        """)
        self.expand_btn.clicked.connect(self.toggle_console_expand)
        console_header.addWidget(self.expand_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedHeight(24)
        close_btn.setFixedWidth(24)
        close_btn.setToolTip("Hide Terminal (Show again via View menu)")
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #7f8c8d;
                border: 1px solid #ced6e0;
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #e74c3c; color: white; border-color: #c0392b; }
        """)
        close_btn.clicked.connect(lambda: self.set_terminal_visible(False))
        console_header.addWidget(close_btn)
        
        console_layout.addLayout(console_header)
        
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setStyleSheet("""
            QTextEdit {
                background-color: #161b22;
                color: #e6edf3;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        console_layout.addWidget(self.log_window)
        
        self.v_splitter.addWidget(self.console_widget)
        self.v_splitter.setStretchFactor(0, 1)
        self.v_splitter.setStretchFactor(1, 0)
        self.v_splitter.setSizes([600, 135])
        self.v_splitter.splitterMoved.connect(self.on_splitter_moved)

        self.is_terminal_fullscreen = False
        self.saved_splitter_sizes = [600, 135]

        # Connect signals between panels
        self.left_panel.signal_subject_selected.connect(self.right_panel.display_subject)
        self.left_panel.signal_mesh_selected.connect(self.right_panel.display_mesh)
        self.left_panel.signal_log_message.connect(self.log)
        self.right_panel.signal_log_message.connect(self.log)
        self.left_panel.signal_template_toggled.connect(self.right_panel.viewer.set_template_visible)
        self.left_panel.signal_overlay_all_toggled.connect(self.on_overlay_all_toggled)
        self.right_panel.signal_overlay_toggled.connect(self.left_panel.set_overlay_visible)
        self.left_panel.signal_side_changed.connect(self.right_panel.set_side_filter)
        if hasattr(self.left_panel, 'signal_gradcam_mesh_requested'):
            self.left_panel.signal_gradcam_mesh_requested.connect(self.right_panel.display_gradcam_mesh)
            self.left_panel.signal_patient_overlay_requested.connect(self.right_panel.set_patient_overlay)
            self.left_panel.signal_clear_gradcam_requested.connect(self.right_panel.clear_gradcam_view)
        if hasattr(self.left_panel, 'signal_diagnostic_info'):
            self.left_panel.signal_diagnostic_info.connect(self.right_panel.viewer.set_diagnostic_info)
        if hasattr(self.right_panel, 'signal_step_sd'):
            self.right_panel.signal_step_sd.connect(self.on_step_sd_requested)
        if hasattr(self.right_panel, 'signal_reset_sd'):
            self.right_panel.signal_reset_sd.connect(self.on_reset_sd_requested)
        self.module_combo.currentTextChanged.connect(self.on_module_changed)

        # Set initial view mode based on current module selection
        self.on_module_changed(self.module_combo.currentText())

        # Global event filter to ensure arrow keys scrub SD cleanly without scrolling Left UI
        QApplication.instance().installEventFilter(self)

        self.log("SlicerSALT-style UI initialized successfully.")
        
        self.check_slicer_salt()
        self.showMaximized()

    def create_menus_and_toolbar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Add basic Edit actions
        undo_action = QAction("Undo", self)
        redo_action = QAction("Redo", self)
        preferences_action = QAction("Preferences...", self)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(preferences_action)
        
        # Add basic View actions
        toggle_toolbar_action = QAction("Toggle Toolbar", self, checkable=True)
        toggle_toolbar_action.setChecked(True)
        reset_view_action = QAction("Reset View", self)
        
        self.toggle_terminal_action = QAction("Terminal / Console", self, checkable=True)
        self.toggle_terminal_action.setChecked(True)
        self.toggle_terminal_action.triggered.connect(self.set_terminal_visible)

        view_menu.addAction(toggle_toolbar_action)
        view_menu.addAction(self.toggle_terminal_action)
        view_menu.addSeparator()
        view_menu.addAction(reset_view_action)

        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        save_action = QAction("💾 Save", self)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Modules: "))
        self.module_combo = QComboBox()
        self.module_combo.addItems([
            "Main Panel",
            "Data Importer",
            "FastSurfer Segmentation",
            "ICP Registration",
            "SPHARM Processing",
            "Result Panel"
        ])
        self.module_combo.setMinimumWidth(200)
        toolbar.addWidget(self.module_combo)

    def check_slicer_salt(self):
        import glob
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        
        if not slicer_paths:
            QMessageBox.critical(self, "SlicerSALT Required", 
                                "SlicerSALT could not be found in the default installation directory (C:\\Program Files\\SlicerSALT*).\n\n"
                                "This program requires SlicerSALT to function. The application will now close and open the download page.")
            webbrowser.open("https://salt.slicer.org/")
            sys.exit(1)
        else:
            self.log(f"SUCCESS: SlicerSALT detected at {slicer_paths[0]}")

    def copy_console_logs(self):
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_window.toPlainText())
            self.console_status_lbl.setText("Logs copied to clipboard!")
            self.console_status_lbl.setStyleSheet("color: #2ecc71; font-size: 11px;")

    def clear_console_logs(self):
        self.log_window.clear()
        self.console_status_lbl.setText("Console cleared")
        self.console_status_lbl.setStyleSheet("color: #7f8c8d; font-size: 11px;")

    def toggle_console_expand(self):
        sizes = self.v_splitter.sizes()
        total_height = sum(sizes) if sum(sizes) > 0 else 800
        
        # If not full screen (top area > 50px) -> make terminal full screen covering left and right
        if sizes[0] > 50:
            self.saved_splitter_sizes = sizes
            self.v_splitter.setSizes([0, total_height])
            self.is_terminal_fullscreen = True
            self.expand_btn.setText("⤡ Restore")
            self.expand_btn.setToolTip("Restore terminal to original size")
            self.console_status_lbl.setText("Terminal Fullscreen (Covering workspace)")
        else:
            # Restore to previous sizes
            if hasattr(self, 'saved_splitter_sizes') and self.saved_splitter_sizes[0] > 50:
                self.v_splitter.setSizes(self.saved_splitter_sizes)
            else:
                self.v_splitter.setSizes([total_height - 135, 135])
            self.is_terminal_fullscreen = False
            self.expand_btn.setText("⤢ Expand")
            self.expand_btn.setToolTip("Expand terminal to full screen")
            self.console_status_lbl.setText("Ready")

    def on_splitter_moved(self, pos, index):
        sizes = self.v_splitter.sizes()
        if sizes[0] > 50 and getattr(self, 'is_terminal_fullscreen', False):
            self.is_terminal_fullscreen = False
            self.expand_btn.setText("⤢ Expand")
            self.expand_btn.setToolTip("Expand terminal to full screen")

    def on_h_splitter_moved(self, pos, index):
        if self.right_scroll.isVisible():
            sizes = self.h_splitter.sizes()
            if len(sizes) == 2 and sizes[0] > 50 and sizes[1] > 50:
                self.saved_h_splitter_sizes = sizes

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'v_splitter') and not getattr(self, '_initial_layout_done', False):
            self._initial_layout_done = True
            total_h = self.v_splitter.height()
            if total_h > 150:
                terminal_h = 135
                top_h = max(total_h - terminal_h, 540)
                self.v_splitter.setSizes([top_h, total_h - top_h])
                self.saved_splitter_sizes = [top_h, total_h - top_h]

    def set_terminal_visible(self, visible):
        self.console_widget.setVisible(visible)
        if hasattr(self, 'toggle_terminal_action'):
            self.toggle_terminal_action.setChecked(visible)
        if visible:
            sizes = self.v_splitter.sizes()
            total_height = sum(sizes) if sum(sizes) > 0 else 800
            if len(sizes) >= 2 and sizes[1] < 50:
                self.v_splitter.setSizes([total_height - 135, 135])
            self.console_status_lbl.setText("Ready")

    def log(self, message):
        import html
        from datetime import datetime
        
        now_str = datetime.now().strftime("%H:%M:%S")
        escaped_msg = html.escape(str(message))
        
        # Color coding tags for easier monitoring
        if "[ERROR]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #ff7b72; font-weight: bold;">{escaped_msg}</span>'
            self.console_status_lbl.setText("Last status: Error")
            self.console_status_lbl.setStyleSheet("color: #ff7b72; font-weight: bold; font-size: 11px;")
        elif "[WARNING]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #d29922; font-weight: bold;">{escaped_msg}</span>'
            self.console_status_lbl.setText("Last status: Warning")
            self.console_status_lbl.setStyleSheet("color: #d29922; font-size: 11px;")
        elif ">>>" in escaped_msg or "Starting" in escaped_msg or "Running" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #58a6ff; font-weight: bold;">{escaped_msg}</span>'
            self.console_status_lbl.setText("Running...")
            self.console_status_lbl.setStyleSheet("color: #58a6ff; font-size: 11px;")
        elif "SUCCESS" in escaped_msg or "completed successfully" in escaped_msg.lower():
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #3fb950; font-weight: bold;">{escaped_msg}</span>'
            self.console_status_lbl.setText("Ready (Success)")
            self.console_status_lbl.setStyleSheet("color: #3fb950; font-weight: bold; font-size: 11px;")
        elif "[INFO]" in escaped_msg:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #79c0ff;">{escaped_msg}</span>'
        else:
            formatted = f'<span style="color: #8b949e;">[{now_str}]</span> <span style="color: #c9d1d9;">{escaped_msg}</span>'
            
        self.log_window.append(formatted)
        
        if hasattr(self, 'auto_scroll_cb') and self.auto_scroll_cb.isChecked():
            sb = self.log_window.verticalScrollBar()
            sb.setValue(sb.maximum())

    def on_overlay_all_toggled(self, enabled: bool, filepaths: list, side_filter: str):
        self.right_panel.set_overlay_visible(enabled)
        if enabled:
            self.right_panel.display_all_meshes(filepaths, side_filter)
        else:
            self.right_panel.viewer.clear_multi_mesh_actors()
            self.right_panel.viewer.update_legend()
            self.right_panel.viewer.mesh_vtkWidget.GetRenderWindow().Render()

    def on_module_changed(self, module_name):
        index = self.module_combo.findText(module_name)
        self.left_panel.switch_module(index)
        
        # Clear previous module's patient meshes from 3D view so they do not mix
        self.right_panel.viewer.clear_all_patient_meshes()

        # Modules configuration: define which modules have Right UI enabled
        # Currently Main Panel and Result Panel do not have Right UI,
        # but the Right UI system is fully preserved so either can be enabled here in the future.
        modules_with_right_ui = {
            "Main Panel": True,
            "Data Importer": True,
            "FastSurfer Segmentation": True,
            "ICP Registration": True,
            "SPHARM Processing": True,
            "Result Panel": True,
        }

        has_right_ui = modules_with_right_ui.get(module_name, False)

        if not has_right_ui:
            # Hide right UI for panels without right UI
            if self.right_scroll.isVisible():
                sizes = self.h_splitter.sizes()
                if len(sizes) == 2 and sizes[0] > 50 and sizes[1] > 50:
                    self.saved_h_splitter_sizes = sizes
            self.right_scroll.setVisible(False)
        else:
            # Show right UI and restore previous horizontal split size
            if not self.right_scroll.isVisible():
                self.right_scroll.setVisible(True)
                if hasattr(self, 'saved_h_splitter_sizes') and self.saved_h_splitter_sizes:
                    self.h_splitter.setSizes(self.saved_h_splitter_sizes)
                else:
                    self.h_splitter.setSizes([430, 850])

            if module_name in ("ICP Registration", "SPHARM Processing", "Result Panel", "Main Panel"):
                self.right_panel.set_view_mode("full_3d", module_name)
            elif module_name == "FastSurfer Segmentation":
                self.right_panel.set_view_mode("quad", module_name)
                self.right_panel.viewer.set_mesh_view_visible(True)
            else:
                self.right_panel.set_view_mode("quad", module_name)
                self.right_panel.viewer.set_mesh_view_visible(False)

        # If switching away from Result Panel, always clear Grad-CAM scalar actors and colorbar
        if module_name != "Result Panel":
            if hasattr(self.right_panel, 'clear_gradcam_view'):
                self.right_panel.clear_gradcam_view(render_now=False)

        # Sync state with newly active module panel
        active_panel = self.left_panel.get_current_module_panel()
        if module_name == "Data Importer" and hasattr(active_panel, 'display_selected_subject'):
            active_panel.display_selected_subject()
        elif hasattr(active_panel, 'results_table') and active_panel.results_table.rowCount() > 0:
            if not active_panel.results_table.selectedItems():
                active_panel.results_table.selectRow(0)
        if hasattr(active_panel, 'overlay_cb'):
            is_overlay = active_panel.overlay_cb.isChecked()
            self.right_panel.set_overlay_visible(is_overlay)
            if is_overlay:
                active_panel.emit_overlay_meshes()
        if hasattr(active_panel, 'current_side_filter'):
            self.right_panel.set_side_filter(active_panel.current_side_filter)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if self.module_combo.currentText() == "Result Panel":
                if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_step_sd_requested(-0.1)
                        return True
                elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_step_sd_requested(0.1)
                        return True
                elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_reset_sd_requested()
                        return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        curr = self.module_combo.currentText()
        if curr == "Result Panel":
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
                self.on_step_sd_requested(-0.1)
                event.accept()
                return
            elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
                self.on_step_sd_requested(0.1)
                event.accept()
                return
            elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
                self.on_reset_sd_requested()
                event.accept()
                return
        super().keyPressEvent(event)

    def on_step_sd_requested(self, delta: float):
        res_panel = getattr(self.left_panel, 'result_panel', None)
        if res_panel and hasattr(res_panel, 'step_sd'):
            res_panel.step_sd(delta)

    def on_reset_sd_requested(self):
        res_panel = getattr(self.left_panel, 'result_panel', None)
        if res_panel and hasattr(res_panel, 'set_sd_value'):
            res_panel.set_sd_value(0.0)
