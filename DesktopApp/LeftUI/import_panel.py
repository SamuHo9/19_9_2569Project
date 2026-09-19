import os
import glob
import json
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QGroupBox, QMenu, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint

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

class ImportPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)
    signal_directories_changed = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_history()
        self.setup_ui()
        
    def setup_ui(self):
        import_layout = QVBoxLayout(self)
        
        # 1. Import Data Properties
        import_group = QGroupBox("Import Data Properties")
        import_group.setStyleSheet("""
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
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 20, 10, 10)
        ig_layout.setSpacing(8)

        # Row 1: Choose Data Directory with Open File History button
        data_btn_row = QHBoxLayout()
        data_btn_row.setSpacing(6)

        dir_select_btn = QPushButton("Choose Data Directory")
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
        self.history_btn.setToolTip("View and select from previously chosen data directories")
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

        ig_layout.addLayout(data_btn_row)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("Choose Output Directory")
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
        ig_layout.addWidget(out_dir_btn)

        # Row 2: Input Path and Output Path side by side
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("No input directory selected...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("No output directory selected...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ccc; border-radius: 3px; padding: 3px 5px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        ig_layout.addLayout(path_row)

        # Row 3: Standalone Import Data Button
        import_action_btn = QPushButton("Import Data")
        import_action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 10px;
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
        import_action_btn.clicked.connect(self.on_import_clicked)
        ig_layout.addWidget(import_action_btn)
        
        import_layout.addWidget(import_group)

        # 2. Imported Subjects
        subj_group = QGroupBox("Imported Subjects")
        subj_group.setStyleSheet("QGroupBox { margin-top: 15px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        subj_layout = QVBoxLayout(subj_group)
        subj_layout.setContentsMargins(10, 20, 10, 10)
        subj_layout.setSpacing(10)
        self.subjects_table = ToggleTableWidget(0, 1)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subj_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 10px;
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
        self.remove_selected_btn.clicked.connect(self.remove_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.remove_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)

    def load_history(self):
        self.history_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_history.json")
        self.recent_dirs = []
        self.last_input_dir = None
        self.last_output_dir = None
        
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.recent_dirs = data.get("recent_input_dirs", [])
                    self.last_input_dir = data.get("last_input_dir")
                    self.last_output_dir = data.get("last_output_dir")
            except Exception:
                self.recent_dirs = []
        
        # History is kept empty unless user explicitly adds directories

    def save_history(self):
        try:
            data = {
                "recent_input_dirs": self.recent_dirs,
                "last_input_dir": getattr(self, 'last_input_dir', None),
                "last_output_dir": getattr(self, 'last_output_dir', None)
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_to_history(self, path):
        if not path or not os.path.isdir(path):
            return
        norm_p = os.path.normpath(os.path.abspath(path))
        self.recent_dirs = [p for p in self.recent_dirs if os.path.normpath(os.path.abspath(p)) != norm_p]
        self.recent_dirs.insert(0, path)
        self.recent_dirs = self.recent_dirs[:10]
        self.last_input_dir = path
        self.save_history()

    def show_history_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 1px solid #ced6e0;
                border-radius: 6px;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item {
                padding: 6px 14px;
                border-radius: 4px;
                color: #2c3e50;
            }
            QMenu::item:selected {
                background-color: #f1f2f6;
                color: #2980b9;
                font-weight: bold;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e4e7eb;
                margin: 4px 8px;
            }
        """)

        if not self.recent_dirs:
            empty_act = menu.addAction("No recent directories found")
            empty_act.setEnabled(False)
        else:
            title_act = menu.addAction("Recent Data Directories:")
            title_act.setEnabled(False)
            font = title_act.font()
            font.setBold(True)
            title_act.setFont(font)
            menu.addSeparator()

            for path in self.recent_dirs:
                act = menu.addAction(f"📁  {path}")
                act.triggered.connect(lambda checked, p=path: self.select_history_path(p))

            menu.addSeparator()
            clear_act = menu.addAction("🧹  Clear History")
            clear_act.triggered.connect(self.clear_history)

        menu.exec(self.history_btn.mapToGlobal(QPoint(0, self.history_btn.height())))

    def select_history_path(self, path):
        if os.path.isdir(path):
            self.folder_input.setText(path)
            self.add_to_history(path)
            self.signal_log_message.emit(f"Selected from history: {path}")
            self.load_subjects_from_directory(path)
            self.signal_log_message.emit(">>> Data loaded from history successfully.")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())
        else:
            self.signal_log_message.emit(f"[WARNING] Directory no longer exists: {path}")
            if path in self.recent_dirs:
                self.recent_dirs.remove(path)
                self.save_history()

    def clear_history(self):
        self.recent_dirs = []
        self.save_history()
        self.signal_log_message.emit("File history cleared.")

    def get_folder(self):
        return self.folder_input.text()

    def get_output_folder(self):
        return self.out_folder_input.text()

    def select_directory(self):
        initial_dir = self.last_input_dir if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Directory with NIFTI files", initial_dir)
        if folder:
            self.folder_input.setText(folder)
            self.add_to_history(folder)
            self.signal_log_message.emit(f"Selected input directory: {folder}")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    def select_out_directory(self):
        initial_dir = self.last_output_dir if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            self.save_history()
            self.signal_log_message.emit(f"Selected output directory: {folder}")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    def on_import_clicked(self):
        directory = self.folder_input.text()
        if directory and os.path.isdir(directory):
            self.add_to_history(directory)
            self.load_subjects_from_directory(directory)
            self.signal_log_message.emit(">>> Data Imported Successfully.")
        else:
            self.signal_log_message.emit("[ERROR] Please select a valid directory first.")

    def load_subjects_from_directory(self, directory, clear_existing=False):
        if not directory or not os.path.isdir(directory):
            return

        if clear_existing or getattr(self, 'loaded_directory', None) != directory:
            self.subjects_table.setRowCount(0)
        self.loaded_directory = directory

        search_patterns = ["*.nrrd", "*.nii.gz", "*.nii", "*.vtk", "*.mgz"]
        files = []
        for pattern in search_patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
            files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
            
        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk"]
        for f in files:
            norm = os.path.normpath(f)
            if norm not in seen:
                seen.add(norm)
                fname = os.path.basename(norm).lower()
                if not any(kw in fname for kw in skip_keywords):
                    unique_files.append(norm)
        files = unique_files

        if not files:
            self.signal_log_message.emit(f"No valid MRI image files found in {directory}")
            return
            
        # Prevent duplicates
        existing_paths = set()
        for i in range(self.subjects_table.rowCount()):
            item = self.subjects_table.item(i, 0)
            if item:
                existing_paths.add(item.data(Qt.ItemDataRole.UserRole))
                
        new_files = [f for f in files if f not in existing_paths]
        
        if not new_files:
            self.signal_log_message.emit(f"All files in {directory} are already imported.")
            if self.subjects_table.rowCount() > 0:
                if not self.subjects_table.selectedItems():
                    self.subjects_table.selectRow(0)
                self.display_selected_subject()
            return
            
        self.signal_log_message.emit(f"Importing {len(new_files)} new files...")
        
        current_row_count = self.subjects_table.rowCount()
        self.subjects_table.setRowCount(current_row_count + len(new_files))
        
        for i, filepath in enumerate(new_files):
            row = current_row_count + i
            filename = os.path.basename(filepath)
            item = QTableWidgetItem(filename)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            
            self.subjects_table.setItem(row, 0, item)

        # Auto-select and display first subject
        if self.subjects_table.rowCount() > 0:
            if not self.subjects_table.selectedItems():
                self.subjects_table.selectRow(0)
            self.display_selected_subject()

    def on_subject_selection_changed(self):
        if self.display_on_click_cb.isChecked():
            self.display_selected_subject()

    def display_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
        
        selected_rows = list(set([item.row() for item in selected_items]))
        if len(selected_rows) > 1:
            return  # Do not display if multiple subjects are selected
            
        row = selected_rows[0]
        item = self.subjects_table.item(row, 0)
        subject_name = item.text()
        filepath = item.data(Qt.ItemDataRole.UserRole)
        
        self.signal_log_message.emit(f"Displaying subject: {subject_name}")
        self.signal_subject_selected.emit(filepath)

    def remove_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
            
        selected_rows = sorted(list(set([item.row() for item in selected_items])), reverse=True)
        count = len(selected_rows)
        if count == 0:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove {count} selected subject(s) from the list?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply != QMessageBox.StandardButton.Ok:
            return
        
        for row in selected_rows:
            item = self.subjects_table.item(row, 0)
            subject_name = item.text() if item else f"Row {row}"
            self.subjects_table.removeRow(row)
            self.signal_log_message.emit(f"Removed subject from list: {subject_name}")
