import vtk
import os
import numpy as np
from PyQt6.QtWidgets import (QWidget, QGridLayout, QFrame, QVBoxLayout, QSlider, 
                             QHBoxLayout, QPushButton, QLabel, QCheckBox)
from PyQt6.QtCore import pyqtSignal, Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

def create_mask_actor(mask_path, r, g, b, alpha=0.5):
    if not os.path.exists(mask_path):
        return None
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableRange(0, 1)
    lut.SetTableValue(0, 0, 0, 0, 0.0) # Transparent background
    lut.SetTableValue(1, r/255.0, g/255.0, b/255.0, alpha)
    
    map_colors = vtk.vtkImageMapToColors()
    map_colors.SetInputData(reader.GetOutput())
    map_colors.SetLookupTable(lut)
    map_colors.SetOutputFormatToRGBA()
    map_colors.Update()
    
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputData(map_colors.GetOutput())
    return actor

# A subclass that intercepts and drops Left Clicks to prevent VTK's default Window/Level adjustment
class CustomQVTKWidget(QVTKRenderWindowInteractor):
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # Ignore left clicks so vtkImageViewer2 doesn't trigger Window/Level adjustment
            return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            # Ignore left click drags
            return
        super().mouseMoveEvent(ev)

class VtkViewer(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_toggled = pyqtSignal(bool)
    signal_step_sd = pyqtSignal(float)
    signal_reset_sd = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.maximized_frame = None
        self.mesh_view_enabled = False
        self.view_mode = "quad"  # 'quad' (4-view) or 'full_3d' (single large 3D viewport)
        self.current_module_name = ""
        self.current_mesh_path = None
        self.current_side_filter = "all"
        self.current_side = "left"
        self.template_actor = None
        self.template_actors = []
        self.mesh_actor = None
        self.multi_mesh_actors = []
        self.multi_mesh_paths = []
        self.is_overlay_active = False
        self.setup_ui()

    def setup_ui(self):
        # 2x2 Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2) # Small spacing between panels

        # Create 4 Frames with colored borders
        self.axial_frame = self.create_view_frame("#e74c3c")    # Red
        self.coronal_frame = self.create_view_frame("#2ecc71")  # Green
        self.sagittal_frame = self.create_view_frame("#f1c40f") # Yellow
        self.mesh_frame = self.create_view_frame("#3498db")     # Blue

        self.grid_layout.addWidget(self.axial_frame, 0, 0)
        self.grid_layout.addWidget(self.mesh_frame, 0, 1)
        self.grid_layout.addWidget(self.coronal_frame, 1, 0)
        self.grid_layout.addWidget(self.sagittal_frame, 1, 1)

        # Set up VTK Viewers with Sliders, Reset Buttons and 3D Plane Toggles
        self.axial_slider, self.axial_vtkWidget, self.axial_slice_lbl, self.axial_flip_btn, self.axial_3d_btn, self.axial_reset_btn = self.setup_vtk_with_slider(self.axial_frame, "Axial View")
        self.axial_viewer = vtk.vtkImageViewer2()
        self.axial_viewer.SetRenderWindow(self.axial_vtkWidget.GetRenderWindow())
        self.axial_viewer.SetupInteractor(self.axial_vtkWidget.GetRenderWindow().GetInteractor())
        self.axial_viewer.SetSliceOrientationToXY()
        self.axial_slider.valueChanged.connect(lambda val: self.change_slice(self.axial_viewer, val, self.axial_slider, self.axial_slice_lbl))
        self.axial_flip_btn.clicked.connect(lambda: self.toggle_flip(self.axial_viewer))
        self.axial_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("axial", chk))
        self.axial_reset_btn.clicked.connect(lambda: self.reset_slice("axial"))

        self.coronal_slider, self.coronal_vtkWidget, self.coronal_slice_lbl, self.coronal_flip_btn, self.coronal_3d_btn, self.coronal_reset_btn = self.setup_vtk_with_slider(self.coronal_frame, "Coronal View")
        self.coronal_viewer = vtk.vtkImageViewer2()
        self.coronal_viewer.SetRenderWindow(self.coronal_vtkWidget.GetRenderWindow())
        self.coronal_viewer.SetupInteractor(self.coronal_vtkWidget.GetRenderWindow().GetInteractor())
        self.coronal_viewer.SetSliceOrientationToXZ()
        self.coronal_slider.valueChanged.connect(lambda val: self.change_slice(self.coronal_viewer, val, self.coronal_slider, self.coronal_slice_lbl))
        self.coronal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.coronal_viewer))
        self.coronal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("coronal", chk))
        self.coronal_reset_btn.clicked.connect(lambda: self.reset_slice("coronal"))

        self.sagittal_slider, self.sagittal_vtkWidget, self.sagittal_slice_lbl, self.sagittal_flip_btn, self.sagittal_3d_btn, self.sagittal_reset_btn = self.setup_vtk_with_slider(self.sagittal_frame, "Sagittal View")
        self.sagittal_viewer = vtk.vtkImageViewer2()
        self.sagittal_viewer.SetRenderWindow(self.sagittal_vtkWidget.GetRenderWindow())
        self.sagittal_viewer.SetupInteractor(self.sagittal_vtkWidget.GetRenderWindow().GetInteractor())
        self.sagittal_viewer.SetSliceOrientationToYZ()
        self.sagittal_slider.valueChanged.connect(lambda val: self.change_slice(self.sagittal_viewer, val, self.sagittal_slider, self.sagittal_slice_lbl))
        self.sagittal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.sagittal_viewer))
        self.sagittal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("sagittal", chk))
        self.sagittal_reset_btn.clicked.connect(lambda: self.reset_slice("sagittal"))

        # Configure solid black background for 2D slice viewers (Axial, Coronal, Sagittal)
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            ren = viewer.GetRenderer()
            ren.GradientBackgroundOff()
            ren.SetBackground(0.0, 0.0, 0.0)
            viewer.GetImageActor().SetVisibility(False)

        # Set up VTK Viewer for 3D Mesh
        layout_mesh = self.mesh_frame.layout()
        mesh_top_bar = QHBoxLayout()
        mesh_top_bar.setContentsMargins(6, 4, 6, 4)
        mesh_top_bar.setSpacing(8)
        
        self.mesh_title_lbl = QLabel("3D Mesh View")
        self.mesh_title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent; font-size: 12px;")
        
        self.mesh_legend_lbl = QLabel("")
        self.mesh_legend_lbl.setStyleSheet("color: #ecf0f1; font-size: 11px; background: rgba(20,25,35,180); padding: 2px 8px; border-radius: 4px; border: 1px solid #3d4d65;")
        self.mesh_legend_lbl.setVisible(False)
        
        self.template_cb = QCheckBox("Show Reference Template")
        self.template_cb.setToolTip("Overlay standard reference template (mean shape) in 3D")
        self.template_cb.setStyleSheet("""
            QCheckBox {
                color: #f1c40f;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
                background: rgba(241, 196, 15, 0.12);
                border: 1px solid #f39c12;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QCheckBox:hover {
                background: rgba(241, 196, 15, 0.22);
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #f39c12;
                border-radius: 3px;
                background: #1e2230;
            }
            QCheckBox::indicator:checked {
                background: #f39c12;
            }
        """)
        self.template_cb.toggled.connect(self.on_template_cb_toggled)
        self.template_cb.setVisible(False) # Hidden initially in quad mode, shown in full_3d mode

        self.overlay_cb = QCheckBox("Overlay All Meshes")
        self.overlay_cb.setToolTip("Superimpose and view all aligned meshes together in 3D")
        self.overlay_cb.setStyleSheet("""
            QCheckBox {
                color: #1abc9c;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
                background: rgba(26, 188, 156, 0.12);
                border: 1px solid #16a085;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QCheckBox:hover {
                background: rgba(26, 188, 156, 0.22);
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #16a085;
                border-radius: 3px;
                background: #1e2230;
            }
            QCheckBox::indicator:checked {
                background: #1abc9c;
            }
        """)
        self.overlay_cb.toggled.connect(self.on_overlay_cb_toggled)
        self.overlay_cb.setVisible(False)
        
        self.mesh_max_btn = QPushButton("◻")
        self.mesh_max_btn.setFixedSize(24, 24)
        self.mesh_max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        self.mesh_max_btn.clicked.connect(lambda: self.toggle_maximize(self.mesh_frame))
        
        mesh_top_bar.addWidget(self.mesh_title_lbl)
        mesh_top_bar.addWidget(self.mesh_legend_lbl)
        mesh_top_bar.addStretch()
        mesh_top_bar.addWidget(self.template_cb)
        mesh_top_bar.addWidget(self.overlay_cb)
        mesh_top_bar.addWidget(self.mesh_max_btn)
        
        layout_mesh.addLayout(mesh_top_bar)
        
        self.mesh_vtkWidget = QVTKRenderWindowInteractor(self.mesh_frame)
        self.mesh_vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.mesh_vtkWidget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout_mesh.addWidget(self.mesh_vtkWidget)
        
        self.mesh_renderer = vtk.vtkRenderer()
        self.mesh_renderer.GradientBackgroundOn()
        self.mesh_renderer.SetBackground(0.741, 0.749, 0.902)
        self.mesh_renderer.SetBackground2(0.459, 0.475, 0.745)
        self.mesh_vtkWidget.GetRenderWindow().AddRenderer(self.mesh_renderer)
        self.setup_3d_lighting()
        try:
            iren = self.mesh_vtkWidget.GetRenderWindow().GetInteractor()
            iren.AddObserver("KeyPressEvent", self._on_mesh_key_press)
            iren.AddObserver("LeftButtonPressEvent", self._on_mesh_mouse_press)
        except Exception:
            pass
        
        # 3D Orthogonal Slice Plane Actors & Outline Borders
        self.axial_3d_actor = vtk.vtkImageActor()
        self.axial_3d_actor.SetVisibility(False)
        self.axial_voi = vtk.vtkExtractVOI()
        self.axial_outline = vtk.vtkOutlineFilter()
        self.axial_outline.SetInputConnection(self.axial_voi.GetOutputPort())
        axial_out_map = vtk.vtkPolyDataMapper()
        axial_out_map.SetInputConnection(self.axial_outline.GetOutputPort())
        self.axial_outline_actor = vtk.vtkActor()
        self.axial_outline_actor.SetMapper(axial_out_map)
        self.axial_outline_actor.GetProperty().SetColor(0.95, 0.25, 0.25) # Red outline
        self.axial_outline_actor.GetProperty().SetLineWidth(2.5)
        self.axial_outline_actor.SetVisibility(False)

        self.coronal_3d_actor = vtk.vtkImageActor()
        self.coronal_3d_actor.SetVisibility(False)
        self.coronal_voi = vtk.vtkExtractVOI()
        self.coronal_outline = vtk.vtkOutlineFilter()
        self.coronal_outline.SetInputConnection(self.coronal_voi.GetOutputPort())
        coronal_out_map = vtk.vtkPolyDataMapper()
        coronal_out_map.SetInputConnection(self.coronal_outline.GetOutputPort())
        self.coronal_outline_actor = vtk.vtkActor()
        self.coronal_outline_actor.SetMapper(coronal_out_map)
        self.coronal_outline_actor.GetProperty().SetColor(0.25, 0.85, 0.35) # Green outline
        self.coronal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.coronal_outline_actor.SetVisibility(False)

        self.sagittal_3d_actor = vtk.vtkImageActor()
        self.sagittal_3d_actor.SetVisibility(False)
        self.sagittal_voi = vtk.vtkExtractVOI()
        self.sagittal_outline = vtk.vtkOutlineFilter()
        self.sagittal_outline.SetInputConnection(self.sagittal_voi.GetOutputPort())
        sagittal_out_map = vtk.vtkPolyDataMapper()
        sagittal_out_map.SetInputConnection(self.sagittal_outline.GetOutputPort())
        self.sagittal_outline_actor = vtk.vtkActor()
        self.sagittal_outline_actor.SetMapper(sagittal_out_map)
        self.sagittal_outline_actor.GetProperty().SetColor(0.95, 0.8, 0.1) # Yellow outline
        self.sagittal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.sagittal_outline_actor.SetVisibility(False)

        self.mesh_renderer.AddActor(self.axial_3d_actor)
        self.mesh_renderer.AddActor(self.axial_outline_actor)
        self.mesh_renderer.AddActor(self.coronal_3d_actor)
        self.mesh_renderer.AddActor(self.coronal_outline_actor)
        self.mesh_renderer.AddActor(self.sagittal_3d_actor)
        self.mesh_renderer.AddActor(self.sagittal_outline_actor)
        
        self.mesh_actor = None
        self.scalar_actor = None
        self.scalar_bar_actor = None
        self.patient_overlay_actor = None
        self.mesh_frame.hide()

        self.axial_vtkWidget.Initialize()
        self.axial_vtkWidget.GetRenderWindow().Render()
        self.coronal_vtkWidget.Initialize()
        self.coronal_vtkWidget.GetRenderWindow().Render()
        self.sagittal_vtkWidget.Initialize()
        self.sagittal_vtkWidget.GetRenderWindow().Render()
        self.mesh_vtkWidget.Initialize()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def create_view_frame(self, color):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 2px solid {color}; background-color: #1e2230; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return frame
        
    def setup_vtk_with_slider(self, frame, title_str):
        layout = frame.layout()
        
        # Top Title and Maximize Button Toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)
        top_bar.setSpacing(4)
        
        title_lbl = QLabel(title_str)
        title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent;")
        
        plane_3d_btn = QPushButton("🔲 3D")
        plane_3d_btn.setCheckable(True)
        plane_3d_btn.setChecked(False)
        plane_3d_btn.setVisible(False)  # Hidden initially until a mesh file is clicked in FastSurfer
        plane_3d_btn.setToolTip("Show / Hide slice plane in 3D Mesh View")
        plane_3d_btn.setFixedSize(45, 24)
        plane_3d_btn.setStyleSheet("""
            QPushButton {
                background: #2c3e50;
                color: #bdc3c7;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #34495e;
                color: white;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #1f618d;
            }
        """)
        
        flip_btn = QPushButton("⇅")
        flip_btn.setFixedSize(24, 24)
        flip_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        
        max_btn = QPushButton("◻")
        max_btn.setFixedSize(24, 24)
        max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        max_btn.clicked.connect(lambda: self.toggle_maximize(frame))
        
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(plane_3d_btn)
        top_bar.addWidget(flip_btn)
        top_bar.addWidget(max_btn)
        
        layout.addLayout(top_bar)
        
        # Slider and Reset Button Row
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(5, 0, 5, 5)
        slider_layout.setSpacing(6)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setEnabled(False)
        slider.setStyleSheet("""
            QSlider { background: transparent; border: none; }
            QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #333333; margin: 2px 0; }
            QSlider::handle:horizontal { background: #7c7c7c; border: 1px solid #999999; width: 18px; margin: -4px 0; border-radius: 3px; }
            QSlider::handle:horizontal:hover { background: #a0a0a0; }
        """)
        
        slice_lbl = QLabel("Slice: - / -")
        slice_lbl.setStyleSheet("color: white; font-size: 11px; border: none; background: transparent;")
        slice_lbl.setFixedWidth(75)
        slice_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        reset_slice_btn = QPushButton("↺ Reset")
        reset_slice_btn.setToolTip("Reset to initial slice")
        reset_slice_btn.setEnabled(False)
        reset_slice_btn.setFixedSize(58, 22)
        reset_slice_btn.setStyleSheet("""
            QPushButton {
                background: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #415b76;
                color: white;
            }
            QPushButton:pressed {
                background: #2c3e50;
            }
            QPushButton:disabled {
                background: #242735;
                color: #666;
                border: 1px solid #3d414d;
            }
        """)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slice_lbl)
        slider_layout.addWidget(reset_slice_btn)
        layout.addLayout(slider_layout)
        
        # We use our custom widget that ignores left clicks
        vtkWidget = CustomQVTKWidget(frame)
        layout.addWidget(vtkWidget)
        
        return slider, vtkWidget, slice_lbl, flip_btn, plane_3d_btn, reset_slice_btn

    def set_3d_plane_buttons_visible(self, visible: bool):
        self.axial_3d_btn.setVisible(visible)
        self.coronal_3d_btn.setVisible(visible)
        self.sagittal_3d_btn.setVisible(visible)

    def reset_slice(self, orientation):
        if orientation == "axial" and hasattr(self, 'axial_initial_slice'):
            self.axial_slider.setValue(self.axial_initial_slice)
            self.signal_log_message.emit(f"Axial view reset to initial slice {self.axial_initial_slice}.")
        elif orientation == "coronal" and hasattr(self, 'coronal_initial_slice'):
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.signal_log_message.emit(f"Coronal view reset to initial slice {self.coronal_initial_slice}.")
        elif orientation == "sagittal" and hasattr(self, 'sagittal_initial_slice'):
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.signal_log_message.emit(f"Sagittal view reset to initial slice {self.sagittal_initial_slice}.")

    def toggle_3d_plane(self, orientation, visible):
        if visible:
            self.set_mesh_view_visible(True)
        
        if orientation == "axial":
            self.axial_3d_actor.SetVisibility(visible)
            self.axial_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.axial_slider.value()
                dims = self.current_dims
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
        elif orientation == "coronal":
            self.coronal_3d_actor.SetVisibility(visible)
            self.coronal_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.coronal_slider.value()
                dims = self.current_dims
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
        elif orientation == "sagittal":
            self.sagittal_3d_actor.SetVisibility(visible)
            self.sagittal_outline_actor.SetVisibility(visible)
            if visible and hasattr(self, 'current_dims'):
                val = self.sagittal_slider.value()
                dims = self.current_dims
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                
        self.mesh_renderer.ResetCameraClippingRange()
        self.mesh_vtkWidget.GetRenderWindow().Render()
        self.signal_log_message.emit(f"3D Slice Plane ({orientation.capitalize()}) {'shown' if visible else 'hidden'} in 3D View.")

    def clear_all_patient_meshes(self):
        """Clears all patient meshes (single and multi-overlaid) from 3D viewport."""
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        self.clear_multi_mesh_actors()
        self.clear_gradcam_view(render_now=False)
        self.current_mesh_path = None
        self.current_diagnostic_info = None
        self.is_overlay_active = False
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(False)
        self.overlay_cb.blockSignals(False)
        self.update_legend()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def set_diagnostic_info(self, text: str):
        self.current_diagnostic_info = text
        self.update_legend()
        if hasattr(self, 'mesh_vtkWidget'):
            self.mesh_vtkWidget.GetRenderWindow().Render()

    def set_view_mode(self, mode: str, module_name: str = ""):
        """
        Switches between:
        - 'quad': 4-view layout (Axial, Coronal, Sagittal + 3D) for Data Importer & FastSurfer
        - 'full_3d': Single large 3D viewport for ICP Registration & SPHARM Processing
        """
        prev_module = self.current_module_name
        self.view_mode = mode
        self.current_module_name = module_name

        # When switching module, always clear previous module's patient meshes and gradcam actors
        if prev_module and prev_module != module_name:
            if self.mesh_actor is not None:
                self.mesh_renderer.RemoveActor(self.mesh_actor)
                self.mesh_actor = None
            self.clear_multi_mesh_actors()
            self.clear_gradcam_view(render_now=False)
            self.current_mesh_path = None
            self.is_overlay_active = False
            self.overlay_cb.blockSignals(True)
            self.overlay_cb.setChecked(False)
            self.overlay_cb.blockSignals(False)
            self.clear_template_actors()
        
        if mode == "full_3d":
            self.maximized_frame = None
            
            # Hide 2D slice frames
            self.axial_frame.hide()
            self.coronal_frame.hide()
            self.sagittal_frame.hide()
            
            # Reposition mesh_frame to occupy entire 2x2 grid (0, 0, 2, 2)
            self.grid_layout.removeWidget(self.mesh_frame)
            self.grid_layout.addWidget(self.mesh_frame, 0, 0, 2, 2)
            self.mesh_frame.show()
            
            # Hide 3D slice plane buttons on hidden slice frames
            self.set_3d_plane_buttons_visible(False)
            self.mesh_max_btn.setVisible(False)
            
            is_result = ("result" in str(module_name).lower()) or ("main" in str(module_name).lower())
            self.template_cb.setVisible(not is_result)
            self.overlay_cb.setVisible(not is_result)
            if is_result:
                self.template_cb.blockSignals(True)
                self.template_cb.setChecked(False)
                self.template_cb.blockSignals(False)
                self.clear_template_actors()
                self.overlay_cb.blockSignals(True)
                self.overlay_cb.setChecked(False)
                self.overlay_cb.blockSignals(False)
                self.clear_multi_mesh_actors()
            
            # Update title
            if "main" in str(module_name).lower():
                disp = "3D View — SPHARM Shape & Diagnosis"
            elif module_name:
                disp = f"3D View — {module_name}"
            else:
                disp = "3D View"
            self.mesh_title_lbl.setText(disp)
            
            # If template checkbox was checked, refresh overlay for this module (ICP/SPHARM)
            if not is_result and self.template_cb.isChecked():
                self.update_template_overlay()
                
            self.reset_3d_camera()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            
        else: # quad mode
            self.maximized_frame = None
            
            # Move mesh_frame back to cell (0, 1)
            self.grid_layout.removeWidget(self.mesh_frame)
            self.grid_layout.addWidget(self.mesh_frame, 0, 1, 1, 1)
            
            # Show 2D slice frames
            self.axial_frame.show()
            self.coronal_frame.show()
            self.sagittal_frame.show()
            self.mesh_frame.setVisible(self.mesh_view_enabled)
            
            self.mesh_title_lbl.setText("3D Mesh View")
            self.mesh_max_btn.setVisible(True)
            self.template_cb.setVisible(False)
            self.overlay_cb.setVisible(False)
            
            # Remove template actor and multi mesh actors in quad mode unless requested
            self.clear_template_actors()
            if self.multi_mesh_actors:
                self.clear_multi_mesh_actors()
                
            if self.mesh_view_enabled:
                self.set_3d_plane_buttons_visible(True)
                
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()

    def clear_template_actors(self):
        for act in self.template_actors:
            self.mesh_renderer.RemoveActor(act)
        self.template_actors = []
        self.template_actor = None

    def set_side_filter(self, side_filter: str):
        self.current_side_filter = side_filter
        if self.template_cb.isChecked():
            if side_filter in ("rh", "right"):
                chosen_side = "right"
            elif side_filter in ("lh", "left"):
                chosen_side = "left"
            else:
                chosen_side = "all"
            self.update_template_overlay(side=chosen_side)

    def get_template_paths(self, side=None):
        """Returns (candidates_list, resolved_side, mod_type) where candidates_list is a list of (filepath, side_str)"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        templates_dir = os.path.join(project_root, "Templates")
        
        is_spharm = ("spharm" in self.current_module_name.lower()) or (
            self.current_mesh_path and "spharm" in self.current_mesh_path.lower()
        ) or (
            self.multi_mesh_paths and any("spharm" in p.lower() for p in self.multi_mesh_paths[:3])
        )
        mod_type = "SPHARM" if is_spharm else "ICP"
        sub_dir = "SPHARM" if is_spharm else "ICP"
        
        detected_side = None
        if side:
            s_low = str(side).lower()
            if s_low == "all":
                detected_side = "all"
            elif s_low in ("right", "rh") or "right" in s_low or "rh" in s_low:
                detected_side = "right"
            elif s_low in ("left", "lh") or "left" in s_low or "lh" in s_low:
                detected_side = "left"

        # 1. Prioritize active side filter
        if detected_side is None:
            if self.current_side_filter in ("rh", "right"):
                detected_side = "right"
            elif self.current_side_filter in ("lh", "left"):
                detected_side = "left"
            elif self.current_side_filter == "all":
                if self.is_overlay_active or self.mesh_actor is None:
                    detected_side = "all"

        # 2. If still None and a single mesh is loaded, infer from its filename
        if detected_side is None and self.current_mesh_path and not self.is_overlay_active:
            mesh_to_check = self.current_mesh_path
            fp_low = mesh_to_check.replace("\\", "/").lower()
            bn = os.path.basename(mesh_to_check).lower()
            if "rh_" in bn or "_rh." in bn or "_rh_" in bn or "right" in bn or "/right/" in fp_low or "/rh/" in fp_low or "right_hippocampus" in fp_low:
                detected_side = "right"
            elif "lh_" in bn or "_lh." in bn or "_lh_" in bn or "left" in bn or "/left/" in fp_low or "/lh/" in fp_low or "left_hippocampus" in fp_low:
                detected_side = "left"

        # 3. Default fallback
        if detected_side is None:
            detected_side = "all" if self.current_side_filter == "all" else "left"

        def get_cand(s):
            if is_spharm:
                c = os.path.join(templates_dir, sub_dir, f"template_spharm_{s}.vtk")
                if not os.path.isfile(c):
                    c = os.path.join(templates_dir, sub_dir, f"template_spharm_{s}.ply")
            else:
                c = os.path.join(templates_dir, sub_dir, f"template_mean_{s}.vtk")
                if not os.path.isfile(c):
                    c = os.path.join(templates_dir, sub_dir, f"template_mean_{s}.ply")
            return c if os.path.isfile(c) else None

        results = []
        if detected_side == "all":
            left_cand = get_cand("left")
            right_cand = get_cand("right")
            if left_cand:
                results.append((left_cand, "left"))
            if right_cand:
                results.append((right_cand, "right"))
        else:
            cand = get_cand(detected_side)
            if cand:
                results.append((cand, detected_side))

        return results, detected_side, mod_type

    def get_template_path(self, side=None):
        results, detected_side, mod_type = self.get_template_paths(side=side)
        if results:
            return results[0][0], detected_side, mod_type
        return None, detected_side, mod_type

    def on_template_cb_toggled(self, checked: bool):
        self.signal_template_toggled.emit(checked)
        self.update_template_overlay()

    def set_template_visible(self, visible: bool):
        self.template_cb.blockSignals(True)
        self.template_cb.setChecked(visible)
        self.template_cb.blockSignals(False)
        self.update_template_overlay()

    def on_overlay_cb_toggled(self, checked: bool):
        self.signal_overlay_toggled.emit(checked)

    def set_overlay_visible(self, visible: bool):
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(visible)
        self.overlay_cb.blockSignals(False)

    def get_spharm_lh_transform(self):
        """Precomputed Kabsch rigid transform from template_spharm_left to template_spharm_right,
        with 180° view rotation around the horizontal viewing axis so LH SPHARM meshes face
        the canonical upright C-crescent orientation matching RH without altering actual file coordinates."""
        tr = vtk.vtkTransform()
        tr.PostMultiply()
        tr.Translate(0.00260724, 0.05516517, 0.10794069)
        mat = vtk.vtkMatrix4x4()
        R = [
            [ 0.69549754,  0.70316967, -0.14776868],
            [-0.66605966,  0.55378502, -0.49968658],
            [-0.26953237,  0.44595355,  0.85350907]
        ]
        for i in range(3):
            for j in range(3):
                mat.SetElement(i, j, R[i][j])
        tr.Concatenate(mat)
        tr.Translate(-0.05795671, 0.04367124, -0.06888033)

        # 180° rotation around horizontal view axis to correct upside-down & backward orientation
        tr.Translate(-0.039, -0.093, -0.001)
        tr.RotateWXYZ(180, 0.806, 0.0, 0.592)
        tr.Translate(0.039, 0.093, 0.001)
        return tr

    def _on_mesh_key_press(self, obj, event):
        try:
            key = (self.mesh_vtkWidget.GetKeySym() or "").lower()
            is_result = ("result" in str(self.current_module_name).lower()) or (getattr(self, 'scalar_actor', None) is not None)
            if is_result:
                if key in ("right", "bracketright", "period"):
                    self.signal_step_sd.emit(0.1)
                elif key in ("left", "bracketleft", "comma"):
                    self.signal_step_sd.emit(-0.1)
                elif key in ("0", "r", "home", "space"):
                    self.signal_reset_sd.emit()
        except Exception:
            pass

    def _on_mesh_mouse_press(self, obj, event):
        try:
            self.mesh_vtkWidget.setFocus()
        except Exception:
            pass

    def setup_3d_lighting(self):
        """Balanced lighting matching view_gradcam_plsda_top3.py."""
        self.mesh_renderer.RemoveAllLights()

        key_light = vtk.vtkLight()
        key_light.SetLightTypeToCameraLight()
        key_light.SetPosition(0.3, 0.5, 1.0)
        key_light.SetIntensity(0.9)
        self.mesh_renderer.AddLight(key_light)

        fill_light = vtk.vtkLight()
        fill_light.SetLightTypeToCameraLight()
        fill_light.SetPosition(-0.4, -0.3, 0.7)
        fill_light.SetIntensity(0.4)
        self.mesh_renderer.AddLight(fill_light)

    def reset_3d_camera(self, side=None):
        """Sets canonical viewing direction without modifying underlying mesh data.
        For Result Panel Grad-CAM: Upright vertical crescent matching view_gradcam_plsda_top3.py.
        For SPHARM meshes: orients camera to the upright vertical crescent angle (matching auto_rh_front).
        Non-SPHARM meshes (e.g. ICP): maintain their standard coordinate view."""
        self.mesh_renderer.ResetCamera()
        camera = self.mesh_renderer.GetActiveCamera()
        fp = camera.GetFocalPoint()
        dist = camera.GetDistance()
        camera.SetFocalPoint(fp[0], fp[1], fp[2])

        is_result = ("result" in str(self.current_module_name).lower()) or (getattr(self, 'scalar_actor', None) is not None)
        is_spharm = ("spharm" in str(self.current_module_name).lower()) or (
            self.current_mesh_path and "spharm" in self.current_mesh_path.lower()
        ) or (
            self.multi_mesh_paths and any("spharm" in p.lower() for p in self.multi_mesh_paths[:3])
        )

        resolved_side = str(side or getattr(self, 'current_side', None) or getattr(self, 'current_side_filter', None) or "left").lower()

        if is_result:
            # Result Panel Grad-CAM: Canonical upright vertical crescent view matching view_gradcam_plsda_top3.py
            camera.SetParallelProjection(True)
            if "right" in resolved_side or "rh" in resolved_side:
                camera.SetPosition(fp[0], fp[1], fp[2] - dist)
                camera.SetViewUp(1.0, 0.0, 0.0)
            else:
                camera.SetPosition(fp[0], fp[1], fp[2] + dist)
                camera.SetViewUp(-1.0, 0.0, 0.0)
        elif is_spharm:
            camera.SetParallelProjection(False)
            camera.SetPosition(fp[0], fp[1] - dist, fp[2])
            camera.SetViewUp(-0.592, 0.0, 0.806)
        else:
            camera.SetParallelProjection(False)
            camera.SetPosition(fp[0], fp[1] - dist, fp[2])
            camera.SetViewUp(0.0, 0.0, 1.0)

        self.mesh_renderer.ResetCameraClippingRange()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def update_template_overlay(self, side=None):
        if not self.template_cb.isChecked():
            if self.template_actors or self.template_actor is not None:
                self.clear_template_actors()
                self.update_legend()
                self.mesh_vtkWidget.GetRenderWindow().Render()
                self.signal_log_message.emit("Reference template hidden.")
            return

        tmpl_specs, resolved_side, mod_type = self.get_template_paths(side=side)
        if not tmpl_specs:
            self.signal_log_message.emit(f"[WARNING] Reference template not found for {resolved_side} ({mod_type}).")
            return

        # Clear existing template actors first
        self.clear_template_actors()

        is_spharm = ("spharm" in str(self.current_module_name).lower()) or any(
            "spharm" in p[0].lower() for p in tmpl_specs
        )

        for tmpl_path, tmpl_side in tmpl_specs:
            if tmpl_path.endswith(".ply"):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkPolyDataReader()
            reader.SetFileName(tmpl_path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            if is_spharm and tmpl_side == "left":
                actor.SetUserTransform(self.get_spharm_lh_transform())
            
            # Warm Amber / Gold translucent ghost surface
            prop = actor.GetProperty()
            prop.SetColor(0.95, 0.76, 0.20)
            prop.SetOpacity(0.45)
            prop.SetSpecular(0.3)
            prop.SetSpecularPower(20)
            prop.SetInterpolationToPhong()

            self.mesh_renderer.AddActor(actor)
            self.template_actors.append(actor)

        if self.template_actors:
            self.template_actor = self.template_actors[0]

        self.reset_3d_camera(side=resolved_side)
        if resolved_side == "all":
            tmpl_name = f"{mod_type} Left & Right Mean"
        elif resolved_side == "left":
            tmpl_name = f"{mod_type} Left Mean"
        else:
            tmpl_name = f"{mod_type} Right Mean"

        self.update_legend(tmpl_name=tmpl_name)
        self.mesh_vtkWidget.GetRenderWindow().Render()
        names_str = ", ".join([f"{os.path.basename(p)} ({s.upper()})" for p, s in tmpl_specs])
        self.signal_log_message.emit(f"[INFO] Reference template overlaid: {names_str}")

    def clear_multi_mesh_actors(self):
        for act in self.multi_mesh_actors:
            self.mesh_renderer.RemoveActor(act)
        self.multi_mesh_actors = []
        self.multi_mesh_paths = []
        self.is_overlay_active = False

    def display_all_meshes(self, filepaths, side_filter="all"):
        self.current_side_filter = side_filter
        self.current_mesh_path = None
        if not filepaths:
            self.clear_multi_mesh_actors()
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        # Clear existing single mesh actor and gradcam
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        self.clear_multi_mesh_actors()
        self.clear_gradcam_view(render_now=False)
        self.is_overlay_active = True
        self.multi_mesh_paths = filepaths

        # Pleasing soft translucent color palettes
        left_colors = [
            (0.40, 0.70, 0.95), (0.35, 0.60, 0.90), (0.45, 0.80, 0.98),
            (0.50, 0.65, 0.95), (0.30, 0.75, 0.85), (0.55, 0.70, 1.00)
        ]
        right_colors = [
            (0.95, 0.65, 0.50), (0.90, 0.55, 0.45), (0.98, 0.70, 0.55),
            (0.92, 0.60, 0.40), (0.85, 0.50, 0.45), (0.95, 0.75, 0.60)
        ]

        loaded_count = 0
        for i, fp in enumerate(filepaths):
            if not (fp.endswith(".vtk") or fp.endswith(".ply") or fp.endswith(".nii.gz") or fp.endswith(".mgz") or fp.endswith(".nii")):
                continue
            if fp.endswith(".ply"):
                r = vtk.vtkPLYReader()
                r.SetFileName(fp)
                r.Update()
                output_port = r.GetOutputPort()
            elif fp.endswith(".vtk"):
                r = vtk.vtkPolyDataReader()
                r.SetFileName(fp)
                r.Update()
                output_port = r.GetOutputPort()
            else:
                r = vtk.vtkNIFTIImageReader()
                r.SetFileName(fp)
                r.Update()
                mc = vtk.vtkMarchingCubes()
                mc.SetInputConnection(r.GetOutputPort())
                mc.SetValue(0, 0.5)
                sm = vtk.vtkWindowedSincPolyDataFilter()
                sm.SetInputConnection(mc.GetOutputPort())
                sm.SetNumberOfIterations(15)
                sm.BoundarySmoothingOff()
                sm.FeatureEdgeSmoothingOff()
                sm.SetPassBand(0.1)
                sm.NonManifoldSmoothingOn()
                sm.NormalizeCoordinatesOn()
                sm.Update()
                output_port = sm.GetOutputPort()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(output_port)
            mapper.ScalarVisibilityOff()

            act = vtk.vtkActor()
            act.SetMapper(mapper)

            fp_low = fp.replace("\\", "/").lower()
            bn = os.path.basename(fp).lower()
            is_rh = ("rh" in bn or "right" in bn or "/right/" in fp_low or side_filter == "rh")
            if is_rh:
                c = right_colors[i % len(right_colors)]
            else:
                c = left_colors[i % len(left_colors)]

            is_spharm = ("spharm" in str(self.current_module_name).lower()) or any(
                "spharm" in p.lower() for p in filepaths[:3]
            )
            if is_spharm and not is_rh:
                act.SetUserTransform(self.get_spharm_lh_transform())

            act.GetProperty().SetColor(*c)
            act.GetProperty().SetOpacity(0.38)
            act.GetProperty().SetSpecular(0.25)
            act.GetProperty().SetSpecularPower(15)
            act.GetProperty().SetInterpolationToPhong()

            self.mesh_renderer.AddActor(act)
            self.multi_mesh_actors.append(act)
            loaded_count += 1

        if side_filter in ("rh", "right"):
            chosen_side = "right"
        elif side_filter in ("lh", "left"):
            chosen_side = "left"
        else:
            chosen_side = "all"

        # If template overlay is enabled, update it
        if self.template_cb.isChecked():
            self.update_template_overlay(side=chosen_side)

        self.reset_3d_camera(side=chosen_side)
        self.update_legend()
        self.mesh_vtkWidget.GetRenderWindow().Render()
        self.signal_log_message.emit(f"[INFO] Superimposed {loaded_count} meshes in 3D view (Filter: {side_filter.upper()}).")

    def update_legend(self, tmpl_name=None):
        parts = []
        if self.is_overlay_active and self.multi_mesh_actors:
            side_str = "Right" if self.current_side_filter == "rh" else ("Left" if self.current_side_filter == "lh" else "All")
            parts.append(f'<span style="color: #1abc9c; font-weight: bold;">&#9679; Overlaid Meshes: {len(self.multi_mesh_actors)} ({side_str})</span>')
        elif self.current_mesh_path and self.mesh_actor:
            m_name = os.path.basename(self.current_mesh_path)
            if len(m_name) > 36:
                m_name = m_name[:16] + "..." + m_name[-16:]
            parts.append(f'<span style="color: #79c0ff; font-weight: bold;">&#9679; Patient: {m_name}</span>')
            
        if self.template_cb.isChecked() and (self.template_actors or self.template_actor is not None):
            if not tmpl_name:
                _, side, mod_type = self.get_template_paths()
                if side == "all":
                    tmpl_name = f"{mod_type} Left & Right Mean"
                elif side == "left":
                    tmpl_name = f"{mod_type} Left Mean"
                else:
                    tmpl_name = f"{mod_type} Right Mean"
            parts.append(f'<span style="color: #f1c40f; font-weight: bold;">&#9679; Template: {tmpl_name} (Ghost)</span>')
            
        if getattr(self, 'current_diagnostic_info', None):
            parts.append(self.current_diagnostic_info)
            
        if parts:
            self.mesh_legend_lbl.setText("  |  ".join(parts))
            self.mesh_legend_lbl.setVisible(True)
        else:
            self.mesh_legend_lbl.setText("")
            self.mesh_legend_lbl.setVisible(False)

    def set_mesh_view_visible(self, visible):
        self.mesh_view_enabled = visible
        if self.view_mode != "full_3d" and self.maximized_frame is None:
            self.mesh_frame.setVisible(visible)

    def toggle_maximize(self, frame):
        if self.view_mode == "full_3d":
            # Already full view in full_3d mode
            return

        frames = [self.axial_frame, self.coronal_frame, self.sagittal_frame, self.mesh_frame]
        
        if self.maximized_frame is None:
            # Maximize the selected frame
            self.maximized_frame = frame
            for f in frames:
                if f != frame:
                    f.hide()
        else:
            # Restore all frames
            self.maximized_frame = None
            for f in frames:
                if f == self.mesh_frame:
                    f.setVisible(self.mesh_view_enabled)
                else:
                    f.show()

    def toggle_flip(self, viewer):
        camera = viewer.GetRenderer().GetActiveCamera()
        vu = camera.GetViewUp()
        camera.SetViewUp(-vu[0], -vu[1], -vu[2])
        viewer.Render()

    def change_slice(self, viewer, val, slider, slice_lbl):
        viewer.SetSlice(val)
        min_val = viewer.GetSliceMin()
        max_val = viewer.GetSliceMax()
        slice_lbl.setText(f"Slice: {val} / {max_val}")
        
        # Sync mask actors' display extent
        extent = viewer.GetImageActor().GetDisplayExtent()
        
        if hasattr(self, 'mask_actors'):
            for orientation_actors in self.mask_actors:
                if orientation_actors['viewer'] == viewer:
                    for actor in orientation_actors['actors']:
                        if actor:
                            actor.SetDisplayExtent(extent)

        # Sync 3D slice plane
        if hasattr(self, 'current_dims'):
            dims = self.current_dims
            if viewer == self.axial_viewer:
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
                if self.axial_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.coronal_viewer:
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
                if self.coronal_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.sagittal_viewer:
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                if self.sagittal_3d_actor.GetVisibility():
                    self.mesh_vtkWidget.GetRenderWindow().Render()

    def display_mesh(self, filepath, side_filter="all"):
        if not filepath:
            if self.mesh_actor is not None:
                self.mesh_renderer.RemoveActor(self.mesh_actor)
                self.mesh_actor = None
            if self.multi_mesh_actors:
                self.clear_multi_mesh_actors()
            self.current_mesh_path = None
            if self.template_cb.isChecked():
                self.update_template_overlay()
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        if not (filepath.endswith(".nii.gz") or filepath.endswith(".mgz") or filepath.endswith(".vtk") or filepath.endswith(".ply")):
            self.signal_log_message.emit("[ERROR] Unsupported mesh format. Expected .vtk, .ply, .nii.gz, or .mgz")
            return
            
        self.current_mesh_path = filepath
        self.current_side_filter = side_filter

        if self.view_mode != "full_3d":
            self.set_mesh_view_visible(True)
            self.set_3d_plane_buttons_visible(True)
        else:
            self.set_3d_plane_buttons_visible(False)
        
        # Clear existing mesh actor only (preserve 3D slice plane actors and template actor)
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        self.clear_gradcam_view(render_now=False)

        if self.multi_mesh_actors:
            self.clear_multi_mesh_actors()
            self.overlay_cb.blockSignals(True)
            self.overlay_cb.setChecked(False)
            self.overlay_cb.blockSignals(False)
        
        if filepath.endswith(".vtk") or filepath.endswith(".ply"):
            if filepath.endswith(".ply"):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Soft tint based on hemisphere
            fp_low = filepath.replace("\\", "/").lower()
            bn = os.path.basename(filepath).lower()
            is_rh = ("rh" in bn or "right" in bn or "/right/" in fp_low or side_filter == "rh")
            is_lh = ("lh" in bn or "left" in bn or "/left/" in fp_low or side_filter == "lh")
            
            is_spharm = ("spharm" in str(self.current_module_name).lower()) or ("spharm" in fp_low)
            if is_spharm and is_lh:
                actor.SetUserTransform(self.get_spharm_lh_transform())

            if is_rh:
                actor.GetProperty().SetColor(0.95, 0.65, 0.55) # Soft Coral
            elif is_lh:
                actor.GetProperty().SetColor(0.55, 0.75, 0.95) # Soft Blue
            else:
                actor.GetProperty().SetColor(0.72, 0.82, 0.93) # Cyan
                
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetSpecular(0.25)
            actor.GetProperty().SetSpecularPower(15)
            actor.GetProperty().SetInterpolationToPhong()
            
            self.mesh_actor = actor
            self.mesh_renderer.AddActor(self.mesh_actor)
            
            # If template overlay is enabled, refresh it to align with this mesh's side
            chosen_side = "right" if is_rh else ("left" if is_lh else None)
            if self.template_cb.isChecked():
                self.update_template_overlay(side=chosen_side)
                
            self.reset_3d_camera(side=chosen_side)
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        # Load NIFTI mask (FastSurfer)
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        # Extract surface
        mc2 = vtk.vtkMarchingCubes()
        mc2.SetInputConnection(reader.GetOutputPort())
        mc2.SetValue(0, 0.5)
        
        # Smooth surface
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc2.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(15)
        actor.GetProperty().SetInterpolationToPhong()
        
        self.mesh_actor = actor
        self.mesh_renderer.AddActor(self.mesh_actor)
        self.reset_3d_camera()
        self.update_legend()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def reset_camera(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            if viewer.GetInput():
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()
        self.reset_3d_camera()
        self.signal_log_message.emit("Camera reset to original position.")

    def display_segmentation_overlays(self, base_img_path, lh_mask_path, rh_mask_path, side_filter="all"):
        filter_text = "ALL (Left & Right)" if side_filter == "all" else ("LEFT (LH)" if side_filter == "lh" else "RIGHT (RH)")
        self.signal_log_message.emit(f"Loading 2D MRI with {filter_text} Hippocampus overlays...")
        
        # Load the base image first
        self.display_subject(base_img_path)
        
        # Remove old mask actors if they exist
        if hasattr(self, 'mask_actors'):
            for orientation in self.mask_actors:
                for actor in orientation['actors']:
                    if actor:
                        orientation['viewer'].GetRenderer().RemoveActor(actor)
        
        self.mask_actors = []
        
        # For each orientation, we need unique actors (because an actor can only be in one renderer)
        viewers = [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]
        
        for viewer in viewers:
            actors = []
            
            # Left Hippocampus: #A7BCE9 (167, 188, 233)
            if side_filter in ("all", "lh") and lh_mask_path and os.path.exists(lh_mask_path):
                lh_actor = create_mask_actor(lh_mask_path, 167, 188, 233, alpha=0.55)
                if lh_actor:
                    viewer.GetRenderer().AddActor(lh_actor)
                    actors.append(lh_actor)
                    
            # Right Hippocampus: #FFA87B (255, 168, 123)
            if side_filter in ("all", "rh") and rh_mask_path and os.path.exists(rh_mask_path):
                rh_actor = create_mask_actor(rh_mask_path, 255, 168, 123, alpha=0.55)
                if rh_actor:
                    viewer.GetRenderer().AddActor(rh_actor)
                    actors.append(rh_actor)
                
            self.mask_actors.append({
                'viewer': viewer,
                'actors': actors
            })
            
            # Initialize their display extent to match the current slice
            extent = viewer.GetImageActor().GetDisplayExtent()
            for actor in actors:
                actor.SetDisplayExtent(extent)
            
            viewer.Render()

    def display_subject(self, filepath):
        filename = os.path.basename(filepath)
        ext = filepath.lower()
        
        try:
            if ext.endswith('.nii') or ext.endswith('.nii.gz'):
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mgz'):
                # Convert mgz to temp nii.gz for VTK reader
                import nibabel as nib
                import numpy as np
                temp_nii = filepath.replace('.mgz', '_conformed.nii.gz')
                if not os.path.exists(temp_nii):
                    img = nib.load(filepath)
                    nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                    nib.save(nii_img, temp_nii)
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(temp_nii)
            elif ext.endswith('.nrrd'):
                reader = vtk.vtkNrrdReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mha'):
                reader = vtk.vtkMetaImageReader()
                reader.SetFileName(filepath)
            else:
                self.signal_log_message.emit(f"Unsupported image format for slice viewer: {filepath}.")
                return
            reader.Update()
            
            image_data = reader.GetOutput()
            if not image_data or image_data.GetDimensions()[0] == 0:
                self.signal_log_message.emit(f"Failed to load image data from {filename}")
                return
                
            # Connect image to viewers
            self.axial_viewer.SetInputData(image_data)
            self.coronal_viewer.SetInputData(image_data)
            self.sagittal_viewer.SetInputData(image_data)
            
            dims = image_data.GetDimensions()
            self.current_dims = dims
            
            # Setup Sliders and Reset Buttons
            self.axial_slider.setEnabled(True)
            self.axial_slider.setRange(0, dims[2] - 1)
            self.axial_initial_slice = dims[2] // 2
            self.axial_slider.setValue(self.axial_initial_slice)
            self.axial_slice_lbl.setText(f"Slice: {self.axial_initial_slice} / {dims[2] - 1}")
            self.axial_reset_btn.setEnabled(True)
            
            self.coronal_slider.setEnabled(True)
            self.coronal_slider.setRange(0, dims[1] - 1)
            self.coronal_initial_slice = dims[1] // 2
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.coronal_slice_lbl.setText(f"Slice: {self.coronal_initial_slice} / {dims[1] - 1}")
            self.coronal_reset_btn.setEnabled(True)
            
            self.sagittal_slider.setEnabled(True)
            self.sagittal_slider.setRange(0, dims[0] - 1)
            self.sagittal_initial_slice = dims[0] // 2
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.sagittal_slice_lbl.setText(f"Slice: {self.sagittal_initial_slice} / {dims[0] - 1}")
            self.sagittal_reset_btn.setEnabled(True)
            
            # Auto window/level based on scalar range
            scalar_range = image_data.GetScalarRange()
            window = scalar_range[1] - scalar_range[0]
            level = (scalar_range[0] + scalar_range[1]) / 2.0

            # Connect 3D slice plane actors & VOIs
            self.axial_3d_actor.GetMapper().SetInputData(image_data)
            self.axial_3d_actor.SetDisplayExtent(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)
            self.axial_voi.SetInputData(image_data)
            self.axial_voi.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)

            self.coronal_3d_actor.GetMapper().SetInputData(image_data)
            self.coronal_3d_actor.SetDisplayExtent(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)
            self.coronal_voi.SetInputData(image_data)
            self.coronal_voi.SetVOI(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)

            self.sagittal_3d_actor.GetMapper().SetInputData(image_data)
            self.sagittal_3d_actor.SetDisplayExtent(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)
            self.sagittal_voi.SetInputData(image_data)
            self.sagittal_voi.SetVOI(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)

            for act_3d in [self.axial_3d_actor, self.coronal_3d_actor, self.sagittal_3d_actor]:
                act_3d.GetProperty().SetColorWindow(window)
                act_3d.GetProperty().SetColorLevel(level)
                act_3d.InterpolateOn()
            
            # Explicitly set the flipped ViewUp by default (as it usually loads upside down)
            self.axial_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, -1, 0)
            self.coronal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            self.sagittal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            
            # Update viewers
            for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
                viewer.GetImageActor().SetVisibility(True)
                viewer.SetColorWindow(window)
                viewer.SetColorLevel(level)
                
                # Make sure the image acts correctly 
                viewer.GetImageActor().InterpolateOn()
                
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()

            if any([self.axial_3d_actor.GetVisibility(), self.coronal_3d_actor.GetVisibility(), self.sagittal_3d_actor.GetVisibility()]):
                self.mesh_vtkWidget.GetRenderWindow().Render()
                
            self.signal_log_message.emit(f"Loaded MRI volume: {filename}")
            
        except Exception as e:
            self.signal_log_message.emit(f"Error loading volume: {str(e)}")

    # =========================================================================
    # Grad-CAM and Deformation Scalar Heatmap Engine
    # =========================================================================
    def build_lut_gradcam(self):
        """Jet colormap matching view_gradcam_plsda_top3.py (0.0 Blue to 1.0 Red)."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(0.0, 1.0)
        lut.Build()
        for i in range(256):
            t = i / 255.0
            r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
        return lut

    def build_lut_signed_distance(self, max_val=0.16):
        """Diverging Blue-White-Red colormap (-max to +max mm) matching view_gradcam_plsda_top3.py."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(-max_val, max_val)
        lut.Build()
        for i in range(256):
            t = (i / 255.0) * 2.0 - 1.0
            if t < 0:
                frac = 1.0 + t
                r, g, b = frac, frac, 1.0
            else:
                frac = 1.0 - t
                r, g, b = 1.0, frac, frac
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
        return lut

    def build_lut_distance_mapping(self, max_val=0.16):
        """Jet colormap matching view_gradcam_plsda_top3.py (0.0 Blue to max mm Red)."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(0.0, max_val)
        lut.Build()
        for i in range(256):
            t = i / 255.0
            r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
        return lut

    def clear_gradcam_view(self, render_now=True):
        if hasattr(self, 'scalar_actor') and self.scalar_actor:
            self.mesh_renderer.RemoveActor(self.scalar_actor)
            self.scalar_actor = None
        if hasattr(self, 'scalar_mapper'):
            self.scalar_mapper = None
        if hasattr(self, 'scalar_bar_actor') and self.scalar_bar_actor:
            self.mesh_renderer.RemoveActor(self.scalar_bar_actor)
            self.scalar_bar_actor = None
        if hasattr(self, 'patient_overlay_actor') and self.patient_overlay_actor:
            self.mesh_renderer.RemoveActor(self.patient_overlay_actor)
            self.patient_overlay_actor = None
        self._last_gradcam_side = None
        if render_now:
            self.mesh_vtkWidget.GetRenderWindow().Render()

    def display_gradcam_mesh(self, mesh_path, scalar_mode="DistanceMapping", lut_type="distance_mapping", title="Distance Mapping", side="left", opacity=1.0):
        if not mesh_path or not os.path.isfile(mesh_path):
            self.signal_log_message.emit(f"[ERROR] Mesh file not found: {mesh_path}")
            return

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            self.signal_log_message.emit(f"[ERROR] Invalid polydata in: {mesh_path}")
            return

        # Check scalar array in PointData
        pdata = poly.GetPointData()
        arr = pdata.GetArray(scalar_mode)
        if arr is not None:
            pdata.SetActiveScalars(scalar_mode)
            s_range = arr.GetRange()
        else:
            if pdata.GetNumberOfArrays() > 0:
                first_name = pdata.GetArrayName(0)
                pdata.SetActiveScalars(first_name)
                s_range = pdata.GetScalars().GetRange()
                scalar_mode = first_name
            else:
                s_range = (0.0, 1.0)

        # Build appropriate LUT based on mode
        if lut_type == "signed_distance":
            max_abs = max(0.16, abs(s_range[0]), abs(s_range[1]))
            lut = self.build_lut_signed_distance(max_val=max_abs)
            s_min, s_max = -max_abs, max_abs
            bar_title = "mm"
        elif lut_type == "distance_mapping" or "Distance" in scalar_mode:
            max_val = max(0.16, s_range[1])
            lut = self.build_lut_distance_mapping(max_val=max_val)
            s_min, s_max = 0.0, max_val
            bar_title = "mm"
        else:
            lut = self.build_lut_gradcam()
            s_min, s_max = 0.0, 1.0
            bar_title = "Score"

        # Update title text on top bar
        self.mesh_title_lbl.setText(f"3D View — {title}")

        # Check if we can perform smooth in-place update (SAME side, existing actors)
        is_same_side = (getattr(self, '_last_gradcam_side', None) == side)
        can_update_in_place = (
            hasattr(self, 'scalar_actor') and self.scalar_actor is not None and
            hasattr(self, 'scalar_mapper') and self.scalar_mapper is not None and
            hasattr(self, 'scalar_bar_actor') and self.scalar_bar_actor is not None and
            is_same_side
        )

        if can_update_in_place:
            self.scalar_mapper.SetInputData(poly)
            self.scalar_mapper.SetScalarRange(s_min, s_max)
            self.scalar_mapper.SetLookupTable(lut)
            self.scalar_mapper.SelectColorArray(scalar_mode)
            self.scalar_bar_actor.SetLookupTable(lut)
            self.scalar_bar_actor.SetTitle(bar_title)
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        # Otherwise full setup (first time or when switching hemisphere side)
        self.clear_gradcam_view(render_now=False)
        self.clear_all_patient_meshes()
        self.clear_template_actors()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarRange(s_min, s_max)
        mapper.SetLookupTable(lut)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray(scalar_mode)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # SPHARM LH canonical orientation transform if needed
        is_spharm = "spharm" in str(self.current_module_name).lower()
        if is_spharm and side.lower() == "left":
            actor.SetUserTransform(self.get_spharm_lh_transform())

        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(0.20)
        actor.GetProperty().SetDiffuse(0.80)
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(30)
        actor.GetProperty().SetInterpolationToGouraud()

        self.mesh_renderer.AddActor(actor)
        self.scalar_actor = actor
        self.scalar_mapper = mapper
        self._last_gradcam_side = side

        # Scalar Bar (Colorbar Legend)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle(bar_title)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.86, 0.12)
        scalar_bar.SetWidth(0.11)
        scalar_bar.SetHeight(0.55)
        tprop = scalar_bar.GetTitleTextProperty()
        tprop.SetColor(1.0, 1.0, 1.0)
        tprop.SetFontSize(11)
        tprop.BoldOn()
        lprop = scalar_bar.GetLabelTextProperty()
        lprop.SetColor(0.9, 0.9, 0.9)
        lprop.SetFontSize(10)
        
        self.mesh_renderer.AddActor(scalar_bar)
        self.scalar_bar_actor = scalar_bar

        self.reset_3d_camera(side=side)
        self.mesh_vtkWidget.GetRenderWindow().Render()
        self.signal_log_message.emit(f"SUCCESS: Rendered {title} on 3D mesh ({side.upper()})")

    def set_patient_overlay(self, mesh_path, visible=True, opacity=0.35, side="left"):
        if hasattr(self, 'patient_overlay_actor') and self.patient_overlay_actor:
            self.mesh_renderer.RemoveActor(self.patient_overlay_actor)
            self.patient_overlay_actor = None

        if not visible or not mesh_path or not os.path.isfile(mesh_path):
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            return

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        is_spharm = "spharm" in str(self.current_module_name).lower()
        if is_spharm and side.lower() == "left":
            actor.SetUserTransform(self.get_spharm_lh_transform())

        # Patient mesh: Cyan / Teal ghost outline / surface
        actor.GetProperty().SetColor(0.2, 0.85, 0.95)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetInterpolationToPhong()

        self.mesh_renderer.AddActor(actor)
        self.patient_overlay_actor = actor
        self.mesh_vtkWidget.GetRenderWindow().Render()
        self.signal_log_message.emit(f"Overlaid patient mesh ({os.path.basename(mesh_path)}) in 3D View.")
