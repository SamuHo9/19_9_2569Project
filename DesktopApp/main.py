import sys
import os
import time
import multiprocessing
import warnings

# Suppress harmless scikit-learn unpickle version and feature name warnings globally
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# CLI Runner Dispatch: Allows the frozen .exe to execute Python sub-scripts
# without requiring Python to be installed on the host machine.
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.py') or sys.argv[1] == "-m"):
        if sys.argv[1] == "-m" and len(sys.argv) > 2:
            module_name = sys.argv[2]
            sys.argv = [sys.argv[2]] + sys.argv[3:]
            import runpy
            runpy.run_module(module_name, run_name="__main__")
            sys.exit(0)
        elif sys.argv[1].endswith('.py'):
            script_path = os.path.abspath(sys.argv[1])
            sys.argv = sys.argv[1:]
            script_dir = os.path.dirname(script_path)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            import runpy
            runpy.run_path(script_path, run_name="__main__")
            sys.exit(0)

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QPen
from PyQt6.QtCore import Qt, QLocale

from ui_main_window import MainWindow
import vtk

if __name__ == "__main__":
    # Suppress annoying VTK OpenGL context warnings on Windows (wglMakeCurrent error 6)
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set default locale to C/English to force standard Arabic numerals (1000, 10, 12)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    
    # ----------------------------------------------------
    # Splash Screen (Slicer/SlicerSALT style startup)
    # ----------------------------------------------------
    # Create a dynamic pixmap for the splash screen
    pixmap = QPixmap(600, 350)
    pixmap.fill(QColor("#2c3e50")) # Dark blue Slicer-style background
    
    # Draw a simple logo/title directly on the splash image
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white")))
    painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Shape Analysis Toolbox\nInitializing...")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Arial", 11, QFont.Weight.Bold))
    splash.show()
    
    # Simulate loading steps with visual feedback
    splash.showMessage("Starting Hippocampal Shape Analysis Pipeline...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Loading VTK Rendering Engine...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Checking System Requirements & SlicerSALT...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.6)
    
    splash.showMessage("Initializing User Interface Modules...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    time.sleep(0.4)
    
    # Initialize the heavy main window
    window = MainWindow()
    
    # Close splash and show main window maximized
    splash.finish(window)
    window.showMaximized()
    
    sys.exit(app.exec())
