@echo off
setlocal
set "SLICER_EXE=C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
if defined HIPPO_SLICER_EXE set "SLICER_EXE=%HIPPO_SLICER_EXE%"
if not exist "%SLICER_EXE%" (
    echo [ERROR] SlicerSALT not found: "%SLICER_EXE%"
    echo Set HIPPO_SLICER_EXE to the installed SlicerSALT.exe path.
    exit /b 1
)
if /I "%~1"=="--check" goto check
if /I "%~1"=="left" goto run
if /I "%~1"=="right" goto run
echo Usage: run_spharm.bat left ^| right ^| --check
exit /b 2

:check
"%SLICER_EXE%" --no-main-window --no-splash --disable-scripted-loadable-modules --python-script "%~dp0check_spharm_environment.py"
exit /b %errorlevel%

:run
set "SIDE=%~1"
for %%I in ("%SLICER_EXE%") do set "SLICER_ROOT=%%~dpI"
set "PYTHON_SLICER=%SLICER_ROOT%bin\PythonSlicer.exe"
if not exist "%PYTHON_SLICER%" (
    echo [ERROR] PythonSlicer not found: "%PYTHON_SLICER%"
    echo         Run through SlicerSALT; regular Python does not provide the slicer module.
    exit /b 1
)
echo [INFO] Running shared production SPHARM pipeline for %SIDE% hippocampus
echo [INFO] Input:  %~dp0..\ICP\output_%SIDE%_hippocampus\aligned_nifti
echo [INFO] Output: %~dp0..\ICP\output_%SIDE%_hippocampus\spharm_results
"%PYTHON_SLICER%" "%~dp0run_spharm_parallel.py" ^
  --slicer_exe "%SLICER_EXE%" ^
  --num_workers 5 ^
  --input_dir "%~dp0..\ICP\output_%SIDE%_hippocampus\aligned_nifti" ^
  --output_dir "%~dp0..\ICP\output_%SIDE%_hippocampus" ^
  --reference_template "%~dp0..\Templates\SPHARM\template_spharm_%SIDE%.vtk"
exit /b %errorlevel%
