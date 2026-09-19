@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Starting Parallel SPHARM-PDM Pipeline (Left Hippocampus)
echo ============================================================
call SPHARM\run_spharm.bat left
exit /b %errorlevel%
