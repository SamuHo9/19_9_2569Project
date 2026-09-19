@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Starting Parallel SPHARM-PDM Pipeline (Right Hippocampus)
echo ============================================================
call SPHARM\run_spharm.bat right
exit /b %errorlevel%
