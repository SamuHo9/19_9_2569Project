@echo off
title 3D Interactive ICP Comparison (Before vs After)
set HTML_PATH=%~dp0Visualize\3D_Mesh_Viewers\compare_icp_3d.html
echo =====================================================================
echo  Opening 3D ICP Comparison (Before vs After) in your default browser...
echo =====================================================================
start "" "%HTML_PATH%"
