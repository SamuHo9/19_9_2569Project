param(
    [string]$PythonSlicer = 'C:\Program Files\SlicerSALT 6.0.0\bin\PythonSlicer.exe',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$modelRoot = $PSScriptRoot
$cudaSite = Join-Path $modelRoot '_training_cuda_site'
$cpuSite = Join-Path $modelRoot '_training_site'
$env:PYTHONPATH = "$cudaSite;$cpuSite"
$env:MPLCONFIGDIR = Join-Path $modelRoot '_mplconfig'
$runner = Join-Path $modelRoot 'leakage_free_pointnet_plsda_training.py'

& $PythonSlicer $runner @RunnerArgs
exit $LASTEXITCODE
