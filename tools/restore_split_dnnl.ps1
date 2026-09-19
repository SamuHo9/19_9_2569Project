[CmdletBinding()]
param(
    [string]$PartDirectory = (Join-Path $PSScriptRoot '..\Model\_training_cuda_site\torch\lib')
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$target = Join-Path $PartDirectory 'dnnl.lib'
$parts = @(
    (Join-Path $PartDirectory 'dnnl.lib.part01'),
    (Join-Path $PartDirectory 'dnnl.lib.part02')
)
$expectedSha256 = 'b6ffe087a5ee14a206c56ddaa37380bcdcf8ec4ba7c75aec895cb895d306f05e'
foreach ($part in $parts) {
    if (-not (Test-Path -LiteralPath $part -PathType Leaf)) {
        throw "Missing split file: $part"
    }
}
$temp = "$target.reassembled.tmp"
$buffer = New-Object byte[] (8MB)
$output = [IO.File]::Create($temp)
try {
    foreach ($part in $parts) {
        $input = [IO.File]::OpenRead($part)
        try {
            while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $output.Write($buffer, 0, $read)
            }
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $temp).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    throw "SHA-256 mismatch. Expected $expectedSha256, got $actualSha256"
}
Move-Item -LiteralPath $temp -Destination $target -Force
Write-Output "Restored $target"
Write-Output "SHA256=$actualSha256"
