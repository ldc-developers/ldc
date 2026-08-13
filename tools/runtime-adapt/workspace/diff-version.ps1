# Diff reference druntime/phobos at a tag against this LDC tree and iterate
# toward equivalence using principles.d / adapt.d (not a hand-copied overlay).
param(
    [Parameter(Mandatory = $true)][string] $Version,
    [string] $Against = '',
    [string] $Compiler = 'ldc2'
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tool = Split-Path -Parent $here
Set-Location $tool
$args = @('--diff-version', $Version, '--iterate')
if ($Against) { $args += @('--target', $Against) }
Write-Host "dub run -- $args"
dub run --compiler=$Compiler -- @args
exit $LASTEXITCODE
