# Create detached worktrees of this LDC repo at v1.30.0 … v1.42.0
# under workspace/refs/<tag>, then init runtime/phobos at the tag pin.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tool = Split-Path -Parent $here
$ldc = Split-Path -Parent (Split-Path -Parent $tool)
$refs = Join-Path $here 'refs'
New-Item -ItemType Directory -Force -Path $refs | Out-Null

30..42 | ForEach-Object {
    $tag = "v1.$_.0"
    $dest = Join-Path $refs $tag
    $obj = Join-Path $dest 'runtime\druntime\src\object.d'
    if (Test-Path $obj) {
        Write-Host "have $tag"
        return
    }
    Write-Host "worktree $tag -> $dest"
    git -C $ldc worktree add --detach $dest $tag
    if ($LASTEXITCODE -ne 0) { throw "worktree add $tag failed" }
    git -C $dest submodule update --init runtime/druntime runtime/phobos
    if (-not (Test-Path $obj)) { throw "no object.d after $tag" }
}
Write-Host "done"
