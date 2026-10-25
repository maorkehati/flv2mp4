<#
.SYNOPSIS
  Download and verify the pinned third-party tools listed in third_party\VERSIONS.json, then create the venv.

.DESCRIPTION
  - Video2X 6.4.0 Windows CLI release (includes the Real-ESRGAN realesr-animevideov3 ncnn models)
  - FFmpeg/FFprobe 9.0.2 (gyan.dev essentials build)
  - optional: Video2X source checkout at tag 6.4.0 (reference only; not needed to run)
  - .venv with requirements.txt
  Every download is checked against the SHA-256 in VERSIONS.json; model files are checked too.
  Existing downloads/extractions are reused, nothing is deleted.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup_tools.ps1 -SkipSource -SkipVenv
#>
param([switch]$SkipSource, [switch]$SkipVenv, [string]$Python = "python")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$tp = Join-Path $root "third_party"
$manifest = Get-Content (Join-Path $tp "VERSIONS.json") -Raw | ConvertFrom-Json
$dl = Join-Path $tp "downloads"
New-Item -ItemType Directory -Force $dl | Out-Null

function Get-Pinned([string]$Url, [string]$Sha256, [string]$Dest) {
    if (-not (Test-Path $Dest)) {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
    $hash = (Get-FileHash $Dest -Algorithm SHA256).Hash
    if ($hash -ne $Sha256.ToUpper()) { throw "SHA-256 mismatch for ${Dest}: $hash (expected $Sha256)" }
    Write-Host "verified $(Split-Path -Leaf $Dest)  $hash"
}

# Video2X release
$v = $manifest.video2x
$vzip = Join-Path $dl ("video2x-" + $v.release + "-windows-amd64.zip")
Get-Pinned $v.asset $v.asset_sha256 $vzip
$vdir = Join-Path $root $v.install_dir
if (-not (Test-Path (Join-Path $vdir "video2x.exe"))) { Expand-Archive $vzip -DestinationPath $vdir -Force }

# FFmpeg / FFprobe
$f = $manifest.ffmpeg
$fzip = Join-Path $dl (Split-Path -Leaf $f.asset)
Get-Pinned $f.asset $f.asset_sha256 $fzip
$fbin = Join-Path $root $f.install_dir
if (-not (Test-Path (Join-Path $fbin "ffmpeg.exe"))) { Expand-Archive $fzip -DestinationPath $tp -Force }

# Model files shipped with Video2X
foreach ($name in "realesr-animevideov3-x2", "realesr-animevideov3-x3", "realesr-animevideov3-x4") {
    $want = $manifest.models.$name
    foreach ($ext in "param", "bin") {
        $file = Join-Path $vdir ("models\realesrgan\" + $name + "." + $ext)
        $hash = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $want.$ext) { throw "Model checksum mismatch for ${file}: $hash (expected $($want.$ext))" }
    }
    Write-Host "verified model $name"
}

# Optional reference checkout of the Video2X source at the pinned tag
if (-not $SkipSource) {
    $src = Join-Path $root $v.source_dir
    if (-not (Test-Path $src)) {
        git clone --branch $v.source_tag --depth 1 $v.source_repository $src
    }
    $head = (git -C $src rev-parse HEAD).Trim()
    if ($head -ne $v.source_commit) { throw "Video2X source is at $head, expected $($v.source_commit)" }
    Write-Host "verified Video2X source $head"
}

# Python environment
if (-not $SkipVenv) {
    $venvPy = Join-Path $root ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { & $Python -m venv (Join-Path $root ".venv") }
    & $venvPy -m pip install -r (Join-Path $root "requirements.txt")
}
Write-Host "Done. Next: .\.venv\Scripts\python.exe enhance_animation.py doctor"
