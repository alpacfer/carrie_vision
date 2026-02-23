param(
    [switch]$Activate
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$requirementsFile = Join-Path $projectRoot "requirements.txt"

if (-not (Test-Path $requirementsFile)) {
    throw "requirements.txt not found at $requirementsFile"
}

if (-not (Test-Path $venvPath)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    $pyLauncherCmd = Get-Command py -ErrorAction SilentlyContinue

    if ($pythonCmd) {
        & $pythonCmd.Source -m venv $venvPath
    }
    elseif ($pyLauncherCmd) {
        & $pyLauncherCmd.Source -3 -m venv $venvPath
    }
    else {
        throw "Python was not found on PATH. Install Python 3.10+ and rerun this script."
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python executable not found at $pythonExe"
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $requirementsFile

Write-Host "Environment is ready."

if ($Activate) {
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    . $activateScript
    Write-Host "Virtual environment activated in this PowerShell session."
}
else {
    Write-Host "Activate with: . .\.venv\Scripts\Activate.ps1"
}