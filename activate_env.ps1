$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$activateScript = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found: $activateScript"
}

. $activateScript
Write-Host "Activated virtual environment from $projectRoot\.venv" -ForegroundColor Green
