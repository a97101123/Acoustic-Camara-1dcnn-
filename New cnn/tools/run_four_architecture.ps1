param([Parameter(Mandatory=$true)][string]$Config, [string]$PythonPath, [string]$State)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $projectDirectory '.venv-cuda\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = Join-Path $projectDirectory '.venv-review\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = 'python' }
}
$env:PYTHONPATH = Join-Path $projectDirectory 'src'
$env:MPLBACKEND = 'Agg'
if (-not $State) {
    $logDirectory = Join-Path $projectDirectory 'logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $State = Join-Path $logDirectory ('four_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
}
& $PythonPath -u (Join-Path $PSScriptRoot 'run_four_architecture.py') --config $Config --state $State 2>&1 | Tee-Object -FilePath ($State + '.log')
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Batch failed ($code). See $State.log" }
