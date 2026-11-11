param(
    [Parameter(Mandatory=$true)][ValidateSet('turning','drilling')][string]$Process,
    [Parameter(Mandatory=$true)][ValidateSet('features','train','report')][string]$Stage,
    [string]$Config,
    [string]$FeatureSet,
    [string]$ResumeDirectory,
    [switch]$ValidateOnly,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $projectDirectory '.venv-cuda\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = Join-Path $projectDirectory '.venv-review\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = 'python' }
}
$env:PYTHONPATH = Join-Path $projectDirectory 'src'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONFAULTHANDLER = '1'
$env:MPLBACKEND = 'Agg'
$arguments = @('-u','-X','faulthandler','-m','wear_cnn.cli')
if ($Stage -ne 'report' -and -not $Config) {
    throw 'features/train require -Config pointing to an explicit functional launch YAML'
}
if ($Stage -eq 'features') {
    $arguments += @('build-features','--config',$Config)
    if ($ValidateOnly) { $arguments += '--validate' }
} elseif ($Stage -eq 'train') {
    $arguments += @('train-batch','--config',$Config)
    if ($FeatureSet) { $arguments += @('--features',$FeatureSet) }
    if ($ResumeDirectory) { $arguments += @('--resume',$ResumeDirectory) }
} else {
    if (-not $ResumeDirectory) { throw 'report requires -ResumeDirectory' }
    $arguments += @('render-training-report','--run',$ResumeDirectory)
}
$logDirectory = Join-Path $projectDirectory 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory ("${Process}_${Stage}_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host "Log: $logPath"
& $PythonPath @arguments 2>&1 | Tee-Object -FilePath $logPath
$code = $LASTEXITCODE
"ExitCode=$code" | Add-Content -LiteralPath $logPath
if ($code -ne 0) { throw "Stage failed with exit code $code. See $logPath" }
