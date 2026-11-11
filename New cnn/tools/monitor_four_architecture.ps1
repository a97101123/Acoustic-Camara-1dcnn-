param([Parameter(Mandatory=$true)][string]$StatePath, [int]$IntervalSeconds = 10)
$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be positive' }
while ($true) {
    Clear-Host
    Write-Host (Get-Date)
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        $state | ConvertTo-Json -Depth 12
        if ($state.status -in @('complete', 'complete_with_failures', 'failed', 'interrupted')) { break }
    } catch { Write-Host ('Waiting: ' + $_.Exception.Message) }
    Start-Sleep -Seconds $IntervalSeconds
}
