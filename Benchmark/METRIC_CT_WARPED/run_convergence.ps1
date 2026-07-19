$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$generator = Join-Path $scriptRoot "gen_mesh.jl"
$runner = Join-Path $scriptRoot "run_alfven.jl"
$diagnostics = Join-Path $scriptRoot "metric_ct_diagnostics.jl"
$convergenceVerifier = Join-Path $scriptRoot "verify_convergence.jl"
$projectOption = "--project=$projectRoot"
$resolutions = @(8, 16, 32)
$statsPaths = @()
$ownedEnvironmentVariables = @(
    "METRIC_CT_MESH_DIR",
    "METRIC_CT_STATS_FILE",
    "METRIC_CT_FINAL_TIME"
)

function Assert-LastExitCode($operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$operation failed with exit code $LASTEXITCODE"
    }
}

Push-Location $scriptRoot
try {
    foreach ($resolution in $resolutions) {
        $meshDir = Join-Path $scriptRoot "MESH_N$resolution"
        $statsPath = Join-Path $scriptRoot "alfven_N${resolution}_stats.dat"
        $logPath = Join-Path $scriptRoot "alfven_N${resolution}.log"
        $statsPaths += $statsPath

        & julia $projectOption $generator $resolution $resolution $resolution $meshDir
        Assert-LastExitCode "mesh generation for N=$resolution"

        $env:METRIC_CT_MESH_DIR = $meshDir
        $env:METRIC_CT_STATS_FILE = $statsPath
        $env:METRIC_CT_FINAL_TIME = "0.1"

        & julia $projectOption $runner 2>&1 | Tee-Object -FilePath $logPath
        Assert-LastExitCode "Alfven run for N=$resolution"

        & julia $projectOption $diagnostics $statsPath "alfven"
        Assert-LastExitCode "run verification for N=$resolution"
    }

    & julia $projectOption $convergenceVerifier $statsPaths
    Assert-LastExitCode "metric CT convergence verification"
}
finally {
    foreach ($name in $ownedEnvironmentVariables) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    Pop-Location
}
