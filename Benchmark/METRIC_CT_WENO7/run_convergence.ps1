param(
    [string]$Julia = "C:\Users\52402\AppData\Local\Programs\Julia-1.12.1\bin\julia.exe"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ArtifactRoot = Join-Path $PSScriptRoot "artifacts\task7-runtime"
$Resolutions = @(12, 24, 48)
$Geometries = @("cartesian", "warped")

foreach ($Geometry in $Geometries) {
    $Stats = @()
    foreach ($N in $Resolutions) {
        $Output = Join-Path $ArtifactRoot "$Geometry\N$N"
        New-Item -ItemType Directory -Force -Path $Output | Out-Null
        & $Julia --project=$Root (Join-Path $PSScriptRoot "run.jl") $Geometry $N $Output
        if ($LASTEXITCODE -ne 0) {
            throw "WENO7 runtime failed for geometry=$Geometry N=$N"
        }
        $Stats += Join-Path $Output "stats.dat"
    }
    & $Julia --project=$Root (Join-Path $PSScriptRoot "run.jl") `
        verify $Geometry $Stats[0] $Stats[1] $Stats[2]
    if ($LASTEXITCODE -ne 0) {
        throw "WENO7 convergence verification failed for geometry=$Geometry"
    }
}
