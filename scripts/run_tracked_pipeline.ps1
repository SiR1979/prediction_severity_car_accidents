#requires -Version 7.2
<#
Launch exactly one new reference pipeline run against the installed local server
or, with -TrackingTarget dagshub, against the shared DagsHub MLflow server.
DagsHub requires MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD (access token)
in the calling process; their values are passed to the container but never printed.
This script never builds images, starts dependencies, or changes Git state.
#>
[CmdletBinding()]
param(
    [string]$Repository = 'C:\Users\simon\MLOps_car_accident',
    [string]$RuntimeImage = 'car-accident-pipeline-tracking:local',
    [ValidateSet('local', 'dagshub')]
    [string]$TrackingTarget = 'local'
)
$ErrorActionPreference = 'Stop'
$trackingUris = @{
    local   = 'http://mlflow:5000'
    dagshub = 'https://dagshub.com/SiR1979/prediction_severity_car_accidents.mlflow'
}
$trackingUri = $trackingUris[$TrackingTarget]
if ($TrackingTarget -eq 'dagshub') {
    foreach ($name in @('MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD')) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name, 'Process'))) { throw ($name + ' is required for DagsHub tracking; value withheld.') }
    }
}
$repoPath = [IO.Path]::GetFullPath($Repository).TrimEnd([IO.Path]::DirectorySeparatorChar)
$gitExecutable = (Get-Command git -CommandType Application | Select-Object -First 1).Source
$dockerExecutable = (Get-Command docker -CommandType Application | Select-Object -First 1).Source
function Invoke-Native {
    param([string]$Executable, [string[]]$NativeArguments)
    $nativeLines = @(& $Executable @NativeArguments)
    $nativeExit = $LASTEXITCODE
    if ($nativeExit -ne 0) { throw ('Native command failed; exit=' + $nativeExit + '. Preserve reached state.') }
    return [pscustomobject]@{ Lines = [string[]]@($nativeLines | ForEach-Object { $_.ToString() }); ExitCode = $nativeExit }
}
function Git-Read {
    param([string[]]$GitArguments)
    Invoke-Native -Executable $gitExecutable -NativeArguments (@('--no-optional-locks', '-C', $repoPath) + $GitArguments)
}
function Identity {
    param($Result)
    [string[]]$identityLines = @($Result.Lines)
    if ($identityLines.Count -ne 1 -or [string]::IsNullOrWhiteSpace($identityLines[0])) { throw 'Expected exactly one complete identity line.' }
    return [string]$identityLines[0]
}
function Evidence-Bytes {
    param([string]$RelativePath)
    $path = Join-Path $repoPath $RelativePath
    $item = Get-Item -LiteralPath $path -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw ('Linked or invalid evidence: ' + $RelativePath) }
    for ($parent = $item.Directory; $null -ne $parent; $parent = $parent.Parent) {
        if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw ('Linked evidence ancestor: ' + $RelativePath) }
    }
    return [pscustomobject]@{ Bytes = [IO.File]::ReadAllBytes($path) }
}
function Hash-Bytes {
    param([byte[]]$Bytes)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
$root = Identity (Git-Read @('rev-parse', '--show-toplevel'))
if (-not [IO.Path]::IsPathRooted($root) -or [IO.Path]::GetFullPath($root) -ne $repoPath) { throw 'Repository root differs.' }
if ((Identity (Git-Read @('branch', '--show-current'))) -notin @('work/level1-foundation', 'master')) { throw 'Unexpected branch; no branch change attempted.' }
$origin = Identity (Git-Read @('config', '--get', 'remote.origin.url'))
# The repository was renamed; GitHub still redirects the old name, so existing clones keep it.
$allowedOrigins = foreach ($name in @('Template_MLOps_accidents', 'prediction_severity_car_accidents')) {
    'https://github.com/SiR1979/' + $name; 'https://github.com/SiR1979/' + $name + '.git'; 'git@github.com:SiR1979/' + $name + '.git'
}
if ($origin -notin $allowedOrigins) { throw 'Unexpected origin; URL withheld.' }
if ((Git-Read @('diff', '--cached', '--name-only', '--no-ext-diff', '--no-textconv', '--')).Lines.Count -ne 0) { throw 'Existing staged changes were preserved.' }
$gitHead = Identity (Git-Read @('rev-parse', 'HEAD'))
$worktree = (Git-Read @('status', '--porcelain=v1', '--untracked-files=all')).Lines
$imageId = Identity (Invoke-Native -Executable $dockerExecutable -NativeArguments @('image', 'inspect', '--format', '{{.Id}}', $RuntimeImage))
if ($imageId -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Unexpected image identity.' }
$sourceHashes = [ordered]@{}
foreach ($relative in @('src/run_pipeline.py', 'src/evaluate_run.py', 'src/reference_contract.py', 'src/data/make_dataset.py', 'src/models/train_model.py', 'src/tracking/track_pipeline_run.py', 'src/tracking/import_reference_run.py')) {
    $sourceHashes[$relative] = Hash-Bytes (Evidence-Bytes $relative).Bytes
}
$configuration = [ordered]@{}
foreach ($relative in @('Dockerfile', 'requirements-runtime.txt', 'compose.yaml', 'compose.tracking.yaml', 'scripts/run_tracked_pipeline.ps1')) {
    $content = (Evidence-Bytes $relative).Bytes
    $configuration[$relative] = @{ sha256 = (Hash-Bytes $content); content_base64 = [Convert]::ToBase64String($content) }
}
$invocationId = [Guid]::NewGuid().ToString('N')
$runName = 'reference_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + $invocationId
$runsPath = Join-Path $repoPath 'data\runs'
if (-not (Test-Path -LiteralPath $runsPath -PathType Container)) { throw 'Existing data/runs directory required.' }
foreach ($relative in @('data/raw', 'data/runs', 'data/preprocessed', 'src/models', 'versioning', 'data')) {
    $directory = Get-Item -LiteralPath (Join-Path $repoPath $relative) -Force
    if (-not $directory.PSIsContainer) { throw 'Expected existing mount directory.' }
    for ($parent = $directory; $null -ne $parent; $parent = $parent.Parent) {
        if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked mount directory was preserved.' }
    }
}
if (Test-Path -LiteralPath (Join-Path $runsPath $runName)) { throw 'Run already exists; no resume attempted.' }
$context = [ordered]@{
    schema_name = 'reference_pipeline_invocation_v1'; invocation_id = $invocationId
    evidence_scope = 'observed_host_launch'
    observed_at_utc = [DateTime]::UtcNow.ToString('o'); git_head = $gitHead
    worktree_status = [string[]]$worktree; image_id = $imageId
    source_sha256 = $sourceHashes; configuration_evidence = $configuration
    raw_directory = '/app/data/raw'; run_directory = '/app/data/runs/' + $runName
}
$requestDirectory = Join-Path $repoPath ('data\run_invocations\' + $invocationId)
$requestParent = Split-Path -Parent $requestDirectory
if (Test-Path -LiteralPath $requestParent) {
    $item = Get-Item -LiteralPath $requestParent -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Invalid invocation parent was preserved.' }
}
if (Test-Path -LiteralPath $requestDirectory) { throw 'Invocation directory exists; preserve it.' }
[IO.Directory]::CreateDirectory($requestDirectory) | Out-Null
$contextPath = Join-Path $requestDirectory 'invocation_context.json'
$contextBytes = [Text.UTF8Encoding]::new($false).GetBytes(($context | ConvertTo-Json -Depth 12) + "`n")
$stream = [IO.File]::Open($contextPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $stream.Write($contextBytes, 0, $contextBytes.Length) } finally { $stream.Dispose() }
$previousImage = [Environment]::GetEnvironmentVariable('PIPELINE_IMAGE_ID', 'Process')
$previousContext = [Environment]::GetEnvironmentVariable('PIPELINE_CONTEXT_PATH', 'Process')
try {
    $env:PIPELINE_IMAGE_ID = $imageId
    $env:PIPELINE_CONTEXT_PATH = $contextPath
    $arguments = @('compose', '--project-directory', $repoPath, '-f', (Join-Path $repoPath 'compose.yaml'),
        '-f', (Join-Path $repoPath 'compose.tracking.yaml'), 'run', '--rm', '--no-deps', '-T',
        '--interactive=false', '--pull', 'never', 'pipeline-run', 'python', '-B', 'src/run_pipeline.py',
        '--raw-dir', '/app/data/raw', '--run-dir', $context.run_directory,
        '--tracking-uri', $trackingUri, '--tracking-context', '/app/invocation_context.json')
    Write-Output ('TRACKING_TARGET=' + $TrackingTarget)
    Write-Output ('INVOCATION_ID=' + $invocationId)
    Write-Output ('INVOCATION_CONTEXT_SHA256=' + (Hash-Bytes $contextBytes))
    Write-Output ('IMAGE_ID=' + $imageId)
    $nativeLines = @(& $dockerExecutable @arguments)
    $nativeExit = $LASTEXITCODE
    $result = [pscustomobject]@{ Lines = [string[]]@($nativeLines | ForEach-Object { $_.ToString() }); ExitCode = $nativeExit }
    $result.Lines
    Write-Output ('PIPELINE_EXIT_CODE=' + $result.ExitCode)
    if ($result.ExitCode -ne 0) { throw 'Pipeline failed. Preserve invocation, run directory and remote state; do not retry this invocation.' }
} finally {
    [Environment]::SetEnvironmentVariable('PIPELINE_IMAGE_ID', $previousImage, 'Process')
    [Environment]::SetEnvironmentVariable('PIPELINE_CONTEXT_PATH', $previousContext, 'Process')
}
