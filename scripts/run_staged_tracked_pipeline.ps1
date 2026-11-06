#requires -Version 7.2
<#
Run the existing prepare, train and evaluate stages in separate containers.
Record the finished workflow, including a failed gate, in a fourth container.
Images and tracking services must already be prepared. No automatic retry.
#>
[CmdletBinding()]
param(
    [string]$Repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')),
    [string]$RuntimeImage = 'car-accident-runtime:local',
    [ValidateSet('local', 'dagshub')][string]$TrackingTarget = 'local'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repoPath = [IO.Path]::GetFullPath($Repository).TrimEnd('\', '/')
$gitPath = (Get-Command git -CommandType Application | Select-Object -First 1).Source
$dockerPath = (Get-Command docker -CommandType Application | Select-Object -First 1).Source
$trackingUri = if ($TrackingTarget -eq 'local') { 'http://mlflow:5000' } else {
    'https://dagshub.com/SiR1979/prediction_severity_car_accidents.mlflow'
}
if ($TrackingTarget -eq 'dagshub') {
    foreach ($name in @('MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD')) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($name, 'Process'))) {
            throw "$name is required; value withheld."
        }
    }
}
function Invoke-Native([string]$Executable, [string[]]$Arguments) {
    $lines = @(& $Executable @Arguments 2>&1)
    $code = $LASTEXITCODE
    [pscustomobject]@{ ExitCode = $code; Lines = [string[]]@($lines | ForEach-Object { [string]$_ }) }
}
function Read-Git([string[]]$Arguments) {
    foreach ($name in @('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES')) {
        if ([Environment]::GetEnvironmentVariable($name)) { throw "Git environment override: $name" }
    }
    $result = Invoke-Native $gitPath (@('--no-optional-locks','--no-replace-objects','-c','core.fsmonitor=false','-C',$repoPath) + $Arguments)
    if ($result.ExitCode -ne 0) { throw "Git read failed; exit=$($result.ExitCode)." }
    $result.Lines -join "`n"
}
function Assert-PlainPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if (([long]$item.Attributes -band 0x441400) -ne 0) { throw 'Linked, offline or cloud path rejected.' }
        $item = if ($item -is [IO.DirectoryInfo]) { $item.Parent } else { $item.Directory }
    }
}
function Get-Identity([string]$Path) {
    Assert-PlainPath $Path
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) { throw 'Expected evidence file.' }
    $length = $item.Length
    $modified = $item.LastWriteTimeUtc.Ticks
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $item.Refresh()
    if ($item.Length -ne $length -or $item.LastWriteTimeUtc.Ticks -ne $modified) { throw 'Evidence changed while hashing.' }
    @{ sha256 = $hash; size_bytes = $length }
}
function Write-NewJson([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 20) + "`n")
    $stream = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
Assert-PlainPath $repoPath
if ([IO.Path]::GetFullPath((Read-Git @('rev-parse','--show-toplevel'))) -ine $repoPath) { throw 'Repository root differs.' }
$branch = Read-Git @('branch','--show-current')
if ($branch -cne 'master' -and $branch -notmatch '^work/[A-Za-z0-9._/-]+$') { throw 'The launcher requires master or a work/ branch.' }
$head = Read-Git @('rev-parse','HEAD')
if ((Read-Git @('diff','--cached','--name-only','--no-ext-diff','--no-textconv','--'))) { throw 'Existing staged changes were preserved.' }
$imageResult = Invoke-Native $dockerPath @('image','inspect','--format','{{.Id}}',$RuntimeImage)
if ($imageResult.ExitCode -ne 0 -or $imageResult.Lines.Count -ne 1 -or $imageResult.Lines[0] -notmatch '^sha256:[0-9a-f]{64}$') {
    throw 'Prepared runtime image required; no image build or pull attempted.'
}
$imageId = $imageResult.Lines[0]
$network = 'bridge'
if ($TrackingTarget -eq 'local') {
    $local = Invoke-Native $dockerPath @('container','inspect','--format','{{json .State}}','car-accident-mlflow-1')
    if ($local.ExitCode -ne 0) { throw 'Existing local tracking service required.' }
    $serviceState = ($local.Lines -join "`n") | ConvertFrom-Json
    if (!$serviceState.Running -or $serviceState.Health.Status -ne 'healthy') { throw 'Existing tracking service is not running and healthy.' }
    $network = 'car-accident_default'
    $networkResult = Invoke-Native $dockerPath @('network','inspect',$network)
    if ($networkResult.ExitCode -ne 0) { throw 'Existing tracking network required.' }
}
$sources = @(
    'src/pipeline_stages.py','src/run_pipeline.py','src/accept_run.py','src/evaluate_run.py',
    'src/reference_contract.py','src/data/make_dataset.py','src/models/train_model.py',
    'src/tracking/import_reference_run.py','src/tracking/track_pipeline_run.py','src/tracking/track_staged_run.py'
)
$configurations = @('Dockerfile','requirements-runtime.txt','src/acceptance_criteria.json','scripts/run_staged_tracked_pipeline.ps1')
$sourceHashes = [ordered]@{}
foreach ($relative in $sources) { $sourceHashes[$relative] = (Get-Identity (Join-Path $repoPath $relative)).sha256 }
$configurationEvidence = [ordered]@{}
foreach ($relative in $configurations) {
    $path = Join-Path $repoPath $relative
    $identity = Get-Identity $path
    $bytes = [IO.File]::ReadAllBytes($path)
    if ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant() -ne $identity.sha256) { throw 'Configuration changed.' }
    $configurationEvidence[$relative] = @{ sha256=$identity.sha256; content_base64=[Convert]::ToBase64String($bytes) }
}
$rawPath = Join-Path $repoPath 'data\raw'
$runsPath = Join-Path $repoPath 'data\runs'
$dataPath = Join-Path $repoPath 'data'
foreach ($path in @($rawPath,$runsPath,$dataPath)) { Assert-PlainPath $path }
$rawIdentities = [ordered]@{}
foreach ($name in @('usagers-2021.csv','caracteristiques-2021.csv','lieux-2021.csv','vehicules-2021.csv')) {
    $rawIdentities[$name] = Get-Identity (Join-Path $rawPath $name)
}
$invocationId = [guid]::NewGuid().ToString('N')
$runName = 'staged_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '_' + $invocationId
$runPath = Join-Path $runsPath $runName
$invocationParent = Join-Path $dataPath 'run_invocations'
if (Test-Path -LiteralPath $invocationParent) { Assert-PlainPath $invocationParent }
$invocationPath = Join-Path $invocationParent $invocationId
if ((Test-Path -LiteralPath $runPath) -or (Test-Path -LiteralPath $invocationPath)) { throw 'Invocation path already exists.' }
[void][IO.Directory]::CreateDirectory($invocationPath)
[void][IO.Directory]::CreateDirectory($runPath)
$containerRun = '/app/data/runs/' + $runName
$contextPath = Join-Path $invocationPath 'invocation_context.json'
$receiptPath = Join-Path $invocationPath 'stage_receipt.json'
$containerInvocation = '/app/data/run_invocations/' + $invocationId
$context = [ordered]@{
    schema_name='staged_pipeline_invocation_v1'; evidence_scope='observed_host_launch'; invocation_id=$invocationId;
    observed_at_utc=[DateTime]::UtcNow.ToString('o'); git_head=$head; worktree_status=[string[]]@((Read-Git @('status','--porcelain=v1','--untracked-files=all')) -split "`n" | Where-Object { $_ });
    image_id=$imageId; source_sha256=$sourceHashes; configuration_evidence=$configurationEvidence;
    raw_directory='/app/data/raw'; run_directory=$containerRun; raw_identities=$rawIdentities
}
Write-NewJson $contextPath $context
$common = @('run','--rm','--pull','never','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true','--tmpfs','/tmp')
$mounts = @('--mount',"type=bind,source=$rawPath,target=/app/data/raw,readonly",'--mount',"type=bind,source=$runPath,target=$containerRun",'--mount',"type=bind,source=$invocationPath,target=$containerInvocation,readonly")
$imageCheck = 'import hashlib,json,pathlib,sys; import src.tracking.track_staged_run; c=json.loads(pathlib.Path(sys.argv[1]).read_bytes()); assert all(hashlib.sha256(pathlib.Path("/app",p).read_bytes()).hexdigest()==h for p,h in c["source_sha256"].items()); assert hashlib.sha256(pathlib.Path("/app/src/acceptance_criteria.json").read_bytes()).hexdigest()==c["configuration_evidence"]["src/acceptance_criteria.json"]["sha256"]'
$check = Invoke-Native $dockerPath ($common + @('--network','none') + $mounts + @($imageId,'python','-B','-c',$imageCheck,($containerInvocation+'/invocation_context.json')))
if ($check.ExitCode -ne 0) { $check.Lines; throw 'Image/source verification failed before training; preserve invocation files.' }
$stages = @()
foreach ($stageName in @('prepare','train','evaluate')) {
    $command = @('python','-B','src/pipeline_stages.py',$stageName)
    if ($stageName -eq 'prepare') { $command += @('--raw-dir','/app/data/raw') }
    $command += @('--run-dir',$containerRun)
    if ($stageName -eq 'evaluate') { $command += @('--criteria','/app/src/acceptance_criteria.json') }
    $stage = [ordered]@{ name=$stageName; image_id=$imageId; command=$command; started_at_utc=[DateTime]::UtcNow.ToString('o'); finished_at_utc=$null; exit_code=$null; launch_error_type=$null }
    try {
        $result = Invoke-Native $dockerPath ($common + @('--network','none') + $mounts + @($imageId) + $command)
        $stage.exit_code = $result.ExitCode
        $result.Lines
    }
    catch { $stage.launch_error_type=$_.Exception.GetType().Name; Write-Output "STAGE_LAUNCH_FAILED=$($stage.launch_error_type)" }
    $stage.finished_at_utc=[DateTime]::UtcNow.ToString('o')
    $stages += [pscustomobject]$stage
    if ($null -eq $stage.exit_code -or $stage.exit_code -ne 0) { break }
}
Write-NewJson $receiptPath ([ordered]@{ schema_name='staged_pipeline_receipt_v1'; invocation_id=$invocationId; stages=$stages })
$authentication = if ($TrackingTarget -eq 'dagshub') { @('--env','MLFLOW_TRACKING_USERNAME','--env','MLFLOW_TRACKING_PASSWORD') } else { @() }
$tracking = Invoke-Native $dockerPath ($common + @('--network',$network) + $authentication + $mounts + @($imageId,'python','-B','-m','src.tracking.track_staged_run','--context',($containerInvocation+'/invocation_context.json'),'--receipt',($containerInvocation+'/stage_receipt.json'),'--tracking-uri',$trackingUri))
$tracking.Lines
Write-Output "INVOCATION_ID=$invocationId"
Write-Output "STAGE_RECEIPT=$receiptPath"
Write-Output "TRACKING_EXIT_CODE=$($tracking.ExitCode)"
if ($tracking.ExitCode -ne 0) { throw 'Staged workflow or tracking failed; preserve outputs and remote state. No automatic retry.' }
