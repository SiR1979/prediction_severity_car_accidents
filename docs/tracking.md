# Local reference-run and new pipeline tracking

Project: car_accident  
Status: NON_FINAL  
Implementation addition: 2026-10-08

## Scope

`compose.tracking.yaml` adds an isolated MLflow 3.17.0 service and a temporary
`tracking-import` tool. The first tracked run imports the already completed
2021 reference run. It does not execute preprocessing, fit a model, evaluate
new predictions, register a model version, or change delivery approval.

Open the UI at `http://127.0.0.1:5000/` after successful installation. Select
the `car_accident_reference` experiment and the `2021-reference-20261003` run.
Inspect its metrics, parameters, tags and `reference/` artifacts.

The server uses a separate image. Its only build context is `src/tracking/`.
MLflow and NumPy are pinned in that image; training and serving dependencies
remain in their existing images. The resolved tracking requirements are saved
inside the image at `/opt/tracking-requirements-resolved.txt`. Transitive
dependencies and the base-image digest are not fully locked.

## Storage and boundaries

| Location | Purpose |
| --- | --- |
| `data/tracking/mlflow.db` | New local SQLite metadata store |
| `data/tracking/artifacts/` | New artifact copies managed by the MLflow proxy |
| `data/runs/validation_20261003T193310Z_56c49087/` | Preserved reference inputs, mounted read-only by the import tool |
| `data/code_backups/tracking_<UTC>_<suffix>/architecture.md` | Exact architecture backup before installation |

The server publishes only `127.0.0.1:5000`; the container listens on its own
network interface. Host headers and browser origins are explicitly restricted.
The server has no raw-data, model-source, API or application bind mounts.
Its writable storage is `/tracking`; `/tmp` is temporary. Telemetry and MLflow
background job execution are disabled for this local tracking step.

The import tool has no writable reference mount or tracking-storage mount. It
sends requests to `http://mlflow:5000` on the existing Compose project network.
Uploads and downloads pass through that server's artifact proxy. Neither
credentials nor external tracking destinations are accepted by the importer.
This is a local, unauthenticated service; keep its published address on loopback.

`data/` is ignored by Git. Source files and documents can be versioned later;
the tracking database, artifact copies and protected originals remain local.
Do not delete or replace existing tracking storage to repeat an installation.

## Historical evidence and import provenance

The selected sources are:

- `validation_report.json`:
  `d1d3200d428701c96f715f99fcd4912c92bb4e528023f7fe7ff5475aedfdcf79`
- `reference_run/run_status.json`:
  `bfd69f4edf47b66ac5b8b15196f1d545182d70b10e8d92988dcc77b09438040e`
- `reference_run/model/trained_model.joblib`:
  `5e8775b69782640c0ee244bfa7c427928ece0f01be877688102313a8bc7a1a2b`

All three files are copied as unchanged bytes. The model is an opaque joblib
artifact; the importer never deserialises or reserialises it. The original
reports remain the canonical historical evidence. Their source and raw-data
hashes are historical observations, not new verification of all those files.
The selected model is distinct from the preserved model in `src/models/`.

| Recorded time | Meaning |
| --- | --- |
| `2026-10-03T19:33:12.381422+00:00` | Original training start |
| `2026-10-03T19:33:18.640003+00:00` | Original training finish |
| `2026-10-03T19:33:32.426452+00:00` | Separate validation observation |
| `imported_at_utc` | New import time, recorded separately |

MLflow run start/end times use the original training interval, at millisecond
precision. Metric timestamps use the later validation observation. Exact
timestamp strings are retained in tags and original artifacts. Importing the
run does not make the separate evaluation an automatic pipeline gate.

The original training Git revision and container image identity are **unknown**.
The import records its own baseline Git HEAD, dirty worktree state, container
image ID and importer source hash separately in tags and
`reference/import_provenance.json`. The baseline HEAD does not contain the
newly installed, uncommitted tracking sources. Never attribute that revision
or tracking image to the original training run.

Recorded reference settings include the split seed `42`, forest seed `None`,
`n_jobs=-1`, the feature count `28`, and the inherited target mapping
`[1,2,3,4] -> [1,1,1,0]`. They do not claim a complete hyperparameter inventory.
Repeated training can produce different model bytes and predictions.

## Import integrity and repeat behaviour

Input hashes and the historical status, metric and timestamp contracts are
checked before any tracking write. Each import identity derives from the three
selected file hashes. Run exactly one import process at a time: deduplication
is designed for sequential use, not concurrent atomic imports.

A new run receives the reference metrics and artifacts. Verification downloads
each artifact and compares its size and SHA-256 with the original. A sequential
repeat verifies the same finished run without updating it or creating a duplicate.
An existing incomplete, deleted, ambiguous or inconsistent import stops the
operation and preserves the stored state. A failed new import is marked failed
when the server remains reachable. There is no automatic destructive repair or
retry of an uncertain run-creation request.

The single installation test performs the actual import, a sequential repeat,
metric-history checks and artifact hash comparisons. It also checks that invalid
destinations and altered evidence fail before metadata writes. Original input
hashes are checked again at the end. The Linux preparation test uses labelled
opaque fixture bytes against MLflow 3.17.0; only successful Windows installation
output establishes validation with the real selected reference files and Docker.

## Local commands

Use PowerShell 7.2 or newer in the repository after installation. The installer
starts only the new service, performs its functional test, and leaves source
changes uncommitted. API, app and training validation are not rerun by this step.

```powershell
$compose = @('compose', '--project-directory', 'C:\Users\simon\MLOps_car_accident',
    '-f', 'compose.yaml', '-f', 'compose.api.yaml', '-f', 'compose.app.yaml',
    '-f', 'compose.tracking.yaml')
docker @compose ps mlflow
docker @compose logs --tail 60 mlflow
```

To stop the tracking server while preserving its files:

```powershell
docker @compose stop mlflow
```

To restart the installed server without rebuilding or starting other services:

```powershell
docker @compose up -d --no-deps --no-build --pull never --wait --wait-timeout 60 mlflow
```

Do not rerun the one-time installer on an existing installation. Review the
recorded run ID and storage first if an import was interrupted. A future
import invocation must supply its actual Git baseline and image identity;
do not copy an earlier import's identity into a new invocation.

## Limits and next work

`approved_for_delivery=false` and `quality_thresholds_enforced=false` are retained.
Observed metrics do not establish operational suitability or quality acceptance.
The inherited severity-based row selection limits claims about prediction before
intervention. Domain labels for class codes 0 and 1 remain unconfirmed.

The historical importer does not instrument new training runs or change the
API's selected artifact. The explicit new-run mode below is separate from that
historical import. Automatic DVC output registration, orchestration, monitoring,
controlled delivery and model-registry policy remain separate work.
Level 1 remains incomplete; Level 2 is a separate extension.

## Explicit new pipeline-run mode

Implementation addition: 2026-10-10. Code presence alone does not prove a new
real-data training run, a current server, or successful artifact recording.
The original historical importer, experiment, report and model remain unchanged.

`pipeline-run` is a finite Tools-profile service. It uses the existing training
Dockerfile and a separate `car-accident-pipeline-tracking:local` image. The
standard-library `LocalClient` transport is reused; the MLflow SDK is not added
to training or serving dependencies. Historical metric/report validation is not
called. No autologging, metrics, registry or serving promotion is performed.

The normal pipeline CLI remains available. Tracking requires both
`--tracking-uri` and `--tracking-context`; supplying only one is a CLI error.
The tracked mode requires the reference raw directory and a new immediate child
of `data/runs`. Existing run directories and invocation identities are rejected.

### Launcher and runtime preparation

Build only the new pipeline image when this source changes. This does not rebuild
the installed MLflow, API or app images or start their services:

```powershell
docker compose --project-directory 'C:\Users\simon\MLOps_car_accident' `
  -f 'C:\Users\simon\MLOps_car_accident\compose.yaml' `
  -f 'C:\Users\simon\MLOps_car_accident\compose.tracking.yaml' build pipeline-run
```

The existing local MLflow 3.17.0 server must already be reachable. The launcher
does not install dependencies, build images, start services, or retry a run:

```powershell
& 'C:\Users\simon\MLOps_car_accident\scripts\run_tracked_pipeline.ps1'
```

This command performs real inherited preprocessing and training. Use it only
for an authorized new reference run, never as a transport or context probe.
Run one launcher process at a time; experiment/identity checks are sequential,
not a concurrent atomic transaction.

The launcher records its observed Git baseline, worktree status, source hashes,
configuration byte snapshots and actual resolved image ID. The temporary
container uses that immutable image ID. A new invocation context is exclusively
created under `data/run_invocations/<invocation_id>/`; it is preserved on failure.
The context is read-only in the container. Git metadata, the Docker socket and
the tracking database are not mounted. Only `data/runs` is writable; raw data,
reference preprocessing, model sources and versioning metadata are read-only.

Executed source hashes are compared with launcher observations before tracking
writes and rechecked at recording. Raw-file hashes are captured before processing
and checked again before completion. Python, installed package versions and
`/opt/requirements-resolved.txt` describe the actual runtime. The image ID remains
a launcher observation, not independent in-container attestation. A dirty
baseline does not mean its commit contains the executed source bytes; code
snapshots and their hashes identify those bytes separately.

### Lifecycle, provenance and artifacts

New runs use `car_accident_reference_pipeline`, with scope
`new_reference_pipeline_run` and a unique invocation tag. After claiming a new
local run directory, the tracker checks context and server version, creates a
new `RUNNING` MLflow run, and records its ID locally. Phase tags/timestamps cover
preprocessing, data validation, training, model validation, report writing and
artifact recording. The inherited transformations, split seed 42, unseeded
forest, target mapping and prepared-feature contract remain unchanged.

After technical validation, new exclusive files under `tracking_evidence/`
contain provenance and an artifact manifest. The unchanged technical report,
opaque model bytes, four prepared CSVs, invocation context, runtime dependency
evidence and small code/configuration snapshots are uploaded under `pipeline/`.
Every upload is downloaded and checked for exact size and SHA-256. The manifest
binds run/invocation/MLflow IDs and artifact hashes; it excludes its own entry.
Its own SHA-256 is stored separately in MLflow tags. No local status file is
uploaded as a prematurely completed terminal snapshot.

The existing raw DVC pointer and registration manifest are read, never executed
or changed. Pointer SHA-256, declared DVC version, object ID and manifest SHA-256
are retained. The association is `matched` only when the pointer matches its
manifest and all four actual raw-file identities match the registered entries.
Otherwise it is `unmatched` or metadata is `unavailable`; neither means verified
DVC association. This does not verify current recovery/store availability or
version new prepared outputs/models. Registration identities never replace
unknown original historical training identities.

`completed` in tracked mode requires successful technical validation, all
artifact round trips, confirmed MLflow `FINISHED` metadata and final local status
writing. Evaluation remains separate and is never an acceptance gate.
`approved_for_delivery=false` is retained in local and recorded evidence.

### Failure and uncertainty

A context, processing, tracking or artifact error causes local `failed` and exit
1. Local outputs and partial remote artifacts are preserved. A known, identified
new MLflow run is marked `FAILED` and checked best-effort. Error type, failed
phase and secondary tracking errors are recorded without replacing the primary
exception. No automatic replay, cleanup, resume or second create request occurs.

An uncertain create response leaves the remote state unknown and the run ID
unset; the invocation ID is the diagnosis key. An existing invocation is never
modified or reused. There is no atomic transaction between MLflow and the local
status file. If final local writing fails after confirmed remote completion,
remote failure marking is attempted; unavailable confirmation remains unknown.
Storage failure may prevent persistence even though the CLI returns nonzero.

### Targeted verification

Only `src/test_tracking_pipeline.py` is selected for this slice. It uses isolated
files, opaque inputs/models and unfitted doubles, with fitting, inference and
deserialisation guarded or replaced. Fake transport cases cover lifecycle,
identity rejection, metadata/transfer/finalisation failures, uncertain creation,
duplicate preservation, local completion failure and unchanged untracked mode.

When `PIPELINE_TRACKING_TEST_URI=http://mlflow:5000` is supplied, one case uses the
existing server and a new `car_accident_reference_pipeline_test_<uuid>` experiment.
That case verifies the real artifact proxy and final metadata using explicitly
labelled `opaque_test_fixture` evidence. It never trains or loads a real model
and leaves its test run/artifacts preserved. Server unavailability is reported
as `TODO-VERIFY`, not successful HTTP evidence. The historical importer tests,
earlier eleven pipeline tests, API/app tests and DVC recovery are not repeated.

The earlier eleven-test result lacks contemporaneous candidate hashes; later
source hashes cannot retroactively bind that historical test run.

### Historical implementation verification (2026-10-10)

The initial targeted suite ran from `2026-10-10T17:31:14.8418701Z` to
`2026-10-10T17:31:23.7800964Z`: 19 selected tests, 18 passed and one skipped,
5.194 seconds of unittest execution, process exit 0. It used the separately
built image ID
`sha256:c10da6636ebe3dc0254a1a43902c8316912a78d76d3d29cd4aa94e4c1361d2da`.
The wrapper printed SHA-256 of the actual executed source files before selecting
only `test_tracking_pipeline.py`:

| Executed file | SHA-256 |
| --- | --- |
| `src/run_pipeline.py` | `e76f46170704d2c74aab285151af4719acf52b84f7474fc1e05fb8270e19340a` |
| `src/tracking/track_pipeline_run.py` | `b5768dc7abe5a7409116694aea6eda7a4a1a204032636cb5e4e19b42f7c8cfcb` |
| `src/test_tracking_pipeline.py` | `6b6a93230d9b16925da1296a0536497169348a04f5e319a9e17052fc0fa87ed9` |
| `src/data/make_dataset.py` | `cb834b3a60d84a19540c4e8ac2c0e3d0027f3b447121ed7aff7515780330541e` |
| `src/models/train_model.py` | `301e0c5305974d16af53f31eb01775e76ecb59d7a91331dbc9ee7e694cac4edb` |
| `src/tracking/import_reference_run.py` | `9eea9f91ad8305b3f3791cb604d2be5073e369ec7add4bee6326aa3c7636e8ef` |

The retained invocation used `docker compose` with the explicit repository,
`compose.yaml` and `compose.tracking.yaml`, `run --rm --no-deps -T
--interactive=false --pull never`, a read-only override for `/app/data/runs`,
`PIPELINE_TRACKING_TEST_URI=http://mlflow:5000`, and `pipeline-run python -B -c`
with this wrapper:

```python
import hashlib, pathlib, sys, unittest
paths = ['src/run_pipeline.py', 'src/data/make_dataset.py', 'src/models/train_model.py', 'src/tracking/track_pipeline_run.py', 'src/tracking/import_reference_run.py', 'src/test_tracking_pipeline.py']
for name in paths:
    print('EXECUTED_SOURCE_SHA256 ' + name + '=' + hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest(), flush=True)
suite = unittest.defaultTestLoader.discover('src', pattern='test_tracking_pipeline.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
```

During that historical run, the real HTTP artifact round trip was skipped with DNS
`gaierror`; no HTTP test experiment/run was created. Read-only container
inspection at `2026-10-10T17:30:04.2387758Z` found the existing MLflow container
stopped (`Exited (255)`). It was preserved; no service was started or restarted
in that phase. The later bounded HTTP evidence below supersedes that integration
gap without changing the historical result or its source identities.

### Corrected checkpoint evidence (2026-10-10)

The bounded correction suite ran from `2026-10-10T18:55:42.3581882Z` to
`2026-10-10T18:56:03.6959299Z`: 24 selected tests, 23 passed and one explicitly
skipped, 6.108 seconds of unittest execution, exit 0. HTTP was disabled without
a server probe. Regression cases cover artifact changes after manifest creation
and during transfer, report/model linkage, empty tracking URIs and omitted options.

The separate HTTP phase selected only
`test_tracking_pipeline.TrackingPipelineTests.test_opaque_round_trip_against_existing_mlflow`.
It ran from `2026-10-10T19:35:51.760358Z` to `2026-10-10T19:36:02.843605Z`:
one test passed, no skip, 6.665 seconds of unittest execution, exit 0.
The existing local container was started separately; its published endpoint
returned MLflow 3.17.0. In the existing project network, `mlflow` resolved only
to that container's verified IP. The test exercised the corrected `PipelineTracker`
and real HTTP client, including verified artifact round trips. Its assertions
confirmed local `completed`, remote `FINISHED`, `opaque_test_fixture`, the expected
opaque-model SHA-256 and absence of metrics. Processing, training and model loading
were replaced by doubles; no real training or evaluation occurred.

- Experiment: `car_accident_reference_pipeline_test_0ea8f353d8714819be3a21c8b4e27993`.
- Experiment ID: `2`.
- Run ID: `f57ed1d4d90548d8a439169c5e2c35b2`; fixture artifacts remain preserved.

Both phases used the immutable image ID recorded above, current sources mounted
read-only and disabled bytecode writes. Runtime-visible source hashes matched
the host candidate; these identities do not retrospectively bind earlier runs:

| Corrected working file | Raw SHA-256 |
| --- | --- |
| `src/run_pipeline.py` | `8367da5ec30a9036c330f1e2ef3fa1ffeca501596fc1ce150ef5630c8af6be09` |
| `src/tracking/track_pipeline_run.py` | `260e94c42f93ef2474a5a076706f47c64001daccecee0cde2cfb14fc39b68983` |
| `src/test_tracking_pipeline.py` | `b67bbd8b5e496d85be28ea91b2f728129ff15014e83c086297eb042e3877e0ff` |

Launcher end-to-end operation, real training and serving verification remain
unconfirmed. No model quality or delivery approval follows from this checkpoint.
Level 1 remains `NON_FINAL`; `approved_for_delivery=false`.

## DagsHub shared tracking target

Updated: 2026-10-14.

New pipeline runs can be recorded on the shared DagsHub MLflow server instead
of the local one. The tracking URI selects the target:

| Target | Tracking URI | Transport and authentication | Server version check |
| --- | --- | --- | --- |
| `local` | `http://mlflow:5000` (or `localhost`/`127.0.0.1`) | Plain HTTP, no credentials | Exactly MLflow 3.17.0 |
| `dagshub` | `https://dagshub.com/SiR1979/prediction_severity_car_accidents.mlflow` | HTTPS with certificate and hostname verification; HTTP Basic with username and access token | Any MLflow 3.x; the observed version is recorded |

`LocalClient` accepts HTTPS only for `dagshub.com` on port 443 with an
`/<owner>/<repo>.mlflow` path. Other hosts, schemes, ports, credentials in the
URI, query strings and extra path segments are rejected before any request.
Redirects are never followed; every non-200 response stops the run, and 401/403
responses name the credentials as the likely cause without repeating them.

Credentials come only from `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` (the personal DagsHub access token). Both are required
for DagsHub; local targets never read or send them. DagsHub answers
unauthenticated searches with an empty result instead of an error, so missing
credentials are rejected before the first request.

The token is held only in the request header. It is not written to run tags,
parameters, provenance, the invocation context, configuration evidence or error
messages. `compose.tracking.yaml` passes both variables to `pipeline-run` by name
only; `docker compose config` prints their values when they are set, so do not
share that output.

Each run carries the tags `tracking_target` and `tracking_server_version`. All
provenance, artifact-hash and remote read-back checks are unchanged for both
targets.

### Launching a DagsHub-tracked run

Set the variables in the launching process, then select the target:

```powershell
$env:MLFLOW_TRACKING_USERNAME = Read-Host 'Your DagsHub username'
$env:MLFLOW_TRACKING_PASSWORD = Read-Host 'Your DagsHub access token' -MaskInput
& .\scripts\run_tracked_pipeline.ps1 -TrackingTarget dagshub
```

The launcher fails before any Git, Docker or tracking step when either variable
is missing, and prints `TRACKING_TARGET=` but never the credential values.
`-TrackingTarget local` (the default) keeps the previous behaviour.

### Validation on 2026-10-14

- Compatibility probe: DagsHub runs MLflow 3.5.1. Run creation, tags, parameters,
  `mlflow-artifacts:` artifact URIs, proxied upload and download below the
  `.mlflow` path prefix, `FINISHED` read-back and 401 for a wrong token behaved
  as the tracker expects. The probe experiment `connectivity_check` was deleted.
- `src/tracking/test_tracking_client.py` covers target selection, rejected URIs,
  missing credentials, authentication failures, redirects, upload/download hash
  checks and the version rules without network access.
- The existing opaque round-trip test
  (`test_opaque_round_trip_against_existing_mlflow`) passed against DagsHub with
  `PIPELINE_TRACKING_TEST_URI` set to the DagsHub URI. Its 20 artifacts were
  checked for the token afterwards (none found); the test experiment was then
  deleted. Processing and training were test doubles, as in the local run.
- The Windows launcher and Docker-based DagsHub run were pending at this
  checkpoint. The subsequent Windows run is recorded below.

### Tracked Windows pipeline run (2026-11-12)

The Windows launcher `scripts/run_tracked_pipeline.ps1 -TrackingTarget dagshub`
completed a staged pipeline run on the private DagsHub MLflow server.

| Field | Recorded value |
| --- | --- |
| Date | 2026-11-12 |
| MLflow experiment | `car_accident_staged_pipeline` |
| MLflow run ID | `9cf8a945e70e42989c21438793d8cf9a` |
| Git commit | `3c2c0538cc2dafebdd2beb7c54d3d5a91e323e5e` |
| Stage exit codes | `prepare`: 0, `train`: 0, `evaluate`: 0 |
| Remote run status | `FINISHED` |
| Quality gate | `passed` |
| `approved_for_delivery` | `false` |

Roy independently confirmed that the run, its metrics and its artifacts were
visible on DagsHub (see #24).

## References

- [MLflow 3.17.0 release](https://github.com/mlflow/mlflow/releases/tag/v3.17.0)
- [Tracking server architecture](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [MLflow REST API](https://mlflow.org/docs/latest/api_reference/rest-api.html)
- [MLflow 3.17.0 REST service definitions](https://github.com/mlflow/mlflow/blob/v3.17.0/mlflow/protos/service.proto)
