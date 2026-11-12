# Tracked reference stages

The reference pipeline and its local MLflow tracking were verified in a
Windows/Docker run on 4 November 2026. See the validation record below.

`scripts/run_staged_tracked_pipeline.ps1` executes the existing
`src/pipeline_stages.py` commands in order: prepare, train, evaluate. Each stage
uses a separate finite container from one immutable runtime image. The existing
DVC stages, Compose stages and historical launcher retain their interfaces.

The supplied preprocessing, feature contract, unseeded Random Forest and team
criteria remain unchanged. Evaluate invokes the existing acceptance function,
including its 1e-9 comparison tolerance. A rejected gate stops the stage chain.
The reports continue to carry `approved_for_delivery=false`.

## Recording

A fourth finite container records the completed stage attempt using the existing
HTTP transport. Recording occurs after execution, including after an unsuccessful
stage. It is not live monitoring: an interrupted host or unavailable tracking
server can leave only the retained local invocation and receipt.

The recorder verifies source hashes, actual raw-file identities, configuration
snapshots, stage order, commands, observed image IDs, exit codes and UTC times.
It verifies the gate against the unchanged criteria and links evaluation inputs
to current artifact bytes. No model is deserialized by the recorder. Each upload
is downloaded and checked by SHA-256; terminal tags and metrics are read back.

Successful execution with a passed gate records `FINISHED`. A failed gate or
stage records `FAILED`, including available partial artifacts. A failed workflow
never receives an intermediate `FINISHED` transition. Transport failures and
uncertain run creation retain diagnostic state and are never retried
automatically. Image IDs and commands remain launcher observations, not an
independent in-container attestation.

The experiment is `car_accident_staged_pipeline`. The original experiment and
historical runs are preserved. Local evidence is exclusively created under
`data/run_invocations/<invocation_id>/` and a new `data/runs/staged_<...>/`.
Raw data is read-only. Stage containers have no network. Only the recording
container receives the selected tracking network and, for DagsHub, authentication
variables by name. Neither the Docker socket nor Git metadata is mounted.

## Separately authorized runtime preparation

After installation the runtime image must be rebuilt before a real run. The
launcher does not build or pull images or start services. It verifies image
source bytes before executing the first processing stage. The local target
requires the existing healthy tracking service and its network; the DagsHub
target requires the existing process authentication variables.

An authorized runtime phase must provide its own guarded build and launch block
for the confirmed root and branch. The new launcher's `TrackingTarget` parameter
selects `local` or `dagshub`; both options execute real reference training.

The launcher accepts `master` and `work/` branches and rejects staged changes.
By default it derives the repository root from the script directory. Use
`-Repository` to specify a repository path explicitly.

The validation below covers the staged launcher and the local MLflow service.
DVC/DagsHub version linkage, scheduled execution, monitoring and versioned model
delivery require separate evidence. DVC transfer, model registration,
serving-model selection and deployment require separate authorization.

The workflow is an educational demo. Operational police use would require
separate validation and approval.

## Isolated verification

`src/test_staged_tracking.py` uses opaque files, the existing acceptance function
and fake transport. It covers gate pass/rejection, incomplete stage execution,
unknown launch exits, source/criteria changes, absent gate evidence, artifact
round-trip failure and ambiguous creation. It does not train or load models or
write to an actual tracking service.

## Local Windows/Docker validation

On 4 November 2026, the launcher version published in commit
`08ef364a09fec57f1da906201a8a2fe18f03b7c6` completed `prepare`, `train`, and
`evaluate` with exit code 0 for each stage. Local MLflow run
`567ae80d9bee457d9a002f0ffe48ec95` in `car_accident_staged_pipeline` reported
`FINISHED`. The recorder downloaded every uploaded artifact and verified its
SHA-256 hash.

| Quality criterion | Observed | Minimum |
| --- | ---: | ---: |
| Macro-F1 | 0.744689 | 0.65 |
| Balanced accuracy | 0.736062 | 0.65 |
| Macro-F1 gain over the majority baseline | 0.349175 | 0.20 |

The run met all three team quality criteria. The recorded delivery flag was
`approved_for_delivery=false`.

## Explicit private local post-registration (7 November 2026)

`src/versioning/register_pipeline_run.py` can bind one completed staged run to a
new local DVC workspace outside the repository. This is a separate explicit
operation after recording, with no automatic launcher integration. The four
prepared CSV outputs and opaque model bytes are copied and cached locally.
It does not deserialize models, train, contact MLflow, configure a remote,
transfer data, change serving selection or grant delivery approval.

The caller must select the repository, raw directory, run directory, original
provenance and stage receipt, a nonexistent workspace, and the expected run,
invocation and MLflow IDs explicitly. Optional original `.dvc` pointers may be
supplied for representation evidence. The workspace's parent must already exist;
the workspace must be outside every protected input. Existing workspaces are
preserved, including failed attempts; there is no automatic resume or retry.
DVC 3.67.1 must already be installed in the selected Python environment. No
installation or image build is performed by the tool.

Before any workspace write, the registrar checks the recorded receipt digest,
invocation/run/MLflow association, all three successful ordered stage receipts,
original image and source associations, configuration snapshot digests, actual
raw/prepared/model/report bytes, and evaluation/gate/report associations. These
are checks against retained evidence. Recorded `FINISHED` and passed-gate values
are not a fresh confirmation of any remote service or independent reproduction
of model quality. The original container paths remain original evidence; the
explicit local directories select the current bytes being checked.

A successful workspace contains five run-specific `.dvc` pointers alongside
`artifacts/<run_id>/...`, its own `.dvc/cache`, isolated configuration/site-cache
directories, exact receipt/provenance copies, command records and
`link_manifest.json`. The manifest binds actual SHA-256/size, local DVC MD5/cache
objects, original training Git/image/source/configuration evidence and run IDs.
Later registrar source identity and registration timestamps are recorded
separately. Current code or a newer image never replaces original training
origin. Failure diagnostics remain in the claimed workspace without a success
manifest. Cache objects are checked as actual bytes before success is recorded.

Each supplied original pointer retains its raw SHA-256 and a separately labelled
comparison hash using only byte replacement of CRLF with LF. Original pointers
are never rewritten. Text equivalence establishes no raw/prepared dataset
association; artifact identity comes from actual bytes and the selected run's
recorded SHA-256/size. Historical repository DVC configuration and cache remain
untouched. Global/system DVC configuration, authentication overrides and shared
site caches are not inherited; analytics and experiment sharing are disabled.

Newly produced technical validation reports now include the estimator's
`get_params(deep=False)` configuration under
`model_validation.estimator_configuration`. JSON values, including `None`, are
preserved; unsupported parameter types and non-finite values receive explicit
descriptions. This is configuration evidence, not a fitted-state snapshot.
The reference forest remains unseeded (`random_state=None`, `n_jobs=-1`) and
preprocessing/training/prediction semantics are unchanged. Historical reports
remain immutable. If an original report lacks parameter evidence, the link
manifest records `missing_in_original_validation_report`; the registrar never
loads a model or reconstructs old parameters from current defaults.

The offline fixture suite is `src/versioning/test_register_pipeline_run.py`.
Set `RUN_VERSION_TEST_ROOT` to a fresh private directory outside the checkout,
then invoke it through the selected Python interpreter with `src` on its module
path. It uses opaque artifacts, model doubles and one real isolated local DVC
registration. Fixtures, command logs and the generated manifest/pointers are
retained under that directory. No `fit`, `predict` or `score` is permitted.
The application phase performs one targeted suite invocation without retry.

Local registration proves no DagsHub availability or transfer. Additional MLflow
parameter fields, structured split/test-size logging, automatic registration,
Airflow, configured triggers, monitoring and versioned controlled delivery
remain separate work. Full Level 1 acceptance is not established; Level 2 is a
separate extension. `approved_for_delivery=false` remains in effect. The dated
Windows/Docker validation above is historical evidence, not validation of this
new registrar or a newer runtime image.
