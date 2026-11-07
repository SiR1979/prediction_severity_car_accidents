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
