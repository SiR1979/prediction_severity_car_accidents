# Descriptive reference-run evaluation

Status: integrated local workflow implementation. Delivery approval remains false.

`src/evaluate_run.py` is a noninteractive evaluation command for an existing,
trusted local run produced by `src/run_pipeline.py`. It reads a passed technical
validation report, the matching model, prepared held-out features and labels,
and training labels. It never trains, registers, starts services or switches a
model. The same evaluator is now a required stage of new reference-run workflows.

## Reference-run integration

`src/run_pipeline.py` writes and closes `validation_report.json` before calling
the evaluator. The new, separate `evaluation_report.json` belongs to that run.
Both tracked and untracked workflows require this stage. Pipeline and evaluator
import their feature names and contract version from `src/reference_contract.py`;
there is no circular dependency or second contract definition between them.

`run_status.json` records technical validation and evaluation separately. A passed
technical report remains a technical result if evaluation or tracking later fails.
Overall status becomes `completed` only after evaluation and enabled tracking
finish successfully. Existing failure handling records the original error,
preserves the reports/model/prepared files and attempts the existing tracking
failure record without retrying evaluation or artifact/metric operations.
Secondary failure-recording errors do not replace the original exception.

Tracking includes the evaluation report in the same artifact manifest as the
model and prepared files. Its run ID, contract, model/report identities and all
consumed input hashes must match that run's recorded artifacts. The provenance
also links the full baseline definition, class counts, metrics and input
identities to the same tracking run.

The same run receives six descriptive metrics: `evaluation_accuracy`,
`evaluation_balanced_accuracy`, `evaluation_macro_f1`,
`training_majority_accuracy`, `training_majority_balanced_accuracy` and
`training_majority_macro_f1`. Artifact hash verification and metric/tag/parameter
readback are required before local workflow completion. A transfer, metadata or
readback failure fails the workflow even when its evaluation is already complete.

The launcher's source inventory now includes the evaluator and shared contract.
A future authorised tracked invocation needs those exact sources in its pinned
runtime image; this integration does not rebuild an image or run the launcher.

The report records unrounded accuracy, balanced accuracy and binary macro-F1.
The comparison baseline predicts the training-label majority on the same held-out
rows; the held-out majority does not select the baseline. A training-count tie
selects code 0. Both held-out classes must be present for this binary comparison.
No domain meaning is assigned to codes 0 and 1.

Each report records the observed SHA-256 and byte size of the technical report,
model and three consumed CSVs. The model digest must match technical validation
before deserialization, and the fitted estimator must match the existing
prepared-features-v1 contract. Input changes detected before report creation
fail the command. These checks identify observed files; they do not independently
prove historical training/split lineage or establish prediction-time availability.

The output file must be new and its parent directory must already exist. Existing
reports, inputs and model bytes are preserved. Linked paths are rejected. Success
returns exit 0; any evaluation or output error returns exit 1. A failed output
write may leave a partial new file; preserve it and select a different output
only in a separately authorised subsequent action.

## Invocation

Start from the repository root with a compatible Python environment activated
and the repository runtime requirements installed. For a selected existing run,
set the two explicit paths before invoking the command:

```powershell
$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$runDirectory = Read-Host 'Existing trusted reference run directory'
$outputPath = Read-Host 'New evaluation JSON path in an existing directory'
& python -B `
    (Join-Path $repo 'src\evaluate_run.py') `
    --run-dir $runDirectory --output $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; preserve all reached evidence.' }
```

The Python command accepts explicit arguments without interactive prompts; the
example prompts only select paths for a manual invocation. An authorised trigger
can supply those same arguments directly.

## Acceptance boundaries

This step integrates evaluation and tracking for a new reference run. Configured
triggering, versioned delivery and agreed monitoring remain separate.
Metrics do not enforce quality thresholds, approve a model or authorise serving.
Every report retains `quality_thresholds_enforced=false`,
`model_quality_approval=NOT_ESTABLISHED` and `approved_for_delivery=false`.
Inherited target mapping, severity-dependent selection and feature-availability
limitations remain. No Level 1 completion or operational suitability is claimed.

The evaluator tests run its complete CLI against a small fitted fixture and
check known metrics, a training-only baseline, identity drift and preservation.
The workflow integration tests use isolated prepared files, estimator/loading
substitutes and a tracking client substitute. They cover stage order, report
separation, tracking linkage, missing metric readback and failure propagation.
No real project training, existing project-model evaluation or live tracking
service write is part of this verification.
