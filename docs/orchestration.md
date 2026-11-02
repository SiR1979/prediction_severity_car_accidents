# Stage pipeline and orchestration

Project: car_accident. Added 2026-11-01. Status: NON_FINAL.
Delivery approval: false.

## Scope

Phase 2 of the roadmap asks for microservices and a simple orchestration. The
reference workflow is split into three stages that run in a fixed order:

| Stage | Reads | Writes into `data/pipeline/` |
| --- | --- | --- |
| prepare | `data/raw/` (four 2021 CSVs) | `preprocessed/`, `data_validation.json` |
| train | `preprocessed/`, `data_validation.json` | `model/trained_model.joblib`, `validation_report.json` |
| evaluate | `preprocessed/`, model, `validation_report.json`, `src/acceptance_criteria.json` | `evaluation_report.json`, `acceptance_report.json`, `metrics.json` |

All stages call `src/pipeline_stages.py`. Preprocessing, training, technical
validation and evaluation are the unchanged reference functions also used by
`src/run_pipeline.py`. No stage overwrites an existing output.

Two entry points run the same commands:

- `dvc.yaml` with `dvc repro`: reruns only changed stages and records hashes in
  `dvc.lock`. This is the orchestration for Phase 2.
- `compose.pipeline.yaml`: one finite container per stage from the runtime
  image. A service starts only after the previous one completed successfully.

The stages record no MLflow run. Tracked runs keep using the `pipeline-run` job
and the launcher (tracking.md). Airflow follows in Phase 3, as confirmed by the
supervisor, and is to call the same stage commands.

## Run with DVC

From the repository root, with the runtime requirements and DVC 3.67.1
installed and the raw data pulled (versioning.md):

```bash
dvc repro
dvc metrics show
```

A second `dvc repro` without changes reports that the pipeline is up to date.
After a change, `dvc metrics diff` compares the metrics with the last committed
`dvc.lock`. Commit `dvc.lock` together with the change that produced it.

The estimator has no fixed seed, so a rerun of train creates a different model
file and new hashes in `dvc.lock`.

## Run with Docker Compose

The run directory must exist and be empty:

```bash
mkdir data/pipeline_compose
PIPELINE_RUN_PATH=./data/pipeline_compose docker compose -f compose.pipeline.yaml up --build --abort-on-container-failure
docker compose -f compose.pipeline.yaml down
```

`PIPELINE_RUN_PATH` defaults to `./data/pipeline`, the DVC run directory. Use a
separate directory so that Compose runs do not change files that DVC manages.
The containers have no network, a read-only root filesystem, read-only raw data
and no added capabilities.

## Quality gate

Evaluate compares the evaluation metrics with the team criteria in `src/acceptance_criteria.json`:

| Criterion | Minimum | Reason |
| --- | --- | --- |
| Macro F1 | 0.65 | Both classes count equally; the training majority baseline reaches about 0.40 |
| Balanced accuracy | 0.65 | Mean recall of both classes; the baseline is 0.50 by construction |
| Macro F1 gain over the training majority baseline | 0.20 | The model must clearly beat always predicting the majority class |

A run on 2026-11-01 (macOS, Python 3.11, real 2021 data) measured macro F1
0.741, balanced accuracy 0.733 and a gain of 0.346. The thresholds leave a
margin for the unseeded estimator. The supervisor left the criteria to the
team (answer reported on 2026-11-01); changes need the agreement of both team
members in a reviewed pull request. The project has no confirmed priority class, so there is no
per-class recall criterion yet; it can be added once the target meaning is
confirmed.

`acceptance_report.json` lists each check with observed value, minimum and
result. Each comparison allows a tolerance of 1e-9 so that floating-point rounding
does not reject a value exactly at a minimum (for example a gain of
0.69 - 0.49 = 0.19999999999999996). A real shortfall such as 0.1999999 still fails. A failed gate makes evaluate exit with code 1: `dvc repro` and the
Compose chain stop, and `dvc.lock` keeps the previous state. Errors exit with
code 2. A passed gate sets `model_quality_approval` to `TEAM_CRITERIA_MET`;
a failed gate keeps `NOT_ESTABLISHED`. `approved_for_delivery` stays false in
both cases: delivery remains a separate step.

## Versioning on DagsHub

The stage outputs are DVC outputs. To share them on the DagsHub remote, push
exactly these stages:

```bash
dvc push --remote dagshub prepare train evaluate
```

A colleague restores the same prepared data and model with
`dvc pull --remote dagshub prepare train evaluate` after checking out the
matching `dvc.lock`. Authentication is described in versioning.md.

## Tests

`src/test_pipeline_stages.py` checks the stages with doubles and that
`dvc.yaml` and `compose.pipeline.yaml` run identical commands.
`src/test_accept_run.py` checks the gate. CI performs no real training.
