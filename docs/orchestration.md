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

Three entry points run the same commands:

- `dvc.yaml` with `dvc repro`: reruns only changed stages and records hashes in
  `dvc.lock`. This is the orchestration for Phase 2.
- `compose.pipeline.yaml`: one finite container per stage from the runtime
  image. A service starts only after the previous one completed successfully.
- `compose.airflow.yaml` with the DAG `reference_pipeline` (Phase 3, #26): one
  Airflow task per stage, started manually. See "Run with Airflow" below.

The stages themselves record no MLflow run. Airflow records each run after
evaluate with `src/tracking/track_stage_run.py` ("Track Airflow runs in MLflow"
below, #27). The same recorder is meant for `dvc repro` (`--orchestrator dvc`);
that integration is still open in #27. Tracked runs of the launcher keep using
the `pipeline-run` job (tracking.md).

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

## Run with Airflow

Airflow 3.3.2 runs locally with the LocalExecutor and PostgreSQL. The DAG
`reference_pipeline` (`airflow/dags/`) has no schedule: a run starts only when
someone triggers it, and then runs without further input:

```text
create_run_directory -> prepare -> train -> evaluate -> track -> finish
(finish also waits directly for evaluate)
```

`create_run_directory` creates a new directory `data/airflow_runs/<run ID>/`
and writes `airflow_run.json` with the Airflow run ID, the source commit of the
image and the criteria file. It fails if the directory already exists, so no
run reuses outputs. The three stage tasks then call the commands of `dvc.yaml`
with this run directory, using the runtime requirements in their own virtual
environment (`/opt/runtime` in `airflow/Dockerfile`). Tasks are not retried.
`track` runs after evaluate even when a stage failed and records the run in
MLflow (next section). `finish` does nothing itself; it needs both evaluate and
track to succeed, so the DAG run fails when the gate fails or the run could not
be recorded.

Start Airflow from the repository root, with the raw data in `data/raw/`
(versioning.md). Build from a clean, committed state so that the recorded
commit describes the code that ran:

```bash
mkdir -p data/airflow_runs
SOURCE_COMMIT=$(git rev-parse HEAD) docker compose -f compose.airflow.yaml up -d --build --wait
```

The UI is at http://127.0.0.1:8081 (user and password `airflow`, or
`AIRFLOW_ADMIN_USER` and `AIRFLOW_ADMIN_PASSWORD` from `.env` on the first
start). The setup is for local use only: it binds to 127.0.0.1 and uses default
local secrets. On Linux, set `AIRFLOW_UID=$(id -u)` so that run files belong
to your user.

Trigger a run in the UI (DAG `reference_pipeline`, button "Trigger") or with
the CLI:

```bash
docker compose -f compose.airflow.yaml exec airflow-scheduler airflow dags trigger reference_pipeline
docker compose -f compose.airflow.yaml exec airflow-scheduler airflow dags list-runs reference_pipeline
```

Stop with `docker compose -f compose.airflow.yaml down`; add `-v` to also
delete the Airflow database and logs. Run directories stay in
`data/airflow_runs/` (ignored by Git).

### A failed quality gate in Airflow

The parameter `quality_gate` selects the criteria: `team` (default,
`src/acceptance_criteria.json`) or `failure_demo`
(`airflow/gate_failure_demo_criteria.json`, all minima 1.0). Other values are
rejected when the run is triggered. The demo only makes the gate stricter; it
never approves a model.

With the reference data the gate fails in every `failure_demo` run, whatever
the model: the training majority baseline has a macro F1 of about 0.396, so the
macro F1 gain over that baseline can be at most about 1 − 0.396 = 0.604, below
the demo minimum of 1.0. In the run shown in PR #38
(`manual__2026-11-18T14:33:54.044970+00:00`) all three checks failed:

| Check | Observed | Demo minimum |
| --- | --- | --- |
| macro F1 | 0.7429 | 1.0 |
| balanced accuracy | 0.7341 | 1.0 |
| macro F1 gain over training majority (baseline 0.3955) | 0.3474 | 1.0 |

```bash
docker compose -f compose.airflow.yaml exec airflow-scheduler airflow dags trigger reference_pipeline --conf '{"quality_gate": "failure_demo"}'
```

A failed gate shows up in three places:

- DAG grid: the task `evaluate` is red (failed), `finish` is orange
  (upstream failed) and the run is marked failed; `create_run_directory`,
  `prepare`, `train` and `track` stay green.
- Log of `evaluate`: one line per check, for example
  `Gate check macro_f1: observed 0.7429, minimum 1.0000, failed.`, then
  `Quality gate failed.` and `Command exited with return code 1`.
- Run directory: `acceptance_report.json` with `gate_status: failed` and
  `model_quality_approval: NOT_ESTABLISHED`.

- MLflow: the run of this Airflow run has status `FAILED`, tag
  `gate_status=failed` and `failed_stage=evaluate`.

A stage error (exit code 2) also fails the task; its log shows the exception
instead of the gate checks. `track` still records the run as `FAILED` with the
outputs that exist and `failed_stage` set to the stage that did not complete.

### Track Airflow runs in MLflow

The task `track` calls `python -m src.tracking.track_stage_run` in the runtime
environment. The parameter `tracking_target` selects where it records:

| `tracking_target` | MLflow | Needs |
| --- | --- | --- |
| `local` (default) | `http://mlflow:5000`, the service of `compose.tracking.yaml` | MLflow running and `compose.airflow.tracking.yaml` |
| `dagshub` | `https://dagshub.com/SiR1979/prediction_severity_car_accidents.mlflow` | `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in the shell that starts Airflow, and `compose.airflow.tracking.yaml` |
| `none` | nothing recorded; `track` only logs that tracking is off | – |

Start MLflow first, then Airflow with the tracking overlay, which connects the
scheduler to the MLflow network `car-accident_default` and passes the DagsHub
credentials by name:

```bash
mkdir -p data/tracking data/airflow_runs
docker compose -f compose.yaml -f compose.tracking.yaml up -d --wait mlflow
SOURCE_COMMIT=$(git rev-parse HEAD) docker compose -f compose.airflow.yaml -f compose.airflow.tracking.yaml up -d --build --wait
docker compose -f compose.airflow.yaml exec airflow-scheduler airflow dags trigger reference_pipeline --conf '{"tracking_target": "local"}'
```

Each Airflow run becomes one MLflow run in the experiment
`car_accident_stage_runs`, named after the run directory. Find it by the
Airflow run ID in the MLflow UI (http://127.0.0.1:5000) with the filter
`tags.airflow_run_id = '<run ID>'`. The run holds:

- Tags: `orchestrator=airflow`, `orchestrator_run_id` and `airflow_run_id`,
  `source_commit`, `gate_status`, `model_quality_approval`, `failed_stage`,
  `approved_for_delivery=false`, the SHA-256 of the criteria file and the
  versions below.
- Parameters: `quality_gate`, criteria version and minima, contract version and
  the estimator configuration from `validation_report.json`.
- Metrics: model and training majority metrics, the macro F1 gain and
  `quality_gate_passed` (1 or 0).
- Artifacts: all run outputs (reports, prepared CSVs, model), `airflow_run.json`
  and `run_versions.json`.

Data and model versions use the same MD5 hash that DVC writes into
`dvc.lock`, plus SHA-256: `raw_data_dvc_md5`, `prepared_data_dvc_md5` (a
`.dir` hash over the four CSVs), `model_dvc_md5` and `model_sha256`.
`run_versions.json` lists every file with SHA-256 and size. A prepared data
hash equal to the one in `dvc.lock` means the bytes are identical to the DVC
output; the model hash differs per run because the forest is unseeded.

Before it creates anything in MLflow, the recorder checks the gate report
against the criteria and the current bytes of the evaluated inputs (the same
check as staged runs, staged_tracking.md), that `airflow_run.json` belongs to
this Airflow run, and that no MLflow run exists for it yet. Every artifact is
downloaded again and compared by SHA-256; tags, parameters and metrics are read
back. The result, including the MLflow run ID, is written to
`mlflow_tracking.json` in the run directory. A run is never recorded twice; if
`track` fails, the run directory keeps the reason and the MLflow run, if one was
created, is set to `FAILED`.

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
`src/test_airflow_dag.py` checks that the Airflow tasks run the `dvc.yaml`
commands in the same order, followed by `track` and `finish`; its DAG checks
run in the CI job "Airflow DAG", which builds the Airflow image and loads the
DAG there. `src/tracking/test_track_stage_run.py` checks the recorder with a
fake MLflow transport, and that its prepared data hash equals `dvc.lock` when
local `dvc repro` outputs exist.
`src/test_accept_run.py` checks the gate. CI performs no real training.
