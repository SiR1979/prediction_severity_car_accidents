Prediction of car accident severity on French streets
==============================

DataScientest MLOps project (road accidents, 2021 data). A reference Random
Forest predicts a binary severity class from 28 prepared features. The project
focuses on the MLOps around it: reproducible stages, data and model versioning,
experiment tracking, a prediction API and CI.

Team: Roy Franke, Simon Ratz. Status: NON_FINAL, educational demonstration;
`approved_for_delivery` is false. Roadmap: Phase 2 (tracking, versioning,
microservices, simple orchestration) due 19 November 2026.

Components
------------

| Component | Where | Documentation |
| --- | --- | --- |
| Stage pipeline prepare → train → evaluate | `dvc.yaml`, `src/pipeline_stages.py` | [docs/orchestration.md](docs/orchestration.md) |
| One container per stage | `compose.pipeline.yaml` | [docs/orchestration.md](docs/orchestration.md) |
| Evaluation and quality gate | `src/evaluate_run.py`, `src/accept_run.py`, `src/acceptance_criteria.json` | [docs/evaluation.md](docs/evaluation.md) |
| Data and model versioning (DVC, DagsHub remote) | `.dvc/config`, `versioning/`, `dvc.lock` | [docs/versioning.md](docs/versioning.md) |
| Experiment tracking (MLflow, local or DagsHub) | `compose.tracking.yaml`, `src/tracking/`, `scripts/run_tracked_pipeline.ps1` | [docs/tracking.md](docs/tracking.md) |
| Prediction API and browser app | `compose.api.yaml`, `compose.app.yaml`, `src/api/`, `src/app/` | [docs/api-contract.md](docs/api-contract.md), [docs/local-app.md](docs/local-app.md) |
| Architecture overview | | [docs/architecture.md](docs/architecture.md) |

Project organization
------------

    ├── dvc.yaml, dvc.lock     <- Stage pipeline and its recorded hashes
    ├── compose*.yaml          <- Runtime, pipeline stages, tracking, versioning, API, app
    ├── Dockerfile             <- Runtime image (Python 3.11, pinned requirements-runtime.txt)
    ├── data/                  <- Git-ignored; raw data via DVC, pipeline and run outputs
    ├── docs/                  <- Project documentation
    ├── scripts/               <- PowerShell launchers (tracked runs, candidate registration)
    ├── versioning/            <- DVC pointers and manifests for reference artifacts
    └── src
        ├── data/              <- Raw-data import and reference preprocessing
        ├── models/            <- Reference training and manual prediction
        ├── pipeline_stages.py <- prepare, train and evaluate stage entry points
        ├── run_pipeline.py    <- Single-run pipeline used for tracked runs
        ├── evaluate_run.py    <- Descriptive evaluation against a majority baseline
        ├── accept_run.py      <- Quality gate (team criteria)
        ├── tracking/          <- MLflow tracking client and historical import
        ├── versioning/        <- DVC registration tools
        ├── api/               <- FastAPI prediction service
        └── app/               <- Browser app for prepared inputs

Quick start
------------

Run all commands from the repository root. Python 3.11 matches the runtime
image and CI.

### 1. Environment

    python3.11 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements-runtime.txt dvc==3.67.1

### 2. Raw data

Set up your own DagsHub access once (see [docs/versioning.md](docs/versioning.md)),
then download the four raw CSV files:

    dvc pull versioning/raw_2021.dvc

Without DagsHub access, `python src/data/import_raw_data.py` downloads the
files from the original source.

### 3. Run the pipeline

    dvc repro
    dvc metrics show

The same stages run as containers with
`docker compose -f compose.pipeline.yaml up --build`; see
[docs/orchestration.md](docs/orchestration.md) for the run directory.

### 4. Tracked run, API and app

- Tracked runs on the local or DagsHub MLflow server: [docs/tracking.md](docs/tracking.md)
- Prediction API on port 8000 and browser app on port 8080: [docs/local-app.md](docs/local-app.md)

Tests
------------

    pip install -r src/api/requirements.txt -r src/tracking/requirements.txt pytest
    pytest
    node --test src/app/test_app_logic.cjs

CI (`.github/workflows/python-app.yml`) runs flake8, pytest and the JavaScript
tests on every pull request to `master`. Tests use doubles and perform no real
training or DVC transfer.

Known limitations
------------

The inherited target mapping, severity-based row selection and the unseeded
estimator are preserved on purpose and documented in
[docs/architecture.md](docs/architecture.md). The supervisor left the quality
thresholds to the team; they are set in `src/acceptance_criteria.json`.

------------------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
