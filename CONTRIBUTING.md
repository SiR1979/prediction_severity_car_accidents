# Contributing

How the three of us work on this repository. Added 2026-11-02 when the team
grew from two to three members. Change these rules like any other file: in a
reviewed pull request.

## Team and areas

Each area has one owner. The owner keeps the overview, decides open questions
in the area and reviews changes to it. Everyone may contribute everywhere.

| Area | Owner | Main files |
| --- | --- | --- |
| Orchestration: stage pipeline, Airflow | Roy Franke (roy-franke) | `dvc.yaml`, `src/pipeline_stages.py`, Airflow DAGs |
| Tracking and versioning: MLflow, DVC, DagsHub, Windows launcher | Simon Ratz (SiR1979) | `src/tracking/`, `src/versioning/`, `versioning/`, `scripts/` |
| Serving and operations: API, app, Kubernetes, monitoring | Mohsen (JamshidiML) | `src/api/`, `src/app/`, `compose.api.yaml`, `compose.app.yaml`, deployment and monitoring files |

The split is a proposal for Phases 3 and 4; confirm or change it in the weekly
sync. `docs/architecture.md` is shared: each owner updates the section for
their area.

## Roadmap

| Phase | Deadline | Content |
| --- | --- | --- |
| 1 | 2026-10-22 | Foundations, baseline model, unit tests, basic API |
| 2 | 2026-11-19 | MLflow tracking, data and model versioning, microservices, simple orchestration |
| 3 | 2026-12-10 | Airflow orchestration, CI, API optimization and security, Docker/Kubernetes scaling |
| 4 | 2027-01-18 | Prometheus/Grafana monitoring, Evidently drift detection, automated updates, final documentation |
| Defense | around 2027-01-25 | Final presentation |

The course prefers DVC/DagsHub and Airflow. Propose an alternative only with a
named reason.

## Onboarding checklist

Work through this list in the first week. Anything that does not work as the
README describes is a good first pull request.

- [ ] Read the README, `docs/architecture.md` and `docs/orchestration.md`.
- [ ] Create a DagsHub account and access token; set up DVC access as in
      `docs/versioning.md` (the token goes only into `.dvc/config.local`).
- [ ] Create a Python 3.11 environment and install the requirements (README).
- [ ] `dvc pull versioning/raw_2021.dvc`, then `dvc repro` and `dvc metrics show`.
- [ ] Run `pytest` and `node --test src/app/test_app_logic.cjs`.
- [ ] Start the API and app as in `docs/local-app.md` and send one prediction.
- [ ] Optional: start local MLflow as in `docs/tracking.md`.
- [ ] Review one open pull request, even if another review already approves it.
- [ ] Agree on your area and first task in the weekly sync.

## Pull requests

- Never commit or push to `master` directly. Work on a branch named
  `work/<short-topic>` and open a pull request to `master`.
- One topic per pull request, small enough to review in one sitting.
  Documentation changes go into the same pull request as the code they describe.
- Merge only with one approval from another team member and green CI
  (Python and JavaScript jobs). The owner of the affected area reviews; if you
  are the owner, ask one of the other two. Rotate reviewers so that knowledge
  spreads.
- Review within two working days. If you cannot, say so in the pull request.
- Describe what changed and how you verified it, including what you did not
  verify (for example "Compose chain not run on Windows").
- Write commit messages in English, imperative mood, with a short summary line.

## Code and data

- Match the style of the surrounding code and keep the existing checks:
  outputs are created new and never overwritten, linked paths are rejected,
  hashes are recorded.
- Tests use doubles. CI performs no real training, DVC transfer or deployment.
  Add tests with every behaviour change.
- Never commit credentials, tokens, `.env` files or data. `data/` and model
  files are ignored by Git and versioned with DVC.
- Push DVC data only with explicit targets, for example
  `dvc push --remote dagshub prepare train evaluate`. A plain `dvc push` can
  upload more than intended.
- Keep the approval fields honest. `approved_for_delivery` stays false until
  the team decides on delivery. A passed quality gate (`src/acceptance_criteria.json`)
  is not delivery approval. Change the criteria only in a pull request that both
  other members approve.
- Preserve the inherited target mapping and row selection unless the team
  agrees on a documented change.

## Communication

- **Weekly sync (30 minutes):** what is done, what comes next, where someone is
  blocked. Decisions from the sync go into the affected document or pull request.
- **Task list:** one GitHub issue per task with a phase label and an assignee,
  so everyone can see who works on what.
- **Chat:** quick questions and coordination. Anything that needs to be found
  again later belongs in an issue, a pull request or the documentation.
