# Local reference artifact versioning

Project: car_accident  
Status: NON_FINAL  
Implementation addition: 2026-10-08

Configuration repair: 2026-10-09; successful local repair output is required.

## Scope

The separate finite tool in `compose.versioning.yaml` pins DVC 3.67.1 and uses
Python 3.12. It registers the existing reference artifacts, transfers their
contents to a new local store, and verifies an isolated recovery. It never fits
or loads a model, changes delivery approval, or replaces the serving artifact.
Successful Windows installation output establishes validation with real files.

| Metadata file under `versioning/` | Selected artifact |
| --- | --- |
| `raw_2021.dvc` | Four supplied files in `data/raw/` |
| `reference_preprocessed.dvc` | Four original files in `data/preprocessed/` |
| `preserved_reference_model.dvc` | `src/models/trained_model.joblib` |
| `validated_reference_run.dvc` | Six files under `data/runs/validation_20261003T193310Z_56c49087/reference_run/` |
| `validation_report.dvc` | `data/runs/validation_20261003T193310Z_56c49087/validation_report.json` |

These are five groups and 16 source files. The validated run consists of four
prepared CSVs, the selected model and its run status. Unexpected source files or
linked paths stop registration. The original and validated prepared files must
match. The two model files retain distinct roles and are never substituted.

## Storage and boundaries

| Location | Purpose |
| --- | --- |
| `.dvc/cache/` | Ignored working cache; copy mode |
| `/tmp/dvc-site-cache/` in the tool container | Temporary internal state cache, selected by `DVC_SITE_CACHE_DIR` |
| `versioning/*.dvc` | Small Git-versioned pointers using `wdir: ..` |
| `versioning/reference_manifest.json` | File SHA-256 values, roles and registration provenance |
| `.dvc/config.local` | Ignored machine-specific local-store configuration |
| `C:\Users\simon\MLOps_car_accident_dvc_store\objects\` | Separate local DVC object store, mounted at `/store/objects` |
| `C:\Users\simon\MLOps_car_accident_dvc_store\validation\<UTC>_<suffix>\` | Preserved recovery directory and validation report |
| `data/code_backups/versioning_<UTC>_<suffix>/architecture.md` | Exact architecture backup before installation |
| `data/code_backups/versioning_resume_<UTC>_<suffix>/` | Exact repair backups of four updated files, the partial config and DVC ignore rules |

The existing `/data/` and model exclusions remain. Metadata sits outside the
ignored data directory. DVC's own content identifiers are separate from the
SHA-256 values used to verify preservation and recovery.

The tool has no port or runtime network. Docker build requires dependency
downloads. At runtime, the repository metadata and cache are writable, while
`data/` and the preserved model file are mounted read-only. The local object
store is writable. Global host configuration and credentials are not mounted.
No Git hooks are installed. Normal DVC initialization temporarily stages three
new metadata files; only those are unstaged after an empty-index guard.

The internal state cache is separate from the persistent content cache and
local object store. DVC 3.67.1 defaults to `/var/tmp/dvc`, which is outside the
tool's writable temporary mounts. The October 9 read-only diagnosis observed
that default path on a read-only filesystem, functioning Git backends, an empty
Git index, an empty `.dvc/config`, the generated `.dvcignore`, and no content
cache or versioning pointers. This explains the incomplete initialisation.
The environment setting redirects internal state to the existing `/tmp` mount.

The bounded repair completes only that recognised partial state. It preserves
the existing DVC files and empty storage directories, creates new evidence in
another directory, and rejects any existing object, validation file or pointer.
The empty configuration is populated after its original bytes are backed up;
the existing `.dvcignore` remains unchanged.
It never uses `dvc init --force`. Recovery uses a new repository and normal
initialisation. Initialization errors remain visible in captured command output.
The repair builds `car-accident-versioning:resume-20261009`; the previous local
image remains preserved. Only the versioning tool is rebuilt.

The local store is on the same drive as the repository. It is not protection
against drive loss and is not accessible to external CI. A different approved
storage destination and release policy are required for such later use.
MLflow storage, reference snapshots, code backups and assets are excluded.

## Integrity and provenance

The four raw file hashes, selected model, report and status are bound to the
previously supplied validation evidence. Other selected file hashes are captured
at registration. The preserved original model's historical hash was not supplied
independently; this phase records its current bytes without claiming its original
training identity. The current raw and model content is never reinterpreted.

Registration records its own baseline Git HEAD, dirty worktree, DVC version and
actual tool image identity, executed tool hash and registration mode. It does
not assign those values to the historical training run. Original training Git
revision and image remain unknown.
The reports retain `approved_for_delivery=false`.

The resolved tool requirements remain inside its image at
`/opt/versioning-requirements-resolved.txt`. The tool image uses the base tag
`python:3.12-slim-bookworm`; full transitive and base-image locking remain open.

## Validation and failure handling

One phase registers the five groups, checks DVC status, transfers objects only to
the local store and pulls them into a new recovery repository with an initially
empty cache. All 16 restored files must match the initial SHA-256 values. The
original files, Git exclusion rules, baseline HEAD and empty index are checked
again. The recovery cache cannot fall back to the primary cache.

An opaque preparation fixture exercises real DVC 3.67.1 with the same 16-file
layout. It is not a model or evidence of Windows, Docker or real-file execution.
The repair preparation exercises the diagnosed partial initialisation and
preservation of existing metadata and empty evidence directories.

Installation leaves source files and generated metadata uncommitted. Existing
pipeline, API, app and MLflow services are not rebuilt or tested. A failure
preserves new source files, cache, local store, recovery files, backup and output.
Do not rerun the one-time installer on a partial installation or clear storage
to make it pass. Diagnose the reported phase first.

## October 11 candidate: bounded registration and recovery

The separate candidate tool selects one opaque model, not all 19 artifacts of
the completed tracked training invocation. Historical pointers, the one-time
reference installer, configuration, primary cache and Compose selection remain
unchanged. Training receipts retain `approved_for_delivery=false`.

| Identity | Value |
| --- | --- |
| Run | `reference_20261011T144150Z_19d3274fd12e49d5b3cb9abbd73da5ea` |
| Invocation | `19d3274fd12e49d5b3cb9abbd73da5ea` |
| MLflow run | `dcaea4d2f6ef4b3087f86e8bde3365d7` |
| Model relative path | `data/runs/reference_20261011T144150Z_19d3274fd12e49d5b3cb9abbd73da5ea/model/trained_model.joblib` |
| Size | `107180009` bytes |
| SHA-256 | `9dcd169ed58e0b09a2367b17055abee4b7424741d1b02fa60aa196fe049f6b99` |
| DVC MD5 | `b2d9206c3d45a8c77599915c46e8cacb` |
| Pointer | `versioning/candidate_20261011_19d3274f_model.dvc` |
| Identity manifest | `versioning/candidate_20261011_19d3274f_manifest.json` |

`scripts/register_candidate.ps1` requires fresh explicit authorization and the
unpublished local branch `work/candidate-recovery` at selection commit
`9f7028d9702f006159d8d68b067863f6db77a766`, with an empty index. Its bounded
worktree guard accepts only the two new tools and this documentation change
before generated candidate metadata exists, plus the three preserved local PNGs.
It rejects existing candidate metadata, phase directories or candidate objects.
Do not rerun it against any partial or completed attempt.

Prepare a unique directory beneath the existing store's `validation/`, named
`candidate_20261011_19d3274f_<UTC>_<UUID>`, and preserve the original documentation
bytes as `original_versioning.md` before editing. The launcher takes this absolute
directory as its required `-EvidenceDirectory` argument. It retains command
records, a sanitized context, original-file hashes, container mount inventories
and `report.json` there. It records allowlisted identities and hashes without
publishing complete configurations or environments.

The actual tool image is pinned to
`sha256:02d4a4e4c447f07446899843b89bb459edf6e79f561ccb8008974fccd1abbfb9`.
The new `src/versioning/register_candidate.py` is mounted read-only; no build or
image pull occurs. Before DVC writes, the tool requires actual DVC version
`3.67.1`. Containers have no network or ports, a read-only root, dropped
capabilities and no host credentials or repository mount. They are retained
after exit so their identities and partial state remain inspectable.

Registration uses a new `registration/` directory and its own empty content
cache. Only the selected original model is mounted read-only at its relative
path. DVC initializes this workspace with `--no-scm`, uses copy mode, creates one
new pointer with `wdir: ..`, then runs targeted
`commit --force --no-relink`, `status --json` and `push --remote reference`.
This DVC commit fills the new pointer and content cache; it is not a Git commit.
The store is the existing `/store/objects`, not a replacement historical
baseline. Pointer and identity-manifest publication into the local worktree
uses exclusive creation. Their names are never overwritten.

The manifest binds the one model and pointer hash to all three run identities
and the paths/hashes of existing technical, training, provenance, artifact,
invocation and local-delivery receipts. Registration HEAD, tool hashes, actual
image and DVC version are separate from historical training identity.
Registration alone does not prove recovery; that phase has its own records.

Exactly one recovery runs in new `recovery/` with an initially empty private
cache. Its container can access only that workspace, the new identity metadata,
the read-only tool and the read-only store objects. Neither original model nor
primary or registration cache is mounted. After detached initialization it runs
one targeted `dvc pull --remote reference`. The restored model must have the
exact size and SHA-256 above; Windows independently checks those bytes.
No model is deserialized and no service is started. Host guards also verify that
historical source files, receipts, pointers, PNGs, primary-cache objects and
previous store objects remain unchanged, with exactly one new store model blob.

Any failed command or identity guard ends the attempt. Preserve its UTC time,
exit code, original error, unknown transfer outcome, metadata, caches, objects,
container and partially restored bytes. There is no automatic replay, cleanup
or overwrite. Diagnose that phase before requesting another attempt.

A later, separately authorized restoration uses the same isolated retrieval
route: local store to a new empty recovery workspace, verify SHA-256 and size,
then exclusively copy to the unchanged Compose source path only if it is absent.
Never overwrite an existing selected model. This implementation performs only
the isolated check and leaves the serving path unchanged. Recovery is local
technical evidence, not model-quality acceptance, delivery approval, protection
against drive loss or Level 1 completion. Git staging, commit, push and public
publication require separate authorization.

## Publication metadata projection

The candidate identity manifest is a metadata projection of the preserved
original manifest, whose SHA-256 is
`79bb97da8322cbc89c1b8b0d4f0f143641152786e9052b3e215d1b24e1d593b6`.
`registration.branch` is deliberately `null`, with `branch_disclosure: "withheld"`.
The neutral branch required by the corrected tools is not asserted to be the
branch used for the historical registration.

Registration HEAD, UTC time, context hash, image identity, DVC version and
executed-tool hashes remain historical identities. Those tool hashes identify
the original sources executed during the recorded registration and recovery;
they do not identify the corrected sources. The corrected tools have no new
runtime validation. Original source bytes, manifest and recovery/QA receipts
are preserved privately; their recorded hashes are not rewritten.

The recorded local recovery completed at `2026-10-12T14:05:50.8165957Z` using
the existing local store. This metadata correction repeats no registration
or recovery and adds no model-quality, service-readiness or Level 1 evidence.
`approved_for_delivery=false` and Level 1 `NON_FINAL` remain unchanged.


## DagsHub shared raw-data remote

Updated: 2026-10-14.

The project configuration in `.dvc/config` selects `dagshub` as its default
remote. The existing remote URL is:

`https://dagshub.com/SiR1979/prediction_severity_car_accidents.dvc`

This setting selects a destination; it does not upload additional artifacts.
The historical local store and machine-specific `.dvc/config.local` remain
separate from the shared project configuration.

### Shared scope and recorded validation

The documented transfer contains only the four supplied raw 2021 CSV files:

- `caracteristiques-2021.csv`
- `lieux-2021.csv`
- `usagers-2021.csv`
- `vehicules-2021.csv`

The pointer is `versioning/raw_2021.dvc`. Five DVC objects were uploaded: one
directory object and four file objects. An authenticated owner-account restore
into a new empty isolated cache recovered four files totalling 31,082,914 bytes.
All MD5 and SHA-256 hashes matched. The transfer used DVC 3.67.1.

Anonymous access returned HTTP 401 in the recorded attempt. Read access with a
second team account (`roy-franke`) was confirmed on 2026-10-14: `dvc status --cloud
versioning/raw_2021.dvc` reported cache and remote in sync. Prepared data, model files, run artifacts and
reports were not uploaded as part of this shared raw-data transfer.

### Authentication on a colleague's machine

Run these commands from the repository root with DVC available. Each person
must use their own DagsHub account and access token.

```powershell
$dagshubUser = Read-Host 'Your DagsHub username'
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user $dagshubUser
dvc remote modify --local dagshub ask_password true
```

These commands store authentication settings in the Git-ignored
`.dvc/config.local`. When DVC prompts for a password, enter your own DagsHub
access token. The prompt avoids putting the token in the command text or
history. An existing `password` setting takes precedence over `ask_password`.

Never commit credentials or send token values to another person. This
interactive setup is for manual use; unattended jobs need a separately
configured secret mechanism.

### Download the shared raw data

On a clone without a local default override:

```powershell
dvc pull versioning/raw_2021.dvc
```

The project default removes the need to specify `--remote dagshub`.

A machine-specific `core.remote` in `.dvc/config.local` overrides the project
default. The existing owner's local default remains `reference`. To select
DagsHub without changing that local configuration, use:

```powershell
dvc pull --remote dagshub versioning/raw_2021.dvc
```

A plain `dvc pull` considers all current DVC pointers, including prepared data
and models. Setting the default does not establish that those objects exist
on DagsHub. Full repository restoration from this shared remote has not been
validated; use the targeted raw-data command for the documented scope.

### Upload boundary and historical workflows

For a separately authorised update of the four raw files, the bounded transfer
command is:

```powershell
dvc push --remote dagshub versioning/raw_2021.dvc
```

An unqualified `dvc push` can select other tracked artifacts. Changing the
default is not permission to upload models or expand the shared scope.

Historical registration and recovery remain completed one-time procedures.
Do not rerun them to configure this remote. Raw-data transfer and restore
evidence does not establish model quality, delivery approval or Level 1
completion. `approved_for_delivery=false` remains unchanged.

Configuration references:

- [DVC default remote](https://doc.dvc.org/command-reference/remote/default)
- [DVC pull targets](https://doc.dvc.org/command-reference/pull)
- [DVC HTTP authentication](https://doc.dvc.org/user-guide/data-management/remote-storage/http)

## Remaining work

The raw 2021 dataset now has a shared DagsHub remote. Automatic registration of
new training outputs, shared storage for prepared data and models, colleague
access validation, CI access, model acceptance and controlled delivery remain
separate steps. Level 1 remains incomplete; Level 2 is tracked separately.

## References

- [DVC 3.67.1 release](https://github.com/treeverse/dvc/releases/tag/3.67.1)
- [DVC metadata schema](https://doc.dvc.org/user-guide/project-structure/dvc-files)
- [DVC commit](https://doc.dvc.org/command-reference/commit)
- [DVC local remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage)
