# ICS435 Assignment 1 Task List

## Status Legend
- `Todo`: not started
- `In Progress`: currently executing
- `Done`: completed and verified
- `Blocked`: needs user input or external dependency

## Task Dependency Graph
- `T02` depends on `T01`
- `T03` depends on `T02`
- `T04` depends on `T03`
- `T05` depends on `T04`
- `T06` depends on `T04`, `T05`
- `T07` depends on `T06`

## Progress Table

| ID | Task | Depends On | Priority | Status | Output |
|---|---|---|---|---|---|
| T01 | Environment + project scaffold | - | High | Done | `requirements.txt`, folders, run entrypoint |
| T02 | Baseline model training pipeline | T01 | High | Done | training/evaluation script |
| T03 | Metrics table + confusion matrices | T02 | High | Done | CSV/JSON metrics + matrix images |
| T04 | Hyperparameter ablation study | T03 | High | Done | ablation results table/plots |
| T05 | Repository polish for GitHub submission | T04 | Medium | Done | updated `README.md` + clear structure |
| T06 | Report drafting + PDF export | T04, T05 | High | Done | report source + PDF |
| T07 | Final submission checklist and handoff | T06 | Medium | Done | LMS-ready checklist |

## Update Log
- 2026-02-11: Initial task breakdown created.
- 2026-02-11: T01-T04 completed. Generated experiment artifacts under `outputs/`.
- 2026-02-11: T05-T07 completed. Report PDF generated and final manual checklist added.
