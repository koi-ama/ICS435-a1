# T01: Environment and Project Scaffold

## Purpose
Create a reproducible Python project structure and dependencies for the assignment pipeline.

## Scope
- Create folder layout for code, outputs, and report.
- Create dependency file.
- Define how to run experiments from one command.

## Dependencies
- None

## Work Steps
1. Create folders: `src/`, `outputs/`, `report/`.
2. Create `requirements.txt`.
3. Create initial run script (`src/main.py`).

## Acceptance Criteria
- Project has executable entry script.
- Required packages are listed.
- Folder structure supports downstream tasks.

## Verification
- `python3 -m pip install -r requirements.txt`
- `python3 src/main.py` (temporary smoke run is acceptable)

## Status
- `Done`

## Completion Notes
- Created `requirements.txt`.
- Implemented entrypoint `src/main.py`.
- Confirmed script execution via `python3 src/main.py`.
