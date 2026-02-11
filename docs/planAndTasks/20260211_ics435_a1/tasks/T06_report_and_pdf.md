# T06: Report Authoring and PDF Export

## Purpose
Produce assignment report in the required structure and export to PDF.

## Scope
- Fill Sections 1-4 per template.
- Include metrics table, confusion matrices, ablation findings.
- Add GitHub repository link.
- Export to PDF (max 4 pages, font 10, single space).

## Dependencies
- T04, T05

## Work Steps
1. Draft report source (`report/report.md` or template-based doc).
2. Insert generated results and discussion.
3. Export to `report/ICS435_Assignment1_Report.pdf`.

## Acceptance Criteria
- All required sections are present.
- GitHub link is included.
- PDF exists and respects formatting limits.

## Verification
- Open generated PDF and check pagination/layout.

## Status
- `Done`

## Completion Notes
- Authored report content in `report/report.md` with template section mapping.
- Included baseline results, confusion matrices, and ablation findings.
- Exported `report/ICS435_Assignment1_Report.pdf` via `pandoc + xelatex`.
- Confirmed PDF length is 4 pages.
