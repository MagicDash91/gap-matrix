"""Bangun berkas gap matrix final (Excel + Markdown) dari matriks referensi yang sudah diverifikasi.

    python scripts/build_reference.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import export, gap  # noqa: E402
from src.config import ASPECTS, DATA_DIR, FIELDS  # noqa: E402
from src.ingest import load_papers  # noqa: E402


def main() -> None:
    ref = json.loads((DATA_DIR / "reference_matrix.json").read_text())
    papers = load_papers().set_index("id")
    rows, arows = [], []
    for pid, cells in ref["rows"].items():
        p = papers.loc[pid]
        row = {"ID": pid, "Paper": f"{p['authors']} ({p['year']}). {p['title']}. {p['venue']}",
               "DOI": p["doi"] or p["url"]}
        row.update({f.label: cells[f.key] for f in FIELDS})
        rows.append(row)
        arows.append({"ID": pid, **{a.label: pid in ref["aspects"][a.key] for a in ASPECTS}})
    matrix, aspects = pd.DataFrame(rows), pd.DataFrame(arows)
    cov = gap.coverage(aspects)
    analysis = (DATA_DIR / "reference_analysis.md").read_text().strip()
    print(export.to_excel(matrix, aspects, cov, analysis, name="gap_matrix_final"))
    print(export.to_markdown(matrix, cov, analysis, name="gap_matrix_final"))
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
