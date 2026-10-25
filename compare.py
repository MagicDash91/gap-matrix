"""Bandingkan keluaran pipeline (outputs/extraction_cache.jsonl) dengan matriks referensi pada aspek biner.

    python scripts/compare.py
Menghasilkan akurasi, presisi, recall per aspek — dipakai untuk melaporkan keandalan ekstraksi LLM.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ASPECTS, DATA_DIR, OUTPUT_DIR  # noqa: E402


def main() -> None:
    ref = json.loads((DATA_DIR / "reference_matrix.json").read_text())["aspects"]
    cache = [json.loads(l) for l in (OUTPUT_DIR / "extraction_cache.jsonl").read_text().splitlines() if l.strip()]
    pred = {r["key"].split("::")[0]: r["aspects"] for r in cache if r["key"].endswith("::aspects")}
    out = []
    for a in ASPECTS:
        tp = fp = fn = tn = 0
        for pid, asp in pred.items():
            v = asp.get(a.key, {})
            p = bool(v.get("present")) if isinstance(v, dict) else bool(v)
            t = pid in ref[a.key]
            tp += p and t; fp += p and not t; fn += (not p) and t; tn += (not p) and (not t)
        n = tp + fp + fn + tn
        out.append({"Aspek": a.label, "Akurasi": round((tp + tn) / n, 2) if n else None,
                    "Presisi": round(tp / (tp + fp), 2) if tp + fp else None,
                    "Recall": round(tp / (tp + fn), 2) if tp + fn else None})
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "pipeline_vs_reference.csv", index=False)


if __name__ == "__main__":
    main()
