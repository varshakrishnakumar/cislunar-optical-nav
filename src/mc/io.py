from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import List

from .types import TrialResult


def save_results_csv(results: List[TrialResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        path.write_text("")
        return path

    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path
