from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.io import ensure_dir, write_json


class ExperimentTracker:
    def __init__(self, root_dir: str | Path = "experiments/results") -> None:
        self.root_dir = ensure_dir(root_dir)
        self.summary_path = self.root_dir / "summary.csv"

    def log_run(
        self,
        run_name: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(self.root_dir / f"{timestamp}_{run_name}")
        payload = {
            "run_name": run_name,
            "timestamp_utc": timestamp,
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts or {},
        }
        write_json(run_dir / "run_summary.json", payload)

        row = {
            "timestamp_utc": timestamp,
            "run_name": run_name,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        }
        self._append_summary(row)
        return run_dir

    def _append_summary(self, row: dict[str, Any]) -> None:
        file_exists = self.summary_path.exists()
        with self.summary_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
