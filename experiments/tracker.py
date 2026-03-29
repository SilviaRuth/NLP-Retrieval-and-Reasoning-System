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
        artifacts = artifacts or {}
        run_dir = ensure_dir(self.root_dir / f"{timestamp}_{run_name}")
        payload = {
            "run_name": run_name,
            "timestamp_utc": timestamp,
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        write_json(run_dir / "run_summary.json", payload)

        row = self._build_summary_row(
            timestamp=timestamp,
            run_name=run_name,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
        )
        self._append_summary(row)
        return run_dir

    def _build_summary_row(
        self,
        timestamp: str,
        run_name: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        test_metrics = artifacts.get("test_metrics") or {}
        robustness = artifacts.get("robustness") or {}
        effective_batch_size = artifacts.get("effective_batch_size")
        if effective_batch_size is None and config.get("batch_size") and config.get("gradient_accumulation_steps"):
            effective_batch_size = config["batch_size"] * config["gradient_accumulation_steps"]

        return {
            "timestamp_utc": timestamp,
            "run_name": run_name,
            "status": artifacts.get("status"),
            "model_name": config.get("model_name"),
            "output_dir": artifacts.get("output_dir"),
            "checkpoint_dir": artifacts.get("checkpoint_dir"),
            "git_commit": artifacts.get("git_commit"),
            "best_epoch": metrics.get("best_epoch"),
            "val_accuracy": metrics.get("accuracy"),
            "val_macro_f1": metrics.get("macro_f1"),
            "val_loss": metrics.get("loss"),
            "train_loss": metrics.get("train_loss"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_loss": test_metrics.get("loss"),
            "typo_macro_f1": (robustness.get("typo") or {}).get("macro_f1"),
            "paraphrase_macro_f1": (robustness.get("paraphrase") or {}).get("macro_f1"),
            "shuffle_macro_f1": (robustness.get("shuffle") or {}).get("macro_f1"),
            "batch_size": config.get("batch_size"),
            "effective_batch_size": effective_batch_size,
            "max_length": config.get("max_length"),
            "learning_rate": config.get("learning_rate"),
            "epochs": config.get("epochs"),
            "early_stopping_patience": config.get("early_stopping_patience"),
            "warmup_steps": artifacts.get("warmup_steps", config.get("warmup_steps")),
            "total_optimizer_steps": artifacts.get("total_optimizer_steps"),
        }

    def _append_summary(self, row: dict[str, Any]) -> None:
        if not self.summary_path.exists():
            with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        with self.summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []

        merged_fieldnames = list(existing_fieldnames)
        for key in row.keys():
            if key not in merged_fieldnames:
                merged_fieldnames.append(key)

        if merged_fieldnames != existing_fieldnames:
            with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({field: existing_row.get(field, "") for field in merged_fieldnames})
                writer.writerow({field: row.get(field, "") for field in merged_fieldnames})
            return

        with self.summary_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
            writer.writerow(row)
