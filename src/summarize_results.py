#!/usr/bin/env python3
"""Collect per-run training summaries into comparison CSV files.

The training jobs write one summary.json in each independent save_dir.
This script consumes either the launcher manifest (preferred, because it also
reports missing jobs) or recursively discovered summary files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_COLUMNS = [
    "case",
    "model",
    "alpha",
    "seed",
    "split_seed",
    "status",
    "selection_metric",
    "metric_space",
    "best_epoch",
    "trainL2",
    "trainL2h",
    "valL2",
    "testL2",
    "group_n_expected",
    "group_n_completed",
    "group_best_valL2_mean",
    "group_best_valL2_std",
    "group_testL2_at_best_val_mean",
    "group_testL2_at_best_val_std",
    "valL2h",
    "testL2h",
    "checkpoint",
    "dataset",
    "model_name",
    "train_samples",
    "val_samples",
    "test_samples",
    "job_id",
    "success_marker",
    "save_dir",
    "summary_path",
    "error",
]

METRICS = (
    "best_epoch",
    "trainL2",
    "trainL2h",
    "valL2",
    "testL2",
    "valL2h",
    "testL2h",
)
MODEL_ORDER = {"FNO": 0, "IMNO": 1, "RNO": 2}
SUMMARY_COLUMNS = [
    "case",
    "model",
    "alpha",
    "seeds",
    "best_valL2_mean_std",
    "testL2_at_best_val_mean_std",
    "metric_space",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-run summary.json files into a seed-level comparison "
            "CSV and a case/model/alpha mean-and-standard-deviation CSV."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Launcher manifest.csv. If omitted, summary files are discovered recursively.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Per-run summary filename (default: summary.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Seed-level CSV (default: <experiment-dir>/comparison.csv).",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help="Mean/std CSV (default: <experiment-dir>/comparison_aggregate.csv).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Compact key-metrics CSV (default: <experiment-dir>/comparison_summary.csv).",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit non-zero after writing CSVs if a run is missing, invalid, or incomplete.",
    )
    return parser.parse_args()


def normalize_alpha(value: Any) -> Any:
    """Use one numeric representation so alpha groups sort and compare reliably."""
    if value in ("", None) or isinstance(value, bool):
        return "" if value in ("", None) else value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return value
    return result if math.isfinite(result) else value


def infer_run_fields(summary_path: Path, experiment_dir: Path) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    try:
        parts = summary_path.relative_to(experiment_dir).parts
    except ValueError:
        parts = summary_path.parts

    if "runs" in parts:
        offset = parts.index("runs")
        if len(parts) >= offset + 5:
            inferred["case"] = parts[offset + 1]
            inferred["model"] = parts[offset + 2]
            seed_match = re.fullmatch(r"seed[_-]?(\d+)", parts[offset + 3], flags=re.IGNORECASE)
            if seed_match:
                inferred["seed"] = int(seed_match.group(1))
    return inferred


def manifest_rows(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "model", "seed", "save_dir"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def discovered_rows(experiment_dir: Path, summary_name: str) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(experiment_dir.rglob(summary_name)):
        inferred = infer_run_fields(summary_path, experiment_dir)
        inferred["save_dir"] = str(summary_path.parent)
        rows.append(inferred)
    return rows


def canonical_row(
    expected: dict[str, Any],
    experiment_dir: Path,
    summary_name: str,
) -> dict[str, Any]:
    save_dir = Path(str(expected.get("save_dir", "")))
    if not save_dir.is_absolute():
        save_dir = experiment_dir / save_dir
    summary_path = save_dir / summary_name

    row = {column: "" for column in RUN_COLUMNS}
    row.update(
        {
            "case": expected.get("case", ""),
            "model": str(expected.get("model", "")).upper(),
            "alpha": normalize_alpha(expected.get("alpha")),
            "seed": expected.get("seed", ""),
            "metric_space": expected.get("metric_space", ""),
            "job_id": expected.get("job_id", ""),
            "success_marker": expected.get("success_marker", ""),
            "save_dir": str(save_dir),
            "summary_path": str(summary_path),
        }
    )
    if row["model"] != "IMNO":
        row["alpha"] = ""

    if not summary_path.is_file():
        row["status"] = "missing"
        row["error"] = f"{summary_name} not found"
        return row

    try:
        with summary_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"top-level JSON must be an object, got {type(payload).__name__}")
    except (OSError, ValueError, TypeError) as exc:
        row["status"] = "invalid"
        row["error"] = str(exc)
        return row

    inferred = infer_run_fields(summary_path, experiment_dir)
    row["case"] = row["case"] or payload.get("case") or inferred.get("case", "")
    row["model"] = (
        row["model"]
        or str(payload.get("model_family") or inferred.get("model", "")).upper()
    )
    if row["model"] == "IMNO":
        if row["alpha"] in ("", None):
            row["alpha"] = normalize_alpha(payload.get("alpha"))
    else:
        # Alpha is an IMNO loss hyperparameter, not an FNO/RNO experiment axis.
        row["alpha"] = ""
    row["seed"] = row["seed"] or payload.get("seed")
    for field in ("split_seed", "selection_metric", "best_epoch", "checkpoint", "dataset", "model_name"):
        row[field] = payload.get(field)
    summary_metric_space = payload.get("metric_space")
    if summary_metric_space not in ("", None):
        row["metric_space"] = summary_metric_space
    best = payload.get("best") or {}
    for metric in ("trainL2", "trainL2h", "valL2", "testL2", "valL2h", "testL2h"):
        row[metric] = best.get(metric)
    splits = payload.get("splits") or {}
    for split in ("train", "val", "test"):
        row[f"{split}_samples"] = (splits.get(split) or {}).get("size")

    missing = [
        field
        for field in ("selection_metric", "best_epoch", "valL2", "testL2")
        if row[field] in ("", None)
    ]
    if row["model"] == "IMNO":
        # Older summaries may omit trainL2h.
        missing.extend(
            field for field in ("valL2h", "testL2h") if row[field] in ("", None)
        )

    # All families are selected by validation rollout error. IMNO's latent
    # errors are useful diagnostics, but do not choose the checkpoint.
    expected_selection = "valL2"
    success_marker = str(row["success_marker"])
    marker_missing = bool(success_marker) and not Path(success_marker).is_file()
    if marker_missing:
        row["status"] = "failed"
        row["error"] = f"success marker not found: {success_marker}"
    elif missing:
        row["status"] = "incomplete"
        row["error"] = "missing fields: " + ", ".join(missing)
    elif row["selection_metric"] != expected_selection:
        row["status"] = "incomplete"
        row["error"] = (
            f"selection_metric={row['selection_metric']!r}; expected "
            f"{expected_selection!r} for {row['model']}"
        )
    else:
        row["status"] = "ok"
    return row


def numeric(value: Any) -> float | None:
    if value in ("", None) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def alpha_sort_key(value: Any) -> tuple[int, float, str]:
    parsed = numeric(value)
    if parsed is not None:
        return (0, parsed, "")
    if value in ("", None):
        return (1, 0.0, "")
    return (2, 0.0, str(value))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["case"]),
                str(row["model"]),
                row["alpha"],
                str(row["metric_space"]),
            )
        ].append(row)

    aggregate = []
    for (case_name, model, alpha, metric_space), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            MODEL_ORDER.get(item[0][1], 99),
            item[0][1],
            alpha_sort_key(item[0][2]),
            item[0][3],
        ),
    ):
        completed = [row for row in group if row["status"] == "ok"]
        output: dict[str, Any] = {
            "case": case_name,
            "model": model,
            "alpha": alpha,
            "metric_space": metric_space,
            "n_expected": len(group),
            "n_completed": len(completed),
            "selection_metric": completed[0]["selection_metric"] if completed else "",
        }
        for metric in METRICS:
            values = [value for row in completed if (value := numeric(row[metric])) is not None]
            output[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        # Explicit aliases document the model-selection semantics: every
        # seed contributes the testL2 measured at that seed's minimum-valL2
        # checkpoint. Keep the shorter historical column names as well.
        output["best_valL2_mean"] = output["valL2_mean"]
        output["best_valL2_std"] = output["valL2_std"]
        output["testL2_at_best_val_mean"] = output["testL2_mean"]
        output["testL2_at_best_val_std"] = output["testL2_std"]
        aggregate.append(output)
    return aggregate


def add_group_summary_columns(
    rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]
) -> None:
    """Repeat the key across-seed summary on each detailed comparison row."""
    summaries = {
        (
            str(item["case"]),
            str(item["model"]),
            item["alpha"],
            str(item["metric_space"]),
        ): item
        for item in aggregate
    }
    for row in rows:
        summary = summaries[
            (
                str(row["case"]),
                str(row["model"]),
                row["alpha"],
                str(row["metric_space"]),
            )
        ]
        row["group_n_expected"] = summary["n_expected"]
        row["group_n_completed"] = summary["n_completed"]
        row["group_best_valL2_mean"] = summary["best_valL2_mean"]
        row["group_best_valL2_std"] = summary["best_valL2_std"]
        row["group_testL2_at_best_val_mean"] = summary["testL2_at_best_val_mean"]
        row["group_testL2_at_best_val_std"] = summary["testL2_at_best_val_std"]


def format_mean_std(mean: Any, std: Any) -> str:
    """Format one human-facing metric without the full float representation."""
    mean_value = numeric(mean)
    std_value = numeric(std)
    if mean_value is None or std_value is None:
        return ""
    return f"{mean_value:.6g} ± {std_value:.6g}"


def compact_summary_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the full aggregate into the small table used for quick reading."""
    return [
        {
            "case": row["case"],
            "model": row["model"],
            "alpha": row["alpha"],
            "seeds": f"{row['n_completed']}/{row['n_expected']}",
            "best_valL2_mean_std": format_mean_std(
                row["best_valL2_mean"], row["best_valL2_std"]
            ),
            "testL2_at_best_val_mean_std": format_mean_std(
                row["testL2_at_best_val_mean"], row["testL2_at_best_val_std"]
            ),
            "metric_space": row["metric_space"],
        }
        for row in aggregate
    ]


def seed_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output = (args.output or experiment_dir / "comparison.csv").resolve()
    aggregate_output = (
        args.aggregate_output or experiment_dir / "comparison_aggregate.csv"
    ).resolve()
    summary_output = (
        args.summary_output or experiment_dir / "comparison_summary.csv"
    ).resolve()

    if args.manifest:
        manifest = args.manifest.resolve()
        if not manifest.is_file():
            print(f"error: manifest not found: {manifest}", file=sys.stderr)
            return 2
        try:
            expected = manifest_rows(manifest)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read manifest: {exc}", file=sys.stderr)
            return 2
    else:
        if not experiment_dir.is_dir():
            print(f"error: experiment directory not found: {experiment_dir}", file=sys.stderr)
            return 2
        expected = discovered_rows(experiment_dir, args.summary_name)

    if not expected:
        print("error: no expected runs or summary files found", file=sys.stderr)
        return 2

    rows = [canonical_row(item, experiment_dir, args.summary_name) for item in expected]
    rows.sort(
        key=lambda row: (
            str(row["case"]),
            MODEL_ORDER.get(str(row["model"]), 99),
            str(row["model"]),
            alpha_sort_key(row["alpha"]),
            str(row["metric_space"]),
            seed_sort_key(row["seed"]),
        )
    )
    aggregate = aggregate_rows(rows)
    add_group_summary_columns(rows, aggregate)
    compact_summary = compact_summary_rows(aggregate)
    aggregate_columns = [
        "case",
        "model",
        "alpha",
        "metric_space",
        "n_expected",
        "n_completed",
        "selection_metric",
        "best_valL2_mean",
        "best_valL2_std",
        "testL2_at_best_val_mean",
        "testL2_at_best_val_std",
        *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")),
    ]

    write_csv_atomic(output, rows, RUN_COLUMNS)
    write_csv_atomic(aggregate_output, aggregate, aggregate_columns)
    write_csv_atomic(summary_output, compact_summary, SUMMARY_COLUMNS)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["status"])] += 1
    status_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"Wrote {len(rows)} runs to {output} ({status_text})")
    print(f"Wrote {len(aggregate)} case/model/alpha/metric-space rows to {aggregate_output}")
    print(f"Wrote {len(compact_summary)} compact key-metric rows to {summary_output}")

    unsuccessful = sum(count for status, count in counts.items() if status != "ok")
    return 1 if args.require_all and unsuccessful else 0


if __name__ == "__main__":
    raise SystemExit(main())
