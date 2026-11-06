from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def split_metadata(*datasets) -> dict:
    result = {}
    for dataset in datasets:
        result[dataset.split] = {
            "size": len(dataset),
            "index_hash": dataset.sample_indices_hash,
        }
    return result


class ExperimentRecorder:
    """Small JSON/JSONL artifact writer shared by all model families."""

    def __init__(self, save_dir: str, model_name: str, continue_training: bool) -> None:
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.save_dir / f"{model_name}.pt"
        self.last_path = self.save_dir / f"{model_name}_last.pt"
        self.history_path = self.save_dir / "history.jsonl"
        self.summary_path = self.save_dir / "summary.json"
        if not continue_training:
            self.history_path.write_text("", encoding="utf-8")

    def log_epoch(self, metrics: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")

    def write_summary(self, summary: dict) -> None:
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.summary_path)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=device) may move these CPU RNG ByteTensors to
    # CUDA along with the optimizer tensors. Move them back before restoring.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def load_trusted_checkpoint(path, *, map_location):
    """Load one of this repository's own checkpoints."""
    return torch.load(path, map_location=map_location, weights_only=False)


def validate_resume_metadata(checkpoint: dict, expected: dict) -> None:
    """Refuse to resume when the saved run identity differs from this run."""
    saved = checkpoint.get("run_metadata")
    if not isinstance(saved, dict):
        raise ValueError(
            "Resume checkpoint has no run_metadata and cannot be safely matched "
            "to the requested experiment; start a fresh run instead"
        )

    mismatches = []
    for key, expected_value in expected.items():
        saved_value = saved.get(key, "<missing>")
        if saved_value != expected_value:
            mismatches.append(f"{key}: saved={saved_value!r}, requested={expected_value!r}")
    if mismatches:
        raise ValueError("Resume checkpoint metadata mismatch: " + "; ".join(mismatches))
