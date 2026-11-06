from __future__ import annotations

import hashlib
import os
import random
from copy import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute relative L2 error (batch-mean).
    rel_l2 = sqrt(sum((pred - target)^2) / sum(target^2))
    The sqrt makes it comparable to RMSE / true_L2.
    """
    reduce_dims = tuple(range(1, target.ndim))
    num = ((pred - target) ** 2).sum(dim=reduce_dims)
    den = (target ** 2).sum(dim=reduce_dims)
    return torch.sqrt(num / (den + eps)).mean()


_MULTI_FIELD_NAMES = {
    1: ("density", "pressure", "Vx"),
    2: ("density", "pressure", "Vx", "Vy"),
}
_COORD_AXES = "xyz"


def make_split_indices(
    num_samples: int,
    split_seed: int = 0,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    """Return deterministic, disjoint train/val/test indices.

    Counts use the largest-remainder rule and every non-zero split receives at
    least one sample.
    """
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"split_ratios must be three non-negative values summing to 1; got {split_ratios}")

    positive = ratios > 0
    if num_samples < int(positive.sum()):
        raise ValueError(
            f"Need at least {int(positive.sum())} samples for non-empty train/val/test splits; got {num_samples}"
        )

    exact = ratios * num_samples
    counts = np.floor(exact).astype(np.int64)
    counts[positive & (counts == 0)] = 1

    while int(counts.sum()) > num_samples:
        removable = np.flatnonzero(counts > positive.astype(np.int64))
        if len(removable) == 0:
            raise ValueError(f"Cannot allocate {num_samples} samples with split ratios {split_ratios}")
        surplus = counts[removable] - exact[removable]
        counts[removable[int(np.argmax(surplus))]] -= 1

    while int(counts.sum()) < num_samples:
        deficit = exact - counts
        counts[int(np.argmax(deficit))] += 1

    permutation = np.random.default_rng(int(split_seed)).permutation(num_samples)
    n_train, n_val, _ = (int(v) for v in counts)
    return {
        "train": np.sort(permutation[:n_train]),
        "val": np.sort(permutation[n_train : n_train + n_val]),
        "test": np.sort(permutation[n_train + n_val :]),
    }


def _read_field(f, name, sample_indices, reduced_resolution_t, reduced_resolution, ndim_spatial):
    """Read only the selected/strided HDF5 slab, then move time behind space."""
    slicer = (sample_indices, slice(None, None, reduced_resolution_t)) + (
        slice(None, None, reduced_resolution),
    ) * ndim_spatial
    arr = np.asarray(f[name][slicer], dtype=np.float32)
    perm = (0, *range(2, 2 + ndim_spatial), 1)
    return np.transpose(arr, perm)


def _build_grid(f, ndim_spatial, reduced_resolution):
    """Coordinate grid from f["x-coordinate"][, "y-coordinate", ...], strided by reduced_resolution."""
    coords = [
        torch.tensor(np.array(f[f"{axis}-coordinate"], dtype=np.float32), dtype=torch.float)
        for axis in _COORD_AXES[:ndim_spatial]
    ]
    if ndim_spatial == 1:
        return coords[0][::reduced_resolution].unsqueeze(-1)
    mesh = torch.meshgrid(*coords, indexing="ij")
    grid = torch.stack(mesh, dim=-1)
    return grid[(slice(None, None, reduced_resolution),) * ndim_spatial]


class PDEDataset(Dataset):
    """
    Loads a single-file PDEBench-style HDF5 dataset. Two on-disk layouts:
      - multi-field (1D/2D CFD): separate "density"/"pressure"/"Vx"[/"Vy"]
        arrays, each (batch, time, x[, y]).
      - scalar (Burgers/KS/ReactionDiffusion/...): a single "tensor" array,
        (batch, time, x).
    """

    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        split=None,
        split_seed=0,
        split_ratios=(0.8, 0.1, 0.1),
        skip_initial_time=0,
        normalize=False,
        norm_mean=None,
        norm_std=None,
        norm_eps=1e-8,
    ):
        reduced_resolution = 1 if reduced_resolution is None else int(reduced_resolution)
        reduced_resolution_t = 1 if reduced_resolution_t is None else int(reduced_resolution_t)
        reduced_batch = 1 if reduced_batch is None else int(reduced_batch)
        if min(reduced_resolution, reduced_resolution_t, reduced_batch) <= 0:
            raise ValueError("reduced_resolution, reduced_resolution_t, and reduced_batch must be positive")

        if split is None:
            split = "test" if if_test else "train"
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all; got {split!r}")

        root_path = Path(saved_folder).expanduser().resolve() / filename
        if not root_path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {root_path}")

        with h5py.File(root_path, "r") as f:
            sample_key = "tensor" if "tensor" in f else "density"
            raw_num_samples = int(f[sample_key].shape[0])
            reduced_indices = np.arange(0, raw_num_samples, reduced_batch, dtype=np.int64)
            if split == "all":
                effective_indices = np.arange(len(reduced_indices), dtype=np.int64)
            else:
                split_map = make_split_indices(len(reduced_indices), split_seed, tuple(split_ratios))
                effective_indices = split_map[split]
            sample_indices = reduced_indices[effective_indices]

            self.split = split
            self.split_seed = int(split_seed)
            self.split_ratios = tuple(float(v) for v in split_ratios)
            self.sample_indices = sample_indices.copy()
            self.sample_indices_hash = hashlib.sha256(sample_indices.astype("<i8").tobytes()).hexdigest()[:16]

            if "tensor" not in f:
                raw_shape = f["density"].shape  # (batch, time, *spatial)
                ndim_spatial = len(raw_shape) - 2
                field_names = _MULTI_FIELD_NAMES[ndim_spatial]

                first = _read_field(
                    f, field_names[0], sample_indices, reduced_resolution_t, reduced_resolution, ndim_spatial
                )
                self.data = np.empty((*first.shape, len(field_names)), dtype=np.float32)
                self.data[..., 0] = first
                for i, name in enumerate(field_names[1:], start=1):
                    self.data[..., i] = _read_field(
                        f, name, sample_indices, reduced_resolution_t, reduced_resolution, ndim_spatial
                    )
                self.grid = _build_grid(f, ndim_spatial, reduced_resolution)

            else:  # scalar equations; data dim = [batch, time, x1, ..., xd]
                ndim_spatial = f["tensor"].ndim - 2
                _data = _read_field(
                    f, "tensor", sample_indices, reduced_resolution_t, reduced_resolution, ndim_spatial
                )
                self.data = _data[..., None]  # batch, x[, y], t, ch
                self.grid = _build_grid(f, ndim_spatial=ndim_spatial, reduced_resolution=reduced_resolution)

        # Time steps used as initial conditions
        self.initial_step = initial_step

        if skip_initial_time > 0:
            self.data = self.data[..., skip_initial_time:, :]

        self.data = torch.from_numpy(self.data)

        # Per-channel normalization (e.g. for u, rho, p with different scales)
        if normalize:
            data = self.data.float()
            nc = data.shape[-1]
            if norm_mean is None or norm_std is None:
                axis = tuple(range(data.ndim - 1))
                self._norm_mean = data.mean(dim=axis)
                self._norm_std = data.std(dim=axis) + norm_eps
            else:
                self._norm_mean = torch.as_tensor(norm_mean, dtype=torch.float32).to(data.device)
                self._norm_std = torch.as_tensor(norm_std, dtype=torch.float32).to(data.device)
                if self._norm_mean.dim() == 0:
                    self._norm_mean = self._norm_mean.unsqueeze(0).expand(nc)
                if self._norm_std.dim() == 0:
                    self._norm_std = self._norm_std.unsqueeze(0).expand(nc)
            self.data = (data - self._norm_mean) / self._norm_std
        else:
            self._norm_mean = None
            self._norm_std = None

        print(
            f"Dataset split={self.split!r}, samples={len(self.data)}, shape={tuple(self.data.shape)}, "
            f"split_seed={self.split_seed}, index_hash={self.sample_indices_hash}, path={root_path}"
        )

    def get_norm_stats(self):
        """Return (mean, std) per channel for use by val/test loaders. None if not normalized."""
        if self._norm_mean is None:
            return None, None
        return self._norm_mean.clone(), self._norm_std.clone()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ..., : self.initial_step, :], self.data[idx], self.grid


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, CPU/CUDA Torch, and deterministic backend settings."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_train_val_test_loaders(
    *,
    flnm,
    base_path,
    initial_step,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    skip_initial_time,
    normalize,
    batch_size,
    num_workers,
    seed,
    split_seed,
    split_ratios,
):
    """Build reproducible loaders with train-only normalization statistics."""
    common = dict(
        filename=flnm,
        reduced_resolution=reduced_resolution,
        reduced_resolution_t=reduced_resolution_t,
        reduced_batch=reduced_batch,
        initial_step=initial_step,
        saved_folder=base_path,
        skip_initial_time=skip_initial_time,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )
    train_data = PDEDataset(**common, split="train", normalize=normalize)
    norm_mean, norm_std = train_data.get_norm_stats()

    evaluation_norm = dict(normalize=normalize, norm_mean=norm_mean, norm_std=norm_std)
    if float(split_ratios[1]) == 0.0:
        if float(split_ratios[2]) <= 0.0:
            raise ValueError("A zero validation ratio requires a non-zero test ratio")
        test_data = PDEDataset(**common, split="test", **evaluation_norm)
        # The trainers select checkpoints on valL2. A legacy 9:1 train/test
        # experiment has no independent validation split, so make that leakage
        # explicit while avoiding a second in-memory copy of the test tensor.
        val_data = copy(test_data)
        val_data.split = "val"
        val_data.split_alias_of = "test"
        print(
            "Dataset split='val' aliases split='test' for checkpoint selection "
            f"(samples={len(val_data)}, index_hash={val_data.sample_indices_hash})"
        )
    else:
        val_data = PDEDataset(**common, split="val", **evaluation_norm)
        test_data = PDEDataset(**common, split="test", **evaluation_norm)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_common)
    val_loader = DataLoader(val_data, shuffle=False, **loader_common)
    test_loader = DataLoader(test_data, shuffle=False, **loader_common)
    return (
        train_data,
        val_data,
        test_data,
        train_loader,
        val_loader,
        test_loader,
    )
