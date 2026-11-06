"""Evaluate a saved checkpoint with its resolved training config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from src.experiment_utils import load_trusted_checkpoint
from src.utils import PDEDataset, seed_everything


def build_model(args, dimensions):
    common = dict(num_channels=args.num_channels, width=args.width, initial_step=args.initial_step)
    equivariant = args.get("shift_equivariant", None)
    phase_mode = args.get("phase_mode", "xy")
    modes2 = args.get("modes2", args.modes)
    modes2 = args.modes if modes2 is None else modes2
    if args.model_name == "IMNO":
        from src.IMNO.IMNO import IMNO1d, IMNO2d
        from src.IMNO.IMNO_SE import IMNO1d as IMNO1dSE
        from src.IMNO.IMNO_SE2d import IMNO2d as IMNO2dSE
        from src.IMNO.IMNO_SEx import IMNO2dx

        common.update(modes=args.modes,
                      modes3=args.modes if args.get("modes3") is None else args.modes3,
                      modes4=args.modes if args.get("modes4") is None else args.modes4,
                      hidden_states=args.hidden_states,
                      ode_substeps=args.get("ode_substeps", 1))
        if dimensions == 1:
            return (IMNO1dSE if equivariant else IMNO1d)(**common)
        model_class = IMNO2d if not equivariant else (IMNO2dx if phase_mode == "x" else IMNO2dSE)
        return model_class(**common, modes2=modes2)
    if equivariant is not None:
        common["shift_equivariant"] = equivariant
    if args.model_name == "FNO":
        from src.FNO.FNO import FNO1d, FNO2d

        if dimensions == 1:
            return FNO1d(**common, modes=args.modes)
        return FNO2d(**common, modes1=args.modes, modes2=modes2, phase_mode=phase_mode)
    if args.model_name == "RNO":
        from src.RNO.RNO import RNO1d, RNO2d

        if dimensions == 1:
            return RNO1d(**common, modes=args.modes)
        return RNO2d(**common, modes_x=args.modes, modes_y=modes2, phase_mode=phase_mode)
    raise ValueError(f"Unknown model_name: {args.model_name!r}")


def evaluate(config, checkpoint_path, data_path=None, split="test"):
    args = config.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.get("seed", 1), deterministic=args.get("deterministic", False))
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    mean, std = checkpoint.get("norm_mean"), checkpoint.get("norm_std")
    if args.get("normalize", False) and (mean is None or std is None):
        raise ValueError("Evaluation needs the training normalization statistics; use the best checkpoint.")
    dataset = PDEDataset(
        args.filename, saved_folder=data_path or args.data_path,
        initial_step=args.initial_step,
        skip_initial_time=args.get("skip_initial_time", 0),
        reduced_resolution=args.reduced_resolution, reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=args.reduced_batch, normalize=args.get("normalize", False),
        norm_mean=mean.cpu() if mean is not None else None,
        norm_std=std.cpu() if std is not None else None,
        split=split, split_seed=args.get("split_seed", 0),
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    model = build_model(args, dataset.data.ndim - 3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    t_train = min(int(args.t_train), dataset.data.shape[-2])
    result = {"split": split, "epoch": checkpoint["epoch"],
              "metric_space": "normalized" if args.get("normalize", False) else "physical"}
    if args.model_name == "IMNO":
        from src.IMNO.train_common import _evaluate

        l2, l2h, ratio = _evaluate(
            model, loader, has_phi=args.get("shift_equivariant", False),
            initial_step=args.initial_step, t_train=t_train, num_channels=args.num_channels,
        )
        result.update(L2=l2, L2h=l2h, v_ratio=ratio)
    else:
        if args.model_name == "FNO":
            from src.FNO.train import _evaluate
        else:
            from src.RNO.train import _evaluate
        result["L2"] = _evaluate(model, loader, args.initial_step, t_train, args.training_type)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--data-path", type=Path, help="Override the dataset directory.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    config = OmegaConf.load(args.config or args.checkpoint.parent / "config.yaml")
    print(json.dumps(evaluate(config, args.checkpoint, args.data_path, args.split), indent=2))


if __name__ == "__main__":
    main()
