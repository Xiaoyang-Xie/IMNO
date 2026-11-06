from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _common_kwargs(args):
    """kwargs shared by every model family's run_training()."""
    return dict(
        continue_training=args.continue_training,
        num_workers=args.num_workers,
        modes=args.modes,
        width=args.width,
        initial_step=args.initial_step,
        skip_initial_time=args.skip_initial_time,
        t_train=args.t_train,
        training_type=args.training_type,
        num_channels=args.num_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        flnm=args.filename,
        base_path=args.data_path,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=args.reduced_batch,
        save_dir=args.save_dir,
        normalize=args.get("normalize", False),
        seed=args.get("seed", 1),
        split_seed=args.get("split_seed", 0),
        train_ratio=args.get("train_ratio", 0.8),
        val_ratio=args.get("val_ratio", 0.1),
        test_ratio=args.get("test_ratio", 0.1),
        deterministic=args.get("deterministic", True),
        run_name=args.get("run_name", None),
    )


@hydra.main(version_base="1.2", config_path="config", config_name="config_base")
def main(cfg: DictConfig):
    args = cfg.args
    model_name = args.model_name

    run_dir = Path(args.save_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_name = "config_resume.yaml" if args.continue_training else "config.yaml"
    OmegaConf.save(cfg, run_dir / config_name, resolve=True)
    print("Training configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True), end="", flush=True)

    if model_name == "FNO":
        from src.FNO.train import run_training

        logger.info("FNO")
        run_training(
            **_common_kwargs(args),
            shift_equivariant=args.get("shift_equivariant", None),
            phase_mode=args.get("phase_mode", "xy"),
            modes2=args.get("modes2", args.modes),
        )

    elif model_name == "IMNO":
        from src.IMNO.train import run_training

        logger.info("IMNO")
        run_training(
            **_common_kwargs(args),
            modes2=args.get("modes2", args.modes),
            modes3=args.get("modes3", args.modes),
            modes4=args.get("modes4", args.modes),
            hidden_states=args.hidden_states,
            alpha=args.alpha,
            gamma=args.gamma,
            w_max=args.w_max,
            ode_substeps=args.get("ode_substeps", 1),
            shift_equivariant=args.get("shift_equivariant", False),
            phase_mode=args.get("phase_mode", "xy"),
        )

    elif model_name == "RNO":
        from src.RNO.train import run_training

        logger.info("RNO")
        run_training(
            **_common_kwargs(args),
            modes2=args.get("modes2", None),
            shift_equivariant=args.get("shift_equivariant", None),
            phase_mode=args.get("phase_mode", "xy"),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


if __name__ == "__main__":
    main()
