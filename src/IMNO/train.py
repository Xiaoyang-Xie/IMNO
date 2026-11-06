"""Training entry point for plain and shift-equivariant IMNO models."""

from __future__ import annotations

from pathlib import Path

from src.IMNO.IMNO import IMNO1d, IMNO2d
from src.IMNO.IMNO_SE import IMNO1d as IMNO1dSE
from src.IMNO.IMNO_SE2d import IMNO2d as IMNO2dSE
from src.IMNO.IMNO_SEx import IMNO2dx
from src.IMNO.train_common import device, run_training_core
from src.utils import build_train_val_test_loaders, seed_everything


def run_training(
    continue_training,
    num_workers,
    modes,
    width,
    initial_step,
    skip_initial_time,
    t_train,
    num_channels,
    hidden_states,
    batch_size,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    flnm,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    alpha,
    gamma,
    w_max,
    save_dir,
    base_path="../data/",
    training_type="autoregressive",
    normalize=False,
    modes2=None,
    modes3=None,
    modes4=None,
    shift_equivariant=False,
    phase_mode="xy",
    ode_substeps=1,
    run_name=None,
    seed=1,
    split_seed=0,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    deterministic=True,
):
    if training_type != "autoregressive":
        raise ValueError("IMNO requires training_type='autoregressive'")
    if t_train <= initial_step:
        raise ValueError(f"t_train ({t_train}) must be greater than initial_step ({initial_step})")

    seed_everything(seed, deterministic=deterministic)
    split_ratios = (float(train_ratio), float(val_ratio), float(test_ratio))

    if shift_equivariant and phase_mode == "x":
        default_suffix = "_IMNOSE_2Dx"
    elif shift_equivariant:
        default_suffix = "_IMNOSE"
    else:
        default_suffix = "_IMNO"
    model_name = run_name or f"{Path(flnm).stem}{default_suffix}"

    (
        train_data,
        val_data,
        test_data,
        train_loader,
        val_loader,
        test_loader,
    ) = build_train_val_test_loaders(
        flnm=flnm,
        base_path=base_path,
        initial_step=initial_step,
        reduced_resolution=reduced_resolution,
        reduced_resolution_t=reduced_resolution_t,
        reduced_batch=reduced_batch,
        skip_initial_time=skip_initial_time,
        normalize=normalize,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )

    _, sample_data, _ = next(iter(val_loader))
    dimensions = sample_data.ndim
    # modes2 is the second spatial-direction mode count and only applies to
    # 2D IMNO models. 1D IMNO always learns all Fourier modes in [0, modes).
    if dimensions == 5:
        modes2 = modes if modes2 is None else modes2
    modes3 = modes if modes3 is None else modes3
    modes4 = modes if modes4 is None else modes4

    if not shift_equivariant:
        if dimensions == 4:
            model = IMNO1d(
                num_channels=num_channels,
                width=width,
                modes=modes,
                modes3=modes3,
                modes4=modes4,
                initial_step=initial_step,
                hidden_states=hidden_states,
                ode_substeps=ode_substeps,
            ).to(device)
        elif dimensions == 5:
            model = IMNO2d(
                num_channels=num_channels,
                width=width,
                modes=modes,
                modes2=modes2,
                modes3=modes3,
                modes4=modes4,
                initial_step=initial_step,
                hidden_states=hidden_states,
                ode_substeps=ode_substeps,
            ).to(device)
        else:
            raise ValueError(f"IMNO expects yy.ndim in (4, 5); got {dimensions}")
    elif dimensions == 4:
        model = IMNO1dSE(
            num_channels=num_channels,
            width=width,
            modes=modes,
            modes3=modes3,
            modes4=modes4,
            initial_step=initial_step,
            hidden_states=hidden_states,
            ode_substeps=ode_substeps,
        ).to(device)
    elif dimensions == 5 and phase_mode == "x":
        model = IMNO2dx(
            num_channels=num_channels,
            width=width,
            modes=modes,
            modes2=modes2,
            modes3=modes3,
            modes4=modes4,
            initial_step=initial_step,
            hidden_states=hidden_states,
            ode_substeps=ode_substeps,
        ).to(device)
    elif dimensions == 5:
        model = IMNO2dSE(
            num_channels=num_channels,
            width=width,
            modes=modes,
            modes2=modes2,
            modes3=modes3,
            modes4=modes4,
            initial_step=initial_step,
            hidden_states=hidden_states,
            ode_substeps=ode_substeps,
        ).to(device)
    else:
        raise ValueError(f"Shift-equivariant IMNO expects yy.ndim in (4, 5); got {dimensions}")

    t_train = min(int(t_train), int(sample_data.shape[-2]))
    if t_train <= initial_step:
        raise ValueError(
            f"Reduced dataset has only {sample_data.shape[-2]} time steps; "
            f"need more than initial_step={initial_step}"
        )

    run_training_core(
        model=model,
        model_name=model_name,
        has_phi=shift_equivariant,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        continue_training=continue_training,
        initial_step=initial_step,
        t_train=t_train,
        num_channels=num_channels,
        epochs=epochs,
        learning_rate=learning_rate,
        scheduler_step=scheduler_step,
        scheduler_gamma=scheduler_gamma,
        alpha=alpha,
        gamma=gamma,
        w_max=w_max,
        save_dir=save_dir,
        normalize=normalize,
        seed=seed,
        split_seed=split_seed,
        dataset_name=flnm,
    )


if __name__ == "__main__":
    run_training()
