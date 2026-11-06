from __future__ import annotations

from pathlib import Path
from timeit import default_timer

import torch
from torch import nn

from src.experiment_utils import (
    ExperimentRecorder,
    capture_rng_state,
    load_trusted_checkpoint,
    restore_rng_state,
    split_metadata,
    validate_resume_metadata,
)
from src.FNO.FNO import FNO1d, FNO2d
from src.utils import build_train_val_test_loaders, rel_l2, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forecast(model, xx, yy, grid, initial_step, t_train, training_type, loss_fn):
    inp_shape = [*xx.shape[:-2], -1]
    if training_type == "autoregressive":
        predictions = []
        loss = torch.zeros((), device=xx.device) if loss_fn is not None else None
        for t in range(initial_step, t_train):
            pred = model(xx.reshape(inp_shape), grid)
            target = yy[..., t : t + 1, :]
            if loss_fn is not None:
                batch_size = pred.size(0)
                loss = loss + loss_fn(
                    pred.reshape(batch_size, -1), target.reshape(batch_size, -1)
                )
            predictions.append(pred)
            xx = torch.cat((xx[..., 1:, :], pred), dim=-2)
        return torch.cat(predictions, dim=-2), yy[..., initial_step:t_train, :], loss

    if training_type == "single":
        pred = model(xx.reshape(inp_shape), grid)
        target = yy[..., t_train - 1 : t_train, :]
        if loss_fn is None:
            loss = None
        else:
            batch_size = pred.size(0)
            loss = loss_fn(pred.reshape(batch_size, -1), target.reshape(batch_size, -1))
        return pred, target, loss

    raise ValueError(f"Unknown training_type={training_type!r}")


@torch.no_grad()
def _evaluate(model, loader, initial_step, t_train, training_type):
    model.eval()
    metric_total = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = 0
    for xx, yy, grid in loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)
        pred, target, _ = _forecast(
            model, xx, yy, grid, initial_step, t_train, training_type, loss_fn=None
        )
        batch_size = yy.size(0)
        metric_total += rel_l2(pred, target) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Cannot evaluate an empty loader")
    return (metric_total / sample_count).item()


def run_training(
    continue_training,
    num_workers,
    modes,
    width,
    initial_step,
    skip_initial_time,
    t_train,
    num_channels,
    batch_size,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    flnm,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    save_dir,
    base_path="../data/",
    training_type="autoregressive",
    normalize=False,
    shift_equivariant=None,
    phase_mode="xy",
    modes2=None,
    seed=1,
    split_seed=0,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    deterministic=True,
    run_name=None,
):

    if training_type == "autoregressive" and t_train <= initial_step:
        raise ValueError(f"t_train ({t_train}) must be greater than initial_step ({initial_step})")
    if training_type not in {"autoregressive", "single"}:
        raise ValueError(f"Unknown training_type={training_type!r}")

    seed_everything(seed, deterministic=deterministic)
    split_ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
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
    norm_mean, norm_std = train_data.get_norm_stats()

    _, sample_data, _ = next(iter(val_loader))
    dimensions = sample_data.ndim
    t_train = min(int(t_train), int(sample_data.shape[-2]))
    if training_type == "autoregressive" and t_train <= initial_step:
        raise ValueError(
            f"Reduced dataset has only {sample_data.shape[-2]} time steps; "
            f"need more than initial_step={initial_step}"
        )

    if shift_equivariant is True and phase_mode == "x":
        suffix = "_FNO_SEx"
    elif shift_equivariant is True:
        suffix = "_FNO_SE"
    elif shift_equivariant is False:
        suffix = "_FNO_x"
    else:
        suffix = "_FNO"
    model_name = run_name or f"{Path(flnm).stem}{suffix}"

    se_kwargs = {} if shift_equivariant is None else {"shift_equivariant": shift_equivariant}
    if dimensions == 4:
        model = FNO1d(
            num_channels=num_channels,
            width=width,
            modes=modes,
            initial_step=initial_step,
            **se_kwargs,
        ).to(device)
    elif dimensions == 5:
        model = FNO2d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes if modes2 is None else modes2,
            initial_step=initial_step,
            phase_mode=phase_mode,
            **se_kwargs,
        ).to(device)
    else:
        raise ValueError(f"FNO expects yy.ndim in (4, 5); got {dimensions}")

    recorder = ExperimentRecorder(save_dir, model_name, continue_training)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    loss_fn = nn.MSELoss(reduction="mean")
    best_score = float("inf")
    best_metrics = None
    start_epoch = 0
    splits = split_metadata(train_data, val_data, test_data)
    run_metadata = {
        "model_family": "FNO",
        "model_name": model_name,
        "dataset": flnm,
        "metric_space": "normalized" if normalize else "physical",
        "seed": int(seed),
        "split_seed": int(split_seed),
        "t_train": int(t_train),
        "training_type": training_type,
        "splits": splits,
    }

    if continue_training:
        if not recorder.last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {recorder.last_path}")
        checkpoint = load_trusted_checkpoint(recorder.last_path, map_location=device)
        validate_resume_metadata(checkpoint, run_metadata)
        generator_state = checkpoint.get("train_loader_generator_state")
        if generator_state is None:
            raise ValueError("Resume checkpoint has no train_loader_generator_state")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_metrics = checkpoint.get("best_metrics")
        restore_rng_state(checkpoint.get("rng_state"))
        train_loader.generator.set_state(generator_state.cpu())
        print(f"Resumed from epoch {start_epoch}; best valL2={best_score:.6f}")

    total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total parameters = {total_params}")

    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = default_timer()
        online_totals = torch.zeros(2, device=device, dtype=torch.float64)
        train_sample_count = 0

        for xx, yy, grid in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pred, target, loss = _forecast(
                model, xx, yy, grid, initial_step, t_train, training_type, loss_fn
            )
            batch_size_actual = yy.size(0)
            train_l2_batch = rel_l2(pred, target)
            online_totals += (
                torch.stack((loss.detach(), train_l2_batch.detach())) * batch_size_actual
            )
            train_sample_count += batch_size_actual

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if train_sample_count == 0:
            raise RuntimeError("Cannot train on an empty loader")
        online_loss_value, train_l2 = (online_totals / train_sample_count).cpu().tolist()
        val_l2 = _evaluate(model, val_loader, initial_step, t_train, training_type)
        test_l2 = _evaluate(model, test_loader, initial_step, t_train, training_type)
        selected = val_l2 < best_score
        epoch_metrics = {
            "epoch": epoch,
            "loss": online_loss_value,
            "metric_space": run_metadata["metric_space"],
            "trainL2": train_l2,
            "valL2": val_l2,
            "testL2": test_l2,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": default_timer() - start_time,
            "selected": selected,
        }

        if selected:
            best_score = val_l2
            best_metrics = dict(epoch_metrics)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": best_score,
                "selection_metric": "valL2",
                "best_score": best_score,
                "best_metrics": best_metrics,
                "seed": int(seed),
                "split_seed": int(split_seed),
                "splits": splits,
                "run_metadata": run_metadata,
            }
            if norm_mean is not None:
                checkpoint["norm_mean"] = norm_mean.cpu()
                checkpoint["norm_std"] = norm_std.cpu()
            torch.save(checkpoint, recorder.best_path)
            print(f"Model saved at epoch {epoch} (valL2={val_l2:.6f}, testL2={test_l2:.6f})")

        scheduler.step()
        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "rng_state": capture_rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "run_metadata": run_metadata,
        }
        torch.save(last_checkpoint, recorder.last_path)
        recorder.log_epoch(epoch_metrics)
        recorder.write_summary(
            {
                "model_family": "FNO",
                "model_name": model_name,
                "dataset": flnm,
                "metric_space": run_metadata["metric_space"],
                "seed": int(seed),
                "split_seed": int(split_seed),
                "selection_metric": "valL2",
                "best_epoch": int(best_metrics["epoch"]),
                "best": best_metrics,
                "checkpoint": str(recorder.best_path),
                "history": str(recorder.history_path),
                "splits": splits,
            }
        )
        print(
            f"epoch: {epoch}, loss: {online_loss_value:.6f}, time: {epoch_metrics['seconds']:.2f}s, "
            f"trainL2: {train_l2:.6f}, valL2: {val_l2:.6f}, testL2: {test_l2:.6f}"
        )

    if recorder.best_path.is_file():
        best_checkpoint = load_trusted_checkpoint(recorder.best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        final_test = _evaluate(model, test_loader, initial_step, t_train, training_type)
        print(
            f"FINAL best_epoch={best_checkpoint['epoch']}, "
            f"best_valL2={best_checkpoint['best_score']:.6f}, testL2={final_test:.6f}"
        )


if __name__ == "__main__":
    run_training()
