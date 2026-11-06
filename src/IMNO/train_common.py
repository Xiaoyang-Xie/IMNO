"""Shared data, rollout, evaluation, and checkpoint logic for the IMNO family."""

from __future__ import annotations

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
from src.utils import rel_l2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_time_step_if_tc(im: torch.Tensor, v: torch.Tensor, initial_step: int, num_channels: int):
    """Extract the final time slice when a model returns T*C output channels."""
    if initial_step <= 1 or im.shape[-1] == num_channels:
        return im, v
    if im.shape[-1] != initial_step * num_channels:
        raise ValueError(
            f"Expected output channels C={num_channels} or T*C={initial_step * num_channels}; "
            f"got {im.shape[-1]}"
        )
    if im.dim() == 4:
        batch, length, _, _ = im.shape
        im_c = im.reshape(batch, length, 1, initial_step, num_channels)[..., -1, :]
        v_c = v.reshape(batch, length, 1, initial_step, num_channels)[..., -1, :]
        return im_c, v_c
    if im.dim() == 5:
        batch, height, width, _, _ = im.shape
        im_c = im.reshape(batch, height, width, 1, initial_step, num_channels)[..., -1, :]
        v_c = v.reshape(batch, height, width, 1, initial_step, num_channels)[..., -1, :]
        return im_c, v_c
    raise ValueError(f"Unexpected im.ndim={im.dim()} for initial_step={initial_step}")


def _forecast(
    model,
    xx,
    yy,
    grid,
    *,
    has_phi,
    initial_step,
    t_train,
    num_channels,
    loss_fn,
    alpha,
    gamma,
    w_max,
    compute_loss,
    compute_ratio,
):
    inp_shape = [*xx.shape[:-2], -1]
    hidden = velocity = phase = None
    predictions = []
    manifold_predictions = []
    loss = torch.zeros((), device=xx.device) if compute_loss else None
    ratio_sum = torch.zeros((), device=xx.device) if compute_ratio else None
    ratio_count = 0

    for t in range(initial_step, t_train):
        inp = xx.reshape(inp_shape)
        if has_phi:
            im, hidden, velocity, phase = model(inp, grid, hidden, velocity, phase)
        else:
            im, hidden, velocity = model(inp, grid, hidden, velocity)
        im_c, velocity_c = last_time_step_if_tc(im, velocity, initial_step, num_channels)
        manifold = im_c - velocity_c

        batch_size = im_c.size(0)
        if compute_loss:
            target = yy[..., t : t + 1, :]
            loss = loss + loss_fn(
                im_c.reshape(batch_size, -1), target.reshape(batch_size, -1)
            )
            if alpha != 0:
                step = t - initial_step
                weight = min(gamma**step, w_max) / w_max
                loss = loss + alpha * weight * loss_fn(
                    manifold.reshape(batch_size, -1), target.reshape(batch_size, -1)
                )

        predictions.append(im_c)
        manifold_predictions.append(manifold)
        xx = torch.cat((xx[..., 1:, :], im_c), dim=-2)

        if compute_ratio:
            velocity_flat = velocity_c.reshape(batch_size, -1)
            manifold_flat = manifold.reshape(batch_size, -1)
            velocity_norm = torch.sqrt((velocity_flat**2).mean(dim=1) + 1e-12)
            manifold_norm = torch.sqrt((manifold_flat**2).mean(dim=1) + 1e-12)
            ratio_sum = ratio_sum + (velocity_norm / (manifold_norm + 1e-12)).sum()
            ratio_count += batch_size

    ratio = ratio_sum / ratio_count if compute_ratio else None

    return (
        torch.cat(predictions, dim=-2),
        torch.cat(manifold_predictions, dim=-2),
        yy[..., initial_step:t_train, :],
        loss,
        ratio,
    )


@torch.no_grad()
def _evaluate(
    model,
    loader,
    *,
    has_phi,
    initial_step,
    t_train,
    num_channels,
):
    model.eval()
    metric_totals = torch.zeros(3, device=device, dtype=torch.float64)
    sample_count = 0
    for xx, yy, grid in loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)
        pred, pred_h, target, _, ratio = _forecast(
            model,
            xx,
            yy,
            grid,
            has_phi=has_phi,
            initial_step=initial_step,
            t_train=t_train,
            num_channels=num_channels,
            loss_fn=None,
            alpha=0.0,
            gamma=1.0,
            w_max=1.0,
            compute_loss=False,
            compute_ratio=True,
        )
        batch_size = yy.size(0)
        batch_metrics = torch.stack(
            (
                rel_l2(pred, target),
                rel_l2(pred_h, target),
                ratio,
            )
        )
        metric_totals += batch_metrics * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Cannot evaluate an empty data loader")
    return tuple((metric_totals / sample_count).cpu().tolist())


def run_training_core(
    model: nn.Module,
    model_name: str,
    has_phi: bool,
    train_data,
    val_data,
    test_data,
    train_loader,
    val_loader,
    test_loader,
    continue_training: bool,
    initial_step: int,
    t_train: int,
    num_channels: int,
    epochs: int,
    learning_rate: float,
    scheduler_step: int,
    scheduler_gamma: float,
    alpha: float,
    gamma: float,
    w_max: float,
    save_dir: str,
    normalize: bool,
    seed: int,
    split_seed: int,
    dataset_name: str,
) -> None:
    norm_mean, norm_std = train_data.get_norm_stats()
    alpha_value = float(alpha)
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
        "model_family": "IMNO",
        "model_name": model_name,
        "dataset": dataset_name,
        "metric_space": "normalized" if normalize else "physical",
        "seed": int(seed),
        "split_seed": int(split_seed),
        "t_train": int(t_train),
        "training_type": "autoregressive",
        "has_phi": bool(has_phi),
        "alpha": alpha_value,
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
        train_metric_totals = torch.zeros(3, device=device, dtype=torch.float64)
        train_sample_count = 0

        for xx, yy, grid in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pred, pred_h, target, loss, _ = _forecast(
                model,
                xx,
                yy,
                grid,
                has_phi=has_phi,
                initial_step=initial_step,
                t_train=t_train,
                num_channels=num_channels,
                loss_fn=loss_fn,
                alpha=alpha,
                gamma=gamma,
                w_max=w_max,
                compute_loss=True,
                compute_ratio=False,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = yy.size(0)
            batch_metrics = torch.stack(
                (
                    loss.detach(),
                    rel_l2(pred, target),
                    rel_l2(pred_h, target),
                )
            )
            train_metric_totals += batch_metrics * batch_size
            train_sample_count += batch_size

        if train_sample_count == 0:
            raise RuntimeError("Cannot train on an empty data loader")
        loss_value, train_l2, train_l2h = (
            train_metric_totals / train_sample_count
        ).cpu().tolist()
        train_seconds = default_timer() - start_time
        val_start_time = default_timer()
        val_l2, val_l2h, val_ratio = _evaluate(
            model,
            val_loader,
            has_phi=has_phi,
            initial_step=initial_step,
            t_train=t_train,
            num_channels=num_channels,
        )
        val_seconds = default_timer() - val_start_time
        test_start_time = default_timer()
        test_l2, test_l2h, test_ratio = _evaluate(
            model,
            test_loader,
            has_phi=has_phi,
            initial_step=initial_step,
            t_train=t_train,
            num_channels=num_channels,
        )
        test_seconds = default_timer() - test_start_time
        selected = val_l2 < best_score
        epoch_metrics = {
            "epoch": epoch,
            "loss": loss_value,
            "metric_space": run_metadata["metric_space"],
            "trainL2": train_l2,
            "trainL2h": train_l2h,
            "valL2": val_l2,
            "testL2": test_l2,
            "valL2h": val_l2h,
            "testL2h": test_l2h,
            "val_v_ratio": val_ratio,
            "test_v_ratio": test_ratio,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": default_timer() - start_time,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "test_seconds": test_seconds,
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
                "alpha": alpha_value,
                "splits": splits,
                "run_metadata": run_metadata,
            }
            if normalize and norm_mean is not None:
                checkpoint["norm_mean"] = norm_mean.cpu()
                checkpoint["norm_std"] = norm_std.cpu()
            torch.save(checkpoint, recorder.best_path)
            print(
                f"Model saved at epoch {epoch} "
                f"(valL2={val_l2:.6f}, valL2h={val_l2h:.6f}, testL2={test_l2:.6f}, testL2h={test_l2h:.6f})"
            )

        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "best_metrics": best_metrics,
                "alpha": alpha_value,
                "rng_state": capture_rng_state(),
                "train_loader_generator_state": train_loader.generator.get_state(),
                "run_metadata": run_metadata,
            },
            recorder.last_path,
        )
        recorder.log_epoch(epoch_metrics)
        recorder.write_summary(
            {
                "model_family": "IMNO",
                "model_name": model_name,
                "dataset": dataset_name,
                "metric_space": run_metadata["metric_space"],
                "seed": int(seed),
                "split_seed": int(split_seed),
                "alpha": alpha_value,
                "selection_metric": "valL2",
                "best_epoch": int(best_metrics["epoch"]),
                "best": best_metrics,
                "checkpoint": str(recorder.best_path),
                "history": str(recorder.history_path),
                "splits": splits,
            }
        )
        print(
            f"epoch: {epoch}, loss: {loss_value:.6f}, time: {epoch_metrics['seconds']:.2f}s "
            f"(train={train_seconds:.2f}s, val={val_seconds:.2f}s, test={test_seconds:.2f}s), "
            f"trainL2: {train_l2:.6f}, trainL2h: {train_l2h:.6f}, valL2: {val_l2:.6f}, "
            f"testL2: {test_l2:.6f}, valL2h: {val_l2h:.6f}, testL2h: {test_l2h:.6f}"
        )

    if recorder.best_path.is_file():
        best_checkpoint = load_trusted_checkpoint(recorder.best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        final_test, final_test_h, _ = _evaluate(
            model,
            test_loader,
            has_phi=has_phi,
            initial_step=initial_step,
            t_train=t_train,
            num_channels=num_channels,
        )
        print(
            f"FINAL best_epoch={best_checkpoint['epoch']}, "
            f"best_valL2={best_checkpoint['best_score']:.6f}, "
            f"testL2={final_test:.6f}, testL2h={final_test_h:.6f}"
        )
