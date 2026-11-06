# IMNO: Inertial Manifold Neural Operator

Research code for **Inertial Manifold Neural Operator for Dissipative
Partial Differential Equations**, with FNO and RNO baselines.
Includes plain and shift-equivariant models in 1D and 2D.

## Install

Use Python 3.10:

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps -e .
```

## Data and pretrained models

[Download datasets, pretrained checkpoints, and results](https://drive.google.com/drive/folders/1DL6u7yrbzrSHQyECJ1XqeUNUh2pFi3-y).
Ordinary Burgers and compressible Navier–Stokes datasets are available from
[PDEBench](https://doi.org/10.18419/darus-2986).
See [data instructions](docs/data.md) and the [paper experiment list](docs/experiments.csv).

## Train and evaluate

```bash
export IMNO_DATA_ROOT=/path/to/data
imno-train +experiment=paper/ks4pi_t40_v1 \
  args.model_name=IMNO args.seed=1 args.alpha=0.1 \
  args.save_dir=outputs/ks_seed1 args.run_name=ks_seed1
imno-evaluate --checkpoint outputs/ks_seed1/ks_seed1.pt
```

Set `args.model_name=FNO` or `RNO` for a baseline. Experiment settings are in
[`src/config/experiment/`](src/config/experiment/); use `paper/…` presets to
reproduce paper runs. Override parameters with `args.<name>=<value>`.

Runs save the configuration, checkpoints, training history, and result summary.
The best checkpoint is selected by validation error. Resume with the same
configuration and `args.continue_training=true`.

## Slurm

Activate the installed environment, then preview a sweep:

```bash
DRY_RUN=1 CASES=ks EXPERIMENT=paper/ks4pi_t40_v1 \
  TAG=ks4pi_t40_v1 IMNO_ALPHAS='0 0.1' \
  bash scripts/submit_sweep.sh
```

Remove `DRY_RUN=1` to submit. Set paths and cluster resources in
[`scripts/submit_sweep.sh`](scripts/submit_sweep.sh) or through environment
variables. Results are collected into per-run and across-seed CSV files;
`imno-summarize --help` describes manual aggregation.

## Code

- `src/IMNO/`, `src/FNO/`, `src/RNO/`: models and trainers.
- `src/config/`: shared defaults and experiment settings.
- `scripts/`: Slurm launchers.
- `docs/`: dataset and experiment details.

MIT licensed. Based on PDEBench and Zongyi Li's FNO implementation.

## AI Assistance Disclosure

OpenAI Codex assisted with code cleanup, repository organization, and packaging.
