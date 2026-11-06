# Data and checkpoints

Set `IMNO_DATA_ROOT` to a directory containing the datasets below. The
repository contains configuration and loading code, not the large datasets.
Download the custom datasets and all 252 three-seed best checkpoints from
[Google Drive](https://drive.google.com/drive/folders/1DL6u7yrbzrSHQyECJ1XqeUNUh2pFi3-y).


Ordinary Burgers and compressible NS are available from the
[PDEBench data archive](https://doi.org/10.18419/darus-2986); see the
[official downloader instructions](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download).
Download the exact filenames below. These public datasets are not duplicated in our Drive release.

| Experiment | Filename |
| --- | --- |
| Burgers | `1D_Burgers_Sols_Nu0.002.hdf5`, `1D_Burgers_Sols_Nu0.01.hdf5` |
| Nonlocal Burgers | `nonlocal_burgers_nu0.01_n10000.hdf5`, `nonlocal_burgers_nu0.01_forcing2_n10000.hdf5` |
| KS | `ks_4pi_n10000.hdf5`, `ks_8pi_n10000.hdf5` |
| 1D compressible NS | `1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5` |
| 2D compressible NS | `2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` |
| Kolmogorov flow, ν=0.1 | `kolmogorov_nu0.1_n5000.hdf5` |

Set `IMNO_EXPERIMENT_ROOT` to choose the experiment output root for the Slurm
sweep launcher. For example:

```text
experiments/
  ks4pi_t40_v1/runs/ks/IMNO/alpha_0.1/seed_1/
    ks_IMNO_alpha0.1_seed1.pt
  ns2d_t20_v1/runs/ns2d/
    IMNO/alpha_0.1/seed_1/ns2d_IMNO_alpha0.1_seed1.pt
    FNO/seed_1/ns2d_FNO_seed1.pt
    RNO/seed_1/ns2d_RNO_seed1.pt
```

Evaluate a checkpoint with `imno-evaluate --checkpoint <path>`; pass
`--config <path>` if its resolved configuration is not stored alongside it.
