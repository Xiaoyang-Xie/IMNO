#!/usr/bin/env bash
# Activate the intended environment before submitting. Numerical settings live
# in the YAML experiment preset; this script only supplies run metadata.
#SBATCH --job-name=imno-train
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this script with sbatch, inside a Slurm allocation." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH:+$ENV_PATH/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-${IMNO_DATA_ROOT:-$REPO_ROOT/data}}"
CASE="${CASE:?Set CASE}"
MODEL="${MODEL:?Set MODEL}"
MODEL="${MODEL^^}"
SEED="${SEED:?Set SEED}"
SAVE_DIR="${SAVE_DIR:?Set an independent SAVE_DIR}"
EXPERIMENT="${EXPERIMENT:-$CASE}"
RUN_NAME="${RUN_NAME:-${CASE}_${MODEL}_seed${SEED}}"

export PYTHONHASHSEED="$SEED"
cd "$REPO_ROOT"
cmd=(
  "$PYTHON_BIN" -u -m src.train
  "+experiment=$EXPERIMENT"
  "++args.model_name=$MODEL"
  "++args.seed=$SEED"
  "++args.data_path=$DATA_ROOT"
  "++args.save_dir=$SAVE_DIR"
  "++args.run_name=$RUN_NAME"
)
if [[ "$MODEL" == IMNO && -n "${ALPHA:-}" ]]; then
  cmd+=("++args.alpha=$ALPHA")
fi
cmd+=("$@")
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
test -f "$SAVE_DIR/summary.json"
touch "$SAVE_DIR/_SUCCESS_$SLURM_JOB_ID"
