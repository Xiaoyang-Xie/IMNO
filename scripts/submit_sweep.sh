#!/usr/bin/env bash
# Submit a reproducible case x model x seed x IMNO-alpha benchmark sweep.
#
# Quick start:
#   bash scripts/submit_sweep.sh
#
# Preview without submitting or creating directories:
#   DRY_RUN=1 bash scripts/submit_sweep.sh
#
# Job selection is configured here. Training values come from config YAMLs and
# the experiment presets in src/config/experiment; this launcher does not
# redefine optimizer, architecture, split, or preprocessing parameters.
# Slurm resources can still use CASE/MODEL suffixes (for example,
# SBATCH_GRES_NS2D_IMNO=gpu:l40:1).

set -Eeuo pipefail

# =============================================================================
# USER CONFIGURATION: edit the default values in this block, then run
#   bash scripts/submit_sweep.sh
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SBATCH="${TRAIN_SBATCH:-$SCRIPT_DIR/train.sbatch}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_PATH="${ENV_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PATH:+$ENV_PATH/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DATA_ROOT="${DATA_ROOT:-${IMNO_DATA_ROOT:-$REPO_ROOT/data}}"
EXP_ROOT="${EXP_ROOT:-${IMNO_EXPERIMENT_ROOT:-$REPO_ROOT/experiments}}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs/slurm}"

# Default matrix for the NS1d batch-size-100 comparison:
# 3 FNO + 3 RNO + (3 seeds x 3 IMNO alphas) = 15 jobs.
CASES="${CASES:-ns1d_eta01}"
MODELS="${MODELS:-FNO IMNO RNO}"
SEEDS="${SEEDS:-1 2 3}"
IMNO_ALPHAS="${IMNO_ALPHAS:-0 0.1 1.0}"
TAG="${TAG:-ns1d_t50_v1_b100}"

# Training-job resources. They also accept case/model suffixes, e.g.
# SBATCH_MEM_NS2D=64G or SBATCH_GRES_NS2D_IMNO=gpu:l40:1.
SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_GRES_NS2D="${SBATCH_GRES_NS2D:-gpu:l40:1}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

# The aggregation job is CPU-only and runs even if one training job fails, so
# comparison.csv records missing/failed runs. Set REQUIRE_ALL=1 if that job
# should exit non-zero when any summary.json is missing.
SUBMIT_SUMMARY="${SUBMIT_SUMMARY:-1}"
SUMMARY_DEPENDENCY="${SUMMARY_DEPENDENCY:-afterany}"
SUMMARY_PARTITION="${SUMMARY_PARTITION:-$SBATCH_PARTITION}"
SUMMARY_TIME="${SUMMARY_TIME:-00:15:00}"
SUMMARY_MEM="${SUMMARY_MEM:-2G}"
SUMMARY_CPUS="${SUMMARY_CPUS:-1}"
SUMMARY_ACCOUNT="${SUMMARY_ACCOUNT:-$SBATCH_ACCOUNT}"
SUMMARY_QOS="${SUMMARY_QOS:-}"
SUMMARY_CONSTRAINT="${SUMMARY_CONSTRAINT:-}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"

DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-1}"

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

die() {
  echo "ERROR: $*" >&2
  exit 2
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

alpha_slug() {
  local slug="$1"
  slug="${slug//./p}"
  slug="${slug//-/m}"
  slug="${slug//+/p}"
  slug="${slug//E/e}"
  printf '%s' "$slug"
}

canonical_case() {
  case "${1,,}" in
    burgers|bgs) printf 'burgers' ;;
    nonlocalbg|burgersnl|nonlocal_burgers|nonlocalburgers) printf 'nonlocalbg' ;;
    ks) printf 'ks' ;;
    ns1d|1dns|ns1d_eta01) printf 'ns1d_eta01' ;;
    ns1d_eta001) printf 'ns1d_eta001' ;;
    ns2d|2dns) printf 'ns2d' ;;
    ns_2dincom|ns2dincom) printf 'ns_2dincom' ;;
    *) return 1 ;;
  esac
}

case_key() {
  case "$1" in
    burgers) printf 'BURGERS' ;;
    nonlocalbg) printf 'NONLOCALBG' ;;
    ks) printf 'KS' ;;
    ns1d_eta01) printf 'NS1D_ETA01' ;;
    ns1d_eta001) printf 'NS1D_ETA001' ;;
    ns2d) printf 'NS2D' ;;
    ns_2dincom) printf 'NS_2DINCOM' ;;
    *) return 1 ;;
  esac
}

# Reporting metadata only: completed summaries provide their actual metric space.
# This fallback keeps missing/failed seeds in the correct aggregate group.
manifest_metric_space() {
  case "$1" in
    burgers|nonlocalbg|ks) printf "physical" ;;
    ns1d_eta01|ns1d_eta001|ns2d) printf "normalized" ;;
    ns_2dincom) printf "physical" ;;
    *) return 1 ;;
  esac
}

pick_value() {
  local name="$1"
  local key="$2"
  local model="$3"
  local fallback="$4"
  local candidate
  local value

  for candidate in \
    "${name}_${key}_${model}" \
    "${name}_${key}" \
    "${name}_${model}" \
    "$name"; do
    value="${!candidate-}"
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
  done
  printf '%s' "$fallback"
}

validate_export_value() {
  local name="$1"
  local value="$2"
  if [[ "$value" == *','* || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    die "$name may not contain commas or newlines because Slurm --export uses comma separators"
  fi
}

[[ -f "$TRAIN_SBATCH" ]] || die "Missing training script: $TRAIN_SBATCH"
[[ -f "$REPO_ROOT/src/summarize_results.py" ]] || die "Missing summarizer: $REPO_ROOT/src/summarize_results.py"
[[ -x "$PYTHON_BIN" ]] || die "Missing Python: $PYTHON_BIN"
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "TAG may contain only letters, digits, dot, underscore, and dash"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "$ALLOW_EXISTING" == 0 || "$ALLOW_EXISTING" == 1 ]] || die "ALLOW_EXISTING must be 0 or 1"
[[ "$SUBMIT_SUMMARY" == 0 || "$SUBMIT_SUMMARY" == 1 ]] || die "SUBMIT_SUMMARY must be 0 or 1"
[[ "$REQUIRE_ALL" == 0 || "$REQUIRE_ALL" == 1 ]] || die "REQUIRE_ALL must be 0 or 1"
[[ "$SUMMARY_DEPENDENCY" == afterany || "$SUMMARY_DEPENDENCY" == afterok ]] || \
  die "SUMMARY_DEPENDENCY must be afterany or afterok"
if [[ "$DRY_RUN" == 0 ]]; then
  command -v sbatch >/dev/null || die "sbatch is not available on PATH"
fi

for item in "$REPO_ROOT" "$PYTHON_BIN" "$DATA_ROOT" "$EXP_ROOT" "$LOG_ROOT"; do
  validate_export_value path "$item"
done

# Accept comma-separated or whitespace-separated lists.
read -r -a requested_cases <<< "${CASES//,/ }"
read -r -a requested_models <<< "${MODELS//,/ }"
read -r -a requested_seeds <<< "${SEEDS//,/ }"
read -r -a requested_imno_alphas <<< "${IMNO_ALPHAS//,/ }"
(( ${#requested_cases[@]} > 0 )) || die "CASES is empty"
(( ${#requested_models[@]} > 0 )) || die "MODELS is empty"
(( ${#requested_seeds[@]} > 0 )) || die "SEEDS is empty"
(( ${#requested_imno_alphas[@]} > 0 )) || die "IMNO_ALPHAS is empty"

cases=()
declare -A seen_cases=()
for requested_case in "${requested_cases[@]}"; do
  canonical="$(canonical_case "$requested_case")" || die "Unknown case: $requested_case"
  [[ -z "${seen_cases[$canonical]-}" ]] || die "Duplicate case after alias normalization: $canonical"
  seen_cases["$canonical"]=1
  cases+=("$canonical")
done

models=()
declare -A seen_models=()
for requested_model in "${requested_models[@]}"; do
  model="${requested_model^^}"
  case "$model" in
    FNO|IMNO|RNO)
      [[ -z "${seen_models[$model]-}" ]] || die "Duplicate model: $model"
      seen_models["$model"]=1
      models+=("$model")
      ;;
    *) die "Unknown model: $requested_model" ;;
  esac
done

declare -A seen_seeds=()
for seed in "${requested_seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Every seed must be a non-negative integer; got $seed"
  [[ -z "${seen_seeds[$seed]-}" ]] || die "Duplicate seed: $seed"
  seen_seeds["$seed"]=1
done

declare -A seen_alphas=()
for alpha in "${requested_imno_alphas[@]}"; do
  [[ "$alpha" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || \
    die "Every IMNO alpha must be a non-negative finite number; got $alpha"
  [[ -z "${seen_alphas[$alpha]-}" ]] || die "Duplicate IMNO alpha: $alpha"
  seen_alphas["$alpha"]=1
done

jobs_per_case_seed=0
for model in "${models[@]}"; do
  if [[ "$model" == "IMNO" ]]; then
    jobs_per_case_seed=$((jobs_per_case_seed + ${#requested_imno_alphas[@]}))
  else
    jobs_per_case_seed=$((jobs_per_case_seed + 1))
  fi
done
expected_tasks=$((
  ${#cases[@]} * ${#requested_seeds[@]} * jobs_per_case_seed
))

EXPERIMENT_DIR="$EXP_ROOT/$TAG"
RUNS_DIR="$EXPERIMENT_DIR/runs"
LOG_DIR="$LOG_ROOT/$TAG"
MANIFEST="$EXPERIMENT_DIR/manifest.csv"
COMPARISON="$EXPERIMENT_DIR/comparison.csv"
AGGREGATE="$EXPERIMENT_DIR/comparison_aggregate.csv"

if [[ "$DRY_RUN" == 0 ]]; then
  if [[ -e "$EXPERIMENT_DIR" && "$ALLOW_EXISTING" != 1 ]]; then
    die "$EXPERIMENT_DIR already exists; choose a new TAG or set ALLOW_EXISTING=1"
  fi
  mkdir -p "$RUNS_DIR" "$LOG_DIR"
  printf 'case,model,seed,alpha,metric_space,save_dir,job_id,success_marker\n' > "$MANIFEST"
fi

echo "Experiment: $EXPERIMENT_DIR"
echo "Slurm logs: $LOG_DIR"
echo "Cases:      ${cases[*]}"
echo "Models:     ${models[*]}"
echo "Seeds:      ${requested_seeds[*]}"
echo "IMNO alpha: ${requested_imno_alphas[*]}"
echo "Tasks:      $expected_tasks expected"
[[ "$DRY_RUN" == 1 ]] && echo "Mode:       dry-run (nothing will be submitted or created)"

job_ids=()
dry_job_id=900000
for case_name in "${cases[@]}"; do
  key="$(case_key "$case_name")"
  for model in "${models[@]}"; do
    metric_space="$(manifest_metric_space "$case_name")"
    experiment="$(pick_value EXPERIMENT "$key" "$model" "$case_name")"

    partition="$(pick_value SBATCH_PARTITION "$key" "$model" all)"
    gres="$(pick_value SBATCH_GRES "$key" "$model" gpu:1)"
    time_limit="$(pick_value SBATCH_TIME "$key" "$model" 24:00:00)"
    memory="$(pick_value SBATCH_MEM "$key" "$model" 64G)"
    cpus="$(pick_value SBATCH_CPUS "$key" "$model" 4)"
    account="$(pick_value SBATCH_ACCOUNT "$key" "$model" "")"
    qos="$(pick_value SBATCH_QOS "$key" "$model" "")"
    constraint="$(pick_value SBATCH_CONSTRAINT "$key" "$model" "")"

    if [[ "$model" == "IMNO" ]]; then
      run_alphas=("${requested_imno_alphas[@]}")
    else
      run_alphas=("")
    fi

    for seed in "${requested_seeds[@]}"; do
      for run_alpha in "${run_alphas[@]}"; do
        if [[ "$model" == "IMNO" ]]; then
          alpha_id="$(alpha_slug "$run_alpha")"
          save_dir="$RUNS_DIR/$case_name/$model/alpha_$run_alpha/seed_$seed"
          job_name="imno_${case_name}_${model,,}_a${alpha_id}_s${seed}"
          run_name="${case_name}_${model}_alpha${run_alpha}_seed${seed}"
          alpha_label=" / alpha $run_alpha"
        else
          save_dir="$RUNS_DIR/$case_name/$model/seed_$seed"
          job_name="imno_${case_name}_${model,,}_s${seed}"
          run_name="${case_name}_${model}_seed${seed}"
          alpha_label=""
        fi
        validate_export_value SAVE_DIR "$save_dir"
        validate_export_value RUN_NAME "$run_name"

        export_spec="ALL,CASE=$case_name,MODEL=$model,SEED=$seed,SAVE_DIR=$save_dir,RUN_NAME=$run_name,REPO_ROOT=$REPO_ROOT,PYTHON_BIN=$PYTHON_BIN,DATA_ROOT=$DATA_ROOT,EXPERIMENT=$experiment"
        if [[ "$model" == "IMNO" ]]; then
          export_spec+=",ALPHA=$run_alpha"
        fi

        sbatch_cmd=(
          sbatch
          --parsable
          "--job-name=$job_name"
          "--partition=$partition"
          "--gres=$gres"
          "--time=$time_limit"
          "--mem=$memory"
          "--cpus-per-task=$cpus"
          "--output=$LOG_DIR/%x-%j.out"
          "--error=$LOG_DIR/%x-%j.err"
          "--export=$export_spec"
        )
        [[ -n "$account" ]] && sbatch_cmd+=("--account=$account")
        [[ -n "$qos" ]] && sbatch_cmd+=("--qos=$qos")
        [[ -n "$constraint" ]] && sbatch_cmd+=("--constraint=$constraint")
        sbatch_cmd+=("$TRAIN_SBATCH")

        echo "Submitting $case_name / $model / seed $seed$alpha_label"
        if [[ "$DRY_RUN" == 1 ]]; then
          print_command "${sbatch_cmd[@]}"
          dry_job_id=$((dry_job_id + 1))
          job_id="$dry_job_id"
        else
          submission="$("${sbatch_cmd[@]}")"
          job_id="${submission%%;*}"
          [[ "$job_id" =~ ^[0-9]+$ ]] || die "Could not parse sbatch job id from: $submission"
          success_marker="$save_dir/_SUCCESS_$job_id"
          printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$case_name" "$model" "$seed" "$run_alpha" "$metric_space" \
            "$save_dir" "$job_id" "$success_marker" >> "$MANIFEST"
          echo "  job_id=$job_id"
        fi
        job_ids+=("$job_id")
      done
    done
  done
done

(( ${#job_ids[@]} > 0 )) || die "No training jobs were generated"

if [[ "$SUBMIT_SUMMARY" == 1 ]]; then
  dependency_ids="$(IFS=:; printf '%s' "${job_ids[*]}")"
  summary_cmd=(
    "$PYTHON_BIN"
    -m src.summarize_results
    --experiment-dir "$EXPERIMENT_DIR"
    --manifest "$MANIFEST"
    --output "$COMPARISON"
    --aggregate-output "$AGGREGATE"
  )
  [[ "$REQUIRE_ALL" == 1 ]] && summary_cmd+=(--require-all)
  printf -v summary_wrap '%q ' "${summary_cmd[@]}"

  summary_sbatch=(
    sbatch
    --parsable
    "--job-name=imno_summary_${TAG:0:24}"
    "--partition=$SUMMARY_PARTITION"
    "--time=$SUMMARY_TIME"
    "--mem=$SUMMARY_MEM"
    "--cpus-per-task=$SUMMARY_CPUS"
    "--dependency=$SUMMARY_DEPENDENCY:$dependency_ids"
    "--output=$LOG_DIR/summary-%j.out"
    "--error=$LOG_DIR/summary-%j.err"
    "--wrap=$summary_wrap"
  )
  [[ -n "$SUMMARY_ACCOUNT" ]] && summary_sbatch+=("--account=$SUMMARY_ACCOUNT")
  [[ -n "$SUMMARY_QOS" ]] && summary_sbatch+=("--qos=$SUMMARY_QOS")
  [[ -n "$SUMMARY_CONSTRAINT" ]] && summary_sbatch+=("--constraint=$SUMMARY_CONSTRAINT")

  echo "Submitting dependent result aggregation"
  if [[ "$DRY_RUN" == 1 ]]; then
    print_command "${summary_sbatch[@]}"
  else
    summary_submission="$("${summary_sbatch[@]}")"
    summary_job_id="${summary_submission%%;*}"
    echo "  summary_job_id=$summary_job_id ($SUMMARY_DEPENDENCY)"
  fi
fi

echo "Manifest:   $MANIFEST"
echo "Per-run:    $COMPARISON"
echo "Aggregated: $AGGREGATE"
echo "Logs:       $LOG_DIR"
