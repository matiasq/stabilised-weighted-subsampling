#!/usr/bin/env bash
set -uo pipefail

# Run all regression-GARCH DC-BATS experiments sequentially.
# Each run gets its own log. A failed run is recorded, but later runs still proceed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="dcbats_overnight_logs"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
master_log="$LOG_DIR/master_${timestamp}.log"

runs=(
  "25 10"
  "25 20"
  "25 37"
  "50 10"
  "50 20"
  "50 36"
)

failures=()

{
  echo "============================================================"
  echo "Regression-GARCH DC-BATS overnight batch"
  echo "Started: $(date)"
  echo "Repository: $SCRIPT_DIR"
  echo "============================================================"
} | tee -a "$master_log"

for run in "${runs[@]}"; do
  read -r p K <<< "$run"
  run_log="$LOG_DIR/p${p}_K${K}_${timestamp}.log"

  {
    echo
    echo "============================================================"
    echo "Starting p=${p}, K=${K}: $(date)"
    echo "Log: $run_log"
    echo "============================================================"
  } | tee -a "$master_log"

  if Rscript run_regression_garch_dc_bats.R --p="$p" --K="$K" 2>&1 | tee "$run_log"; then
    echo "COMPLETED p=${p}, K=${K}: $(date)" | tee -a "$master_log"
  else
    status=${PIPESTATUS[0]}
    echo "FAILED p=${p}, K=${K} with exit status ${status}: $(date)" | tee -a "$master_log"
    failures+=("p=${p},K=${K},status=${status}")
  fi
done

{
  echo
  echo "============================================================"
  echo "Batch finished: $(date)"
  if ((${#failures[@]} == 0)); then
    echo "All six DC-BATS runs completed successfully."
  else
    echo "Some runs failed:"
    printf '  %s\n' "${failures[@]}"
  fi
  echo "Master log: $master_log"
  echo "============================================================"
} | tee -a "$master_log"

if ((${#failures[@]} > 0)); then
  exit 1
fi
