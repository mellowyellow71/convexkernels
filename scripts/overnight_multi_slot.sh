#!/bin/bash
# Multi-slot overnight autoresearch session.
#
# Runs three sequential autoresearch sessions on the Mac MLX evaluator,
# logging each one's output to its own file. Intended for a 4-6 hour
# overnight budget on a single Mac (no concurrency — the Mac sleeps poorly
# under two simultaneous tall_medium synth jobs per `tasks/program_kernel.md`).
#
# Usage on Mac:
#   cd ~/convexkernels && ./scripts/overnight_multi_slot.sh
#
# Output:
#   overnight_logs/pdhg_tv_continuation.log
#   overnight_logs/admm_lasso_extension.log
#   overnight_logs/pdhg_bp_medium.log
#
# Lineage / champions land under the per-run state-roots; tail each .log to
# see live progress.

set -e

mkdir -p overnight_logs

DATE=$(date +%Y%m%d_%H%M)
COMMON_FLAGS="--proposer openai --model gpt-5.5 --reasoning-effort medium --api-timeout-s 240 --warmup-runs 1 --speedup-margin 0.97"

echo "=== overnight session $DATE ==="
date

# 1. PDHG-TV-1D continuation: 80 proposals on tv1d_medium under corrected eval.
#    Push past the 256-step temporal fusion ceiling. ~1.5-2 hr.
echo "--- pdhg_tv_continuation ---"
date
.venv/bin/python -m convexkernels.synth.run \
  --slot total_variation_1d/pdhg/apple_silicon/fp32 \
  --shape tv1d_medium \
  $COMMON_FLAGS \
  --n-proposals 80 --reps 3 --max-iters 5000 \
  --problem-backend mlx --variant basic \
  --state-root ./synth_run_overnight_pdhg_tv_${DATE} \
  --program-md convexkernels/synth/program.md \
  --timeout-s 240 --fitness-tol 1e-6 \
  > overnight_logs/pdhg_tv_continuation.log 2>&1 || echo "pdhg_tv exited $?"
date

# 2. ADMM-LASSO extension: 30 proposals on tall_small. Past the 5-prop smoke.
#    ~30-45 min.
echo "--- admm_lasso_extension ---"
date
.venv/bin/python -m convexkernels.synth.run \
  --slot lasso_admm/admm/apple_silicon/fp32 \
  --shape tall_small \
  $COMMON_FLAGS \
  --n-proposals 30 --reps 3 --max-iters 1000 \
  --problem-backend mlx --variant adaptive \
  --state-root ./synth_run_overnight_admm_${DATE} \
  --program-md convexkernels/synth/program.md \
  --timeout-s 180 --fitness-tol 1e-5 \
  > overnight_logs/admm_lasso_extension.log 2>&1 || echo "admm exited $?"
date

# 3. PDHG-BP at bp_medium: 50 proposals. Extends the 5-prop smoke on bp_small
#    to a more meaningful problem size. ~1-1.5 hr.
echo "--- pdhg_bp_medium ---"
date
.venv/bin/python -m convexkernels.synth.run \
  --slot basis_pursuit/pdhg/apple_silicon/fp32 \
  --shape bp_medium \
  $COMMON_FLAGS \
  --n-proposals 50 --reps 3 --max-iters 30000 \
  --problem-backend mlx --variant basic \
  --state-root ./synth_run_overnight_bp_${DATE} \
  --program-md convexkernels/synth/program.md \
  --timeout-s 240 --fitness-tol 1e-6 \
  > overnight_logs/pdhg_bp_medium.log 2>&1 || echo "bp exited $?"
date

echo "=== overnight session complete ==="
date
