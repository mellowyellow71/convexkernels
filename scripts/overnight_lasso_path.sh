#!/bin/bash
# Overnight autoresearch session on the lasso_path/fista_gram slot.
#
# Usage (on the M3 Pro Mac, from convexkernels/ repo root):
#   ./scripts/overnight_lasso_path.sh path_wide_hero 100
#       └ runs 100 proposals on the hero shape, the Phase 3 falsification test
#   ./scripts/overnight_lasso_path.sh path_tall_medium 50
#       └ runs 50 proposals on the regression shape we already win on
#
# Lineage goes to ./synth_run_lasso_path_${SHAPE}_${TIMESTAMP}/.
# Rerun the lineage dashboard:
#   .venv/bin/python scripts/lineage_dashboard.py synth_run_lasso_path_${SHAPE}_*/ \
#     --out /tmp/dashboard.html --refresh 30
#
# Reqs: $OPENAI_API_KEY must be set. Mac M3 Pro, mlx installed in .venv.

set -e

SHAPE="${1:-path_wide_hero}"
N_PROPOSALS="${2:-100}"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
STATE_ROOT="./synth_run_lasso_path_${SHAPE}_${TIMESTAMP}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set" >&2
    exit 1
fi

cd "$(dirname "$0")/.."  # repo root

echo "[overnight] starting lasso_path autoresearch"
echo "[overnight] shape: $SHAPE"
echo "[overnight] proposals: $N_PROPOSALS"
echo "[overnight] state root: $STATE_ROOT"
echo "[overnight] $(date +%H:%M:%S)"

# Reps: more on hero (4131.8 ms baseline) for tighter median; warmup 2 to
# absorb subprocess cold-start. Timeout per proposal: 600 s (allows hero's
# slower candidates to actually finish so the loop sees their wall_ms).
case "$SHAPE" in
    path_wide_hero)
        REPS=3
        MAX_ITERS=10000
        TIMEOUT=900
        ;;
    *)
        REPS=5
        MAX_ITERS=5000
        TIMEOUT=300
        ;;
esac

.venv/bin/python -m convexkernels.synth.run \
    --slot lasso_path/fista_gram/apple_silicon/fp32 \
    --shape "$SHAPE" \
    --proposer openai \
    --model gpt-5.5 \
    --reasoning-effort medium \
    --n-proposals "$N_PROPOSALS" \
    --reps "$REPS" \
    --max-iters "$MAX_ITERS" \
    --problem-backend mlx \
    --problem-dtype fp32 \
    --state-root "$STATE_ROOT" \
    --program-md convexkernels/synth/program.md \
    --warmup-runs 2 \
    --timeout-s "$TIMEOUT" \
    --fitness-tol 1e-6 \
    --speedup-margin 0.97

echo "[overnight] $(date +%H:%M:%S) — session complete"
echo "[overnight] lineage: $STATE_ROOT/lineage.jsonl"
echo "[overnight] champion: $STATE_ROOT/champion.py"
echo ""
echo "Render dashboard:"
echo "  .venv/bin/python scripts/lineage_dashboard.py $STATE_ROOT --out /tmp/lasso_path_dashboard.html"
