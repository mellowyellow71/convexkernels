#!/bin/bash
# Run N independent autoresearch chains on the same slot (sequentially on a
# single Mac to avoid GPU contention; concurrency in the LLM API calls comes
# for free since they're network-bound).
#
# Each chain has its own state-root and lineage.jsonl. At the end we compute
# per-chain best champion and the cross-chain best.
#
# Path-dependence finding (2026-05-10): the same slot ran twice produced
# 99x and 9.21x speedups respectively, hitting different lever points.
# This wrapper is the test for whether parallel exploration finds a more
# reliable best champion.
#
# Usage on Mac:
#   cd ~/convexkernels && ./scripts/parallel_chains.sh \
#     <slot> <shape> <n_chains> <n_per_chain> [extra_flags...]
#
# Example:
#   ./scripts/parallel_chains.sh \
#     total_variation_1d/pdhg/apple_silicon/fp32 tv1d_medium 4 30 \
#     --variant basic --max-iters 5000 --fitness-tol 1e-6

set -e

SLOT="$1"
SHAPE="$2"
N_CHAINS="$3"
N_PER_CHAIN="$4"
shift 4

DATE=$(date +%Y%m%d_%H%M)
TAG=$(echo "${SLOT}" | tr '/' '_')
ROOT_PREFIX="synth_run_parallel_${TAG}_${SHAPE}_${DATE}"

mkdir -p parallel_logs

COMMON_FLAGS="--proposer openai --model gpt-5.5 --reasoning-effort medium --api-timeout-s 240 --warmup-runs 1 --speedup-margin 0.97 --reps 3 --problem-backend mlx --program-md convexkernels/synth/program.md --timeout-s 240"

echo "=== parallel_chains: ${N_CHAINS} chains x ${N_PER_CHAIN} props on ${SLOT}/${SHAPE} ==="
date

for i in $(seq 0 $((N_CHAINS - 1))); do
    STATE_ROOT="./${ROOT_PREFIX}_chain_${i}"
    LOG="parallel_logs/${ROOT_PREFIX}_chain_${i}.log"
    echo
    echo "--- chain ${i}/$((N_CHAINS - 1)) -> ${STATE_ROOT} ---"
    date

    # OpenAI Responses API doesn't expose a deterministic seed; run-to-run
    # diversity comes from the model's natural sampling stochasticity. We
    # nonetheless pass a per-chain MARKER via OPENAI_USER_TAG which the
    # model sees in metadata for telemetry isolation.
    OPENAI_USER_TAG="chain_${i}_${DATE}" \
    .venv/bin/python -m convexkernels.synth.run \
        --slot "${SLOT}" --shape "${SHAPE}" \
        $COMMON_FLAGS \
        --n-proposals "${N_PER_CHAIN}" \
        --state-root "${STATE_ROOT}" \
        "$@" \
        > "${LOG}" 2>&1 || echo "chain ${i} exited $?"
done

echo
echo "=== aggregating ==="
date

# Use the lineage_summary.py we already have, plus a cross-chain best pick.
.venv/bin/python scripts/lineage_summary.py "${ROOT_PREFIX}_chain_"*/

echo
echo "=== cross-chain best champion ==="
.venv/bin/python -c "
import json, sys
from pathlib import Path
import math

best = None
best_ms = math.inf
best_chain = None
for d in sorted(Path('.').glob('${ROOT_PREFIX}_chain_*')):
    lp = d / 'lineage.jsonl'
    if not lp.exists():
        continue
    rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
    chain_kept = [r for r in rows if (r.get('decision') or {}).get('accepted')]
    chain_best, chain_best_ms = None, math.inf
    for r in chain_kept:
        s = r.get('score') or {}
        ms = s.get('solve_ms_median')
        if isinstance(ms, (int, float)) and ms < chain_best_ms:
            chain_best, chain_best_ms = r, ms
    print(f\"  {d.name}: kept={len(chain_kept)}/{len(rows)} best_ms={chain_best_ms:.2f}\" if math.isfinite(chain_best_ms) else f\"  {d.name}: kept={len(chain_kept)}/{len(rows)} best_ms=none\")
    if math.isfinite(chain_best_ms) and chain_best_ms < best_ms:
        best = chain_best
        best_ms = chain_best_ms
        best_chain = d.name

if best is not None:
    s = best.get('score') or {}
    print()
    print(f\"cross-chain best: {best_chain} gen={best.get('generation')} id={best['id'][:8]}\")
    print(f\"  wall_ms={s.get('solve_ms_median'):.2f} +/- {s.get('solve_ms_std'):.2f}\")
    print(f\"  iters={s.get('iters')}, fitness={s.get('fitness_final'):.2e}\")
    rationale = (best.get('edit') or {}).get('rationale', '')[:200].replace(chr(10), ' ')
    print(f\"  rationale: {rationale}\")
else:
    print('  (no champions across any chain)')
"

echo
echo "=== rendering comparison dashboard ==="
.venv/bin/python scripts/lineage_dashboard.py "${ROOT_PREFIX}_chain_"*/ \
    --out "parallel_logs/${ROOT_PREFIX}_dashboard.html" --refresh 0
echo "dashboard at parallel_logs/${ROOT_PREFIX}_dashboard.html"
date
echo "=== done ==="
