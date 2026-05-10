# synth/program.md — autoresearch instruction layer

This is the Karpathy-style "research org code" the autoresearch loop reads
each iteration. It defines the active target, what the LLM is allowed to
edit, the correctness contract, the performance metric, and the keep/discard
rule. Tightening this file is part of doing better autoresearch — when
results plateau, the program description is usually the lever, not the
search engine.

---

## Active target

```
problem_family: total_variation_1d
algorithm:      pdhg
variant:        basic
hardware:       apple_silicon
backend:        mlx
dtype:          fp32
shape:          tv1d_medium  (n=2048, 16 piecewise-constant pieces, noise=0.1)
workload:       single solve (no setup amortization for now; PDHG has no
                 problem-dependent precompute analogous to FISTA-Gram)
gate:           gap < 1e-6 (primal-dual gap on TVDenoising1D, scale-free)
metric:         median solve_time_ms over 5 reps
margin:         0.97 (must be ≥3% faster than current best to promote)
```

## Current best (regression baselines, 2026-05-10)

```
seed kernel:    convexkernels/kernels/mlx/seeds/pdhg_step_v0.py
mlx fp32 (5-rep median, tv1d_medium):  3802 ms (5000 iters, gap 1e-6)
mlx fp32 (5-rep median, tv1d_small):   2188 ms (2900 iters, gap 1e-6)

numpy fp64 reference (for context, not the gating baseline):
numpy fp64 (5-rep median, tv1d_medium): 146 ms
numpy fp64 (5-rep median, tv1d_small):   57 ms
```

The MLX seed is launch-overhead bound (5+ ops per iter × ~5000 iters × ~150 µs
Metal launch ≈ measured time). The autoresearch loop's job is to fuse those
launches.

## What the LLM may change

The seed kernel is a single file. Edit it freely, subject to:

1. **Public interface** — must expose `init_state(problem, *, tau=None,
   sigma=None, theta=1.0)` and `pdhg_step(state, problem)`. The `state`
   object must have attributes `x`, `y`, `x_bar`, `tau`, `sigma`, `theta`.
2. **Algorithm semantics** — the math must remain Chambolle-Pock 2011 PDHG
   (or a documented variant with a known convergence proof). No silent
   tolerance relaxations, no algorithmic shortcuts that break for the
   specimen at hand. Specifically:
   - The y update must apply prox of g* (clip to [-lam, lam] for TV-L1).
   - The x update must apply prox of f (closed form for f = ½||x - b||²).
   - The extrapolation step must be x_bar = x_new + theta (x_new - x_old).
3. **dtype** — fp32 storage is the default. fp16 storage with fp32 accumulators
   is allowed if and only if the gap < 1e-6 acceptance gate still passes;
   end-to-end fp16 will fail the gate and is silently rejected.

## What is rich search surface (encouraged)

The most valuable mutations for this specimen are:

- **Op fusion**: collapse the K_apply + scaling + clip + K_T_apply chain into
  one or two Metal kernels. Right now there are 5+ launches per iter; the
  ceiling is 1.
- **Single-pass kernel**: with one threadgroup per element of x, each thread
  can read x_bar[i-1..i+1], y[i-1..i], x[i], b[i], and write
  x_new[i], x_bar_new[i], y_new[i-1] in one launch.
- **Threadgroup memory**: cache a slab of x_bar in TG memory before computing
  K_apply * sigma + clip; saves global memory traffic on the K_T leg.
- **Precision experiments** that preserve gap < 1e-6: e.g. fp16 storage with
  fp32 accumulator on K_apply, periodic fp32 correction on x.
- **Algorithm variant** flags: `variant="accelerated"` or `"restart"`. These
  change the host driver's outer loop only, not the kernel; but the kernel
  may exploit a known step-size schedule (e.g. fixed tau = 1/L_K) when the
  outer driver promises it.

## What is low-value (avoid burning budget on)

- **Threadgroup size sweeps** beyond {128, 256, 512}: covered by the
  deterministic grid; the LLM should only revisit if structural changes make
  a different size optimal.
- **Cosmetic refactors** (variable renames, comment churn, identical math
  rewritten with `mx.add` vs `+`): no measurable effect, eats budget.
- **Boundary-condition swaps** that change the answer: TV-1D forward
  difference is the contract; circular or periodic boundaries are a
  different problem and will fail the CVXPY-anchored correctness check.

## Hard rules

- **Don't relax the tolerance** to make a candidate pass. gap < 1e-6 is the
  contract. If a candidate's gap_final is 1.2e-6, it's rejected — propose
  more iters or a tighter formulation, not a looser gate.
- **Don't propose `kernel_step` and `init_state` removals**. The synth-loop
  evaluator imports those by name from the candidate file.
- **Don't pickle MLX arrays into the candidate source**. Build them in
  `init_state` or `prepare_problem`.

## Logging

Every proposal lands as a row in:

```
synth_state/lineage.jsonl
```

Per-row schema is in `convexkernels/synth/lineage.py`. The fields the LLM
should consult in subsequent proposals (via the loop's history-with-
rationales context) are:

- `decision.accepted` — bool
- `decision.reason` — short string (e.g. "discard:not_faster_than_baseline",
  "tier_failed:gap", "kept:passed")
- `tier1.solve_time_ms` (median of N reps)
- `tier1.gap_final`
- `edit.rationale` — what the proposer claimed; useful to see what was tried

Per-iter trajectories of gap, primal, dual residual go into the eval result
and are summarized in the next proposer prompt. Looking only at the final
scalar misses divergence/stall information.

## Autonomy instruction

Once a session begins:

- Do not stop after one proposal. Run until the configured proposal budget
  finishes or the user interrupts.
- Crashes → record + continue. Same-source duplicates → discard + continue.
- Periodically (every ~10 proposals) update `tasks/results.md` with a
  summary of accepts / rejects / fastest-by-gap-margin.
- After ~50 unsuccessful proposals on the same target, flag stagnation and
  pause for user input. If the loop hasn't found a fusion in 50 attempts,
  the seed or program.md is the bug, not the engine.

## Open invitations

Mutations that would expand the autoresearch territory (file as new edits if
a clean Pareto improvement is found):

1. **2D specimen**: TVDenoising2D with isotropic prox is a natural sibling
   slot. Same algorithm; the K operator becomes a 2-tuple of grad images.
2. **Constrained LASSO via PDHG**: $\min ½\|Ax-b\|^2 + \lambda\|x\|_1$
   s.t. $Ex = d$ is a PDHG target with a different K = E.
3. **Anderson acceleration**: provably converging variant of PDHG with
   Anderson-style mixing of past iterates. Requires a small extra state
   buffer; correctness check is still gap < 1e-6.

These are speculative — the bottom line for the current target is closing
the 30× gap to numpy fp64 via fusion.
