# synth/program.md — autoresearch instruction layer

This is the Karpathy-style "research org code" the autoresearch loop reads
each iteration. It defines the active target, what the LLM is allowed to
edit, the correctness contract, the performance metric, and the keep/discard
rule. Tightening this file is part of doing better autoresearch.

---

## Active target

```
problem_family: lasso_path
algorithm:      fista_gram   (FISTA in the Gram form with adaptive Gram-or-direct dispatch)
variant:        basic        (the seed already includes O'Donoghue-Candes per-column gradient restart)
hardware:       apple_silicon (M3 Pro, ~150 GB/s unified-memory bandwidth, ~5 TF fp32 GPU)
backend:        mlx
dtype:          fp32
shape:          path_wide_hero  (m=1000 samples, n=50000 features, K=50 log-spaced lambdas, sparsity 1%)
workload:       full reg-path  (one batched solve over all K lambdas; cold-start each session)
gate:           max-over-lambdas KKT-batched residual < 1e-6  (scale-free, per-lambda)
metric:         median solve_time_ms over 3 reps (including setup = Gram precompute or direct A upload)
margin:         0.97 (must be ≥3% faster than current best to promote)
```

## Why this slot exists

This is the headline regime of the project — "fastest full-regularization-path
LASSO on Apple Silicon, beat Adelie." Adelie (Stanford glmnet for grouped
elastic net) is the dominant CPU LASSO solver. On `path_wide_hero` it does
the full 50-λ path in 4131.8 ms median (5-rep) on M3 Pro under tol=1e-12,
early_exit=False. Our hand-written batched FISTA-Gram seed does it in
34731.6 ms — 8.4× slower than Adelie. The autoresearch loop's job: bring
that under 4131.8 ms.

On `path_tall_medium` (m=5000, n=2000) the seed already beats Adelie 1.82×
(80.1 ms vs 145.7 ms). On `path_wide_small` (m=500, n=2000) the seed is at
0.42× (97.6 vs 40.6 ms). On `path_square` (m=10000, n=10000) the seed is
at 0.87× (1240 vs 1077 ms). The hero is the bottleneck.

## Where Adelie wins on hero (the autoresearch's diagnosis)

Adelie's advantages on wide p≫n:
- **Per-cycle coordinate descent**: only touches features whose coefficient
  could change at this λ (SAFE/STRONG screening rules).
- **Warm-start along the path**: solution at λ_k is initialized from
  solution at λ_{k-1}, so per-λ iteration count is small.
- **Active-set restriction**: most features are inactive most of the time.

The batched FISTA-Gram seed has none of these:
- Cold-starts all K=50 columns from zero simultaneously.
- Iterates every (i, k) ∈ (50000, 50) cell every iter (no masking).
- G doesn't fit on hero (10 GB > Metal cap), so it uses direct A-form
  (`A.T @ (A @ Y - b[:, None])`, two matmuls/iter).

The autoresearch loop should compose techniques from Adelie's playbook
**within the FISTA framework**, not replace FISTA with CD.

## What the LLM may change

The seed kernel is `convexkernels/kernels/mlx/seeds/gram_fista_path_v0.py`.
It defines `prepare_problem(LassoPathMLX) -> LassoPathGramMLX`,
`init_state(problem) -> FistaPathStateMLX`,
`fista_path_step(state, problem, t) -> FistaPathStateMLX`, and
`kkt_max(state, problem) -> float`. Edit the file freely, subject to:

1. **Public interface** — must expose those four names with the same
   call signatures. `state.X`, `state.Y`, `state.theta` attributes are
   required (the driver inspects them). Returned state must be the same
   shape `(n, K)`.

2. **Algorithm semantics** — math must remain FISTA on the LASSO objective
   `0.5 ||Ax - b||² + lam_k * ||x_k||_1` per column. The proximal step
   must be soft-threshold; the smooth gradient must be `A^T(Ax - b)` or
   `Gx - c` (Gram form). The momentum recursion must be a Nesterov
   sequence (or a restart-modified variant). No silent tolerance
   relaxations.

3. **dtype** — fp32 storage is default. fp16 storage with fp32 KKT
   verification is allowed if and only if `kkt_max < 1e-6` still passes
   on the multi-rep eval.

## What is rich search surface (encouraged)

For `path_wide_hero` specifically:

- **Per-column convergence masking**: when a column's KKT residual is
  below tol, freeze its X and Y to current values (no further update).
  Iterating all K columns to the slowest's convergence is wasted work
  — most columns at large λ converge in tens of iterations while small-λ
  columns need thousands.
- **SAFE / STRONG screening rules** (El Ghaoui 2010, Tibshirani 2012):
  prove feature j is inactive at λ from the dual, drop it from the
  active matrix in the inner iter. For sparsity 1% on hero (~500 active
  out of 50000), screening can reduce per-iter cost 50–100×. **This is
  the single biggest lever for wide problems and the discovery the
  autoresearch loop is most likely to find.**
- **fp16 inner with fp32 KKT**: matmuls in fp16 (~2× bandwidth on M3),
  KKT verification in fp32. Soft-threshold is naturally a denoiser so
  ADMM/FISTA tolerate this well.
- **Algorithm variants with proofs**: e.g. accelerated proximal gradient
  with adaptive step (Scheinberg-Goldfarb-Bai 2014), Nesterov's
  composite gradient (2007). Cite the variant by name in the rationale.
- **Warm-start across the path**: instead of cold-starting all K columns
  at zero, initialize column k from column k-1 (sequential along the
  decreasing-λ path). Convergence for column k starts much closer to its
  optimum. Combine with per-column masking and you've recovered most of
  Adelie's edge while keeping the matmul-batched form.
- **Fused Metal kernel** for the soft-threshold + momentum + restart tail:
  the current seed uses MLX elementwise ops which the lazy graph fuses
  but with reduction overhead. A single hand-written kernel reading
  (Y, g, X_prev, lambdas/L, theta) and writing (X_next, Y_next, theta_next)
  is ~3× faster than the current op chain on (n, K) workloads.

## What is low-value (avoid burning budget on)

- **Threadgroup-size tweaks** without changing the underlying op count.
- **Cosmetic refactors** (variable renames, equivalent math rewritten).
- **Replacing `mx.matmul`** with hand-rolled matmul kernels. MLX's matmul
  is already bandwidth-tuned by Apple; you will not beat it for the
  generic (m, n) × (n, K) matmul. Lever is the *surrounding* ops.

## Hard rules

- **Don't relax the tolerance**. `kkt_max_final < 1e-6` is the contract.
  If a candidate's value is 1.2e-6, it is rejected — propose more iters
  or screening, not a looser gate.
- **Don't drop `init_state` or `fista_path_step` or `kkt_max` or
  `prepare_problem`** from the module. The eval harness imports them by
  name.
- **Don't pickle Adelie's solution or any precomputed answer into the
  candidate source**. The candidate must produce the answer from `(A, b,
  lambdas)` alone.

### No algorithm replacement

The autoresearch loop's job is to **make batched FISTA on the LASSO path
faster**, not to replace it with a different solver. The active algorithm
is FISTA-Gram on the path (with O'Donoghue-Candes restart and the listed
allowed amortizations). The model is expected to optimize the
implementation of *that* algorithm — fuse ops, mask columns, add
screening, exploit structure — not to swap in coordinate descent and wrap
it in a fake `fista_path_step` for contract compliance.

Forbidden patterns (non-exhaustive):

- **Importing Adelie, sklearn, glmnet, or any third-party LASSO solver**
  in the candidate source. The candidate must use only numpy + mlx +
  python stdlib.
- **Pre-solving with coordinate descent / Adelie / sklearn** inside
  `init_state` and then doing trivial FISTA "verification" iterations.
- **Hardcoded paths or X_adelie arrays** in the candidate source.
- **Closed-form solutions that bypass the iteration** (e.g. computing
  `X = soft(A^T b / L, lam/L)` per column without iterating to
  convergence).

Distinguish from **allowed amortizations** (encouraged, explicitly):

- **Gram precompute**: `G = A^T A` and `c = A^T b` once for the whole
  path. Already in the seed for moderate n.
- **Cholesky / inverse caching** for ADMM-style reformulations (if the
  candidate switches to an ADMM-equivalent formulation that converges
  to the same per-column LASSO optimum, document it; the autoresearch
  doesn't ban algorithm-equivalent reformulations, only direct CD
  imports).
- **Per-column convergence masking**, **SAFE/STRONG screening rules**,
  **warm-start across the path**, **fp16 inner with fp32 KKT** — all
  explicitly listed above as the rich search surface.
- **Fused Metal kernels** for the per-iter tail.

Litmus test: if you replaced the public `fista_path_step` body with
`return state` (no-op) and your solution still passed because the work
was all in `prepare_problem` or `init_state`, you have replaced the
algorithm. Reject your own proposal and try again. The timing contract
is enforced (`t0 = perf_counter()` before `kernel_init`), so stuffing
work into setup will show up as setup_ms in the wall_time anyway.

## Logging

Every proposal lands as a row in `synth_state/lineage.jsonl`. Schema in
`convexkernels/synth/lineage.py`. Fields the LLM should consult in
subsequent proposals via the loop's history-with-rationales context:

- `decision.accepted` — bool
- `decision.reason` — short string (e.g. "discard:not_faster_than_baseline",
  "discard:kkt_above_tol", "kept:passed")
- `tier1.solve_time_ms` (median of 3 reps)
- `tier1.kkt_final` (max-over-columns)
- `edit.rationale` — what the proposer claimed

Per-iter KKT trajectory goes into `trajectory.json` and is summarized
into the next proposer prompt (compressed to ~9 anchor points).

## Autonomy instruction

Once a session begins:

- Do not stop after one proposal. Run until the configured budget
  finishes or the user interrupts.
- Crashes → record + continue. Duplicates → discard + continue.
- Each session begins from the current best (`champion.py` in the state
  root) and explores from there. Cross-chain experiments are run in
  parallel via `scripts/parallel_chains.sh`.
- After ~50 unsuccessful proposals on the same target, flag stagnation.
  If nothing has gone in after 50 attempts, the seed or program.md is
  likely the bug.

## Open invitations

Mutations expanding the autoresearch territory beyond what is listed:

1. **Different inner-solve form** (e.g. ADMM-Woodbury for wide problems):
   if the candidate provides a different but mathematically equivalent
   reformulation with documented convergence (e.g. ADMM with Woodbury
   identity for tall-rewriting), it is allowed. The KKT gate still
   enforces correctness; speedup gate enforces relevance.
2. **Block-coordinate FISTA**: alternate FISTA on subsets of columns
   instead of all K together. Variant of the algorithm; provably
   convergent.
3. **Adaptive λ-ordering**: solve large λ first (sparse → small active
   set → fast), then progressively smaller λ with the previous solution
   as warm start. This is the Adelie strategy expressed as
   path-ordering, *not* as algorithm replacement.

The bottom line for the current target: close the 8.4× gap to Adelie on
`path_wide_hero`. Per-column masking + screening + warm-start across the
path is the path of least resistance.
