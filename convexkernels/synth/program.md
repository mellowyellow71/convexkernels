# synth/program.md — autoresearch instruction layer

This is the "research org code" the autoresearch loop reads each iteration. It
defines the active target, what you (the proposer) may change, the correctness
contract, the performance metric, and the keep/discard rule.

---

## Active target

```
problem:   lasso_path   (full regularization path: per-column LASSO over K lambdas)
hardware:  apple_silicon (M3 Pro, ~150 GB/s unified memory, ~5 TF fp32 GPU; MLX)
```

**That is the whole spec.** Algorithm, precision, quantization format, kernel
layout, and data structure are *not* fixed — they are the search space. Your job
is to discover the solver/kernel stack that reaches the optimality target
fastest, by any convex-correct means.

## Metric: time to a KKT-verified optimality target

The loop measures every method — your candidate and every classical baseline —
on one identical ruler: the **trusted, scale-free KKT residual** of the
canonical numpy problem (`bench/metrics.py:trusted_kkt`). It is a self-certifying
optimality certificate, so no `f*` reference solve is needed.

```
target:  trusted KKT residual < 1e-5        (the convergence contract)
metric:  total_time_s = setup_time_s + time_to_kkt_s   (median over reps)
         — cold-start wall-clock to reach the target, lower is better
margin:  0.97 — must be ≥3% faster than the current champion to promote
```

**On the target (1e-5) and the fp32 floor.** End-to-end fp32 FISTA on these
shapes plateaus at a trusted KKT of ~1–3e-6 (the fp32 rounding floor of the
gradient `A^T(Ax-b)`), so 1e-6 is *not reachable* by a pure-fp32 solver and is
not the contract. Reaching 1e-6 or tighter is a real, separately-creditable
research direction: do the bulk of the work in fast fp32/fp16, then add a short
fp64 (or mixed-precision iterative-refinement) polish on the gradient/residual
to drive past the floor. A candidate that reaches 1e-6 honestly is a headline
result; the everyday gate is 1e-5.

The bar to beat is the **baseline panel**: CLARABEL (interior point), SCS, OSQP,
ECOS, plus adelie / sklearn where applicable. Their `time_to_kkt` values appear
in the research state each iteration. The headline claim is "reach the target
faster than the classical solvers."

## The contract you implement

Your candidate is a single Python module exposing **one entry point**:

```python
def solve(problem, recorder, *, kkt_tol, max_time_s) -> X:
    # run ANY algorithm; report progress; return the final iterate
```

and (optionally) `prepare_problem(problem[, config])` for one-time setup (Gram
precompute, device upload, quantization) — its cost is timed as `setup_time_s`.

Rules of the contract:

1. **Report progress through the recorder.** Call `recorder.record(x)` every few
   iterations. The recorder timestamps the iterate and evaluates the trusted
   KKT (you cannot compute or fake the metric yourself). Stop when
   `recorder.should_stop(kkt_tol)` is true (target reached or `max_time_s`
   budget spent).
2. **Return the final iterate `X`** (shape `(n, K)` for the path). The harness
   recomputes the trusted KKT on it as the final anti-gaming gate.
3. **Stay convex-correct.** The returned `X` must genuinely reach
   `kkt < 1e-6`. No tolerance relaxation, no pickled/precomputed answers, no
   importing third-party LASSO solvers (adelie/sklearn/glmnet) into the
   candidate. Use only numpy + mlx + python stdlib.

## Rich search surface (encouraged — this is wide on purpose)

The previous iteration's search space was too narrow (tail-edits to a fixed
FISTA-Gram kernel). Go broad:

- **Algorithm family is open.** FISTA / accelerated proximal gradient, ADMM,
  PDHG/Chambolle-Pock, (block) coordinate descent, prox-Newton, screening-
  augmented variants. Pick what fits the shape; cite the variant in your
  rationale. The KKT gate enforces correctness regardless of which you choose.
- **Custom proximal-operator / soft-threshold Metal kernels**; fuse the
  elementwise tail (soft-threshold + momentum + restart) into one pass.
- **Operation fusion** to cut HBM traffic on the `(n, K)` workload.
- **Reduced precision**: fp16/bf16 inner matmuls with an fp32 KKT check.
- **Quantization (Pilanci-style matrix compression). THIS IS A PRIORITY LEVER
  ON THE HERO SHAPE.** The hero gradient is bandwidth-bound on two matmuls with
  the `(1000, 50000)` matrix `A` (G = AᵀA is 10 GB, does not fit). Quantizing
  `A` to 4/8-bit cuts the bytes moved per iter ~4–8×, the dominant cost.
  Concrete MLX API:
    - `A_q, scales, biases = mx.quantize(A, bits=4, group_size=64)` — quantize
      once in `prepare_problem` (its cost is one-time setup).
    - `y = mx.quantized_matmul(x, A_q, scales, biases, transpose=False, bits=4, group_size=64)`
      for `A @ x`; use `transpose=True` for `Aᵀ @ r`. So the smooth gradient
      `Aᵀ(A y − b)` becomes two `quantized_matmul` calls — no custom kernel
      needed (Pilanci's point: the quantized dot product IS the kernel).
    - Keep the iterate/soft-threshold path in fp32. The harness recomputes the
      trusted KKT in fp64 on your returned `X`, so quantization that's too
      coarse will simply fail the gate — tune `bits`/`group_size` so you still
      reach `kkt < 1e-5`. Mixed schemes (e.g. 8-bit early iters → fp32 polish)
      are fair game.
  The same quantized-dot-product machinery is the bridge to a future second
  specimen (compressed RAG embedding / nearest-neighbor search) — design for it.
- **Path structure**: per-column convergence masking, SAFE/STRONG screening,
  warm-start across decreasing lambdas, adaptive lambda-ordering. These recover
  Adelie's edge while staying inside a batched-matmul form.

## Low-value (don't burn budget)

- Threadgroup-size tweaks that don't change op count.
- Cosmetic refactors / equivalent math rewrites.
- Hand-rolled replacements for `mx.matmul` (already bandwidth-tuned by Apple).

## Durable memory (no context rot)

Every experiment is a node in a durable tree, not a line in a rotting chat:

- Accepted candidates become **checkpoints** (`checkpoints/<id>/`, with source,
  score, trajectory, and a `parent_id`). The loop can resume/branch from any
  checkpoint.
- Each iteration you receive a **curated research state** (current champion, the
  ranked baseline bar, and a deduplicated digest of tried directions) rebuilt
  from the tree — plus the current checkpoint's source. Use it: don't repeat a
  direction the digest shows already failed; branch toward what's unexplored.

## Logging

Every proposal lands in `lineage.jsonl` with its `parent_id` (the experiment
tree). Consult in `research_state.json`: `champion`, `bar_to_beat`,
`tried_directions`. The per-iterate KKT-vs-time trajectory is saved and plotted
(`plots/`) against the baseline panel.

## Autonomy

Once a session begins, run until the budget finishes or the user interrupts.
Crashes → recorded + continue. Duplicates → discarded + continue. Each session
starts from the current best checkpoint and explores from there. After ~50
unproductive proposals on one target, flag stagnation — the seed or this
program.md is likely the constraint.

The bottom line: **specify (problem, hardware); search everything else; win on
time-to-KKT against the classical solvers.**
