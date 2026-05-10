# Lessons

- 2026-05-09: The project goal is not to optimize "a problem type" in isolation.
  The user specifies a problem type and a targeted algorithm, and the harness
  should autoresearch kernels for that specific setting, like kernel
  optimization for any other workload. Cross-problem transfer is only a source
  of priors/candidate edits; it is not the main objective or acceptance target.

- 2026-05-10: Don't drift from "custom kernels for optimization algorithms with
  autoresearch" into "compile CVXPY problems to GPU." Moreau (Boyd's group)
  already owns the latter territory commercially. The autoresearch loop is the
  centerpiece, not a backstage tuner of a library. When a re-pivot starts
  feeling Moreau-shaped, pull back.

- 2026-05-10: An algorithm with too small a per-iter surface (FISTA on LASSO:
  one fused O(n) tail) starves the autoresearch loop. AlgoTune (NeurIPS 2025)
  found LLMs do "surface-level recombination, not novel discovery"; with no
  surface to recombine, you get nothing. The fix is a richer algorithm (PDHG,
  ALM) with multiple coupled updates, step parameters, and operator-splitting
  choices — not a richer search engine.

- 2026-05-10: One-rep Mac timings can be optimistic by 20-30% on small
  evaluators. The recorded FISTA Gram champion was 16.4 ms (1-rep); 5-rep
  median is 21.6 ms. Always confirm with multi-rep timing before treating a
  number as a regression baseline. CV around 5% is realistic; deltas under
  ~5-10% should not promote.

- 2026-05-10: An MLX seed that fans out into 5+ separate ops per iteration is
  launch-overhead bound for problem sizes the autoresearch loop actually
  cares about (n <= 2048). The PDHG-TV-1D seed is 26-38x SLOWER than numpy
  fp64 because of this. Lesson: when writing a seed for the autoresearch
  loop, fuse aggressively from the start unless the search surface
  specifically depends on having pre-fused ops to recombine. For algorithms
  with many small per-iter ops (PDHG, ADMM x/z/u updates), the LLM needs
  ROOM to find fusions — but it also needs a believable starting point, not
  one that's 30x off the floor. The fix is a moderately-fused seed (e.g.
  fuse 2-3 ops manually) so the LLM can search the remaining 2-3 fusions.

- 2026-05-10: OpenAI Responses API (openai>=2.x) uses `text={"format": {...}}`
  for structured-output JSON schema, NOT `response_format={...}` (that's the
  Chat Completions API). Symptom: `Responses.create() got an unexpected
  keyword argument 'response_format'` on every call. Fix:
      response = client.responses.create(
          model=...,
          input=prompt,
          text={"format": {"type": "json_schema", "name": ..., "schema": {...}, "strict": True}},
      )

- 2026-05-10: Recording the per-iter fitness (KKT or gap) for trajectory
  feedback is expensive on MLX — each `float(mx.sum(...))` call forces a
  full GPU sync. With 5000 iters that's 5000 sync points and 26-38x slowdown
  vs not recording. The fix is `convergence_check_every=10`: only check
  convergence (and record the gap/kkt) every Nth iter. Default 10 in the
  sandbox path. For a 5000-iter run that's 500 sync points and ~7x faster.

- 2026-05-10: Eval-contract loophole. If the algorithm driver sets
  `t0 = perf_counter()` AFTER `state = kernel_init(...)`, an LLM proposer
  WILL exploit it: stuff arbitrary per-step work into `init_state`, report
  iters=10 with solve_ms=0.98, beat the speedup gate while doing more total
  compute than the seed. Symptom: "iters" drops dramatically while
  rationales mention moving work into init_state or warm-start prefixes.
  Fix: time `kernel_init` as part of `wall_time_s` (move `t0 =
  perf_counter()` before `kernel_init`). Defense in depth: gate on
  `setup + solve` (=`single_solve_time`) when cost_model="single". Same
  failure mode as Sakana CUDA Engineer benchmark cheating; expect every
  evolutionary code-search loop to need this audit.

- 2026-05-10: Algorithm surface determines autoresearch yield. FISTA-LASSO
  on tall_medium: 0 non-trivial wins after dozens of proposals (one fused
  tail kernel, ~5 knobs, deterministic grid covers the space). PDHG-TV-1D
  on tv1d_medium: 99x speedup over MLX seed in 10 proposals via
  rediscovered time-skewing (256 steps/launch with threadgroup-cached
  halo). Picking the right specimen — one with multiple coupled per-iter
  ops and a small stencil — matters more than the search engine.

- 2026-05-10: MLX `linalg.cholesky` and `linalg.solve_triangular` are
  CPU-only (mlx 2.36+). Symptom: `[linalg::solve_triangular] This op is
  not yet supported on the GPU. Explicitly pass a CPU stream to run it.`
  Fix: `mx.linalg.cholesky(H, stream=mx.cpu)` and
  `mx.linalg.solve_triangular(L, rhs, upper=False, stream=mx.cpu)`. The ALM
  inner solve runs on CPU because of this; an autoresearch lever for the
  ALM specimen is replacing it with a custom Metal trisolve when launch
  overhead doesn't dominate.

- 2026-05-10: For PURE equality-ALM (no z splitting, x-update is exact
  linear solve), the dual residual `||rho A^T (Ax - Ax_prev)||` is not the
  right convergence gate. The x-update enforces stationarity at every iter;
  only feasibility (||Ax - b||) needs to converge. Boyd 2011 §3.3.1 dual
  residual applies to splitting-based ADMM, not pure equality-ALM. Gating
  on primal residual alone gives the right behavior. Document this
  distinction; future ADMM/splitting-based specimens need both gates.

- 2026-05-10: ALM with fixed rho is affine. Given rho fixed, the ALM
  iteration `(x, lam) -> (x_new, lam_new)` is a single linear transition
  `M @ (x, lam) + offset`. The autoresearch loop discovered this in
  generation 1 of the ALM smoke: it precomputes M and offset in
  init_state, then does one matvec per `alm_step` instead of trisolve +
  matvec. Real win, ~5% under the test problem. (Limit: relies on rho
  being constant across iters; adaptive-rho variants invalidate it.)
