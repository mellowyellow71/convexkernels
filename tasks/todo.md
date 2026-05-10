# convexkernels — KKT-gated LLM kernel synthesis for convex optimization

## Context

We are building a **kernel synthesis harness** for convex optimization algorithms on accelerator hardware. The pitch is **CVXGEN-for-GPU-kernels**: input a convex problem type, a targeted algorithm, and a hardware target; output a specialized solver/kernel stack synthesized by an LLM agent against a closed-form fitness signal.

The thing that makes this tractable and novel is **convexity → KKT residual is an oracle-free scalar fitness signal computable per-iterate at near-zero marginal cost**. No ground-truth solver needed. KKT residual collapses correctness verification and convergence detection into one cheap check, which is precisely the bottleneck for LLM-driven code synthesis.

LASSO + FISTA is **specimen #1**. The optimization target is a concrete setting: `(problem type, targeted algorithm, hardware, dtype)`. Subsequent problem types (non-negative LASSO, elastic net, group LASSO, basis pursuit denoising, ridge, isotonic regression) prove users can swap the problem input; ADMM proves users can swap the targeted algorithm. Cross-problem transfer is a bootstrapping prior, not the core objective. Mac/MLX is the first hardware target; NVFP4-on-Blackwell is a future dtype slot in the same architecture.

Advisor pointers:
- adelie (prox-Newton + CD with structured matrix wrappers) — borrow the typed-matrix interface idea
- alpaqa lasso-jax (PANOC + JIT'd gradients) — borrow the problem-object ergonomics
- huggingface/ml-intern (autonomous ML-engineering agent with sandbox + traces) — reference architecture for the proposer/sandbox/lineage layout
- nvfp4 on Blackwell — future dtype, deferred from MVP but designed-for

## Goals (MVP)

1. A correct, KKT-verified LASSO FISTA solver in Python (NumPy + MLX) on Mac.
2. A minimal end-to-end synthesis loop that takes a hand-written seed Metal kernel and produces a measurably faster KKT-gated successor.
3. The full custom synthesis architecture: hierarchical proposer bank, multi-fidelity gating, structured edit grammar with lineage, and optional transfer priors.
4. Apples-to-apples benchmarks vs sklearn `Lasso`, adelie, alpaqa lasso-jax, and CVXPY ground truth on a fixed 4–6 shape problem suite, logged from day one.
5. Demonstrate generality: a second algorithm (ADMM) and a second problem family drop into the same harness while preserving the same slot-level optimization loop.

## Non-goals (deferred)

- Triton / NVFP4 / Blackwell backend — designed-for, not built.
- Metal hardware counters — designed-for via the roofline interface, real counter capture deferred to post-MVP.
- GLMs (logistic, Cox), distributed solvers, web UI.

## Architecture

```
convexkernels/
├── frontend/
│   ├── problem.py           # base Problem with matvec/rmatvec/prox/kkt_residual/lambda_max
│   ├── lasso.py             # specimen #1: Lasso(A, b, lam) ergonomic constructor
│   ├── cvxpy_adapter.py     # (later) consumes cvxpy.Problem, dispatches to specimen if recognized
│   └── (later) nn_lasso.py, elastic_net.py, group_lasso.py
├── algorithms/
│   ├── fista.py             # algorithm template (host-side iter loop, calls into kernel)
│   ├── admm.py              # second algorithm template (added P5)
│   └── kkt.py               # closed-form KKT residual per problem family
├── kernels/
│   ├── numpy_ref.py         # correctness oracle
│   ├── mlx/
│   │   ├── champions/       # current best per (problem, algo, hw, dtype) — synthesized
│   │   ├── seeds/           # hand-written starting points the proposer mutates from
│   │   └── lib.py           # shared MLX utilities (e.g. mx.fast.metal_kernel registration)
│   └── registry.py          # named-variant registry; keyed (problem, algo, hw, dtype)
├── synth/
│   ├── loop.py              # custom synthesis driver
│   ├── proposers/
│   │   ├── algorithm.py     # algo-level mutations (FISTA → FISTA+restart, ADMM → adaptive ρ, …)
│   │   ├── kernel.py        # kernel-level mutations (fusion, op partition)
│   │   └── impl.py          # impl-level mutations (tile, dtype, threadgroup memory, layout)
│   ├── prompts/
│   │   ├── algorithm.md
│   │   ├── kernel.md
│   │   └── impl.md
│   ├── edits.py             # edit grammar: structured mutation objects, not raw replacement
│   ├── lineage.py           # schema (see docs/schema.md), append/load/priors/seed_from_neighbors
│   ├── champion_store.py    # Pareto front, slot-keyed champions, atomic symlink updates
│   ├── fitness.py           # structured fitness vector + gating logic
│   ├── roofline.py          # analytical bytes-moved / ops-performed → utilization signal
│   ├── tiers.py             # 3-tier evaluator (smoke / convergence / full bench)
│   ├── checkpoint.py        # WAL: started.json marker, crash-resume scan
│   └── sandbox.py           # subprocess + timeout + resource.setrlimit memory cap
├── bench/
│   ├── shapes.py            # 4–6 problem-shape suite (small/large × tall/wide × dense/sparse)
│   ├── datasets.py          # synthetic Boyd-style + 1 real (rcv1.binary)
│   ├── baselines.py         # sklearn, adelie, alpaqa, cvxpy adapters
│   └── run.py               # apples-to-apples driver
└── tests/
    ├── test_kkt.py          # KKT residual = 0 at cvxpy solution
    ├── test_fista.py        # FISTA matches cvxpy on small problems
    ├── test_kernels.py      # functional equivalence: assert_equivalent (KKT-gated, drift logged)
    └── test_synth_loop.py   # one full proposer cycle with a stub LLM (deterministic)

# Runtime state (gitignored, repo-root, not in package)
synth_state/                 # lineage.jsonl, edits.json, champions/  (see docs/schema.md)
runs/                        # per-proposal artifacts: source, compile.log, tier{1,2,3}.json, started.json
```

### Interface contracts

- `Problem` exposes `matvec`, `rmatvec`, `prox(v, kappa)`, `kkt_residual(x)`, `lambda_max`. Backend-agnostic (numpy or mlx).
- **Frontend (DSL)** is hybrid: ergonomic per-specimen constructors (`Lasso(A, b, lam)`, `NonnegLasso(A, b, lam)`) are the canonical entry points and the unit the synthesis loop dispatches over. A future `cvxpy_adapter.py` consumes a `cvxpy.Problem`, recognizes known structures, and routes to the matching specimen — *unrecognized structures raise `NotImplementedError`*. We do not re-build a general convex solver; we accept arbitrary problems only when a specialization exists.
- `Algorithm` is a host-side iteration template: takes a `Problem` + a `KernelStep`, runs until KKT < ε or iter cap. Returns `RunResult` (iters, KKT trajectory, x, wall-time).
- `KernelStep` is a named callable: `(state, problem, dtype) -> next_state`. Variants register by name. Synth loop never sees Metal source — sees variant names + structured config.
- `Edit` is a structured mutation: `{type: "tile_change", from: 128, to: 256, target_kernel: "fista_step_v3", rationale: "..."}`. NOT a raw source replacement. Proposer outputs Edits; an applier turns Edits into source. Lineage logs Edits, not source diffs.
- `FitnessVector = (converged: bool, N_ε: int, T_ε: float, KKT_final: float, peak_mem: int, roofline_pct: float)`. Pareto over `(N_ε, T_ε)` gated on `converged ∧ KKT_final < ε`.
- **Test contract**: `assert_equivalent(x_kernel, x_ref, problem, kkt_tol=1e-6, drift_warn=1e-2)` — both must satisfy KKT; iterate drift `‖x_k - x_ref‖∞ / ‖x_ref‖∞` is logged and warned-on but does NOT fail the test. Reasoning in `docs/schema.md` is the deeper "why this and not numerical."
- **Persistence schema** is fully specified in `docs/schema.md`: lineage records, edit priors, champion store, write-ahead log for checkpointing. The synth loop's data model is a load-bearing artifact, not implementation detail.

## Algorithm reference (FISTA — Beck & Teboulle 2009)

LASSO: $\min_x \tfrac12\|Ax-b\|_2^2 + \lambda\|x\|_1$, Lipschitz $L = \|A\|_2^2$, step $t = 1/L$.

$$
\begin{aligned}
g^{k} &= A^\top(Ay^{k} - b) \\
x^{k+1} &= S_{\lambda t}(y^{k} - t\,g^{k}) \\
\theta^{k+1} &= \tfrac{1+\sqrt{1+4(\theta^k)^2}}{2} \\
y^{k+1} &= x^{k+1} + \tfrac{\theta^k - 1}{\theta^{k+1}}(x^{k+1} - x^{k})
\end{aligned}
$$

Hot ops per iter: one $Ay$ (m·n), one $A^\top r$ (m·n), one soft-threshold + axpy combo (n). Pure SIMT, bandwidth-bound — exactly what Metal wants.

### KKT residual (the fitness function)

For $g = A^\top(Ax - b)$:
$$
v_i(x) = \begin{cases} |g_i + \lambda\,\mathrm{sign}(x_i)| & x_i \ne 0 \\ \max(|g_i|-\lambda,\,0) & x_i = 0 \end{cases},\quad
\mathrm{KKT}(x) = \frac{\|v(x)\|_\infty}{\lambda + \|A^\top b\|_\infty}.
$$
Scale-free, oracle-free, $\mathrm{KKT}(x) = 0 \iff x = x^\star$. The numerator is free if KKT is checked the iter we already compute $g$.

## Custom synthesis loop (the research artifact)

This is the part we are NOT delegating to evo. Specifics that justify the build:

1. **Per-iter KKT trajectory feedback.** Proposer sees the KKT sequence + fp32 baseline trajectory, not just final score. "Diverges at iter 41" → targeted edit.
2. **Three-tier evaluation.**
   - Tier-1: tiny instance (n=100, m=200), 200 iters, smoke test KKT decreases. <1s. Kills ~95% of bad proposals.
   - Tier-2: mid instance (n=2000, m=5000), full convergence to ε=1e-6.
   - Tier-3: full bench suite (4–6 shapes × 3 reps for variance).
3. **Roofline-from-source.** Each kernel's bytes-moved and ops-performed are computed analytically from the source (we wrote it). Wall-time + roofline → "X% of peak bandwidth." Replaces Metal counters in MVP; deeper than counters because it forces an explicit ideal.
4. **Structured edit grammar.** Proposer emits `Edit` objects (`tile_change`, `fuse_op`, `dtype_swap`, `hoist_to_threadgroup`, `vectorize`, `swap_layout`, `algo_variant`, …). Applier turns them into source. Lineage tracks Edit → outcome → prior. When an edit wins, it becomes a seed candidate for sibling problems.
5. **Hierarchical proposer bank.** Three LLM proposer roles with separate prompts and mutation rates: algorithm-level (FISTA → FISTA-with-restart), kernel-level (fusion topology), impl-level (tile, dtype, threadgroup memory). Different success criteria per level.
6. **Convex invariants as kill switches.** Primal-objective non-decrease beyond warmup, KKT residual blow-up, NaN. Mid-run termination saves compute.
7. **Champion store with cross-problem transfer.** Pareto front keyed by `(problem_family, algorithm, hw, dtype)`. New problem → seed proposals from edit-success priors of nearest-neighbor problems.

State on disk:
- `synth/champions/<problem>/<algo>/<hw>/<dtype>/champion.{py,msl}` — current best
- `synth/lineage.jsonl` — every proposal: parent, edit, profile-before, scores, accepted?, reason
- `synth/edits.json` — per-edit-type success stats (proposer ranking prior)
- `synth/runs/<id>/` — full trace per attempt (KKT trajectory, roofline, source, diff)

## Phases

Each phase ends with: code green, tests green, numbers logged in `tasks/results.md`. **Never mark complete without proving it works** (CLAUDE.md).

### P0 — Scaffold + capability probes
- [x] Create `pyproject.toml` (uv/hatch), `.gitignore`, package layout per architecture above.
- [x] Pin dependencies: `numpy`, `scipy`, `cvxpy<1.6`, `scikit-learn`, `polars`, `adelie>=1.1`, `alpaqa==1.1.0a1`, `pytest`. `mlx` is the `[mac]` optional extra (Linux dev box can't install it). Conflict between cvxpy ≥1.6 (numpy ≥2) and adelie (numpy <2) resolved in `tasks/results.md`.
- [x] **Probe MLX linalg surface** — done on M3 Pro. `cholesky`, `solve_triangular`, `solve`, `lu`, etc. all present. ADMM path derisked; no custom trisolve kernel needed.
- [x] **Probe arithmetic intensity** of FISTA hot loop on each bench shape — analytical roofline in `docs/roofline.md`, calibrated against M3 Pro (150 GB/s peak) with measured utilization.
- [x] Sandbox spike: `mx.fast.metal_kernel` compiles and runs correctly on M3 Pro.
- **Acceptance**: `pytest` collects ✓, `python -c "import convexkernels"` works ✓, MLX probe runbook committed ✓, MLX probe executed ✓. **P0 complete.**

### P1 — NumPy reference FISTA + KKT verifier
- [x] `algorithms/fista.py`: clean NumPy FISTA with `basic` and `restart` (O'Donoghue–Candès 2012 gradient-restart) variants. `monotone` (MFISTA) intentionally dropped — restart captures the practical value of monotonization, MFISTA needs an explicit objective evaluator we don't otherwise need.
- [x] `algorithms/kkt.py`: KKT residual via the **prox-residual reformulation** (mathematically equivalent to the case-split formula at non-degenerate points, robust to floating-point near-zeros). Unit-tested against CVXPY-solved $x^\star$ — KKT < 1e-9 in tests.
- [x] `frontend/lasso.py`: `Lasso(A, b, lam)` with `matvec`, `rmatvec`, `grad_smooth`, `prox`, `kkt_residual`, `lambda_max`, `L`. Cached `Atb`, `lambda_max`, `L`.
- [x] `kernels/numpy_ref.py`: `FistaState` + `fista_step(state, problem, t)`. The correctness oracle for P3+.
- [x] `tests/test_kkt.py` (10 tests), `tests/test_fista.py` (8 tests): green. Plus 2 smoke tests = 20/20.
- **Acceptance**: FISTA matches CVXPY at $\|x_\text{fista} - x_\text{cvxpy}\|_\infty / \|x_\text{cvxpy}\|_\infty < 10^{-4}$ on N=200, p=500 ✓. KKT residual at CVXPY solution < 1e-8 (CLARABEL with `tol_gap_abs=1e-12`) ✓. **P1 complete.**

### P2 — Baseline shootout (apples-to-apples from day one)
- [x] `bench/shapes.py`: 4 dense shapes (`tall_small`, `tall_medium`, `wide_small`, `wide_large`). `rcv1.binary` deferred — 4 dense shapes cover the regime spread for MVP. Add as P2.5 if synth-loop training requires more diversity.
- [x] `bench/baselines.py`: adapters for `numpy_fista_restart`, `sklearn`, `adelie`, `cvxpy`. **alpaqa deferred** — its Python API requires CasADi/JAX bindings to construct a Problem; significant detour for one baseline. Documented in `tasks/results.md`.
- [x] `bench/run.py`: runs all baselines on the suite, logs `(iters, T_ε, KKT_final, primal_obj)` to `tasks/results.md`.
- **Acceptance**: All baselines run on all shapes ✓. Numbers logged ✓. NumPy FISTA's KKT residual ≤ 1e-6 across the suite (max observed: 9.2e-8) ✓. **All four solvers reach identical primal_obj to 4 decimal places on every shape — strong cross-validation. P2 complete.**

#### Follow-ups (not blocking)
- Cache CVXPY ground-truth results to disk; current bench rerun takes ~10 min dominated by `wide_large` CVXPY (448 s).
- Replace `Lasso.L` SVD with power iteration; currently dominates FISTA wall time on Linux x86 (Mac/Accelerate is faster).

### P3 — Minimal end-to-end synthesis cycle

#### P3.1 — Seed MLX kernel + MLX-path FISTA + functional equivalence ✓
- [x] `kernels/mlx/seeds/fista_step_v0.py`: hand-written `mx.fast.metal_kernel` fusing `z=y-tg`, `soft(z, t*lam)`, and momentum axpy.
- [x] `kernels/mlx/lib.py`: `LassoMLX` (MLX-backed problem, duck-typed to `Problem`).
- [x] `algorithms/fista.py`: refactored to take optional `kernel_init`, made restart-indicator backend-agnostic. MLX path verified.
- [x] `tests/test_kernels.py`: functional equivalence at fp32 (drift 1e-7, KKT 6e-8) and fp16 (drift 2e-3, KKT 2e-3 — precision floor as expected). MLX-only; skipped on Linux.
- [x] `algorithms/kkt.py::assert_equivalent`: implemented per the test contract in todo.md (KKT-gated, drift logged not gated).
- [x] Bench: MLX seed kernel is **2–27× faster than numpy_ref per-iter on M3 Pro** with iter counts matching within 6%. See `tasks/results.md → P3.1`.

#### P3.2 — Sandbox + checkpoint infra ✓
- [x] `synth/sandbox.py`: subprocess + timeout + `RLIMIT_AS` (Linux only — Mac documented as timeout-only). Eval driver `synth/_eval_kernel.py` loads a kernel module by name, runs FISTA, writes JSON result + `x.npy`.
- [x] `synth/checkpoint.py`: WAL via `mark_started()` (writes `runs/<id>/started.json`) + `find_orphans()` (scans `runs/` for started-without-lineage entries).
- [x] `synth/lineage.py`: schema implementation per `docs/schema.md`. `LineageRecord`, `Slot`, `Edit`, `Tier{1,2,3}Result`, `Decision` dataclasses. `append_record(record, jsonl_path)` + `load_records(jsonl_path)`.
- [x] `tests/test_sandbox.py` (4 tests): end-to-end subprocess eval, runtime-error reporting, orphan detection, lineage round-trip.

#### P3.3 — Minimal synth loop with stub proposer ✓
- [x] `synth/proposers/stub.py`: `DeterministicStubProposer` that cycles edit types deterministically.
- [x] `synth/loop.py`: full driver — orphan check on startup, per-proposal WAL → eval → tier-1 gate → lineage append. Hardware auto-detected (`apple_silicon` / `linux_x86_64`).
- [x] `tests/test_synth_loop.py` (6 tests): KKT-only compatibility path, tier-1 failure path, orphan-after-crash detection, speed-ratchet keep/discard behavior, proposer-error resilience, tier2 promotion.

#### P3.4 — Wire in OpenAI proposer + ratchet ✓
- [x] `synth/applier.py`: take parent_source + Edit → child source (full_source mode for MVP).
- [x] `synth/proposers/openai.py`: OpenAI Responses API proposer. Reads champion source + recent lineage history. Emits an `Edit` with `payload.full_source` (the new kernel source).
- [x] `synth/prompts/impl_openai.md`: JSON-schema synthesis prompt template.
- [x] Update `synth/loop.py` and `synth/_eval_kernel.py` to handle file-path kernel modules (so LLM-emitted source can be evaluated).
- [x] Convert canonical NumPy `Lasso` to `LassoMLX` inside the sandbox evaluator for MLX kernel runs.
- [x] `pyproject.toml`: `[ai] = ["openai>=1.109"]` extra.
- [x] Add untimed warmup evaluation before Tier-1 timing so first Metal compile overhead does not dominate proposal scores.
- [x] Add Karpathy-style speed ratchet: measure warmed seed/champion baseline, keep only KKT-valid proposals faster than the target margin, discard slower valid proposals, and promote kept full-source proposals to `synth_state/champions/.../champion.py`.
- [x] Run on M3 Pro for ~30 proposals; produce ≥1 accepted variant.

- **Acceptance** (P3 overall): A synthesized variant is faster than `fista_step_v0` on Tier-1 with `KKT_final < 1e-6` on the tiny instance, AND passes functional equivalence (`assert_equivalent`) vs `numpy_ref`. Iterate drift logged for diagnostics, not gated.

### P4 — Full synthesis architecture
- [x] `synth/tiers.py`: 3-tier evaluator scaffold (Tier-1 ratchet, Tier-2 convergence, Tier-3 shape-suite medians).
- [x] Add repeated Tier-1/Tier-2 timing medians and a Tier-2 speed gate so noisy Tier-1 wins do not promote unless they also improve full-convergence wall time.
- [x] Expose FISTA `restart` in the synth evaluator/CLI and default CLI synth runs to restart.
- [x] Split Tier-1 escalation from Tier-2 promotion, and confirm Tier-2 speed wins with a paired champion remeasurement before promotion.
- [x] Regenerate `synth_state/edits.json` edit-outcome priors from lineage and feed slot-specific prior summaries into the proposer prompt.
- [x] Skip exact duplicate full-source proposals before sandbox evaluation.
- [x] Add payload-level structured priors (`edits.json` v2), exact-payload avoid lists, and Tier-2 near-miss summaries for proposer context.
- [x] Add a deterministic `StructuredGridProposer` to sweep the supported structured payload space without spending API calls.
- [x] `synth/roofline.py`: analytical bytes/ops, utilization %, integrated into Tier-3 per-shape records.
- [ ] `synth/edits.py`: full edit grammar (`tile_change`, `fuse_op`, `dtype_swap`, `hoist_to_threadgroup`, `vectorize`, `swap_layout`, `algo_variant`).
  - [x] First structured applier slice: deterministic `threadgroup_size`, `remove_bounds_check`, `branchless_soft_threshold`, and `kernel_name_suffix` transforms.
  - [x] Structured vectorization slice: deterministic `items_per_thread` transform for 2/4 contiguous coefficients per Metal thread.
  - [x] Structured edit composition hardening: vectorization + branchless soft-threshold and applier-error lineage records.
  - [x] Problem/prox-aware structured applier: vectorization preserves LASSO vs NN-LASSO prox semantics, and unsafe no-bounds/vectorized compositions are rejected before evaluation.
- [ ] `synth/proposers/{algorithm,kernel,impl}.py` + corresponding prompts. **`impl.py` proposer uses size-aware mutation distributions** (per the M3 Pro two-regime finding in `tasks/results.md`): small/launch-bound shapes weight fusion edits, large/BW-bound shapes weight dtype edits.
- [x] `synth/champion_store.py`: slot-keyed champion store with atomic symlink promotion, `index.json`, metadata, and `pareto.jsonl`.
- [x] `synth/lineage.py::seed_from_neighbors(records, slot, k=3)`: implements cross-problem transfer per `docs/schema.md`.
- [x] `synth/fitness.py`: first structured vector/report layer for correctness, speed ratios, timing noise, roofline hints, bottleneck hints, and proposer recommendations.
- [x] FISTA gradient-strategy search slice: prepared-problem hook, setup/solve timing split, Gram-precompute MLX seed, and dtype-strategy plumbing.
- [x] Extend structured/OpenAI proposer payloads to actively propose `gradient_strategy` and `dtype_strategy` edits, not just run them through CLI-selected seeds.
  - [x] Materialize gradient/dtype strategy edits as replayable `source.py` candidates with a generated `prepare_problem()` hook, so accepted strategy edits can be promoted and resumed from champion state.
  - [x] Seed proposer history from persisted same-slot lineage so restarted structured sweeps continue past already-tried payloads instead of spending a round on duplicate-source rejection.
- [x] Make champion selection workload-aware by cost model, so setup-inclusive `single` and solve-amortized champions do not overwrite each other; select the fastest recorded matching champion when legacy and workload-specific records coexist.
- [x] Feed current kernel strategy and Gram-specific search focus into OpenAI proposer context.
- [x] Add large fresh-agent context docs for the Karpathy-style loop transition:
  - `tasks/autoresearch_context.md`
  - `tasks/program_kernel.md`
- [ ] Promote `synth/fitness.py` from proposer diagnostics into full Pareto/champion gating.
- [ ] Convex-invariant kill switches: primal monotone (post-warmup), KKT non-explosion, NaN.
- [ ] Run loop overnight with budget; populate champions for all bench shapes.
- **Acceptance**: (a) For each bench shape, a synthesized champion beats `mx.matmul`-only baseline on $T_\epsilon$ by ≥1.3× while preserving KKT < 1e-6 and $N_\epsilon \le 1.2 \cdot N_\epsilon^{fp32}$. (b) `seed_from_neighbors` returns sensible top-k for any populated slot (manual review). (c) Crash-resume works: kill the loop mid-run, restart, verify orphaned `started.json` records are detected and logged.

### P5 — ADMM as second algorithm template
- [ ] `algorithms/admm.py`: cached-Cholesky ADMM (Boyd §6.4), tall + wide branches, adaptive $\rho$.
- [ ] Resolve MLX trisolve question from P0 probe (custom kernel if needed, or fall back to NumPy for the factor with MLX for everything else).
- [ ] Wire into synth loop as a second algorithm-bank entry.
- [ ] Run synth loop for ADMM champions on the same bench suite.
- **Acceptance**: Synth loop produces an ADMM champion for at least 3 shapes; harness slot architecture unchanged. Cross-problem transfer (FISTA edit-priors → ADMM seed proposals) demonstrated.

### P6 — Second specimen (proves problem-frontend slot)
- [x] Pick one: non-negative LASSO (simplest — same KKT with extra constraint), elastic net (KKT with two penalties), or basis pursuit denoising. **Selected non-negative LASSO** (smallest delta from LASSO; isolates the slot test).
- [x] `frontend/nonnegative_lasso.py` + KKT for it.
- [x] MLX backend adapter + NN-LASSO FISTA seed kernel.
- [x] Synth CLI/sandbox problem-family switch and NN-LASSO smoke run.
- [x] Problem/prox-aware transfer seeds from LASSO structured edits into NN-LASSO.
- [ ] Produce a strict speed-gated NN-LASSO champion.
- [ ] Run transfer-seeded and no-transfer counterfactuals as a diagnostic, not as the core success criterion.
- **Acceptance**:
  - (a) Synth loop produces a champion for the new specimen using only the harness's existing infra (no specimen-specific synth code).
  - (b) The champion is for the specified setting `(nonnegative_lasso, fista, apple_silicon, fp32)` and improves KKT-gated full-convergence time under the strict repeated Tier-2 gate.
  - (c) Transfer effectiveness is logged separately as a diagnostic table; it can inform proposer priors but does not define whether the harness succeeded for the target setting.

### P7 — Writeup + future-work hooks
- [ ] Tech note: "KKT-gated LLM kernel synthesis for convex optimization." Include perf table, edit-success stats, lineage stories.
- [ ] Document the NVFP4/Blackwell extension path (new dtype + new hw target = same architecture).
- [ ] Document Metal-counter extension path (new signal in `synth/profile.py`).

## Critical files

| Path | Why critical |
|---|---|
| `algorithms/kkt.py` | The fitness function. Bug here corrupts every promotion decision. |
| `docs/schema.md` | The persistent data model. Wrong schema = the proposer can't learn from history. |
| `synth/loop.py` | The thing we are arguing for. If this is messy, the research artifact is messy. |
| `synth/edits.py` | Structured mutations are what differentiates this from a random walk. |
| `synth/lineage.py` | Implements the schema, including `seed_from_neighbors` (cross-problem transfer). |
| `synth/champion_store.py` | Slot-keyed champions, atomic symlink updates, Pareto front maintenance. |
| `synth/fitness.py` | Gating logic: converged ∧ KKT < ε ∧ iter-floor. Wrong gates = bad data. |
| `kernels/mlx/seeds/fista_step_v0.py` | The starting point. Quality of seed sets the synthesis ceiling. |
| `bench/baselines.py` | Adapter consistency determines whether headline numbers mean anything. |

## Reuse from references

- **adelie**: typed-matrix-wrapper pattern (`ad.matrix.dense`, `ad.matrix.snp_unphased`) → mirror for `Problem.matvec/rmatvec` so structure-aware kernels can be added without solver changes.
- **alpaqa**: `Problem` object exposes `eval_objective`, `eval_objective_gradient`, `prox`; solver consumes backend-agnostically. Mirror the ergonomics.
- **ml-intern**: agent + sandbox + benchmark + trace-persistence layout for `synth/sandbox.py` + `synth_state/lineage.jsonl`.
- **Boyd 2011 §6.4 + §3.3.1 + §3.4.1**: ADMM math + stop-criterion + adaptive-$\rho$ verbatim into `algorithms/admm.py` docstring (P5).
- **Beck & Teboulle 2009**: FISTA + restart variant verbatim into `algorithms/fista.py` (P1).

## Verification plan

- **Correctness invariants** (every phase): `tests/test_kkt.py` and `tests/test_fista.py` green. Every kernel variant passes **functional equivalence** vs `numpy_ref` (KKT-gated; iterate drift logged but non-blocking).
- **Performance invariants** (P2+): `bench/run.py` regenerates `tasks/results.md`. Each new champion strictly Pareto-dominates the previous on the bench suite.
- **Synth-loop sanity** (P3+): every promoted variant must (a) pass functional equivalence (`assert_equivalent`, KKT-gated), (b) KKT-converge on all bench shapes within iter-floor, (c) Pareto-dominate previous champion.
- **Harness generality** (P5–P6): adding ADMM and adding a second specimen requires *no* change to `synth/loop.py`. If it does, the slot abstraction is wrong.

## Open questions / things to flag

- **MLX trisolve availability** — RESOLVED in P0: `mx.linalg.cholesky`, `solve_triangular`, `solve`, `lu`, `lu_factor` all present. ADMM (P5) does not need a custom trisolve kernel. (See `tasks/results.md`.)
- **Real dataset choice** — defaulting to `rcv1.binary` (mid-wide, sparse) for P2. If you want GWAS-shape (n=500K) added, that's another bench shape, doable.
- **DSL design** — RESOLVED: hybrid (ergonomic per-specimen constructors as canonical, optional `cvxpy_adapter.py` that recognizes known structures and dispatches; unrecognized structures raise `NotImplementedError`). See "Interface contracts" above.
- **Sandbox isolation** — RESOLVED: `synth/sandbox.py` uses subprocess + timeout + `resource.setrlimit` memory cap (4 GB). Full container isolation deferred to post-MVP if needed.
- **Test contract (functional vs numerical equivalence)** — RESOLVED: functional (KKT-gated) primary, drift logged as diagnostic. See "Interface contracts."
- **Persistent schema** — RESOLVED: see `docs/schema.md`.
- **Mac availability for long synth runs** — single Mac dependency for P3+. Crash-resume via `synth/checkpoint.py` (WAL `started.json` markers) handles restart, but does not handle a Mac going offline mid-run. Acceptable risk for MVP; document in P7.

## Review (filled in after each phase)

(empty until P0 complete)
