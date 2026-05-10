# convexkernels — agent handoff

This is a complete project handoff for a fresh agent. Read this end-to-end before doing anything else. By the time you finish reading you should know:

1. What this project is
2. What's been built so far
3. What every locked design decision is and *why*
4. Exactly where we paused
5. Two concrete paths forward

The original conversation went through ~50+ user/assistant turns of design dialogue. The substantive prompts and the assistant's substantive responses are included verbatim in §13 ("conversation log"). When making any design call, **prefer what's recorded here** over your priors — many decisions were made for non-obvious reasons that are documented.

## 0. Fresh-Agent Reading Order

The project has become complex enough that a new agent should not rely on this
file alone. Read in this order:

1. `tasks/handoff.md` — this file; global project history and locked decisions.
2. `tasks/autoresearch_context.md` — long-form context for why this is a
   Karpathy-style autoresearch project, what is wrong with the current loop,
   and what the current Gram champion means.
3. `tasks/program_kernel.md` — draft instruction layer for the next
   Karpathy-style kernel-autoresearch run.
4. `tasks/results.md` — measured results and command logs.
5. `tasks/todo.md` — phase checklist and remaining work.
6. `docs/schema.md` — lineage/champion-store persistence model.

Do not start by running more OpenAI proposals. The immediate next work is to
make the current target a sharper autoresearch game and add Gram-specific
benchmark/edit support.

---

## 1. TL;DR

**Project**: `convexkernels` — a kernel synthesis harness ("CVXGEN-for-GPU-kernels"). User inputs a convex optimization problem type, a targeted algorithm, and a hardware target; an LLM-driven synthesis loop produces a specialized solver/kernel stack for that concrete setting using a closed-form KKT-residual fitness signal.

**Vision** (the user's words, paraphrased after several iterations):

> "I think my end goal is some sort of harness where we just input a problem, and our subagent synthesizes super fast kernel code for it using iterative autoresearch. think like how cvxgen software takes a problem and writes super fast C solver code for QPs and LPs"

LASSO + FISTA is **specimen #1**. The deliverable is the *harness*, not the LASSO solver. The optimization target is a concrete slot: `(problem type, targeted algorithm, hardware, dtype)`. Subsequent problem types (non-negative LASSO, elastic net, group LASSO, basis pursuit denoising) prove users can swap the problem input. ADMM proves users can swap the targeted algorithm. Cross-problem transfer is only a source of priors/candidate edits, not the main objective. NVFP4-on-Blackwell is a future hardware/dtype slot.

**Why this is novel**: LASSO is convex, so KKT stationarity is closed-form, oracle-free, computable per-iterate at near-zero cost. This collapses the LLM kernel-synthesis loop's evaluation bottleneck — propose → run → KKT residual gives a tight feedback signal that *doesn't need a ground-truth solver*.

**Status**: P0–P3.4 ✅ complete and tested. P4 is well underway: tiered evaluation, slot-keyed champion store, repeated timing, Tier-2 speed gates, structured edit appliers, deterministic structured sweeps, payload-level priors, analytical roofline reporting, structured fitness diagnostics, `seed_from_neighbors` transfer seeds, the first larger FISTA gradient-strategy slice, and proposer-native gradient/dtype strategy edits are implemented. The first P6 second-specimen slice is also implemented: nonnegative LASSO frontend/KKT, synthetic generator, MLX backend, seed kernel, synth CLI switch, prox-aware structured edits, and transfer-smoke runs. Pareto/champion fitness gating, a strict NN-LASSO+FISTA champion, and overnight multi-shape population remain.

**Test count**: latest Linux suite `92 passed, 4 skipped`. Latest Mac targeted suite `33 passed`.

**Current exact pause point**:

- We have a current **amortized** champion for
  `lasso + fista + apple_silicon/mlx + fp32 + tall_medium`.
- Champion id:
  `450c6b8b-49e7-4b57-9a2e-900b7708fbba`.
- Strategy:
  `gradient_strategy=gram`, `dtype_strategy=fp32`.
- Stored Tier-2 solve time:
  `16.396416118368506` ms.
- KKT:
  about `4.29e-07`.
- Paired direct reference in the promotion run:
  `48.46549988724291` ms.
- This is not globally optimal. It is the best recorded candidate for this
  exact workload. The next work is to confirm it with stronger repeated timing,
  then add Gram-specific search dimensions.

**Post-handoff update (2026-05-08)**: User switched provider preference from
Anthropic to OpenAI. `OpenAIProposer` is implemented in
`convexkernels/synth/proposers/openai.py`, the CLI defaults to
`--proposer openai --model gpt-5.5`, and `[ai]` now depends on `openai`.
First warmed OpenAI smoke: 3/3 proposals compiled and passed Tier-1 KKT
(`8.35e-4`), but all were slower than the warmed seed baseline (7.7-7.9 ms
vs 5.5 ms).
The follow-up ratchet patch now records slower KKT-valid proposals as
`discard:not_faster_than_baseline` and promotes only faster full-source
proposals to `synth_state/champions/<slot>/champion.py`.
The 30-proposal ratchet run found champion
`6eea3c75-e80e-43aa-8bbd-78c74c01ef7e` (`fuse_op`), 1.52x faster on the
Tier-1 gate and 1.53x faster at the tighter `tol=1e-6` acceptance check.
P4.1 added `synth/tiers.py`, `synth/champion_store.py`, and tier-aware
promotion via `--promotion-tier tier1|tier2|tier3`.
P4.3 hardened the ratchet: repeated Tier-1/Tier-2 medians, Tier-2 speed gates,
FISTA `restart` exposure in the synth CLI, Tier-1-as-escalation for Tier-2/3
runs, and paired Tier-2 champion remeasurement before promotion. A restart
OpenAI run produced one apparent `wide_small` champion, but a 5-rep paired
check showed it was slower than the seed; paired confirmation was added to
prevent that false positive.
P4.4 added `convexkernels/synth/edits.py`, regenerates
`synth_state/edits.json` edit-outcome priors from lineage, feeds slot-specific
accepted/avoid edit summaries into the OpenAI prompt, and skips exact duplicate
full-source proposals as `discard:duplicate_source`. Linux tests are green
(`46 passed, 2 skipped`); Mac targeted tests are green (`23 passed`).
P4.5 added the first structured edit applier slice. OpenAI can now return
`structured_edit` with `threadgroup_size`, `remove_bounds_check`,
`branchless_soft_threshold`, and `kernel_name_suffix`; `source` can be empty
for those mutations. The loop applies the structured transform to the current
source and evaluates the generated file. Tests are green on Linux
(`52 passed, 2 skipped`) and Mac targeted (`29 passed`). A 5-proposal
structured probe on `wide_small` produced 4/5 structured proposals; none were
accepted, but lineage now contains machine-readable mutation fields.
P4.6 added structured vectorization via `items_per_thread` (2 or 4 contiguous
coefficients per Metal thread), including Metal body rewrite and launch-grid
rewrite. Tests are green on Linux (`54 passed, 2 skipped`) and Mac targeted
(`31 passed`). A structured OpenAI probe emitted `items_per_thread=2` and was
evaluated correctly; it lost on performance, but vectorization no longer
requires a full-source blob.
P4.7 added a deterministic `StructuredGridProposer`, `edits.json` v2
payload-level priors keyed by behavior (ignoring `kernel_name_suffix`), exact
structured-payload avoid summaries, and Tier-2 near-miss summaries. The applier
now composes vectorization with branchless soft-thresholding, and applier
failures become lineage rows instead of aborting the loop. Linux tests are
green (`60 passed, 2 skipped`) and Mac targeted tests are green (`37 passed`).
Structured-grid + OpenAI continuations on `wide_small` found no accepted
champion, but produced useful near misses: e.g.
`remove_bounds_check=true, branchless_soft_threshold=true` reached a Tier-2
speed ratio of `0.9766`, and `threadgroup_size=512` plus no-bounds/branchless
reached `0.9788`; both failed the 3% promotion margin.
P4.8 added `convexkernels/synth/roofline.py` and integrated analytical
bytes/ops/utilization into Tier-3 per-shape records. A forced `tall_small`
Tier-3 smoke on Mac wrote `bytes_per_iter=8080000`, `roofline_floor_ms_per_iter
=0.0539`, `measured_ms_per_iter=1.173`, `achieved_bandwidth_gb_s=6.89`, and
`roofline_pct_med=4.59`, confirming that the small full-convergence smoke is
overhead/setup dominated rather than bandwidth-bound.
P4.9 added `convexkernels/synth/fitness.py`, regenerates
`synth_state/fitness.json` alongside `edits.json`, and feeds structured
fitness diagnostics into OpenAI runtime context. The report classifies records
by `tier1_speed_loss`, `tier2_speed_near_miss`, `duplicate`, etc., adds timing
CV, Tier-2 speed ratios, bottleneck hints, and recommendations. Latest tests:
Linux `67 passed, 2 skipped`; Mac targeted `44 passed`. A 2-proposal OpenAI
continuation on `wide_small` produced 0 accepts and updated fitness for 23
records: 14 `tier1_speed_loss`, 8 `tier2_speed_near_miss`, 1 `duplicate`.
P6.1 added the first second-specimen slot: `NonnegativeLasso`,
`nonnegative_lasso_kkt_residual`, synthetic shape generation,
`NonnegativeLassoMLX`, `nonnegative_fista_step_v0`, and
`--problem-family nonnegative_lasso` in the synth CLI/sandbox path. Tests are
green on Linux (`75 passed, 3 skipped`) and Mac targeted (`32 passed`). A
structured NN-LASSO smoke with the Tier-2 speed gate disabled verified the
slot end-to-end and kept a KKT-converged remove-bounds candidate. A strict
Tier-2 run produced 0 accepted champions; the best tile-change candidates were
near misses (`90.8` and `91.5` ms vs a `91.4` ms target). Key lesson:
problem-agnostic edit transfer is too crude because LASSO branchless
soft-threshold rewrites are invalid for NN-LASSO's nonnegative prox. Next step:
make structured appliers problem/prox-aware before the formal P6
`seed_from_neighbors` transfer test.
P6.2 made structured edits and transfer problem/prox-aware. The applier now
detects LASSO vs NN-LASSO prox semantics before rewriting Metal source, so
`items_per_thread` vectorization preserves `max(zi - thresh, 0)` for NN-LASSO
instead of inserting LASSO soft-thresholding. `StructuredGridProposer` skips
branchless-only LASSO edits for NN-LASSO, and `seed_from_neighbors(records,
slot, k=3)` plus CLI `--transfer-seed-k` now queue accepted edits from
neighbor slots before the normal proposer. Tests are green on Linux
(`84 passed, 3 skipped`) and Mac targeted (`50 passed`). Debug transfer from
LASSO to NN-LASSO queued `tg128`, `tg512`, and `remove_bounds_check`, all
KKT-valid and accepted with the Tier-2 speed gate disabled (`90.0`, `88.0`,
`85.8` ms). Strict repeated Tier-2 transfer still produced 0 champions
(`99.3`, `99.4`, `101.5` ms vs a `96.5` ms baseline and `<93.6` ms target).
Conclusion: transfer plumbing is real, but the current tail-edit family is not
the lever for strict NN-LASSO+FISTA promotion. Do not treat transfer itself as
the goal; the goal is the best kernel for the user-specified
problem+algorithm+hardware setting.
P4.10 added the first larger FISTA search dimension for `lasso + fista +
apple_silicon/mlx`: kernel modules can define `prepare_problem(problem[,
config])`, sandbox/tier records now split setup/solve/single/amortized timing,
`LassoGramMLX` precomputes `G=A.T@A` and `c=A.T@b`, and
`gram_fista_step_v0` reuses the existing fused tail kernel with a Gram-prepared
problem. CLI now exposes `--gradient-strategy direct|gram`,
`--cost-model single|amortized|both`, and
`--dtype-strategy fp32|fp16_storage|mixed_gram`. Tests are green on Linux
(`86 passed, 4 skipped`) and Mac targeted (`30 passed`). Mac probe: on
`tall_medium`, direct fp32 was `5381.4` ms single / `53.6` ms solve; Gram fp32
was `3434.6` ms single / `20.6` ms solve with the same `KKT=4.29e-07` and
22 iterations. `mixed_gram` failed the strict KKT contract (`4.13e-04`), so
dtype search is correctly gated.
P4.11 made gradient/dtype strategy search proposer-native. `StructuredGridProposer`
now starts LASSO sweeps with `gram/fp32`, `gram/mixed_gram`, and
`fp16_storage`; `OpenAIProposer` JSON schema and prompt expose
`gradient_strategy=direct|gram` and
`dtype_strategy=fp32|fp16_storage|mixed_gram`. The applier materializes these
strategy edits as replayable `source.py` candidates with a generated
`prepare_problem()` hook, and the loop promotes config-backed sources while
recording/restoring candidate dtype metadata. The loop also seeds proposer
history from persisted same-slot lineage, so restarted structured sweeps
continue past already-tried payloads. Tests are green on Linux
(`90 passed, 4 skipped`) and Mac targeted (`45 passed`). Mac smoke:
`gram/fp32` passed/promoted on `tall_small` with `KKT=7.18e-07`,
`single=214.6` ms, `solve=16.6` ms; the next persisted-history run skipped
that payload and evaluated `gram/mixed_gram`, which correctly failed the strict
KKT bar (`1.52e-04`).
P4.12 ran the first strict Mac tall-medium strategy search. Under the
setup-inclusive `single` cost model, `gram/fp32` was KKT-correct and cut solve
time (`41.6` ms direct to `18.7` ms Gram) but lost because setup rose enough
to make single-solve time slower (`2149.0` ms direct vs `2655.7` ms Gram).
Under `--cost-model amortized`, the same structured search promoted
`gram/fp32` as champion `450c6b8b-49e7-4b57-9a2e-900b7708fbba`: Tier-2 solve
time `16.4` ms vs paired direct reference `48.5` ms, with strict
`KKT=4.29e-07`. `gram/mixed_gram` and fp16 storage were rejected by KKT.
The Mac non-interactive SSH API-key issue was fixed by copying the existing
`OPENAI_API_KEY` export from `~/.zshrc` into `~/.zshenv` without printing the
key. A 3-proposal OpenAI continuation from the Gram champion then ran on the
Mac. It produced one full-source `fuse_op`, one structured `items_per_thread=4`
branchless/vectorized edit, and one structured Gram/no-bounds edit. All were
KKT-valid but slower than the current Gram champion (`17.6`-`19.1` ms vs a
`14.9` ms Tier-1 solve baseline), so no new champion was promoted.
P4.14 made champion selection workload-aware by cost model. New promotions are
stored under `.../<dtype>/workloads/<cost_model>/champion.py`; legacy champions
are reused only if their metadata `summary.cost_model` matches the requested
workload. If legacy and workload-specific champions both match, selection uses
the fastest recorded metric (`tier2_wall_time_ms` when available, otherwise
`tier1_wall_time_ms`) so a noisy one-rep remeasurement cannot silently replace
a better historical champion. This prevents an amortized Gram champion from
becoming the baseline for a setup-inclusive `single` run. OpenAI runtime
context now reports the current champion strategy and, for Gram champions,
explicitly steers proposals toward `LassoGramMLX.grad_smooth`, `G @ y - c`,
Gram dtype/layout, setup reduction under `single`, and KKT-safe precision
under `amortized`. Tests are green on Linux (`92 passed, 4 skipped`) and Mac
targeted (`33 passed`). A short 2-proposal OpenAI validation promoted a
KKT-valid `fuse_op` by a narrow noisy margin (`tier2=19.54` ms), but the
selector still chooses the older faster Gram champion
`450c6b8b-49e7-4b57-9a2e-900b7708fbba` with stored `tier2=16.40` ms.
P4.15 added `tasks/autoresearch_context.md` and `tasks/program_kernel.md` as
large context documents for future agents. These documents explicitly map
Karpathy's autoresearch loop to this project, define the current active
autoresearch game, record the current Gram champion, list what is wrong with
the loop, and give the next run instructions. Treat them as the source of truth
for the next phase.

---

## 2. Hardware setup

| Box | Role | Access |
|---|---|---|
| Linux x86 (`raygun`, this host) | Development, tests that don't need MLX | Local |
| MacBook Pro M3 Pro 18 GB (`revants-macbook-pro`, `100.98.66.89`) | MLX/Metal evaluation, the actual synth runs | Tailscale + macOS Remote Login |

**SSH setup**: macOS Remote Login is enabled, the Linux box's `~/.ssh/id_ed25519.pub` is in the Mac's `~/.ssh/authorized_keys`. The username on the Mac is `revantkasichainula` (NOT `revant` — early misguess based on email). Tailscale SSH was tried first but `tailscale up --ssh` flow failed on the user's setup; we fell back to standard ssh-over-tailscale.

**Workflow**:
- Edit code on Linux
- `rsync -az --exclude='.venv/' --exclude='__pycache__/' --exclude='synth_state/' --exclude='runs/' /home/ray/convexkernels/ revantkasichainula@100.98.66.89:convexkernels/`
- For Linux-runnable tests: `pytest -q` (here)
- For Mac-only (MLX) tests/runs: `ssh revantkasichainula@100.98.66.89 'cd convexkernels && .venv/bin/python <cmd>'`

**M3 Pro hardware specs that matter**: ~150 GB/s peak memory bandwidth, 18 GB unified memory. The bench problems are bandwidth-bound (FISTA arithmetic intensity ≈ 0.5 ops/byte at fp32). Two regimes apparent from the M3 Pro probe:
- Small shapes (n ≤ 500): launch-overhead bound, BW utilization 16–32%
- Large shapes (n ≥ 2000): BW-bound, utilization 75% on (2000, 10000)

This regime split is baked into P4 acceptance criteria (`impl.py` proposer should weight fusion edits for small shapes, dtype edits for large).

---

## 3. The five locked design decisions

These came out of the user's "lets iterate" sweep. Each was discussed in detail; full reasoning in §13 conversation log. **Treat these as load-bearing constraints — change only with explicit user approval.**

### 3.1 Functional vs numerical equivalence (test contract)

**Locked**: Functional equivalence (KKT-gated) primary; iterate drift logged but non-blocking.

```python
def assert_equivalent(x_kernel, x_ref, problem, *, kkt_tol=1e-6, drift_warn=1e-2):
    """Both must satisfy KKT. Drift logged but does NOT fail."""
    kkt_k = problem.kkt_residual(x_kernel)
    kkt_r = problem.kkt_residual(x_ref)
    assert kkt_k < kkt_tol
    assert kkt_r < kkt_tol
    rel_drift = ||x_k - x_r||_inf / max(||x_r||_inf, 1)
    if rel_drift > drift_warn:
        warnings.warn(...)
```

**Why**: precision-induced drift on fp16 makes numerical thresholds spurious; degenerate active sets give multiple optima; one ε across the project (matches the synthesis convergence gate). Implementation: `convexkernels/algorithms/kkt.py::assert_equivalent`.

### 3.2 Champion store + lineage schema

**Locked**: JSONL + JSON files, full schema in `docs/schema.md`. Slot key = `(problem_family, algorithm, hardware, dtype)`. **Different precision dtypes are separate slots, not winner-take-all.**

Per-proposal lineage record fields: id, parent_id, generation, slot, edit, source (path + sha256), tier1, tier2 (if tier1 passed), tier3 (if tier2 passed), decision. Records can omit later tiers.

`synth_state/`:
- `lineage.jsonl` — append-only, polars-readable
- `edits.json` — cached edit-success priors
- `champions/index.json` — current champion pointers
- `champions/<problem>/<algo>/<hw>/<dtype>/champion.py` (symlink to `runs/<id>/source.py`)

`runs/<lineage_id>/`:
- `source.py`, `compile.log`, `started.json` (WAL marker), `tier1.json`, `tier2.json`, `tier3.json`, `x.npy`

Implementation: `convexkernels/synth/lineage.py`, `synth/checkpoint.py`, and `synth/champion_store.py`. P4.1 now has symlink promotion, `index.json`, metadata, and `pareto.jsonl`; true Pareto-front pruning/ranking remains future P4 work.

### 3.3 DSL design

**Locked**: Hybrid. Ergonomic per-specimen constructors (`Lasso(A, b, lam)`, future `NonnegLasso(A, b, lam)`) are canonical entry points. Future `frontend/cvxpy_adapter.py` consumes a `cvxpy.Problem`, recognizes known structures, dispatches to the matching specimen — **unrecognized structures raise `NotImplementedError`** (we are not building a general convex solver).

### 3.4 Cross-problem transfer test (P6 acceptance)

**Current framing**: transfer is a diagnostic and prior source, not the primary acceptance target. For P6, the primary test is whether the loop can synthesize a strict champion for the specified setting `(nonnegative_lasso, fista, apple_silicon, fp32)`. A WITH/without `seed_from_neighbors(records, slot, k=3)` counterfactual is still useful to measure whether transfer priors reduce search time, but it should not replace the slot-level kernel-optimization goal.

### 3.5 Checkpointing + sandbox

**Locked**: 
- `synth/sandbox.py`: subprocess + timeout + `RLIMIT_AS` memory cap (Linux only — `RLIMIT_AS` is unreliable on macOS; Mac users rely on timeout). Default: 30 s timeout, 4 GB cap.
- `synth/checkpoint.py`: write-ahead log via `runs/<id>/started.json` BEFORE evaluation; on startup scan `runs/` for orphans (started without lineage record). Default policy: log orphans, don't requeue. Policy is a knob.

---

## 4. Repo file map

```
convexkernels/                      # repo root, /home/ray/convexkernels
├── pyproject.toml                  # deps + extras: [mac]=mlx, [baselines]=adelie+alpaqa, [ai]=openai, [dev]=pytest
├── .python-version                 # 3.12 (pinned because cvxpy 1.5 has no cp314 wheels)
├── .gitignore                      # incl. synth_state/ runs/
├── convexkernels/
│   ├── __init__.py                 # __version__ = "0.0.1"
│   ├── frontend/
│   │   ├── problem.py              # ABC: matvec/rmatvec/grad_smooth/prox/kkt_residual/L/lambda_max/n
│   │   └── lasso.py                # Lasso(A, b, lam), cached Atb/L/lambda_max
│   ├── algorithms/
│   │   ├── kkt.py                  # lasso_kkt_residual (prox-residual form), assert_equivalent
│   │   └── fista.py                # fista() driver — "basic" + "restart" variants, kernel_init parameter
│   ├── kernels/
│   │   ├── numpy_ref.py            # FistaState, fista_step, init_state — correctness oracle
│   │   ├── registry.py             # NOT YET WRITTEN (planned P4)
│   │   └── mlx/
│   │       ├── lib.py              # LassoMLX (from_lasso classmethod)
│   │       ├── seeds/
│   │       │   └── fista_step_v0.py     # FIRST MLX SEED — fused soft-threshold + axpy + momentum
│   │       └── champions/          # empty; populated by synth loop
│   ├── synth/
│   │   ├── lineage.py              # full schema dataclasses + JSONL writer
│   │   ├── checkpoint.py           # mark_started, find_orphans
│   │   ├── sandbox.py              # SandboxResult, write_eval_config, run_kernel
│   │   ├── _eval_kernel.py         # subprocess entry point — handles dotted module names AND .py paths
│   │   ├── applier.py              # full-source + structured source transforms
│   │   ├── edits.py                # edit-type and structured-payload priors
│   │   ├── fitness.py              # structured fitness diagnostics
│   │   ├── tiers.py                # Tier-1/2/3 evaluators with repeated timing
│   │   ├── roofline.py             # analytical bytes/ops/utilization estimates
│   │   ├── champion_store.py       # slot-keyed champion promotion metadata/symlinks
│   │   ├── loop.py                 # run_synth_loop — main driver
│   │   ├── run.py                  # CLI driver (`python -m convexkernels.synth.run`)
│   │   ├── proposers/
│   │   │   ├── stub.py             # DeterministicStubProposer
│   │   │   ├── openai.py           # OpenAIProposer (Responses API)
│   │   │   ├── structured.py       # deterministic structured payload sweep
│   │   │   └── claude.py           # legacy ClaudeProposer (not default)
│   │   └── prompts/
│   │       └── impl_openai.md      # OpenAI implementation-level prompt template
│   └── bench/
│       ├── shapes.py               # 4 dense ShapeSpec defaults
│       ├── baselines.py            # numpy_fista, sklearn, adelie, cvxpy adapters
│       └── run.py                  # bench/run.py CLI driver
├── tests/
│   ├── test_smoke.py               # 2 tests
│   ├── test_kkt.py                 # 10 tests
│   ├── test_fista.py               # 8 tests
│   ├── test_kernels.py             # 2 tests (MLX-only, skipped on Linux)
│   ├── test_sandbox.py             # 5 tests
│   ├── test_synth_loop.py          # loop ratchet/tier/structured plumbing
│   ├── test_proposer_openai.py     # mocked OpenAI client and prompt context
│   ├── test_applier.py             # structured source transforms
│   ├── test_edits.py               # priors
│   ├── test_structured_proposer.py # deterministic payload sweep
│   └── test_champion_store.py      # champion promotion
├── tasks/
│   ├── todo.md                     # the master plan; phase checkboxes + acceptance criteria
│   ├── results.md                  # numbers + decisions per phase
│   ├── lessons.md                  # empty (CLAUDE.md-mandated, no entries yet)
│   ├── mac_probe_output.txt        # raw output from the M3 Pro probe
│   └── handoff.md                  # this file
└── docs/
    ├── schema.md                   # full lineage/champion-store schema spec
    ├── roofline.md                 # FISTA AI=0.5 ops/byte + per-shape T_floor table calibrated to M3 Pro
    ├── mac_probe.md                # runbook for the Mac probe
    └── probes/
        ├── mac_probe.py            # the probe script run on the Mac
        └── bench_mlx_seed.py       # MLX seed kernel benchmark
```

**State directories** (gitignored, runtime):
- `synth_state/` — lineage.jsonl, edits.json, champions/
- `runs/` — per-proposal artifacts

---

## 5. Test status

```
$ pytest -q
.........................ssss.............................................
92 passed, 4 skipped in ~2s
```

| File | Tests | Notes |
|---|---|---|
| `test_smoke.py` | 2 | imports |
| `test_kkt.py` | 10 | KKT formula + parametric sweep over seeds |
| `test_fista.py` | 8 | convergence vs CVXPY (P1 acceptance), restart variant, multi-seed |
| `test_kernels.py` | 4 | **MLX-only** — skipped on Linux. LASSO fp32/fp16, Gram fp32, and NN-LASSO fp32 seed kernels pass on M3 Pro |
| `test_sandbox.py` | 6 | subprocess sandbox + prepare hook/timing + lineage + WAL/orphan + path-loader dataclass regression |
| `test_applier.py` | 12 | structured source rewrites plus durable gradient/dtype strategy hooks |
| `test_structured_proposer.py` | 3 | deterministic structured payload ordering and NN-LASSO filtering |
| `test_synth_loop.py` | 18 | loop plumbing, speed ratchet, cost-model timing, duplicate guard, structured/config edits, transfer seeds, applier/proposer errors, Tier-2 promotion |
| `test_proposer_openai.py` | 8 | mocked OpenAI client; structured parsing, prompt/runtime context, error paths |
| `test_transfer.py` | 4 | cross-slot transfer ranking, full-source skip, target dedupe, NN-LASSO payload adaptation |
| `test_applier.py` | 7 | structured source transforms and composition |
| `test_edits.py` | 3 | edit-type and payload-level priors |
| `test_structured_proposer.py` | 2 | deterministic structured payload sweep |
| `test_champion_store.py` | 2 | slot-keyed promotion metadata/symlinks |
| `test_roofline.py` | 3 | analytical roofline estimates and Tier-3 integration |
| `test_fitness.py` | 4 | structured fitness diagnostics and prompt summary |

Run on Mac to verify MLX tests:
```
ssh revantkasichainula@100.98.66.89 'cd convexkernels && .venv/bin/python -m pytest tests/test_kernels.py -v'
```

---

## 6. Origin and reframing

The user opened with this prompt:

> "i am interested in doing a project that involves creating custom kernels for convex optimization algorithms using autoresearch. This is cool because we can easily verify KKT conditions for fast iterates and improvement. I have here some quick notes sent from my advisor nvfp4 for 4 bit precision on nvidia blackwell"
>
> "lasso admm solver. use triton kernels for cpu or mlx kernels for mac to improve the algorithm."
>
> "https://github.com/JamesYang007/adelie is a good lasso solver"
>
> "https://kul-optec.github.io/alpaqa/Sphinx/examples/lasso-jax.html"
>
> "https://github.com/huggingface/ml-intern/tree/main. I would like to plan for next steps. /effort max /plan"

Initial AskUserQuestion produced these scope answers:
- Hardware: **Mac first (MLX)**
- Autoresearch: **Custom research loop (build from scratch)**
- MVP scope: **LASSO ADMM end-to-end with KKT verifier + one fast kernel**
- Baselines: **adelie, alpaqa, sklearn, cvxpy**

Three parallel fork agents researched: (a) MLX custom kernel API + ml-intern repo + Mac hardware constraints; (b) adelie + alpaqa internals; (c) ADMM-LASSO + KKT residual formulation. Results synthesized into the first plan draft.

Then iteration. The user said: "yes all 6 points are really good to look into, i really want to question all of them." — opening a deep design pass. Six axes interrogated:

1. **First kernel choice**: I had picked "fused z+u+KKT-grad pass" but realized this conflated O(n) and O(mn) ops. Settled on a P0 spike to determine what to fuse.
2. **Autoresearch scope**: User said this was the THESIS, not a stretch goal. Agent proposer must be CORE.
3. **Fitness shape**: structured fitness vector instead of single scalar.
4. **Datasets**: 4 dense shapes for MVP, real datasets later if needed.
5. **Phasing**: baselines moved to P2 (day-one numbers, not end).
6. **Precision**: core MVP axis, not deferred.

Plus a bigger question I raised: **is ADMM the right algorithm?** Since the advisor said "lasso ADMM" but neither reference uses ADMM (adelie=prox-Newton+CD, alpaqa=PANOC), and since FISTA is a much better GPU citizen than ADMM (cached-Cholesky trisolve is hostile to GPU)... we agreed to **FISTA first, ADMM second** as algorithm-bank entries.

Then the CVXGEN reframe (user prompt verbatim above). This shifted the whole project from "fast LASSO solver" to "kernel synthesis harness." Major implication: agent proposer in P3, not P6.

User pushed back on using `evo` skill: "i like it, and i gave it as an example, but I feel like it is not that efficient for our case, it is built to iterate on hyperparameters for ML models and I feel we could build something that is better suited to our task even if it takes more effort." → custom synthesis loop with seven specific advantages enumerated (per-iter KKT trajectory feedback, multi-fidelity gating, structured edit grammar, profile-guided proposers, hierarchical proposer bank, convex-invariant kill switches, cross-problem transfer).

Then five-axis lock-down (§3 above).

---

## 7. Phase-by-phase progress

### P0 — Scaffold + capability probes ✅

- pyproject.toml with extras `[mac, baselines, ai, dev]`. Python 3.12 pinned via `.python-version`.
- Resolved cvxpy<1.6 + adelie<numpy2 conflict by pinning cvxpy<1.6.
- alpaqa pinned to `1.1.0a1` (PyPI's `0.0.1` is a placeholder).
- M3 Pro probe (`docs/probes/mac_probe.py`) executed: `mx.fast.metal_kernel` works; `mx.linalg` has cholesky/solve_triangular/solve/lu/qr/svd/inv (full LAPACK surface — **major derisking of ADMM in P5**); matmul utilization 16–75% across bench shapes (two regimes); fp16 saves ~42% wall time vs fp32.
- Roofline calibrated to 150 GB/s peak.
- 2 smoke tests passing.

### P1 — NumPy reference FISTA + KKT verifier ✅

- `Problem` ABC, `Lasso` specimen with cached Atb/L/lambda_max.
- KKT residual via **prox-residual reformulation** (NOT case-split — first attempt failed because CVXPY's ~1e-12 entries got misclassified as "active"). The prox-residual `r = L*x - soft(L*x - g, lam)` is mathematically equivalent at the optimum, robust to floating-point.
- FISTA basic + restart variants. Restart is empirically 3× faster on the test problems.
- MFISTA dropped (would require explicit objective evaluator; restart already monotonizes via momentum reset).
- 18 tests passing (10 KKT + 8 FISTA).
- P1 acceptance: KKT(x_cvxpy) < 1e-8 ✓, FISTA matches CVXPY at rel drift < 1e-4 ✓.

### P2 — Baseline shootout ✅

- 4 dense shapes: tall_small (2000, 500), tall_medium (5000, 2000), wide_small (500, 2000), wide_large (2000, 10000). `rcv1.binary` deferred.
- 4 baselines: numpy_fista_restart (us), sklearn (CD), adelie (prox-Newton+CD), cvxpy (CLARABEL oracle).
- alpaqa **deferred** — its Python API requires CasADi/JAX bindings to construct a Problem object; significant detour for one baseline.
- All four solvers produce identical primal_obj to 4 decimal places on every shape.
- adelie wins wall time on every shape (1.7 ms to 160 ms). CVXPY ~10 min total run time, dominated by wide_large (448 s).
- **Critical CLARABEL gotcha**: tolerance kwargs are `tol_gap_abs`, `tol_gap_rel`, `tol_feas` — NOT `eps_abs`/`eps_rel` (those are SCS-style names). Documented.

### P3 — Minimal end-to-end synthesis cycle (PARTIAL)

#### P3.1 — Seed MLX kernel + functional equivalence ✅

- `kernels/mlx/seeds/fista_step_v0.py`: hand-written `mx.fast.metal_kernel` fusing z = y - t*g, soft-threshold, momentum axpy. Matvecs stay as `mx.matmul`.
- `kernels/mlx/lib.py`: `LassoMLX` duck-typed to `Problem`.
- `algorithms/fista.py`: refactored backend-agnostic via `kernel_init` parameter, restart-indicator uses `(diff_y * diff_x).sum()` which works for both numpy and mlx arrays.
- 2 functional-equivalence tests on M3 Pro: fp32 drift 1.14e-7, fp16 drift 2.34e-3 (precision floor as expected).
- **Headline bench (M3 Pro fp32 vs numpy_ref)**: MLX seed is 2–27× faster per iter (15.5× tall_small, 26.9× tall_medium, 2.1× wide_small, 8.2× wide_large). Iter counts match within 6%.
- Realized: tol=1e-6 is the realistic fp32 KKT bar; below that you hit the precision floor. Updated benchmarks accordingly.

#### P3.2 — Sandbox + checkpoint + lineage ✅

- `synth/sandbox.py`: subprocess + timeout + RLIMIT_AS (Linux), `SandboxResult` dataclass.
- `synth/_eval_kernel.py`: subprocess entry. Loads kernel by **dotted module name OR .py path** (key for P3.4: LLM-emitted source loaded from `runs/<id>/source.py`).
- `synth/checkpoint.py`: `mark_started`, `find_orphans`.
- `synth/lineage.py`: full schema implementation per `docs/schema.md`.
- 4 sandbox tests passing.

Subtle gotcha: `numpy_ref.py` initially had no `init_state` function. The sandbox subprocess imports the module fresh (no monkey-patching from parent process), so I added `init_state` directly to `numpy_ref.py` to match the seed kernel's API contract.

#### P3.3 — Minimal synth loop + stub proposer ✅

- `synth/proposers/stub.py`: `DeterministicStubProposer` cycles through edit types deterministically (used for testing the loop machinery without LLM in the critical path).
- `synth/loop.py`: `run_synth_loop` driver. Orphan check on startup, per-proposal WAL → eval → tier-1 gate → lineage append. Hardware auto-detection (`apple_silicon` vs `linux_x86_64`).
- 3 synth-loop tests passing.

#### P3.4 — OpenAI proposer + ratchet ✅

- `synth/applier.py`: `apply_edit(edit, run_dir)` — writes `edit.payload["full_source"]` to `runs/<id>/source.py`, returns `SourceInfo` with sha256 hash.
- `synth/proposers/openai.py`: `OpenAIProposer` calls the OpenAI Responses API and parses structured JSON `{edit_type, rationale, source}`.
- `synth/prompts/impl_openai.md`: prompt template with `{{champion_source}}`, `{{recent_history}}`, and `{{runtime_context}}` placeholders.
- `synth/loop.py` updated: when Edit has `full_source`, write to disk + use file-path for kernel_module; measure seed/champion baseline; keep only KKT-valid speedups.
- `synth/_eval_kernel.py` updated: handles `.py` paths via `importlib.util.spec_from_file_location`.
- Sandbox evaluator converts canonical NumPy `Lasso` to `LassoMLX` for MLX kernel runs, so seed/proposal modules receive `problem.dtype` and MLX arrays.
- Tier-1 evaluation has an untimed warmup pass by default, preventing first Metal compile overhead from dominating proposal timing.
- Provider/API failures are recorded as `crash:proposer_error:*` and the batch continues.
- `synth/run.py`: CLI driver defaults to `--proposer openai --model gpt-5.5`.
- 6 mocked-OpenAI tests passing (structured parsing, prompt templating, error paths), plus a path-loader regression test for generated dataclasses.
- `pyproject.toml`: `[ai]` extra now uses `openai>=1.109`.
- 30-proposal Mac ratchet run produced a kept champion: `6eea3c75-e80e-43aa-8bbd-78c74c01ef7e` (`fuse_op`), 1.53x faster than the seed at `tol=1e-6`.

P3.4 acceptance is met for `tall_small`. P4 work has continued beyond this
section; see §1 and `tasks/results.md` for the current structured-loop state.

---

## 8. The plan document (`tasks/todo.md`)

Read it. Sections you'll care about:

- Architecture diagram (current) — under "Architecture"
- Interface contracts (DSL, fitness vector, schema reference, assert_equivalent) — under "Interface contracts"
- P-by-P checklists with acceptance criteria
- Critical files table — under "Critical files"
- Reuse-from-references — borrow adelie's matrix-wrapper pattern, alpaqa's Problem ergonomics, ml-intern's agent+sandbox+trace layout, Boyd 2011 §6.4 + §3.3.1 + §3.4.1 for ADMM, Beck-Teboulle 2009 for FISTA
- Verification plan — correctness, performance, synth-loop sanity, harness generality
- Open questions/things to flag — most resolved, some open (real dataset choice, single-Mac risk)

---

## 9. Where we paused

Current state: the Tier-2 ratchet is timing-robust and structured edits are
machine-readable. Roofline and fitness diagnostics are now available in
`synth_state/fitness.json`. The latest `wide_small` work did not produce a
champion, but it did produce useful near-miss payloads just above the 3%
Tier-2 speed margin.

Recommended next step: stop spending many more rounds on the current
soft-threshold tail-kernel payload family alone. Either add a larger search
axis (fp16 storage/fp32 convergence, matvec layout/dtype, or screening/warm
starts), or promote the new fitness diagnostics into Pareto/champion gating
before running a larger population.

---

## 10. Two paths to resume

### Path A: P4 full architecture (primary)

Steps:

1. **Verify `OPENAI_API_KEY` on the Mac**:
   ```
   ssh revantkasichainula@100.98.66.89 'zsh -ic "cd convexkernels && .venv/bin/python -c '\''import os; print(\"key set\" if os.environ.get(\"OPENAI_API_KEY\") else \"not set\")'\''"'
   ```

2. **Install/update the OpenAI SDK on the Mac if needed**:
   ```
   ssh revantkasichainula@100.98.66.89 'cd convexkernels && .venv/bin/python -m pip install -e ".[ai]"'
   ```

3. **Run a larger ratchet batch on another shape**:
   ```
   ssh revantkasichainula@100.98.66.89 'source ~/.zshrc && cd convexkernels && .venv/bin/python -m convexkernels.synth.run --proposer openai --n-proposals 30 --shape wide_small --state-root ./synth_run_openai_wide_small --reasoning-effort low'
   ```
   Expected outcome: populate a second shape's lineage and see whether the `tall_small` edit transfers.

4. **Inspect results**:
   ```
   ssh revantkasichainula@100.98.66.89 'cd convexkernels && cat synth_run/synth_state/lineage.jsonl | python -m json.tool'
   ```

5. **P4 implementation focus**: turn the current one-shape ratchet into a tiered evaluator and champion store so accepted variants must pass Tier-2/Tier-3, not just Tier-1.

6. **Update `tasks/todo.md` to check off P3.4 boxes; populate `tasks/results.md` with**:
   - Number of proposals attempted
   - Number that compiled
   - Number that converged
   - Number that beat seed wall time
   - Best variant's edit + rationale + speedup

7. **Iterate on the prompt** (`convexkernels/synth/prompts/impl_openai.md`) based on what fails: many proposals failing on the same axis means the prompt isn't communicating constraints well.

### Path B: Option 3 — Manual proposer via Claude Code (free, slower)

In this mode, you (the agent) act as the proposer instead of an API call. The synth loop is modified slightly to read the source from a file the user provides, instead of calling the API.

Steps:

1. **Add a `FileProposer`** to `convexkernels/synth/proposers/`:
   ```python
   class FileProposer:
       """Reads the next proposal from a hand-written file. For Claude Code-driven runs."""
       def __init__(self, source_dir: Path):
           self.source_dir = source_dir
           self.counter = 0
       
       def propose(self, slot, parent_id, history):
           src_path = self.source_dir / f"proposal_{self.counter:03d}.py"
           # Block until the file exists, or raise if it doesn't
           if not src_path.exists():
               raise FileNotFoundError(f"expected {src_path}; the user/agent must write it")
           full_source = src_path.read_text()
           edit_type_path = self.source_dir / f"proposal_{self.counter:03d}.meta.json"
           meta = json.loads(edit_type_path.read_text())  # {edit_type, rationale}
           self.counter += 1
           return Edit(
               type=meta["edit_type"],
               payload={"full_source": full_source},
               rationale=meta["rationale"],
               proposer_role="impl",
               proposer_model="claude_code_session",
               source="manual",
           )
   ```

2. **Workflow per proposal**:
   - You read `convexkernels/kernels/mlx/seeds/fista_step_v0.py` (current champion source) and recent lineage.
   - Propose a mutation. Write the new source to `proposals/proposal_NNN.py` and metadata to `proposals/proposal_NNN.meta.json`.
   - Trigger the synth loop on the Mac (it loads the proposal, evaluates, writes lineage).
   - Read the result; iterate.

3. This is slower (~2 min per proposal vs ~30 sec for API) but free and lets you exercise the full pipe without API setup.

**For the next agent**: I'd default to Path A *if the user has already set up API credits*. Otherwise propose Path B and check if they want it.

---

## 11. Known issues + follow-ups (non-blocking)

1. **`Lasso.L` via SVD is slow on Linux** — dominates FISTA wall time on the Linux x86 dev box. On Mac with Accelerate-backed numpy it's faster. Power iteration would close the gap. Documented in `tasks/results.md → P2`. Not blocking.

2. **alpaqa baseline missing** — deferred from P2 due to CasADi/JAX dependency complexity. Add later if PANOC's quasi-Newton story becomes important.

3. **rcv1.binary sparse dataset missing** — deferred from P2. Add as P2.5 if synth loop diversity requires.

4. **CVXPY ground-truth caching** — `bench/run.py` reruns CVXPY every time; that's ~10 min on the wide_large shape. Should pickle to disk. Documented as P2 follow-up.

5. **Champion store pruning/ranking still minimal** — P4.1 has `index.json`, symlink promotion, metadata, and append-only `pareto.jsonl`, but not full Pareto-front pruning/ranking yet.

6. **`synth/registry.py` not yet written** — planned for P4 when we have multiple kernel variants to register.

7. **Strict P6 transfer acceptance not yet met** — `seed_from_neighbors` is implemented and records `edit.source = "transfer:<src_slot>"`, but repeated Tier-2 timing still rejects the current transferred tail edits on NN-LASSO. Need a larger edit family before the formal transfer-effectiveness criterion is likely to pass.

8. **alpaqa stale 0.0.1 placeholder on PyPI** — pinned to alpaqa==1.1.0a1.

9. **Mac `RLIMIT_AS` unreliable** — sandbox documents that Mac users rely on timeout only. Not blocking; could add psutil-based monitoring later.

10. **MLX fp16 hits precision floor at KKT ~1e-3** — `fista_step_v0` is fp16-end-to-end. fp16-storage/fp32-accum is a separate slot the synth loop will populate. Test bar relaxed to kkt_tol=1e-2 for fp16 in `test_kernels.py`.

11. **Vectorizing an already-vectorized champion is still an applier error** — this is now recorded cleanly, but the structured proposer/loop should eventually filter source-incompatible payloads before spending a round.

---

## 12. Operational notes for the next agent

- **The user prefers concise responses.** They opened with `/effort max`; they've set auto-mode off so default to asking clarifying questions when ambiguous, but they explicitly said multiple times "make the reasonable call and continue" when they want autonomy. Read the room.
- **Don't commit unless explicitly asked.** CLAUDE.md says so; the user has not asked yet.
- **Don't skip git hooks (`--no-verify`)** unless explicitly asked.
- **Update `tasks/todo.md`** as items complete, **`tasks/results.md`** with numbers, and **`tasks/lessons.md`** if user corrects something. The user's CLAUDE.md mandates this loop.
- **Test on Linux, run on Mac.** Linux is for development; Mac (M3 Pro via SSH) is for actual MLX runs. Don't try to run MLX on Linux — it doesn't work.
- **`@pytest.mark.skipif(not _has_mlx, ...)` pattern** lets test_kernels.py be Linux-collectable but Mac-runnable.
- **rsync excludes**: `--exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.egg-info/' --exclude='build/' --exclude='synth_state/' --exclude='runs/'`
- **Run pytest after every code change**; the user's CLAUDE.md says "Never mark a task complete without proving it works."
- **The KKT formula is THE fitness function.** A bug there corrupts every promotion decision. It's in the critical-files list. The current implementation (prox-residual form) is correct; tested at fp64 precision against CVXPY.
- **Two-regime mental model**: small shapes are launch-bound (fusion wins), large shapes are BW-bound (precision wins). The proposer prompt mentions this; the impl-level proposer in P4 should weight mutations by shape regime.

---

## 13. Conversation log of substantive prompts and responses

This is *not* the full transcript. It's the design-load-bearing prompts and the assistant's substantive responses. Most operational chitchat (e.g. "ok continue") is omitted. When in doubt about a design call, this is the source of truth.

### 13.1 Opening prompt (user, verbatim)

> "i am interested in doing a project that involves creating custom kernels for convex optimization algorithms using autoresearch. This is cool because we can easily verify KKT conditions for fast iterates and improvement. I have here some quick notes sent from my advisor nvfp4 for 4 bit precision on nvidia blackwell"
>
> "lasso admm solver. use triton kernels for cpu or mlx kernels for mac to improve the algorithm."
>
> "https://github.com/JamesYang007/adelie is a good lasso solver"
>
> "https://kul-optec.github.io/alpaqa/Sphinx/examples/lasso-jax.html"
>
> "https://github.com/huggingface/ml-intern/tree/main. I would like to plan for next steps. /effort max /plan"

### 13.2 Initial AskUserQuestion responses (user)

| Question | Answer |
|---|---|
| Hardware target | Mac first (MLX) |
| Autoresearch mode | Custom research loop (build from scratch) |
| MVP scope | LASSO ADMM end-to-end with KKT verifier + one fast kernel |
| Baselines | adelie, alpaqa, sklearn, cvxpy |

### 13.3 Iteration request (user, verbatim)

> "yes all 6 points are really good to look into, i really want to question all of them."

The assistant then went through each of six axes (first kernel choice, autoresearch scope, fitness shape, datasets, phasing, precision) plus a bigger algorithm question (ADMM vs FISTA).

### 13.4 evo pushback (user, verbatim)

> "ok i like this idea, except I want to push back on using evo, i like it, and i gave it as an example, but I feel like it is not that efficient for our case, it is built to iterate on hyperparameters for ML models and I feel we could build something that is better suited to our task even if it takes more effort."

Assistant response enumerated 7 specific advantages of a custom loop over evo: per-iter KKT trajectory feedback, multi-fidelity gating, structured edit grammar with lineage, profile-guided proposers, hierarchical proposer bank, convex invariants as kill switches, cross-problem transfer.

### 13.5 The CVXGEN reframe (user, verbatim)

> "yes I like this idea, except... [evo pushback above]. I think my end goal is some sort of harness where we just input a problem, and our subagent synthesizes super fast kernel code for it using iterative autoresearch. think like how cvxgen software takes a problem and writes super fast C solver code for QPs and LPs"

Assistant response: this reframes the whole project. LASSO becomes specimen #1 of a kernel-synthesis-harness. The deliverable is the harness, not the LASSO solver. ADMM becomes the second algorithm in a bank. NVFP4 becomes a future dtype slot.

### 13.6 Locks for the three big questions (user, verbatim)

> "yes all three are good to lock down. esp keep in mind this: Recommend: agent-proposer is P4 (after first manual kernel proves the harness works), not P6. Manual templates are a warmup. The writeup is 'KKT-gated LLM-driven kernel synthesis,' not 'we wrote a fast lasso solver.'"

### 13.7 P3.4 path choice (user, verbatim)

> "yes lets do 1"

Confirmed: Anthropic API directly for P3.4.

### 13.8 Billing question (user, verbatim)

> "when you put the API key does it spend usage limit or does it spend money from somewhere else"

This is the LAST substantive user message before "write a massive in depth handoff document". The user had not yet decided whether to actually fund the API call or switch to Option 3.

### 13.9 Substantive agent design responses (paraphrased; full detail in tasks/todo.md and tasks/results.md)

- **Functional vs numerical equivalence (#1 lock)**: detailed why precision-induced drift makes numerical thresholds fragile; proposed `assert_equivalent` with KKT-gating + drift-as-diagnostic.
- **Champion store schema (#2 lock)**: full JSONL+JSON layout, lineage record fields, edit-priors, cross-problem transfer mechanic.
- **DSL hybrid (#3 lock)**: per-specimen constructors canonical; cvxpy adapter raises NotImplementedError on unrecognized structures.
- **Cross-problem transfer test (#4 lock)**: P6 acceptance — WITH-seeding ≤50% rounds vs WITHOUT; ≥1 transferred edit in final Pareto front.
- **Profiler call**: dropped Metal hardware counters from MVP; ship roofline-from-source instead. Three signals: timing, KKT trajectory, analytical roofline. Document concrete fallback if proposer quality plateaus.
- **MLX trisolve risk eliminated** by P0 probe — full LAPACK surface confirmed (cholesky, solve_triangular, solve, lu, lu_factor, qr, svd, inv, pinv, tri_inv, cholesky_inv, eig).
- **KKT formula switched to prox-residual** during P1 implementation — first attempt with case-split formula failed because CVXPY's near-zero entries got misclassified as "active." `r = L*x - soft(L*x - g, lam)` is mathematically equivalent at the optimum and numerically robust.
- **MFISTA dropped, restart kept** — restart already monotonizes via momentum reset and is empirically 3× faster than basic FISTA.
- **alpaqa deferred** during P2 implementation — its Python API requires CasADi/JAX bindings to construct a Problem. Significant detour; skipped.
- **CLARABEL kwargs**: `tol_gap_abs`/`tol_gap_rel`/`tol_feas`, NOT `eps_abs`/`eps_rel`. Documented in `tasks/results.md → P1`.
- **MLX-only tests skipped on Linux** via `pytest.mark.skipif(not _has_mlx, ...)`.

---

## 14. Final checklist before resuming

If you (the next agent) are about to make a code change:

- [ ] Read `tasks/todo.md` end-to-end
- [ ] Read `tasks/results.md` for numbers
- [ ] Read `docs/schema.md` for the persistent data model
- [ ] Look at `tasks/lessons.md` for any user corrections
- [ ] `.venv/bin/python -m pytest -q` here on Linux to confirm the current 90+4 skipped baseline
- [ ] Check whether the Mac is reachable: `nc -zv -w 5 100.98.66.89 22`
- [ ] If touching synth/, add a test in tests/ before claiming done
- [ ] If touching MLX code, rsync to Mac and run there
- [ ] If running synth loop with OpenAI API, watch the lineage.jsonl in real time
- [ ] Update `tasks/todo.md` checkboxes as items complete
- [ ] Update `tasks/results.md` with numbers
- [ ] Don't commit unless asked

Good luck.
