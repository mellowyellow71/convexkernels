# program_kernel.md — Karpathy-style kernel autoresearch program

This is the instruction layer for the next autonomous kernel-research run. It
is modeled after Karpathy's `program.md`, but adapted to KKT-gated convex
optimization kernel synthesis.

The goal is to make each run a tight experiment game:

```
fixed target
fixed evaluator
fixed metric
fixed correctness gate
limited editable/search surface
append-only result log
keep only confirmed improvements
```

Use this as the "research org code" for the next agent or OpenAI/Codex
continuation. Do not treat it as immutable; improving this file is part of
improving the autoresearch system.

---

## 1. Active Run Target

Current active target:

```
problem_family: lasso
algorithm:      fista
variant:        restart
hardware:       apple_silicon
backend:        mlx
dtype:          fp32
shape:          tall_medium
shape dims:     m=5000, n=2000
workload:       amortized repeated solves with fixed A
cost_model:     amortized
accuracy gate:  KKT_final < 1e-6
metric:         median Tier-2 solve_time_ms
```

Current selected champion:

```
lineage id:       450c6b8b-49e7-4b57-9a2e-900b7708fbba
strategy:         Gram-precomputed FISTA
gradient path:    g = G @ y - c
G:                A.T @ A
c:                A.T @ b
dtype_strategy:   fp32
stored Tier-2:    16.396416118368506 ms
KKT:              4.29e-07
paired direct ref: 48.46549988724291 ms
```

This champion is not global-optimal. It is the best recorded candidate for
this exact target and workload.

---

## 2. What May Be Changed

For this run, the most valuable search surface is the Gram gradient path.

Allowed edits should target:

- `LassoGramMLX.grad_smooth`
- `G @ y - c`
- Gram storage dtype
- Gram KKT dtype
- Gram memory layout
- Gram precompute/setup strategy if it impacts repeated-solve timing or future
  `single` workload
- KKT-safe precision experiments
- custom Metal/MLX kernel variants for Gram GEMV if benchmark evidence justifies it

Allowed files for implementation work:

```
convexkernels/kernels/mlx/lib.py
convexkernels/kernels/mlx/seeds/gram_fista_step_v0.py
convexkernels/synth/applier.py
convexkernels/synth/proposers/structured.py
convexkernels/synth/proposers/openai.py
convexkernels/synth/prompts/impl_openai.md
convexkernels/synth/loop.py
convexkernels/synth/tiers.py
tests/test_kernels.py
tests/test_synth_loop.py
tests/test_applier.py
tests/test_proposer_openai.py
tasks/results.md
tasks/handoff.md
tasks/autoresearch_context.md
```

If adding a new structured edit field, update:

```
convexkernels/synth/applier.py
convexkernels/synth/proposers/structured.py
convexkernels/synth/proposers/openai.py
convexkernels/synth/prompts/impl_openai.md
tests/
tasks/results.md
```

---

## 3. What Should Not Be Changed In This Run

Do not pivot to ADMM for this run. ADMM remains important, but the current
experiment is specifically:

```
lasso + fista + apple_silicon/mlx + fp32 + tall_medium + amortized
```

Do not spend proposals on generic tail-kernel edits unless they directly
interact with the Gram bottleneck. Tail edits tried after the Gram champion
were KKT-valid but slower.

Avoid exact repeats of:

- pure branchless/tail rewrites after Gram
- `items_per_thread=4` tail vectorization after Gram
- Gram/no-bounds tail variants
- `mixed_gram` as currently implemented; it fails KKT around `1e-4`
- `fp16_storage` as currently implemented; it fails KKT around `1e-3` to
  `1e-4`

Do not change KKT tolerance to make a candidate pass. KKT < `1e-6` is the
contract for this run.

Do not run multiple tall-medium synth baselines concurrently on the Mac.

---

## 4. Fixed Evaluator

The main evaluator is:

```bash
ssh revantkasichainula@100.98.66.89 'cd convexkernels && \
.venv/bin/python -m convexkernels.synth.run \
  --problem-family lasso \
  --proposer openai \
  --model gpt-5.5 \
  --reasoning-effort medium \
  --api-timeout-s 240 \
  --n-proposals 20 \
  --shape tall_medium \
  --tier2-shape tall_medium \
  --state-root ./synth_run_p4_gram_autoresearch_YYYYMMDD \
  --promotion-tier tier2 \
  --variant restart \
  --cost-model amortized \
  --tier1-tol 1e-6 \
  --tier1-max-iters 5000 \
  --tier1-reps 3 \
  --tier1-escalation-margin 1.0 \
  --tier2-tol 1e-6 \
  --tier2-max-iters 5000 \
  --tier2-reps 3 \
  --tier2-speed-margin 0.97 \
  --timeout-s 240'
```

Before a long run, confirm the currently selected champion:

```bash
ssh revantkasichainula@100.98.66.89 'cd convexkernels && .venv/bin/python -c '\''
from pathlib import Path
from convexkernels.synth.champion_store import ChampionStore
from convexkernels.synth.lineage import Slot
slot = Slot("lasso", "fista", "apple_silicon", "fp32")
store = ChampionStore(Path("synth_run_p4_strategy_tall_medium_amortized_20260509"))
for key in ("single", "amortized"):
    source, meta = store.current_source_for_workload(slot, key)
    print(
        key,
        "source", source,
        "metadata_cost", (meta.get("summary") or {}).get("cost_model"),
        "id", meta.get("id"),
        "tier2", (meta.get("summary") or {}).get("tier2_wall_time_ms"),
    )
'\'''
```

Expected:

```
single source None metadata_cost None id None tier2 None
amortized source ... id 450c6b8b-49e7-4b57-9a2e-900b7708fbba tier2 16.396416118368506
```

---

## 5. Keep / Discard Rule

Hard validity gate:

```
status == completed
converged == true where applicable
KKT_final < 1e-6
no timeout
no NaN
```

Performance gate:

```
candidate Tier-2 median solve_time_ms
    <
paired/current champion Tier-2 median solve_time_ms * tier2_speed_margin
```

Current margin:

```
tier2_speed_margin = 0.97
```

Interpretation:

- a smaller time is better
- a candidate must be at least about 3% faster to promote
- narrow one-rep wins are suspect and should be repeated

If a candidate is KKT-valid but slower, record and discard.

If a candidate fails KKT, record and discard.

If a candidate crashes due to a simple typo/import, fix once and rerun. If the
idea is fundamentally broken, log as crash/invalid and move on.

---

## 6. Logging

The main log is already:

```
synth_state/lineage.jsonl
```

Additional generated summaries:

```
synth_state/edits.json
synth_state/fitness.json
synth_state/champions/**/metadata.json
synth_state/champions/**/pareto.jsonl
runs/<id>/tier1.json
runs/<id>/tier2.json
runs/<id>/eval_config.json
runs/<id>/source.py
```

For human-readable progress, update:

```
tasks/results.md
tasks/handoff.md
tasks/autoresearch_context.md
```

For an overnight run, also create a compact TSV if useful:

```
tasks/gram_autoresearch_results.tsv
```

Suggested TSV columns:

```
id	status	edit_type	kkt	tier1_ms	tier2_ms	speed_ref_ms	description
```

Do not commit runtime state unless the user explicitly asks.

---

## 7. First Experiment Ideas

Prefer these before random full-source rewrites.

### 7.1 Gram gradient microbenchmark

Add a benchmark that isolates repeated calls to:

```
LassoGramMLX.grad_smooth(x)
```

Measure:

- repeated `G @ x - c`
- dtype of `G`, `c`, `x`
- cost of casting
- KKT path if separate dtype is used

This tells us whether MLX GEMV is the bottleneck and whether a custom Metal
GEMV is even worth trying.

### 7.2 Structured field: `gram_gradient_strategy`

Add a structured edit field:

```
gram_gradient_strategy:
    "mlx_matvec"
    "custom_metal_gemv"
    "blocked_metal_gemv"
```

Start with two values only if implementing all three is too much.

### 7.3 Structured field: `gram_kkt_dtype`

Separate solve-gradient dtype from KKT-check dtype:

```
gradient_dtype = fp16 or fp32
kkt_dtype = fp32
```

Current `mixed_gram` uses fp16 gradients and fp32 KKT, but convergence fails at
strict `1e-6`. A possible future variant is mixed storage with correction or
periodic fp32 gradient, not simply fp16 every iteration.

### 7.4 Periodic correction idea

Try:

- cheap approximate gradient most iterations
- full fp32/direct/Gram correction periodically

This changes algorithm semantics and must be KKT-gated carefully. It may be an
algorithm-level edit rather than pure impl-level edit.

### 7.5 Setup optimization for `single`

For single-solve workload only:

- reduce Gram precompute setup
- avoid Gram if setup dominates
- possibly choose direct champion

Do not mix this with the amortized game.

---

## 8. Test Discipline

After code changes:

Linux:

```bash
.venv/bin/python -m pytest
```

Sync:

```bash
rsync -az \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='*.egg-info/' \
  --exclude='build/' \
  --exclude='synth_state/' \
  --exclude='runs/' \
  /home/ray/convexkernels/ \
  revantkasichainula@100.98.66.89:convexkernels/
```

Mac targeted:

```bash
ssh revantkasichainula@100.98.66.89 'cd convexkernels && \
.venv/bin/python -m pytest \
  tests/test_champion_store.py \
  tests/test_synth_loop.py \
  tests/test_proposer_openai.py \
  tests/test_kernels.py'
```

If touching MLX kernels or `LassoGramMLX`, include `tests/test_kernels.py` on
the Mac.

If touching proposer schema, include `tests/test_proposer_openai.py`.

If touching champion selection, include `tests/test_champion_store.py` and
`tests/test_synth_loop.py`.

---

## 9. Autonomy Instruction

Once a serious autoresearch run begins:

- do not stop after one proposal
- do not ask whether to continue
- keep the loop running until the requested proposal budget finishes or the
  user interrupts
- if there are crashes, record them and continue unless the harness itself is
  broken
- periodically summarize results in `tasks/results.md`

The current system is not yet ready for a true overnight `100`-proposal run.
It first needs the Gram-specific benchmark/edit surface and a confirmed
multi-rep baseline.

---

## 10. Current Bottom Line

We have a real current champion, but we have not exhausted the search.

The loop should now become narrower, more Karpathy-like, and more Gram-specific:

```
one workload
one scalar metric
one correctness gate
one bottleneck-focused search surface
many repeated proposals
```

The next agent should not restart broad planning. It should:

1. read `tasks/handoff.md`
2. read `tasks/autoresearch_context.md`
3. read this file
4. confirm the Mac state and tests
5. implement the first Gram-specific benchmark/edit surface
6. run a fixed-budget Mac autoresearch continuation

