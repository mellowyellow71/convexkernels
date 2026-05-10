# convexkernels autoresearch context

This document is the long-form context pack for a fresh agent. It explains why
the project exists, how it should be understood relative to Karpathy-style
`autoresearch`, what has actually been implemented, what the current champion
means, what is still wrong with the loop, and exactly where to continue.

Read this after `tasks/handoff.md` and before making design changes.

---

## 1. Project Vision

The project is a KKT-gated autoresearch system for convex optimization kernels.
The end goal is not "a fast LASSO solver". The end goal is a harness where a
user specifies:

- convex problem type
- targeted algorithm
- hardware backend
- dtype/precision contract
- workload contract
- accuracy contract

and an agentic loop synthesizes a specialized accelerator solver/kernel stack
for that concrete setting.

The mental model is:

> CVXGEN for accelerator kernels, but instead of a hand-written code generator
> only for a narrow class of QPs/LPs, an LLM-driven autoresearch loop proposes
> algorithm/kernel/precision/layout changes and keeps only candidates that pass
> convex optimality checks and improve measured performance.

The initial specimen is:

```
problem:   LASSO
algorithm: FISTA
hardware:  Apple Silicon / MLX / Metal
dtype:     fp32, with dtype strategy as a search dimension
```

The broader trajectory includes:

- CUDA kernels
- Triton kernels
- NVIDIA Blackwell
- NVFP4 or related low-precision formats
- additional convex problem families
- additional algorithms such as ADMM
- transfer of edit priors across problem/algorithm/hardware slots

The point of the project is the synthesis harness, not any one solver.

---

## 2. Why KKT Makes This Feasible

For convex problems, KKT residuals give an oracle-free correctness and
convergence signal. For LASSO, the residual can be computed from the gradient
and prox/stationarity conditions. This means a generated kernel can be judged
without comparing against a ground-truth solver on every proposal.

The loop can therefore use:

```
candidate is valid iff:
    evaluation completed
    KKT_final < tolerance
    no NaN / crash / timeout
    optional convergence flag true
```

Then performance gates apply only to valid candidates.

This is the main reason the project is plausible. It gives the agent a tight
feedback signal that is cheaper and more robust than full numerical comparison
against CVXPY/adelie on every experiment.

Important distinction:

- KKT is the correctness contract.
- Wall time / solve time / setup time are performance objectives.
- Iterate drift is diagnostic, not the primary gate.

---

## 3. Karpathy Autoresearch, Distilled

Karpathy's `autoresearch` is a deliberately small loop:

- `prepare.py` is fixed and contains data/evaluation/runtime utilities.
- `train.py` is the single editable file.
- `program.md` is the human-written instruction layer.
- every experiment runs for a fixed 5-minute wall-clock training budget,
  excluding startup/compilation.
- the scalar metric is `val_bpb`, lower is better.
- the agent commits a change, runs the experiment, logs results, and keeps the
  commit only if the metric improves.
- if the metric does not improve, the branch is reset back.
- the point is to run many comparable experiments overnight.

Primary source:

- `https://github.com/karpathy/autoresearch`
- `https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md`

The important lesson is not the exact LLM training task. The important lesson
is the shape of the experiment game:

```
fixed editable surface
fixed evaluator
fixed budget
one scalar metric
explicit keep/discard rule
append-only experiment log
no human in the loop once started
```

The current `convexkernels` loop is inspired by this, but it is still too broad
and too much like a platform. To become a strong autoresearch loop, each run
needs to be narrowed to a concrete experimental game.

---

## 4. Our Analogy To Karpathy's Loop

The analogous mapping should be:

| Karpathy autoresearch | convexkernels autoresearch |
|---|---|
| `prepare.py` fixed data/eval | problem generator + sandbox + KKT evaluator fixed |
| `train.py` editable file | kernel source or structured edit target |
| `program.md` instruction layer | `tasks/program_kernel.md` / prompt program |
| fixed 5-minute budget | fixed Mac eval command, timeout, reps, shape |
| `val_bpb` scalar metric | median solve ms or single-solve ms, gated by KKT |
| keep/reset git commit | promote/discard champion candidate |
| `results.tsv` | lineage JSONL + optionally compact TSV |
| one GPU | one Mac/MLX target for now |

For our current run, the game should be:

```
setting:
    lasso + fista + apple_silicon/mlx + fp32 + tall_medium

workload:
    amortized repeated solves with fixed A

metric:
    minimize Tier-2 median solve_time_ms

hard gate:
    KKT_final < 1e-6

current champion:
    Gram-precomputed FISTA, fp32

allowed next research direction:
    improve the Gram gradient path, not random tail-only edits
```

This is the clean equivalent of Karpathy's `val_bpb` game.

---

## 5. Current State At A Glance

Latest verified tests:

```
Linux full suite: 92 passed, 4 skipped
Mac targeted suite: 33 passed
```

Latest important state:

```
state root:
    synth_run_p4_strategy_tall_medium_amortized_20260509

slot:
    lasso / fista / apple_silicon / fp32

workload:
    amortized

selected champion:
    450c6b8b-49e7-4b57-9a2e-900b7708fbba

champion strategy:
    gradient_strategy = gram
    dtype_strategy = fp32

Tier-2 solve time:
    16.396 ms

paired direct reference in that run:
    48.465 ms

KKT:
    4.29e-07
```

This is a real result. It is not "optimal"; it is the current best discovered
candidate for one exact workload.

The champion is a Gram-precomputed FISTA variant. It precomputes:

```
G = A.T @ A
c = A.T @ b
```

and then uses:

```
g = G @ y - c
```

inside FISTA iterations.

Why this matters:

- Direct FISTA computes `A.T @ (A @ y - b)` every iteration.
- Gram FISTA has expensive setup but cheaper repeated iterations.
- Therefore Gram is good for amortized/repeated-solve workloads.
- Gram can lose for single-solve workloads because setup is charged.

---

## 6. Current Champion Is Workload-Specific

Do not say "we found the optimal kernel" without qualification.

Correct phrasing:

> We found a current amortized-solve champion for
> `lasso + fista + apple_silicon/mlx + fp32 + tall_medium`, using Gram
> precomputation. It improves stored Tier-2 solve time from roughly 48.5 ms to
> 16.4 ms under KKT < 1e-6.

Incorrect phrasing:

> We found the optimal LASSO kernel.

The result is not global. It is conditional on:

- shape: `tall_medium`
- backend: Apple Silicon / MLX
- dtype: fp32
- algorithm: FISTA restart
- workload: amortized repeated solves
- metric: solve time, not setup-inclusive wall time

For setup-inclusive `single` workloads, direct FISTA may remain better because
Gram precompute setup is expensive.

---

## 7. What We Implemented So Far

### Core solver and verification

- NumPy FISTA reference.
- LASSO frontend with KKT residual.
- Nonnegative LASSO frontend and KKT residual.
- `assert_equivalent()` KKT-gated functional equivalence.
- MLX-backed problem classes.
- MLX seed kernels:
  - direct LASSO FISTA tail kernel
  - nonnegative LASSO tail kernel
  - Gram FISTA seed via `prepare_problem()`

### Synth loop infrastructure

- subprocess sandbox
- timeout and Linux memory cap
- path-loaded kernel modules
- `prepare_problem(problem[, config])` hook
- setup/solve/single/amortized timing split
- tiered evaluator:
  - Tier-1 quick ratchet
  - Tier-2 full convergence
  - Tier-3 shape suite
- lineage JSONL
- edit priors
- structured fitness report
- champion store
- cross-problem transfer seed support
- orphan detection via write-ahead `started.json`

### Proposers

- deterministic stub proposer
- deterministic structured grid proposer
- OpenAI Responses API proposer
- JSON schema for full source or structured edits
- prompt includes runtime context, history, priors, fitness summaries

### Structured edit support

Implemented structured payloads:

- `threadgroup_size`
- `items_per_thread`
- `remove_bounds_check`
- `branchless_soft_threshold`
- `kernel_name_suffix`
- `gradient_strategy`
- `dtype_strategy`

Implemented hardening:

- problem/prox-aware LASSO vs NN-LASSO rewrites
- vectorization composition with branchless soft threshold
- unsafe vectorized/no-bounds combinations rejected before evaluation
- applier errors recorded in lineage
- duplicate-source detection

### Gradient/dtype strategy support

Implemented:

- `LassoGramMLX`
- `gram_fista_step_v0.py`
- config-backed strategy edits materialized as durable `source.py`
- generated `prepare_problem()` hooks for strategy edits
- `gradient_strategy=direct|gram`
- `dtype_strategy=fp32|fp16_storage|mixed_gram`
- KKT correctly rejects current mixed/low-precision attempts

### Workload-aware champion store

Important recent fix:

- new champions are stored under:

```
synth_state/champions/<problem>/<algo>/<hardware>/<dtype>/workloads/<cost_model>/champion.py
```

- legacy champion fallback is allowed only if metadata `summary.cost_model`
  matches the requested workload.
- if both legacy and workload-specific champions match, selection chooses the
  fastest recorded metric:
  - `tier2_wall_time_ms` if present
  - otherwise `tier1_wall_time_ms`

This prevents a noisy one-rep later promotion from displacing an older better
champion.

---

## 8. Key Results So Far

### P4.10 direct vs Gram probe

Mac probe, FISTA restart, `tol=1e-6`, `max_iters=5000`.

| shape | strategy | passed | KKT | iters | setup ms | solve ms | single ms |
|---|---|---|---:|---:|---:|---:|---:|
| `tall_small` | direct fp32 | yes | `7.18e-07` | 17 | `200.5` | `21.4` | `222.8` |
| `tall_small` | Gram fp32 | yes | `7.18e-07` | 17 | `172.7` | `16.1` | `189.7` |
| `tall_small` | Gram mixed | no | `1.52e-04` | 5000 | `171.4` | `1859.8` | `2027.6` |
| `tall_medium` | direct fp32 | yes | `4.29e-07` | 22 | `5327.6` | `53.6` | `5381.4` |
| `tall_medium` | Gram fp32 | yes | `4.29e-07` | 22 | `3414.7` | `20.6` | `3434.6` |
| `tall_medium` | Gram mixed | no | `4.13e-04` | 5000 | `4665.6` | `2991.2` | `7656.8` |

Interpretation:

- Gram fp32 is a real larger lever for tall dense amortized solves.
- Current mixed precision attempts fail KKT and are correctly rejected.

### P4.12 strict structured tall-medium search

Setup-inclusive `single` gate:

| candidate | result | KKT | setup ms | solve ms | single ms |
|---|---|---:|---:|---:|---:|
| direct baseline | startup | `4.29e-07` | `2107.4` | `41.6` | `2149.0` |
| Gram fp32 | rejected: single slower | `4.29e-07` | `2637.0` | `18.7` | `2655.7` |
| Gram mixed | rejected: KKT | `4.13e-04` | `2320.5` | `2750.0` | `5070.6` |
| fp16 storage | rejected: KKT | `1.17e-03` | `3038.0` | `4926.0` | `7964.0` |

Amortized gate:

| candidate | result | KKT | Tier-2 solve ms | paired ref ms |
|---|---|---:|---:|---:|
| direct baseline | startup | `4.29e-07` | `51.4` | - |
| Gram fp32 | promoted | `4.29e-07` | `16.4` | `48.5` |
| Gram mixed | rejected: KKT | `4.13e-04` | `2739.3` | - |
| fp16 storage after Gram | rejected: KKT | `4.93e-04` | `2724.5` | - |

### P4.13 OpenAI continuation

OpenAI from Gram champion produced:

| proposal | form | result | Tier-1 solve ms |
|---|---|---|---:|
| `fuse_op` | full source | rejected: slower | `18.3` |
| `vectorize` | structured tail edit | rejected: slower | `17.6` |
| `other` | structured Gram/no-bounds | rejected: slower | `19.1` |

The model mostly proposed tail edits. That is the core reason we need a
Karpathy-style narrower program and Gram-specific search surface.

### P4.14 validation after workload-aware selector

One short 2-proposal validation promoted a KKT-valid full-source `fuse_op` by a
narrow one-rep paired margin:

```
tier2 candidate: 19.54 ms
paired ref:      23.81 ms
```

But the older Gram champion record had:

```
tier2 stored: 16.40 ms
```

After the fastest-matching workload selector, direct store inspection chooses
the older faster champion:

```
single:    no champion selected from amortized state
amortized: id 450c6b8b-49e7-4b57-9a2e-900b7708fbba, tier2 16.396 ms
```

This proves the selector is protecting against noisy one-rep regressions.

---

## 9. What Is Wrong With The Current Loop

The current loop works, but it is not yet as sharp as Karpathy's autoresearch.

### 9.1 Too many objectives are active at once

We track:

- setup time
- solve time
- single-solve wall time
- amortized solve time
- Tier-1 timing
- Tier-2 timing
- KKT
- iteration count
- shape
- dtype strategy
- gradient strategy

This is valuable telemetry, but an autoresearch run needs one active game.

For example:

```
Game A:
    minimize single_solve_wall_time_ms
    KKT < 1e-6
    target direct or setup-optimized Gram

Game B:
    minimize solve_time_ms
    KKT < 1e-6
    target Gram/repeated-solve path
```

These should not be mentally mixed.

### 9.2 Editable/search surface is too broad

Karpathy's loop edits one file. Our OpenAI proposer can emit arbitrary full
source and structured tail edits. After the Gram champion, random tail edits
are not the right lever.

For the current target, the search surface should be narrowed to:

- `LassoGramMLX.grad_smooth`
- `G @ y - c`
- Gram storage dtype/layout
- KKT dtype
- precompute/setup strategy if `single`
- KKT-safe precision if `amortized`

### 9.3 We do not yet have a `program.md` equivalent

Karpathy's `program.md` is the research organization code. It tells the agent
the exact loop, what can be edited, what cannot, how to log, what metric to
optimize, and when to keep/discard.

We need the same. A draft exists at:

```
tasks/program_kernel.md
```

That should be used as the instruction layer for the next autonomous run.

### 9.4 Timing noise still matters

One-rep Mac timing can promote narrow false positives. Mitigations already
added:

- paired Tier-2 remeasurement
- workload-aware champion selection
- fastest stored matching champion selector

Still needed for serious runs:

- use `tier1_reps=3` or `5`
- use `tier2_reps=3` or `5`
- require meaningful margin
- treat `1-5%` changes with suspicion unless repeated

### 9.5 We lack Gram-specific structured edit grammar

The current structured edit grammar mostly mutates the fused O(n) tail kernel.
For Gram FISTA, the hot path shifts to dense `G @ y - c`. We need structured
fields that represent this space directly.

Candidate fields:

- `gram_gradient_strategy`: `mlx_matvec`, `custom_metal_gemv`, `blocked_metal_gemv`
- `gram_storage_dtype`: `fp32`, `fp16`, `bf16` if available
- `gram_kkt_dtype`: `fp32`, `fp64-ish fallback if available`, or separate KKT path
- `gram_layout`: row-major, transposed, tiled
- `gram_precompute_strategy`: eager, cached, split dtype
- `gram_block_size`
- `gram_accumulator_dtype`

Do not assume these are all good. They are the structured search dimensions to
test.

---

## 10. Immediate Next Steps

### Step 1: make the current autoresearch game explicit

Create or use:

```
tasks/program_kernel.md
```

The game should be:

```
problem_family = lasso
algorithm = fista
hardware = apple_silicon
backend = mlx
dtype = fp32
shape = tall_medium
variant = restart
cost_model = amortized
metric = median Tier-2 solve_time_ms
gate = KKT_final < 1e-6
current champion = 450c6b8b-49e7-4b57-9a2e-900b7708fbba
```

### Step 2: confirm champion with stronger reps

Do not run long autonomous search until the baseline is confirmed with more
than one rep.

Recommended command:

```bash
ssh revantkasichainula@100.98.66.89 'cd convexkernels && \
.venv/bin/python -m convexkernels.synth.run \
  --problem-family lasso \
  --proposer structured \
  --n-proposals 0 \
  --shape tall_medium \
  --tier2-shape tall_medium \
  --state-root ./synth_run_p4_strategy_tall_medium_amortized_20260509 \
  --promotion-tier tier2 \
  --variant restart \
  --cost-model amortized \
  --tier1-tol 1e-6 \
  --tier1-max-iters 5000 \
  --tier1-reps 5 \
  --tier1-escalation-margin 1.0 \
  --tier2-tol 1e-6 \
  --tier2-max-iters 5000 \
  --tier2-reps 5 \
  --tier2-speed-margin 0.97 \
  --timeout-s 240'
```

Caveat: do not run multiple tall-medium baselines concurrently on the Mac. Two
concurrent sanity checks timed out previously.

### Step 3: implement first Gram-specific benchmark/edit surface

Before asking OpenAI for many proposals, make it possible to measure the
actual Gram bottleneck.

Recommended additions:

- a microbenchmark for `LassoGramMLX.grad_smooth` repeated many times
- a structured edit field for Gram gradient implementation
- one seed custom Metal GEMV candidate only if the benchmark suggests MLX GEMV
  overhead/bandwidth is the bottleneck

Potential file targets:

- `convexkernels/kernels/mlx/lib.py`
- `convexkernels/kernels/mlx/seeds/gram_fista_step_v0.py`
- `convexkernels/synth/applier.py`
- `convexkernels/synth/proposers/openai.py`
- `convexkernels/synth/prompts/impl_openai.md`
- `tests/test_kernels.py`
- `tests/test_synth_loop.py`

### Step 4: run a real autoresearch session

Once the game is sharp and the edit surface is Gram-specific:

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

This is closer to Karpathy's overnight loop than the 2-3 proposal probes.

---

## 11. Operational Details

### Machines

Linux dev box:

```
/home/ray/convexkernels
```

Mac MLX box:

```
revantkasichainula@100.98.66.89:convexkernels
```

### Sync command

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

### Test commands

Linux:

```bash
.venv/bin/python -m pytest
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

### OpenAI key

The Mac non-interactive SSH environment has been fixed:

- the key existed in `~/.zshrc`
- it was copied into `~/.zshenv`
- fresh SSH commands now see `OPENAI_API_KEY=set`

Do not print the key. Verify only with:

```bash
ssh revantkasichainula@100.98.66.89 'test -n "$OPENAI_API_KEY" && echo set || echo missing'
```

---

## 12. Current Runtime State

Important state roots on the Mac:

```
synth_run_p4_strategy_tall_medium_20260509
synth_run_p4_strategy_tall_medium_amortized_20260509
```

The amortized state contains the important Gram champion and later noisy
OpenAI continuation. With current code, selection should choose:

```
single:
    no champion from this amortized state

amortized:
    id 450c6b8b-49e7-4b57-9a2e-900b7708fbba
    tier2 16.396416118368506
```

Inspect with:

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

Expected output:

```
single source None metadata_cost None id None tier2 None
amortized source ... id 450c6b8b-49e7-4b57-9a2e-900b7708fbba tier2 16.396416118368506
```

---

## 13. Known Traps

1. Do not run MLX tests on Linux. They skip or fail; the real target is the
   Mac.

2. Do not run multiple `tall_medium` synth baselines concurrently on the Mac.
   Two concurrent sanity checks timed out.

3. Do not conflate `single` and `amortized` champions. They are different
   workload contracts.

4. Do not call the current champion "optimal". It is the best discovered
   candidate in the current limited search space.

5. Do not spend many OpenAI proposals before narrowing the search surface.
   The model has repeatedly proposed tail edits after Gram, which is usually
   not the right bottleneck.

6. Do not use NN-LASSO with Gram strategy yet. The current Gram hook is
   LASSO-specific.

7. Do not accept KKT failures for dtype experiments. Mixed precision is a
   search dimension, but KKT is the hard contract.

8. Do not rely on one-rep narrow wins. Use repeated timing for real claims.

9. Do not print API keys. The Mac key is available to SSH via `~/.zshenv`.

10. Do not make unrelated refactors in broad files unless needed. The repo is
    still accumulating research state and lineage semantics.

---

## 14. What "Done" Means For The Next Milestone

The next milestone is not "run more proposals". It is:

1. Convert the current target into a sharp Karpathy-style autoresearch game.
2. Add or draft the instruction layer (`tasks/program_kernel.md`).
3. Add a Gram-specific benchmark or structured edit dimension.
4. Run a multi-proposal Mac search with repeated timing.
5. Log whether any candidate beats the stored Gram champion under KKT < 1e-6.

Success looks like:

```
same target:
    lasso + fista + apple_silicon/mlx + fp32 + tall_medium + amortized

new champion:
    Tier-2 median solve time improves over 16.396 ms
    KKT < 1e-6
    repeated timing confirms the win
    lineage and champion metadata recorded
```

If no improvement is found, that is also useful. The loop should produce a
clear log of failed ideas and enough context to decide the next search
dimension.

