# Synthesis state schema

The synthesis loop persists everything to disk in three index files plus a
per-run artifact directory. Both `synth_state/` and `runs/` are gitignored.

## Files

```
synth_state/
├── lineage.jsonl                      # append-only, one line per proposal
├── edits.json                         # cached per-edit-type success priors
└── champions/
    ├── index.json                     # {slot_key_str: lineage_id}
    └── <problem>/<algo>/<hw>/<dtype>/
        ├── champion.py                # symlink to runs/<id>/source.py
        ├── metadata.json              # {id, accepted_at, source_hash, summary}
        └── pareto.jsonl               # full Pareto-front history for this slot

runs/<lineage_id>/
├── source.py                          # the kernel as proposed
├── compile.log
├── started.json                       # WAL marker (see Checkpointing below)
├── tier1.json
├── tier2.json
└── tier3.json
```

`champion.py` is a **symlink** so promoting a new champion is atomic via
rename-of-symlink-target.

## Slot

The unit a champion is "best for":

```python
SlotKey = (problem_family, algorithm, hardware, dtype)
# example: ("lasso", "fista", "m3_pro", "fp16_storage_fp32_accum")
```

Different dtype strategies co-exist as separate implementation slots for
reporting and reproducibility. The user-level target is
`problem_family + algorithm + hardware + accuracy contract`; dtype is a search
dimension whose candidates must satisfy the same KKT gate.

## Lineage record (one JSONL line)

```jsonc
{
  "id": "uuid4",
  "parent_id": "uuid4 | null",
  "generation": 7,
  "created_at": "ISO-8601",
  "evaluated_at": "ISO-8601",

  "slot": {
    "problem_family": "lasso",
    "algorithm": "fista",
    "hardware": "m3_pro",
    "dtype": "fp16_storage_fp32_accum"
  },

  "edit": {
    "type": "tile_change",
    "payload": {"from": 128, "to": 256},
    "rationale": "free-text proposer reasoning",
    "proposer_role": "impl",                       // algorithm | kernel | impl
    "proposer_model": "claude-opus-4-7",
    "source": "claude_subagent"                    // | manual | transfer:<src_slot_key>
  },

  "source": {
    "path": "runs/<id>/source.py",
    "hash": "sha256:...",
    "diff_from_parent": "..."                      // capped at 4 KB
  },

  "tier1": {                                       // smoke
    "passed": true,
    "reject_reason": null,                         // compile_failed | kkt_increased | nan | timeout | null
    "wall_time_ms": 12.3,                           // selected cost-model median if n_reps > 1
    "setup_time_ms": 4.1,                           // backend conversion + optional prepare_problem
    "solve_time_ms": 8.2,                           // timed FISTA solve only
    "single_solve_wall_time_ms": 12.3,              // setup + solve
    "amortized_wall_time_ms": 8.2,                  // solve only
    "cost_model": "single",                         // single | amortized | both
    "n_reps": 3,
    "wall_time_min_ms": 11.9,
    "wall_time_max_ms": 12.8,
    "wall_time_std_ms": 0.4
  },
  "tier2": {                                       // convergence on mid instance, present iff tier1.passed
    "passed": true,
    "converged": true,
    "n_iters": 142,
    "wall_time_ms": 87.5,                           // selected cost-model median if n_reps > 1
    "kkt_final": 4.2e-7,
    "setup_time_ms": 30.0,
    "solve_time_ms": 57.5,
    "single_solve_wall_time_ms": 87.5,
    "amortized_wall_time_ms": 57.5,
    "cost_model": "single",
    "kkt_trajectory_downsampled": [/* 50 log-spaced floats */],
    "primal_obj_first_last": [123.4, 45.6],
    "n_reps": 2,
    "wall_time_min_ms": 86.8,
    "wall_time_max_ms": 89.1,
    "wall_time_std_ms": 1.1,
    "speed_ref_wall_time_ms": 86.2,                 // paired champion ref if confirmed
    "speed_ref_margin": 0.97,
    "speed_ref_source": "paired"                    // startup | paired | ""
  },
  "tier3": {                                       // full bench, per shape, k reps; present iff tier2.passed
    "passed": true,
    "per_shape": [
      {"m": 2000, "n": 500, "n_iters_med": 89, "wall_time_med_ms": 12.3,
       "kkt_final_med": 5.1e-7, "roofline_pct_med": 42.0, "peak_mem_mb": 28,
       "setup_time_med_ms": 4.1, "solve_time_med_ms": 8.2,
       "single_solve_wall_time_med_ms": 12.3, "amortized_wall_time_med_ms": 8.2,
       "cost_model": "single",
       "bytes_per_iter": 8080000, "flops_per_iter": 4006000,
       "arithmetic_intensity": 0.496,
       "roofline_floor_ms_per_iter": 0.0539,
       "measured_ms_per_iter": 0.128,
       "achieved_bandwidth_gb_s": 63.1}
    ],
    "rank_summary": {"median_n_iters": 110, "median_wall_time_ms": 45.0,
                     "median_roofline_pct": 42.0,
                     "peak_bandwidth_gb_s": 150.0}
  },

  "decision": {
    "accepted": true,
    "pareto_dominates": ["uuid4", "uuid4"],
    "champion_for_slot": true,
    "reason": "pareto_dominant"                    // | keep:tier1_speedup | keep:tier2_passed | keep:tier3_passed | tier_failed:2 | tier_failed:2_speed | tier_failed:3 | discard:not_faster_than_baseline | not_dominant
  }
}
```

Aggregations should always check `tierK.passed` before reading later tiers.
A failing tier omits all later tier objects entirely.

`kkt_trajectory_downsampled`: 50 log-spaced samples — enough to detect smooth
decrease, late divergence, or early stall. Bump to 100 if proposer prompt
quality requires.

## Edit priors (`synth_state/edits.json`)

Lazily regenerated from `lineage.jsonl` on synth-loop startup. The proposer
reads this to weight mutation choices.

```json
{
  "version": 1,
  "global": {
    "tile_change": {
      "n_proposed": 128,
      "n_accepted": 23,
      "n_tier1_passed": 90,
      "n_tier2_passed": 30,
      "n_tier2_speed_failed": 7,
      "n_invalid": 12,
      "n_crashed": 2,
      "accept_rate": 0.180,
      "tier1_pass_rate": 0.703,
      "tier2_pass_rate": 0.234,
      "median_tier1_wall_ms_when_accepted": 12.3,
      "median_tier2_wall_ms_when_accepted": 87.5
    }
  },
  "by_slot": {
    "lasso/fista/apple_silicon/fp32": {
      "tile_change": {
        "n_proposed": 95,
        "n_accepted": 20,
        "accept_rate": 0.210
      }
    },
    "lasso/fista/apple_silicon/fp16": {}
  }
}
```

## Cross-problem transfer

`synth/lineage.py::seed_from_neighbors(records, slot, k=3)`:

1. Find neighbor slots: same problem-family with different `(dtype | hardware)`,
   or same algorithm with different problem-family.
2. Filter accepted records in those slots.
3. Rank by fitness improvement (post-edit `T_ε` / pre-edit `T_ε`).
4. Return top-k edits, tagged `edit.source = "transfer:<src_slot_key>"`.

Transferred edits are evaluated like any other proposal; their fate is
recorded in their own lineage records, which lets us measure transfer
effectiveness directly.

## Checkpointing (write-ahead log)

Every evaluation:

1. Create `runs/<id>/`, write `started.json` first (problem slot, edit, source path).
2. Run tier 1 → tier 2 → tier 3, writing `tierK.json` as each completes.
3. Append the final lineage record to `lineage.jsonl` (single atomic append).
4. If accepted: update `synth_state/champions/index.json` and rename the
   `champion.py` symlink (both atomic).

On synth-loop startup, scan `runs/` for any directory with `started.json` but
no matching record in `lineage.jsonl` — those are crashed evaluations.
Default policy: log them with `decision.reason = "crashed"` and don't requeue.
Policy is a knob in `synth/checkpoint.py`.

## Format choice

JSONL for `lineage.jsonl` (append-only, polars-readable), JSON for indexes
(atomically rewritten under flock), plain files for sources and per-tier
logs. SQLite/DuckDB is a future migration only if querying becomes slow; the
schema is relational-friendly so the migration is mechanical.
