# algopool — measured behaviour (numpy, this box)

Companion to `algopool_bench.{npz,json}`. All numbers are the trusted KKT
(`kkt_residual_max`) on the canonical numpy `LassoPath`, tol = 1e-6, measured on
CPU (no MLX / no CUDA on this Linux box). Reproduce with
`scripts/run_algopool_bench.py`.

## Tractable path shapes (committed in algopool_bench.json)

| shape | m×n×K | family | reaches 1e-6 | time-to-KKT |
|---|---|---|---|---|
| path_tall_medium | 5000×2000×50 | fista_path_baseline | yes | 0.38 s |
| path_tall_medium | 5000×2000×50 | path_fista_screen | yes | 0.65 s |
| path_tall_medium | 5000×2000×50 | path_cd_screen | yes | 2.69 s |
| path_wide_small | 500×2000×50 | fista_path_baseline | yes | 2.15 s |
| path_wide_small | 500×2000×50 | path_fista_screen | yes | 4.84 s |
| path_wide_small | 500×2000×50 | path_cd_screen | yes | 4.96 s |

On tall/square/small-wide shapes (n = 2000) plain FISTA-path is *fastest* — the
support is a large fraction of n, so screening removes little and its per-sweep
full gemv is pure overhead. Screening is not worth it here, and the bench says
so. This is the regime the standard gate already handles.

## Hero shape — path_wide_hero (m=1000, n=50000, K=50)

| family | best KKT | at time | reached 1e-6 |
|---|---|---|---|
| fista_path_baseline | 4.13e-05 | 100 s (cap) | **no — plateaus** |
| path_fista_screen | 2.26e-03 | 100 s (cap) | **no** — union working set ≈ full active set |
| path_cd_screen | per-column 1e-6 | col40 @ 194 s | full path impractical in numpy |

Two findings, both measured (not extrapolated):

1. **Plain FISTA-path cannot reach the deep-tight region on the wide path.** It
   plateaus around 4e-5 after 100 s; the gate needs 1e-6. This is the structural
   reason the family search points at coordinate descent / screening, not FISTA
   tuning, on p ≫ n.

2. **Screened CD reaches 1e-6 per column, but the full-path homotopy is
   impractical in pure numpy.** Each column is solved to 1e-6, but the
   small-lambda tail columns carry working sets of several hundred features, and
   cyclic CD over them is a Python coordinate loop — cols 31→40 alone took 150 s.
   The max-over-columns gate is not met until every column finishes, so the full
   n = 50000 path does not complete under a practical budget on CPU.

The hero specimen's declared hardware is Apple-Silicon MLX (see
`synth/program.md`), where the CD inner loop maps to a compiled/GPU kernel —
that is where a screening CD pool is meant to run, and where these families are
the natural seed. On CPU/numpy the contribution is: (a) the path-native families
reach the gate on tractable path shapes, and (b) the evidence above that FISTA
alone does not close the wide-path gap. No hero speedup is claimed on numpy.
