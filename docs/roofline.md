# FISTA roofline sketch (analytical)

This file replaces Metal hardware counters in MVP. For each kernel variant
the synthesis loop will report `(measured_time, roofline_time, utilization%)`
where `roofline_time = bytes_moved / peak_bandwidth`. The proposer reads this
to know whether it is bandwidth-bound or has slack.

## Per-iter cost of FISTA on LASSO

Problem: $\min_x \tfrac12\|Ax-b\|^2 + \lambda\|x\|_1$, A is `(m, n)`.

Per iteration:
- $g = A^\top(Ay - b)$: one matvec $Ay$ ($mn$ FMAs), one axpy ($m$), one $A^\top r$ ($mn$ FMAs).
- $x_{k+1} = S_{\lambda t}(y - tg)$: $n$ elementwise ops.
- Momentum: $y_{k+1} = x_{k+1} + \beta(x_{k+1} - x_k)$: $n$ elementwise ops.

### FLOPs per iter

$$
\text{FLOPs} \approx 4mn + O(m+n)
$$

(matvec costs 2 FLOPs per FMA, two matvecs total).

### Bytes moved per iter (dense, fp32)

- `A` is read **twice** (once for $Ay$, once for $A^\top r$): $2 \cdot mn \cdot 4$ bytes.
- Vectors $y, x, x_\text{prev}, g, b, r$ are $O(m{+}n)$, total $\le 32(m+n)$ bytes.

$$
\text{bytes}_\text{fp32} \approx 8mn + 32(m+n)
$$

### Arithmetic intensity (fp32)

$$
\text{AI}_\text{fp32} = \frac{4mn}{8mn} = 0.5 \text{ ops/byte}
$$

This is **bandwidth-bound** on every Apple GPU. Bandwidth utilization is the
single most important kernel metric. Peak FLOPS is irrelevant.

### fp16 / bf16 storage

If $A$ is stored fp16 but accumulated fp32:
- Bytes: $4mn + 32(m+n)$.
- AI: $\approx 1.0$ ops/byte.
- Still bandwidth-bound, but each iter moves half the bytes ⇒ theoretical 2× speedup.
- Convergence is preserved iff the matvec accumulator is fp32 (true for `mx.matmul`).

This is the precision axis the synth loop will explore.

## Roofline numbers for the bench-suite shapes

Calibrated against the actual dev hardware: **MacBook Pro M3 Pro, 150 GB/s
theoretical peak**. Numbers are lower bounds at 100% utilization.

| `(m, n)`         | bytes/iter (fp32) | $T_\text{floor}$ fp32 (ms) | $T_\text{floor}$ fp16 (ms) | measured `mx.matmul` fp32 (ms) | utilization |
|------------------|-------------------|------------------------------|------------------------------|---------------------------------|-------------|
| (2000, 500)      | 8 MB              | 0.053                        | 0.027                        | 0.331 / 0.183 (cold/warm)       | 16% / 29%   |
| (5000, 2000)     | 80 MB             | 0.533                        | 0.267                        | 0.786 / 0.747                   | 68% / 71%   |
| (500, 2000)      | 8 MB              | 0.053                        | 0.027                        | 0.169                           | 32%         |
| (2000, 10000)    | 160 MB            | 1.067                        | 0.533                        | 1.417                           | **75%**     |

**Two regimes apparent from the measurements:**
- **Large shapes**: `mx.matmul` is near the bandwidth wall. Synthesis can only win via precision (fp16 already gives 42% reduction on the (5000,2000) shape).
- **Small shapes**: launch overhead dominates, BW utilization is below 30%. Synthesis wins via kernel fusion (fewer launches per iter).

If FISTA converges in 100–300 iters, total wall-time floor is order 5–320 ms
on this Mac.

The synth loop's $T_\epsilon$ axis is exactly this number, measured. The
roofline ratio `measured / floor` is what the proposer reads to decide
whether further tuning is worthwhile.

## Sparse case (rcv1.binary, ~0.2% nonzero)

- `A` is CSR, ~$2 \cdot \text{nnz} \cdot 4$ bytes (data + col-idx) plus row-pointer $4(m{+}1)$.
- For `news20`-shape (m=20K, n=47K, nnz≈4M), $A$ is ~32 MB ⇒ fits in cache; bandwidth concerns shift to gather efficiency, not raw bytes.
- Sparse matvec is harder to roofline analytically; we will fall back to
  `(measured / dense-equivalent floor)` and accept the imprecision.

## How the synth loop uses this

`synth/roofline.py` exposes:

```python
estimate_dense_fista_roofline(
    m=m,
    n=n,
    dtype_name="fp32",
    wall_time_ms=wall_time_ms,
    n_iters=n_iters,
    peak_bandwidth_gb_s=150.0,
)
```

Tier-3 per-shape records now include:

```
bytes_per_iter
flops_per_iter
arithmetic_intensity
roofline_floor_ms_per_iter
measured_ms_per_iter
achieved_bandwidth_gb_s
roofline_pct_med = roofline_floor_ms_per_iter / measured_ms_per_iter * 100
```

A variant at >80% is near the wall; further mutation is unlikely to help.
A variant at <40% has headroom; the proposer should iterate.

---

# Gram-path cost model

The section above models the **direct** gradient `g = A^T(Ay - b)`, which reads
the `(m, n)` matrix `A` **twice** per iteration. The current `tall_medium`
champion does not use that path: it precomputes `G = A^T A` (`(n, n)`) and
`c = A^T b` once, then each iteration is a single dense matvec `g = G y - c`.
The direct roofline therefore *mis-describes the champion*. `synth/roofline.py`
exposes a Gram model (`gram_fista_*`, `estimate_gram_fista_roofline`) and a
strategy dispatcher (`estimate_fista_roofline(..., gradient_strategy=...)`);
`EvalConfig.gradient_strategy` selects it so Tier-3 reports Gram-path numbers
for Gram champions.

## Per-iter cost (Gram, dense, fp32)

The hot path is one matvec over `G`:

$$
\text{FLOPs} \approx 2n^2, \qquad
\text{bytes}_\text{fp32} \approx 4n^2 + 32n
$$

$$
\text{AI}_\text{fp32} = \frac{2n^2}{4n^2} = 0.5 \text{ ops/byte}
$$

Still bandwidth-bound — but the *quantity* of traffic is what changes. For
`tall_medium` `(m, n) = (5000, 2000)`:

| path | matrix read / iter | bytes / iter | floor @150 GB/s |
|---|---|---:|---:|
| direct | `A` twice = `2mn` | 80 MB | 0.533 ms |
| Gram (dense) | `G` once = `n²` | 16 MB | 0.107 ms |
| Gram (symmetric) | lower-tri `G` = `n²/2` | 8 MB | 0.053 ms |

So for tall problems (`m ≫ n`) Gram moves **~5× fewer bytes per iteration** than
direct — the analytical reason the measured Gram solve (20.6 ms) beats direct
(53.6 ms) on this shape. The model predicts the regime, not the exact constant
(measured ratio 2.6× vs modeled 5× — MLX GEMV efficiency and launch overhead
eat the rest, which is itself the next thing to attack).

## Kernel levers the Gram model exposes

- **Symmetric matvec (`symv`).** `G = A^T A` is symmetric PSD, so a triangular
  kernel reads only the lower triangle and roughly halves per-iter bandwidth.
  Because the Gram solve is bandwidth-bound, this is close to a 2× lever on the
  solve. `gram_fista_bytes_per_iter(..., symmetric=True)` models it; MLX has no
  `symv`, so this is a candidate custom Metal kernel.
- **Amortize the KKT check.** FISTA here evaluates `kkt_residual` every
  iteration, which for the Gram path is a *second* `n²` matvec — it doubles
  per-iter traffic. Checking KKT every `k` iterations (k = 4–8) nearly halves
  Gram per-iter bytes with no effect on the optimum. Pure data-transfer /
  fewer-matvec win.
- **Gram storage precision.** `G` in bf16/fp16 halves bytes again, but
  `mixed_gram` currently fails the KKT contract (`G = A^TA` squares the
  condition number). The lever is real but needs error-feedback / fp32
  accumulation, not naive fp16 storage.

## Setup-amortization crossover

Gram is not free: forming `G = A^T A` costs `2mn²` FLOPs once
(`gram_setup_flops`). With `A` fixed across repeated solves, Gram pays setup
once and saves `Δ = direct_solve − gram_solve` per solve, so it wins after

$$
N^* = \frac{\text{setup}}{\text{direct\_solve} - \text{gram\_solve}}
$$

solves (`amortization_crossover`). For `tall_medium`
(setup ≈ 3414 ms, direct solve ≈ 53.6 ms, Gram solve ≈ 20.6 ms):

$$
N^* = \frac{3414}{53.6 - 20.6} \approx 103 \text{ solves}.
$$

This is exactly the `single` vs `amortized` cost-model split: the `single` gate
charges setup, so Gram loses below ~100 solves; the `amortized` gate excludes
setup, so Gram wins immediately. If Gram's per-solve time is not faster, setup
can never be amortized and the crossover is `inf`.
