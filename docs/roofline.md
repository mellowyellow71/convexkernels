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
