# MLX capability probe (run on a Mac with Apple Silicon)

Single self-contained probe script. Run on a Mac, paste the output into
`tasks/results.md` under `P0 → MLX probe`. The Linux dev box can do everything
else; these probes need real Metal hardware.

## Setup + run

```bash
cd /path/to/convexkernels
uv sync --extra mac --extra dev
python docs/probes/mac_probe.py | tee tasks/mac_probe_output.txt
```

If MLX install fails see https://ml-explore.github.io/mlx/build/html/install.html.

The script runs four probes:

| # | What it checks | Why we care |
|---|----------------|-------------|
| 1 | `mx.fast.metal_kernel` compiles and runs | Whole synthesis loop bets on this API. |
| 2 | `mlx.core.linalg` surface (`cholesky`, `solve_triangular`, …) | Tells us if ADMM (P5) needs a custom trisolve kernel. |
| 3 | `mx.matmul` matvec timing on the bench-suite shapes (fp32) | Floor we synthesize against; calibrates `docs/roofline.md`. |
| 4 | Same timing across fp32 / fp16 / bf16 | Precision axis dry run; quantifies the headroom for P4 mutations. |

## What to paste back

The full stdout of `mac_probe.py`. Once pasted, P0 is complete and P1 (NumPy
reference FISTA) starts.
