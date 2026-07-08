"""Batched FISTA-Gram path seed — torch/CUDA (Blackwell-class GPUs).

Torch analog of `kernels/mlx/seeds/gram_fista_path_v0.py`, the first GPU seed
for the NVIDIA side of the search. On the LASSO path the per-iter work is a
single ``G @ Y`` GEMM (n×n by n×K), which is throughput-bound on a 5090 — so it
is where reduced precision can buy wall-clock (measured bf16/fp16 ≈ 2.7–3.7×
over fp32 on the path GEMM, vs the single-vector GEMV which is launch-bound and
gets *slower* in low precision).

Two coupled facts shape the design:

  1. **The iterate state is kept in fp64.** The trusted gate is a *scale-free*
     KKT < 1e-6. The residual is ``L·X − soft(L·X − G@X + c)``; with a Gram
     spectral norm ``L`` of order 1e2, evaluating that in float32 has absolute
     error ``~L·|X|·ε_fp32`` that normalizes to ~1e-6 — i.e. a pure-fp32 solver
     *floors at the gate* and cannot reliably pass it on moderately-conditioned
     problems. Keeping X/Y/θ and the prox/restart tail in fp64 removes that
     floor; the cost is trivial (an (n,K) elementwise tail).

  2. **Only the heavy ``G@Y`` GEMM carries the precision knob** (``dtype_strategy``):
       - ``fp64``            : fp64 GEMM every iter — the reference that always
                               reaches the gate (but pays full fp64 throughput,
                               which is 1/64 rate on a consumer 5090).
       - ``{fp32,bf16,fp16}``: pure low-precision GEMM — biased, so it *floors*
                               above the gate (kept for measuring the floor).
       - ``{fp32,bf16,fp16}_switch`` : **iterative refinement** — run the cheap
                               low-precision GEMM for the bulk of the descent
                               until its KKT plateaus (detected on-device), then
                               switch to the fp64 GEMM for the endgame. Reaches
                               the exact gate while paying low precision on the
                               majority of iterations; the win grows with the
                               iteration count (ill-conditioned / low-reg paths).

A periodic *stale* SVRG-style correction does **not** work for this deterministic
``E = G_low − G`` bias — the staleness ``E@(anchor − Y)`` stays proportional to
the remaining distance-to-solution, giving the same relative error as no
correction. Iterative refinement is the correct structure.

Correctness: the harness recomputes the *trusted* fp64 KKT on the returned
iterate, so an over-aggressive precision choice is simply rejected. The solver
keeps a cheap fp32 on-device KKT (``view.kkt_device``, matching the gate's
prox-fixed-point formula) only to *detect the plateau* that triggers the switch.

`solve` returns a host (CPU) array — the harness's final ``_materialize`` does
``np.asarray``, which cannot consume a CUDA tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # torch is optional at import time (CPU-only CI without the dep)
    import torch
except Exception:  # pragma: no cover - exercised only where torch is absent
    torch = None  # type: ignore


# dtype_strategy -> (low-precision GEMM dtype, mode). The endgame/exact GEMM is
# always fp64 (the only precision that reaches the scale-free 1e-6 gate).
#   "exact" : fp64 GEMM every iter (reference).
#   "low"   : low-precision GEMM every iter (floors above the gate; diagnostic).
#   "switch": low-precision bulk, then fp64 endgame (iterative refinement).
_STRATEGIES = {
    "fp64": ("float64", "exact"),
    "fp32": ("float32", "low"),
    "bf16": ("bfloat16", "low"),
    "fp16": ("float16", "low"),
    "fp32_switch": ("float32", "switch"),
    "bf16_switch": ("bfloat16", "switch"),
    "fp16_switch": ("float16", "switch"),
}

# Switch trigger: low-precision KKT improved by < (1 - _STALL) over a check
# interval => it has plateaued at its precision floor; hand off to fp64.
_STALL = 0.10


def _resolve_device(config: dict) -> "torch.device":
    want = (config or {}).get("device", "auto")
    if want not in ("auto", None):
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _GramPathView:
    """GPU-resident Gram path problem + precision policy (one-time setup).

    State is fp64; ``G64`` drives the exact/endgame GEMM, ``Glow`` the cheap
    bulk GEMM, ``G32`` the on-device plateau detector.
    """

    G64: "torch.Tensor"             # (n, n) fp64 Gram, G = A^T A
    G32: "torch.Tensor"             # (n, n) fp32 Gram (plateau detector)
    Glow: Optional["torch.Tensor"]  # (n, n) low-precision Gram (None if exact)
    c: "torch.Tensor"               # (n, 1) fp64, c = A^T b
    lambdas: "torch.Tensor"         # (1, K) fp64
    L: float
    lambda_max: float
    n: int
    K: int
    device: "torch.device"
    low_dtype: "torch.dtype"
    mode: str

    # --- gradients (return fp64) ---
    def grad_low(self, Y: "torch.Tensor") -> "torch.Tensor":
        return (self.Glow @ Y.to(self.low_dtype)).to(torch.float64)

    def grad_exact(self, Y: "torch.Tensor") -> "torch.Tensor":
        return self.G64 @ Y

    # --- cheap fp32 on-device KKT (matches the trusted prox-fixed-point formula) ---
    def kkt_device(self, X: "torch.Tensor") -> "torch.Tensor":
        Xf = X.to(torch.float32)
        g = self.G32 @ Xf - self.c.to(torch.float32)
        z = self.L * Xf - g
        lam = self.lambdas.to(torch.float32)
        soft = torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)
        r = self.L * Xf - soft
        r_inf = torch.amax(torch.abs(r), dim=0)
        denom = lam.reshape(-1) + self.lambda_max
        return torch.amax(torch.where(denom > 0, r_inf / denom, r_inf))


def prepare_problem(problem, config: Optional[dict] = None) -> _GramPathView:
    """One-time setup (timed): upload A, build the Gram on-device, set precision."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the CUDA path seed")
    config = config or {}
    strat = str(config.get("dtype_strategy", "fp64"))
    dtype_name, mode = _STRATEGIES.get(strat, ("float64", "exact"))
    low_dtype = getattr(torch, dtype_name)
    device = _resolve_device(config)

    A = torch.as_tensor(np.asarray(problem.A), dtype=torch.float64, device=device)
    b = torch.as_tensor(np.asarray(problem.b), dtype=torch.float64, device=device)
    lambdas = torch.as_tensor(np.asarray(problem.lambdas), dtype=torch.float64, device=device)
    G64 = A.T @ A                                    # (n, n) fp64 on device
    c = (A.T @ b).reshape(-1, 1)
    G32 = G64.to(torch.float32)
    Glow = None if mode == "exact" else G64.to(low_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return _GramPathView(
        G64=G64, G32=G32, Glow=Glow, c=c, lambdas=lambdas.reshape(1, -1),
        L=float(problem.L), lambda_max=float(problem.lambda_max),
        n=int(problem.n), K=int(problem.K), device=device,
        low_dtype=low_dtype, mode=mode,
    )


def solve(view: _GramPathView, recorder, *, kkt_tol, max_time_s, check_every: int = 25):
    """Batched FISTA-Gram on GPU (fp64 state) with per-column gradient restart.

    Per iter: one ``G@Y`` GEMM (configured precision) + an elementwise (n,K)
    fp64 tail. Per-column O'Donoghue–Candès restart (each lambda runs its own
    schedule). For ``*_switch`` strategies, the cheap on-device KKT detects when
    the low-precision GEMM has plateaued and hands the endgame to the fp64 GEMM.
    Progress is recorded on the trusted host recorder every `check_every` iters.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the CUDA path seed")
    n, K = view.n, view.K
    dev = view.device
    f64 = torch.float64
    t = 1.0 / view.L
    X = torch.zeros((n, K), dtype=f64, device=dev)
    Y = X.clone()
    theta = torch.ones((K,), dtype=f64, device=dev)
    kappa = t * view.lambdas                            # (1, K) fp64

    use_low = view.mode in ("low", "switch")
    prev_kkt = float("inf")

    it = 0
    while it < 100000:
        it += 1
        g = (view.grad_low(Y) if use_low else view.grad_exact(Y)) - view.c
        Z = Y - t * g
        Xn = torch.sign(Z) * torch.clamp(torch.abs(Z) - kappa, min=0.0)
        diff_x = Xn - X
        restart = torch.sum((Y - Xn) * diff_x, dim=0) > 0.0   # (K,)
        theta_adv = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * theta * theta))
        theta_new = torch.where(restart, torch.ones_like(theta), theta_adv)
        mom = torch.where(restart, torch.zeros_like(theta), (theta - 1.0) / theta_new)
        Y = Xn + mom.reshape(1, -1) * diff_x
        X = Xn
        theta = theta_new
        if it % check_every == 0:
            kdev = float(view.kkt_device(X))            # cheap fp32 plateau probe
            if view.mode == "switch" and use_low and kdev > prev_kkt * (1.0 - _STALL):
                use_low = False                          # low precision plateaued
            prev_kkt = kdev
            recorder.record(X.detach().to("cpu").numpy())
            if recorder.should_stop(kkt_tol):
                break
    return X.detach().to("cpu").numpy()


__all__ = ["prepare_problem", "solve", "_GramPathView"]
