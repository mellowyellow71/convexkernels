"""MLX-backed problem and helpers.

`LassoMLX` mirrors the `Lasso` interface but stores arrays as `mx.array` and
implements ops with MLX. The synth loop uses this as the per-iter input to
MLX kernels in P3+. The `Lasso` numpy class remains canonical for the
correctness oracle (numpy_ref).
"""

from __future__ import annotations

from functools import cached_property

import mlx.core as mx
import numpy as np

from ...frontend.lasso import Lasso
from ...frontend.nonnegative_lasso import NonnegativeLasso


class LassoMLX:
    """MLX-backed view of a `Lasso` problem.

    A and b are converted to `mx.array` at the requested dtype. L and lambda_max
    are kept as Python floats (computed from the numpy original).

    Construct via `LassoMLX.from_lasso(prob, dtype=mx.float32)` to share the
    expensive `L` (Lipschitz constant) computation.
    """

    def __init__(self, A: mx.array, b: mx.array, lam: float,
                 L: float, lambda_max: float, dtype: mx.Dtype):
        self.A = A
        self.b = b
        self.lam = float(lam)
        self.dtype = dtype
        self._L = float(L)
        self._lambda_max = float(lambda_max)

    @classmethod
    def from_lasso(cls, lasso: Lasso, dtype: mx.Dtype = mx.float32) -> "LassoMLX":
        return cls(
            A=mx.array(lasso.A, dtype=dtype),
            b=mx.array(lasso.b, dtype=dtype),
            lam=lasso.lam,
            L=lasso.L,
            lambda_max=lasso.lambda_max,
            dtype=dtype,
        )

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def L(self) -> float:
        return self._L

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    def matvec(self, x: mx.array) -> mx.array:
        return self.A @ x

    def rmatvec(self, y: mx.array) -> mx.array:
        return self.A.T @ y

    def grad_smooth(self, x: mx.array) -> mx.array:
        return self.rmatvec(self.matvec(x) - self.b)

    def prox(self, v: mx.array, t: float) -> mx.array:
        kappa = t * self.lam
        return mx.sign(v) * mx.maximum(mx.abs(v) - kappa, 0.0)

    def kkt_residual(self, x: mx.array) -> float:
        """Same prox-residual formula as the numpy version, in MLX."""
        g = self.grad_smooth(x)
        z = self._L * x - g
        soft_z = mx.sign(z) * mx.maximum(mx.abs(z) - self.lam, 0.0)
        r = self._L * x - soft_z
        denom = self.lam + self._lambda_max
        if denom == 0.0:
            return float(mx.max(mx.abs(r)))
        return float(mx.max(mx.abs(r)) / denom)


class LassoGramMLX:
    """MLX-backed LASSO view with precomputed dense Gram data.

    This changes the FISTA gradient path from `A.T @ (A @ x - b)` to
    `(A.T @ A) @ x - A.T @ b`. It is useful for tall dense regimes when setup
    can be amortized, and it gives the synth loop a larger gradient-strategy
    search dimension than tail-kernel rewrites alone.
    """

    def __init__(
        self,
        G: mx.array,
        c: mx.array,
        lam: float,
        L: float,
        lambda_max: float,
        dtype: mx.Dtype,
        *,
        m: int,
        kkt_G: mx.array | None = None,
        kkt_c: mx.array | None = None,
        gradient_dtype: mx.Dtype | None = None,
    ):
        self.G = G
        self.c = c
        self.lam = float(lam)
        self.dtype = dtype
        self.gradient_dtype = gradient_dtype or dtype
        self._m = int(m)
        self._kkt_G = kkt_G if kkt_G is not None else G
        self._kkt_c = kkt_c if kkt_c is not None else c
        self._L = float(L)
        self._lambda_max = float(lambda_max)

    @classmethod
    def from_lasso_mlx(
        cls,
        problem: LassoMLX,
        *,
        gradient_dtype: mx.Dtype | None = None,
        kkt_dtype: mx.Dtype | None = mx.float32,
    ) -> "LassoGramMLX":
        gradient_dtype = gradient_dtype or problem.dtype
        A_grad = problem.A.astype(gradient_dtype)
        b_grad = problem.b.astype(gradient_dtype)
        G = A_grad.T @ A_grad
        c = A_grad.T @ b_grad

        kkt_G = None
        kkt_c = None
        if kkt_dtype is not None and kkt_dtype != gradient_dtype:
            A_kkt = problem.A.astype(kkt_dtype)
            b_kkt = problem.b.astype(kkt_dtype)
            kkt_G = A_kkt.T @ A_kkt
            kkt_c = A_kkt.T @ b_kkt

        mx.eval(G, c)
        if kkt_G is not None and kkt_c is not None:
            mx.eval(kkt_G, kkt_c)

        return cls(
            G=G,
            c=c,
            lam=problem.lam,
            L=problem.L,
            lambda_max=problem.lambda_max,
            dtype=problem.dtype,
            m=problem.m,
            kkt_G=kkt_G,
            kkt_c=kkt_c,
            gradient_dtype=gradient_dtype,
        )

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return int(self.G.shape[0])

    @property
    def L(self) -> float:
        return self._L

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    def grad_smooth(self, x: mx.array) -> mx.array:
        x_grad = x.astype(self.gradient_dtype) if x.dtype != self.gradient_dtype else x
        g = self.G @ x_grad - self.c
        return g.astype(self.dtype) if g.dtype != self.dtype else g

    def prox(self, v: mx.array, t: float) -> mx.array:
        kappa = t * self.lam
        return mx.sign(v) * mx.maximum(mx.abs(v) - kappa, 0.0)

    def kkt_residual(self, x: mx.array) -> float:
        x_kkt = x.astype(self._kkt_G.dtype) if x.dtype != self._kkt_G.dtype else x
        g = self._kkt_G @ x_kkt - self._kkt_c
        z = self._L * x_kkt - g
        soft_z = mx.sign(z) * mx.maximum(mx.abs(z) - self.lam, 0.0)
        r = self._L * x_kkt - soft_z
        denom = self.lam + self._lambda_max
        if denom == 0.0:
            return float(mx.max(mx.abs(r)))
        return float(mx.max(mx.abs(r)) / denom)


class NonnegativeLassoMLX:
    """MLX-backed view of `NonnegativeLasso`."""

    def __init__(self, A: mx.array, b: mx.array, lam: float,
                 L: float, lambda_max: float, normalizer: float, dtype: mx.Dtype):
        self.A = A
        self.b = b
        self.lam = float(lam)
        self.dtype = dtype
        self._L = float(L)
        self._lambda_max = float(lambda_max)
        self._normalizer = float(normalizer)

    @classmethod
    def from_problem(
        cls,
        problem: NonnegativeLasso,
        dtype: mx.Dtype = mx.float32,
    ) -> "NonnegativeLassoMLX":
        return cls(
            A=mx.array(problem.A, dtype=dtype),
            b=mx.array(problem.b, dtype=dtype),
            lam=problem.lam,
            L=problem.L,
            lambda_max=problem.lambda_max,
            normalizer=max(problem.lambda_max, float(np.max(np.abs(problem.A.T @ problem.b)))),
            dtype=dtype,
        )

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def L(self) -> float:
        return self._L

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    def matvec(self, x: mx.array) -> mx.array:
        return self.A @ x

    def rmatvec(self, y: mx.array) -> mx.array:
        return self.A.T @ y

    def grad_smooth(self, x: mx.array) -> mx.array:
        return self.rmatvec(self.matvec(x) - self.b)

    def prox(self, v: mx.array, t: float) -> mx.array:
        return mx.maximum(v - t * self.lam, 0.0)

    def kkt_residual(self, x: mx.array) -> float:
        g = self.grad_smooth(x)
        z = self._L * x - g
        prox_z = mx.maximum(z - self.lam, 0.0)
        r = self._L * x - prox_z
        denom = self.lam + self._normalizer
        if denom == 0.0:
            return float(mx.max(mx.abs(r)))
        return float(mx.max(mx.abs(r)) / denom)


class LassoPathMLX:
    """MLX-backed view of a `LassoPath` problem (batched lambdas).

    A, b on device. `lambdas` is an `mx.array` of shape (K,) holding the
    per-column thresholds. L and lambda_max are Python floats (path-independent;
    depend only on (A, b)).
    """

    def __init__(self, A: mx.array, b: mx.array, lambdas: mx.array,
                 L: float, lambda_max: float, dtype: mx.Dtype):
        self.A = A
        self.b = b
        self.lambdas = lambdas
        self.dtype = dtype
        self._L = float(L)
        self._lambda_max = float(lambda_max)

    @classmethod
    def from_lasso_path(cls, prob, dtype: mx.Dtype = mx.float32) -> "LassoPathMLX":
        from ...frontend.lasso_path import LassoPath  # noqa: F401
        return cls(
            A=mx.array(prob.A, dtype=dtype),
            b=mx.array(prob.b, dtype=dtype),
            lambdas=mx.array(prob.lambdas, dtype=dtype),
            L=prob.L,
            lambda_max=prob.lambda_max,
            dtype=dtype,
        )

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def K(self) -> int:
        return int(self.lambdas.shape[0])

    @property
    def L(self) -> float:
        return self._L

    @property
    def lambda_max(self) -> float:
        return self._lambda_max


class LassoPathGramMLX:
    """MLX-backed batched-lambda LASSO view with adaptive Gram-or-direct form.

    Two modes depending on whether `G = A^T A` fits in a single Metal
    allocation budget (Metal caps single buffers at ~9.66 GB on M3 Pro):

      **Gram mode** (n^2 * elem_size fits): precompute G, c on device.
        gradient: G @ Y - c[:, None]                 — one (n,n)x(n,K) matmul.
        Best for path_tall_medium, path_square, path_wide_small.

      **Direct mode** (G doesn't fit): keep A, b on device.
        gradient: A.T @ (A @ Y - b[:, None])          — two matmuls/iter.
        Required for path_wide_hero (n=50000 → G would be 10 GB fp32).

    Mode is decided at construction based on `gram_budget_bytes`.
    """

    def __init__(
        self,
        *,
        mode: str,                # "gram" | "direct"
        lambdas: mx.array,
        L: float,
        lambda_max: float,
        dtype: mx.Dtype,
        m: int,
        n: int,
        gradient_dtype: mx.Dtype | None = None,
        G: mx.array | None = None,
        c: mx.array | None = None,
        kkt_G: mx.array | None = None,
        kkt_c: mx.array | None = None,
        A: mx.array | None = None,
        b: mx.array | None = None,
        kkt_A: mx.array | None = None,
        kkt_b: mx.array | None = None,
    ):
        assert mode in ("gram", "direct"), f"unknown mode {mode!r}"
        if mode == "gram":
            assert G is not None and c is not None, "gram mode requires G, c"
        else:
            assert A is not None and b is not None, "direct mode requires A, b"
        self.mode = mode
        self.lambdas = lambdas
        self.dtype = dtype
        self.gradient_dtype = gradient_dtype or dtype
        self._m = int(m)
        self._n = int(n)
        self._L = float(L)
        self._lambda_max = float(lambda_max)
        self.G = G
        self.c = c
        self._kkt_G = kkt_G if kkt_G is not None else G
        self._kkt_c = kkt_c if kkt_c is not None else c
        self.A = A
        self.b = b
        self._kkt_A = kkt_A if kkt_A is not None else A
        self._kkt_b = kkt_b if kkt_b is not None else b

    @classmethod
    def from_lasso_path_mlx(
        cls,
        problem: LassoPathMLX,
        *,
        gradient_dtype: mx.Dtype | None = None,
        kkt_dtype: mx.Dtype | None = mx.float32,
        gram_budget_bytes: int = 8 * 1024 ** 3,
    ) -> "LassoPathGramMLX":
        gradient_dtype = gradient_dtype or problem.dtype
        elem_size = {mx.float32: 4, mx.float16: 2,
                     mx.bfloat16: 2}.get(gradient_dtype, 4)
        gram_bytes = int(problem.n) * int(problem.n) * elem_size
        use_gram = gram_bytes <= gram_budget_bytes

        if use_gram:
            A_grad = problem.A.astype(gradient_dtype)
            b_grad = problem.b.astype(gradient_dtype)
            G = A_grad.T @ A_grad
            c = A_grad.T @ b_grad
            kkt_G = None
            kkt_c = None
            if kkt_dtype is not None and kkt_dtype != gradient_dtype:
                A_kkt = problem.A.astype(kkt_dtype)
                b_kkt = problem.b.astype(kkt_dtype)
                kkt_G = A_kkt.T @ A_kkt
                kkt_c = A_kkt.T @ b_kkt
            mx.eval(G, c)
            if kkt_G is not None and kkt_c is not None:
                mx.eval(kkt_G, kkt_c)
            return cls(
                mode="gram",
                lambdas=problem.lambdas, L=problem.L,
                lambda_max=problem.lambda_max, dtype=problem.dtype,
                m=problem.m, n=problem.n, gradient_dtype=gradient_dtype,
                G=G, c=c, kkt_G=kkt_G, kkt_c=kkt_c,
            )

        # Direct mode.
        A_grad = problem.A.astype(gradient_dtype) \
            if problem.A.dtype != gradient_dtype else problem.A
        b_grad = problem.b.astype(gradient_dtype) \
            if problem.b.dtype != gradient_dtype else problem.b
        kkt_A = None
        kkt_b = None
        if kkt_dtype is not None and kkt_dtype != gradient_dtype:
            kkt_A = problem.A.astype(kkt_dtype) \
                if problem.A.dtype != kkt_dtype else problem.A
            kkt_b = problem.b.astype(kkt_dtype) \
                if problem.b.dtype != kkt_dtype else problem.b
            mx.eval(kkt_A, kkt_b)
        mx.eval(A_grad, b_grad)
        return cls(
            mode="direct",
            lambdas=problem.lambdas, L=problem.L,
            lambda_max=problem.lambda_max, dtype=problem.dtype,
            m=problem.m, n=problem.n, gradient_dtype=gradient_dtype,
            A=A_grad, b=b_grad, kkt_A=kkt_A, kkt_b=kkt_b,
        )

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    @property
    def K(self) -> int:
        return int(self.lambdas.shape[0])

    @property
    def L(self) -> float:
        return self._L

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    def grad_path_smooth(self, Y: mx.array) -> mx.array:
        """Per-column gradient over a (n, K) iterate. Returns (n, K).

        Gram mode: G @ Y - c[:, None]      — one matmul.
        Direct mode: A.T @ (A @ Y - b[:, None])  — two matmuls. Required
        when n^2 exceeds Metal's single-buffer cap.
        """
        if self.mode == "gram":
            Y_grad = Y.astype(self.gradient_dtype) \
                if Y.dtype != self.gradient_dtype else Y
            g = self.G @ Y_grad - self.c[:, None]
            return g.astype(self.dtype) if g.dtype != self.dtype else g

        # direct
        Y_grad = Y.astype(self.gradient_dtype) \
            if Y.dtype != self.gradient_dtype else Y
        AY = self.A @ Y_grad                    # (m, K)
        resid = AY - self.b[:, None]            # (m, K)
        g = self.A.T @ resid                    # (n, K)
        return g.astype(self.dtype) if g.dtype != self.dtype else g

    def prox_path(self, V: mx.array, t: float) -> mx.array:
        kappa = t * self.lambdas
        return mx.sign(V) * mx.maximum(mx.abs(V) - kappa[None, :], 0.0)

    def kkt_residual_per_col(self, X: mx.array) -> mx.array:
        """Per-column scale-free KKT residual; (K,) on device.

        Uses kkt_dtype buffers if a separate KKT precision was configured.
        """
        if self.mode == "gram":
            X_kkt = X.astype(self._kkt_G.dtype) \
                if X.dtype != self._kkt_G.dtype else X
            g = self._kkt_G @ X_kkt - self._kkt_c[:, None]
        else:
            X_kkt = X.astype(self._kkt_A.dtype) \
                if X.dtype != self._kkt_A.dtype else X
            AX = self._kkt_A @ X_kkt
            g = self._kkt_A.T @ (AX - self._kkt_b[:, None])
        z = self._L * X_kkt - g
        soft_z = mx.sign(z) * mx.maximum(
            mx.abs(z) - self.lambdas[None, :], 0.0,
        )
        r = self._L * X_kkt - soft_z
        r_inf = mx.max(mx.abs(r), axis=0)
        denom = self.lambdas + self._lambda_max
        return r_inf / denom

    def kkt_residual_max(self, X: mx.array) -> float:
        return float(mx.max(self.kkt_residual_per_col(X)))
class TVDenoising1DMLX:
    """MLX-backed view of `TVDenoising1D` for PDHG/Chambolle-Pock kernels.

    Holds ``b`` as an MLX array and provides ``K_apply`` / ``K_T_apply`` for
    the forward-difference operator and its adjoint, plus the closed-form
    proxes (``prox_f``, ``prox_g_conjugate``) PDHG needs each iteration.

    The host gap measurement still uses the canonical numpy `TVDenoising1D`
    via the sandbox's canonical→MLX conversion pattern; this class only
    exists so MLX kernels see MLX arrays.
    """

    def __init__(self, b: mx.array, lam: float, dtype: mx.Dtype):
        self.b = b
        self.lam = float(lam)
        self.dtype = dtype

    @classmethod
    def from_problem(cls, problem: TVDenoising1D, dtype: mx.Dtype = mx.float32) -> "TVDenoising1DMLX":
        return cls(
            b=mx.array(problem.b, dtype=dtype),
            lam=problem.lam,
            dtype=dtype,
        )

    @property
    def n(self) -> int:
        return int(self.b.shape[0])

    @property
    def m(self) -> int:
        return int(self.b.shape[0]) - 1

    @property
    def L_K(self) -> float:
        return 2.0

    def K_apply(self, x: mx.array) -> mx.array:
        return x[1:] - x[:-1]

    def K_T_apply(self, y: mx.array) -> mx.array:
        # Pad y with zeros at both ends so the central difference picks up the
        # boundary terms correctly: (K^T y)[0] = -y[0], (K^T y)[n-1] = y[n-2].
        zero = mx.zeros((1,), dtype=y.dtype)
        y_padded = mx.concatenate([zero, y, zero])
        return y_padded[:-1] - y_padded[1:]

    def prox_f(self, v: mx.array, tau: float) -> mx.array:
        return (v + tau * self.b) / (1.0 + tau)

    def prox_g_conjugate(self, z: mx.array, sigma: float) -> mx.array:
        del sigma
        return mx.clip(z, -self.lam, self.lam)

    def primal_dual_gap(self, x: mx.array, y: mx.array) -> float:
        Kx = self.K_apply(x)
        primal = 0.5 * float(mx.sum((x - self.b) ** 2)) + self.lam * float(mx.sum(mx.abs(Kx)))
        y_proj = mx.clip(y, -self.lam, self.lam)
        KTy = self.K_T_apply(y_proj)
        dual = -0.5 * float(mx.sum(KTy ** 2)) + float(mx.sum(KTy * self.b))
        scale = 0.5 * float(mx.sum(self.b ** 2)) + 1.0
        return float(max(primal - dual, 0.0)) / scale


class EqualityQPMLX:
    """MLX-backed view of `EqualityQP` for ALM kernels.

    Holds P, q, A, b as MLX arrays. The cached Cholesky factor of
    ``(P + rho A^T A)`` is built via ``mx.linalg.cholesky``; per-iter linear
    solves use ``mx.linalg.solve_triangular`` (twice). Both were confirmed
    available in the P0 MLX probe.

    The host primal/dual residual measurements still use the canonical numpy
    `EqualityQP` via the sandbox's canonical->MLX conversion pattern. This
    class exists so MLX kernels see MLX arrays.
    """

    def __init__(
        self,
        P: mx.array, q: mx.array, A: mx.array, b: mx.array,
        dtype: mx.Dtype,
    ):
        self.P = P
        self.q = q
        self.A = A
        self.b_constraint = b
        self.dtype = dtype

    @classmethod
    def from_problem(cls, problem: EqualityQP, dtype: mx.Dtype = mx.float32) -> "EqualityQPMLX":
        return cls(
            P=mx.array(problem.P, dtype=dtype),
            q=mx.array(problem.q, dtype=dtype),
            A=mx.array(problem.A, dtype=dtype),
            b=mx.array(problem.b_constraint, dtype=dtype),
            dtype=dtype,
        )

    @property
    def n(self) -> int:
        return int(self.P.shape[0])

    @property
    def m_constraints(self) -> int:
        return int(self.A.shape[0])

    def A_apply(self, x: mx.array) -> mx.array:
        return self.A @ x

    def A_T_apply(self, y: mx.array) -> mx.array:
        return self.A.T @ y

    def x_rhs(self, lam: mx.array, rho: float) -> mx.array:
        return -self.q + self.A.T @ (rho * self.b_constraint - lam)

    def build_factor(self, rho: float) -> dict:
        # MLX cholesky and solve_triangular are CPU-only as of mlx 0.18+
        # ("This op is not yet supported on the GPU"). The factor lives on
        # the CPU stream; per-iter solves also dispatch to CPU. The autoresearch
        # loop's lever here is replacing this with a custom Metal trisolve
        # kernel for the small/medium sizes where launch overhead doesn't
        # dominate.
        H = self.P + rho * (self.A.T @ self.A)
        L = mx.linalg.cholesky(H, stream=mx.cpu)
        mx.eval(L)
        return {"L": L, "rho": float(rho)}

    def solve_with_factor(self, factor: dict, rhs: mx.array) -> mx.array:
        L = factor["L"]
        # Two triangular solves: L y = rhs, L^T x = y. CPU stream required.
        try:
            y = mx.linalg.solve_triangular(L, rhs, upper=False, stream=mx.cpu)
            x = mx.linalg.solve_triangular(L.T, y, upper=True, stream=mx.cpu)
        except TypeError:
            # older MLX uses `lower=` instead of `upper=`
            y = mx.linalg.solve_triangular(L, rhs, lower=True, stream=mx.cpu)
            x = mx.linalg.solve_triangular(L.T, y, lower=False, stream=mx.cpu)
        return x


class BasisPursuitMLX:
    """MLX-backed view of `BasisPursuit` for PDHG kernels.

    K = A (dense matvec) and K^T = A.T. Per-iter cost is dominated by these
    two matvecs (``mx.matmul`` on m x n and n x m). The autoresearch loop's
    lever is fusing the matvec with the prox / extrapolation tail; temporal
    fusion (ala TV) does NOT work here because the dense K means each step's
    output depends on all of x — full dependency cone.
    """

    def __init__(self, A: mx.array, b: mx.array, dtype: mx.Dtype):
        self.A = A
        self.b = b
        self.dtype = dtype

    @classmethod
    def from_problem(cls, problem: BasisPursuit, dtype: mx.Dtype = mx.float32) -> "BasisPursuitMLX":
        return cls(
            A=mx.array(problem.A, dtype=dtype),
            b=mx.array(problem.b, dtype=dtype),
            dtype=dtype,
        )

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @cached_property
    def L_K(self) -> float:
        """Spectral norm ||A||_2. Computed once via numpy SVD on the materialized A."""
        return float(np.linalg.norm(np.asarray(self.A), ord=2))

    def K_apply(self, x: mx.array) -> mx.array:
        return self.A @ x

    def K_T_apply(self, y: mx.array) -> mx.array:
        return self.A.T @ y

    def prox_f(self, v: mx.array, tau: float) -> mx.array:
        return mx.sign(v) * mx.maximum(mx.abs(v) - tau, 0.0)

    def prox_g_conjugate(self, z: mx.array, sigma: float) -> mx.array:
        return z - sigma * self.b

    def primal_dual_gap(self, x: mx.array, y: mx.array) -> float:
        l1 = float(mx.sum(mx.abs(x)))
        residual = float(mx.linalg.norm(self.A @ x - self.b))
        bty = float(mx.sum(self.b * y))
        scale = max(float(mx.linalg.norm(self.b)), 1.0)
        return (abs(l1 + bty) + residual) / scale


class LassoAdmmMLX:
    """MLX-backed view of `LassoAdmm` for ADMM-on-LASSO kernels.

    Holds A and b as MLX arrays. (A^T A + rho I) Cholesky factor lives on
    the CPU stream (MLX linalg is CPU-only). Per-iter ops:
      - x-update: cached trisolve of A^T b + rho (z - u)
      - z-update: soft-threshold(x + u, lam/rho)
      - u += x - z
    """

    def __init__(self, A: mx.array, b: mx.array, lam: float, dtype: mx.Dtype):
        self.A = A
        self.b = b
        self.lam = float(lam)
        self.dtype = dtype

    @classmethod
    def from_problem(cls, problem: LassoAdmm, dtype: mx.Dtype = mx.float32) -> "LassoAdmmMLX":
        return cls(
            A=mx.array(problem.A, dtype=dtype),
            b=mx.array(problem.b, dtype=dtype),
            lam=problem.lam,
            dtype=dtype,
        )

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @cached_property
    def A_T_b(self) -> mx.array:
        return self.A.T @ self.b

    @cached_property
    def lambda_max(self) -> float:
        """||A^T b||_inf — the regularization above which x* = 0."""
        return float(mx.max(mx.abs(self.A_T_b)))

    @cached_property
    def default_rho(self) -> float:
        """Sensible ADMM penalty: Boyd Box 3.4 suggests rho ~ lambda_max."""
        return max(self.lambda_max, 1.0)

    def x_rhs_admm(self, z: mx.array, u: mx.array, rho: float) -> mx.array:
        return self.A_T_b + rho * (z - u)

    def build_factor(self, rho: float) -> dict:
        H = self.A.T @ self.A + rho * mx.eye(self.n, dtype=self.dtype)
        L = mx.linalg.cholesky(H, stream=mx.cpu)
        mx.eval(L)
        return {"L": L, "rho": float(rho)}

    def solve_with_factor(self, factor: dict, rhs: mx.array) -> mx.array:
        L = factor["L"]
        try:
            y = mx.linalg.solve_triangular(L, rhs, upper=False, stream=mx.cpu)
            x = mx.linalg.solve_triangular(L.T, y, upper=True, stream=mx.cpu)
        except TypeError:
            y = mx.linalg.solve_triangular(L, rhs, lower=True, stream=mx.cpu)
            x = mx.linalg.solve_triangular(L.T, y, lower=False, stream=mx.cpu)
        return x

    def prox_g(self, v: mx.array, t: float) -> mx.array:
        kappa = t * self.lam
        return mx.sign(v) * mx.maximum(mx.abs(v) - kappa, 0.0)
