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
