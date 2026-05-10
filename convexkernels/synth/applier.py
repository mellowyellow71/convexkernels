"""Apply an `Edit` to produce a new kernel source on disk."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

from .lineage import Edit, SourceInfo

_STRUCTURED_KEYS = {
    "threadgroup_size",
    "items_per_thread",
    "remove_bounds_check",
    "branchless_soft_threshold",
    "kernel_name_suffix",
}

CONFIG_PAYLOAD_KEYS = {
    "gradient_strategy",
    "dtype_strategy",
}

GRADIENT_STRATEGY_KERNELS = {
    "direct": "convexkernels.kernels.mlx.seeds.fista_step_v0",
    "gram": "convexkernels.kernels.mlx.seeds.gram_fista_step_v0",
}

DTYPE_STRATEGIES = {"fp32", "fp16_storage", "mixed_gram"}

_CONFIG_BLOCK_START = "# BEGIN convexkernels config strategy hook"
_CONFIG_BLOCK_END = "# END convexkernels config strategy hook"


def has_applicable_payload(edit: Edit) -> bool:
    """Return true if `edit` can produce a child source file."""
    return "full_source" in edit.payload or any(
        key in edit.payload and edit.payload[key] not in (None, False, "")
        for key in _STRUCTURED_KEYS
    )


def has_config_payload(edit: Edit) -> bool:
    """Return true if `edit` changes evaluation config rather than source."""
    return any(
        key in edit.payload and edit.payload[key] not in (None, False, "")
        for key in CONFIG_PAYLOAD_KEYS
    )


def config_edit_hash(edit: Edit) -> str:
    """Stable hash for duplicate detection of config-only edits."""
    import json

    payload = {
        key: edit.payload[key]
        for key in sorted(CONFIG_PAYLOAD_KEYS)
        if key in edit.payload and edit.payload[key] not in (None, False, "")
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"config:{digest}"


def kernel_module_for_gradient_strategy(strategy: str) -> str:
    """Return the canonical seed module for a gradient strategy."""
    if strategy not in GRADIENT_STRATEGY_KERNELS:
        raise ValueError(
            "gradient_strategy must be one of "
            f"{sorted(GRADIENT_STRATEGY_KERNELS)}, got {strategy!r}"
        )
    return GRADIENT_STRATEGY_KERNELS[strategy]


def problem_dtype_for_dtype_strategy(strategy: str, default_dtype: str) -> str:
    """Return MLX problem storage dtype implied by a dtype strategy."""
    if strategy not in DTYPE_STRATEGIES:
        raise ValueError(
            f"dtype_strategy must be one of {sorted(DTYPE_STRATEGIES)}, got {strategy!r}"
        )
    if strategy == "fp16_storage":
        return "fp16"
    return default_dtype


def config_from_edit_payload(
    payload: dict,
    *,
    default_kernel_module: str,
    default_problem_dtype: str,
    default_dtype_strategy: str,
) -> tuple[str, str, str]:
    """Resolve kernel module, problem dtype, and dtype strategy for an edit."""
    dtype_strategy = str(payload.get("dtype_strategy") or default_dtype_strategy)
    problem_dtype = problem_dtype_for_dtype_strategy(
        dtype_strategy,
        default_problem_dtype,
    )
    kernel_module = default_kernel_module
    if payload.get("gradient_strategy") is not None:
        kernel_module = kernel_module_for_gradient_strategy(
            str(payload["gradient_strategy"])
        )
    return kernel_module, problem_dtype, dtype_strategy


def apply_edit(
    edit: Edit,
    run_dir: Path,
    *,
    parent_source_path: Path | None = None,
) -> SourceInfo:
    """Write the edited kernel to `run_dir/source.py` and return SourceInfo."""
    if "full_source" in edit.payload:
        source_text = str(edit.payload["full_source"])
    elif has_applicable_payload(edit):
        if parent_source_path is None:
            raise ValueError("structured edits require parent_source_path")
        source_text = _apply_structured_edit(
            parent_source_path.read_text(),
            edit,
        )
    else:
        raise NotImplementedError(
            "applier supports edit.payload['full_source'] or structured keys; "
            f"got payload keys: {list(edit.payload.keys())}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = run_dir / "source.py"
    source_path.write_text(source_text)
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return SourceInfo(path=str(source_path), hash=f"sha256:{digest}")


def apply_config_edit(
    edit: Edit,
    run_dir: Path,
    *,
    parent_source_path: Path | None = None,
    default_kernel_module: str,
    default_problem_dtype: str,
    default_dtype_strategy: str,
) -> tuple[SourceInfo, str, str, str]:
    """Materialize a gradient/dtype strategy edit as a durable source file.

    The eval config still records the requested storage dtype, but the promoted
    source also carries a `prepare_problem()` hook. That makes accepted strategy
    edits replayable from the champion store instead of depending on transient
    loop state.
    """
    kernel_module, problem_dtype, dtype_strategy = config_from_edit_payload(
        edit.payload,
        default_kernel_module=default_kernel_module,
        default_problem_dtype=default_problem_dtype,
        default_dtype_strategy=default_dtype_strategy,
    )
    gradient_strategy = str(
        edit.payload.get("gradient_strategy")
        or _infer_gradient_strategy(default_kernel_module, parent_source_path)
    )
    if gradient_strategy not in GRADIENT_STRATEGY_KERNELS:
        raise ValueError(
            "gradient_strategy must be one of "
            f"{sorted(GRADIENT_STRATEGY_KERNELS)}, got {gradient_strategy!r}"
        )

    source_path = parent_source_path or _source_path_for_module(kernel_module)
    if source_path is None:
        source_path = _source_path_for_module(
            kernel_module_for_gradient_strategy(gradient_strategy)
        )
    if source_path is None:
        raise ValueError(f"could not resolve source for {kernel_module!r}")

    source_text = _apply_config_strategy_hook(
        _normalize_path_loaded_imports(source_path.read_text()),
        gradient_strategy=gradient_strategy,
        dtype_strategy=dtype_strategy,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "source.py"
    output_path.write_text(source_text)
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return (
        SourceInfo(path=str(output_path), hash=f"sha256:{digest}"),
        str(output_path),
        problem_dtype,
        dtype_strategy,
    )


def _source_path_for_module(module_spec: str) -> Path | None:
    if module_spec.endswith(".py"):
        return Path(module_spec)
    spec = importlib.util.find_spec(module_spec)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _infer_gradient_strategy(
    module_spec: str,
    source_path: Path | None,
) -> str:
    if "gram_fista" in module_spec:
        return "gram"
    if source_path is not None and source_path.exists():
        source = source_path.read_text()
        if 'GRADIENT_STRATEGY = "gram"' in source:
            return "gram"
    return "direct"


def _apply_config_strategy_hook(
    source: str,
    *,
    gradient_strategy: str,
    dtype_strategy: str,
) -> str:
    source = _strip_config_strategy_hook(source).rstrip()
    return (
        source
        + "\n\n"
        + _config_strategy_hook_source(
            gradient_strategy=gradient_strategy,
            dtype_strategy=dtype_strategy,
        )
    )


def _strip_config_strategy_hook(source: str) -> str:
    pattern = (
        rf"\n*{re.escape(_CONFIG_BLOCK_START)}.*?"
        rf"{re.escape(_CONFIG_BLOCK_END)}\n*"
    )
    return re.sub(pattern, "\n", source, flags=re.DOTALL)


def _config_strategy_hook_source(
    *,
    gradient_strategy: str,
    dtype_strategy: str,
) -> str:
    return f'''{_CONFIG_BLOCK_START}
import mlx.core as mx

from convexkernels.kernels.mlx.lib import LassoGramMLX, LassoMLX

GRADIENT_STRATEGY = "{gradient_strategy}"
DTYPE_STRATEGY = "{dtype_strategy}"


def _config_storage_dtype(problem):
    if DTYPE_STRATEGY == "fp16_storage":
        return mx.float16
    return getattr(problem, "dtype", mx.float32)


def _config_cast_lasso_problem(problem, dtype):
    if not isinstance(problem, LassoMLX):
        raise TypeError(
            "gradient/dtype strategy hooks currently require LassoMLX input"
        )
    if getattr(problem, "dtype", None) == dtype:
        return problem
    return LassoMLX(
        A=problem.A.astype(dtype),
        b=problem.b.astype(dtype),
        lam=problem.lam,
        L=problem.L,
        lambda_max=problem.lambda_max,
        dtype=dtype,
    )


def prepare_problem(problem, config=None):
    prepared = _config_cast_lasso_problem(
        problem,
        _config_storage_dtype(problem),
    )
    if GRADIENT_STRATEGY != "gram":
        return prepared

    if DTYPE_STRATEGY == "mixed_gram":
        gradient_dtype = mx.float16
        kkt_dtype = mx.float32
    else:
        gradient_dtype = getattr(prepared, "dtype", mx.float32)
        kkt_dtype = mx.float32 if gradient_dtype != mx.float32 else None
    return LassoGramMLX.from_lasso_mlx(
        prepared,
        gradient_dtype=gradient_dtype,
        kkt_dtype=kkt_dtype,
    )
{_CONFIG_BLOCK_END}
'''


def _apply_structured_edit(source: str, edit: Edit) -> str:
    source = _normalize_path_loaded_imports(source)
    payload = edit.payload
    prox_kind = _detect_prox_kind(source)

    if payload.get("items_per_thread") is not None:
        source = _replace_items_per_thread(
            source,
            int(payload["items_per_thread"]),
            prox_kind=prox_kind,
        )
    if payload.get("threadgroup_size") is not None:
        source = _replace_threadgroup_size(
            source,
            int(payload["threadgroup_size"]),
        )
    if payload.get("remove_bounds_check"):
        source = _remove_bounds_check(source)
    if payload.get("branchless_soft_threshold"):
        source = _replace_branchless_soft_threshold(source, prox_kind=prox_kind)
    if payload.get("kernel_name_suffix"):
        source = _rename_kernel(source, str(payload["kernel_name_suffix"]))
    return source


def _normalize_path_loaded_imports(source: str) -> str:
    source = source.replace(
        "from ..lib import LassoMLX",
        "from convexkernels.kernels.mlx.lib import LassoMLX",
    )
    return source.replace(
        "from ..lib import NonnegativeLassoMLX",
        "from convexkernels.kernels.mlx.lib import NonnegativeLassoMLX",
    )


def _detect_prox_kind(source: str) -> str:
    """Infer the prox semantics of a seed kernel before structured rewrites."""
    if (
        "NonnegativeLassoMLX" in source
        or "nn_lasso_fused_zpos_momentum" in source
        or (
            "metal::max(zi - thresh, T(0))" in source
            and "metal::max(-zi - thresh, T(0))" not in source
        )
    ):
        return "nonnegative_lasso"
    if (
        "LassoMLX" in source
        or "fista_fused_zsoft_momentum" in source
        or "T abs_zi = metal::abs(zi);" in source
        or "T xi_new = xi_pos - xi_neg;" in source
    ):
        return "lasso"
    raise ValueError("could not detect prox kind for structured MLX edit")


def _replace_threadgroup_size(source: str, size: int) -> str:
    if size <= 0 or size > 1024:
        raise ValueError(f"threadgroup_size must be in 1..1024, got {size}")
    pattern = r"threadgroup=\(min\(\d+,\s*([^)]+)\),\s*1,\s*1\)"
    updated, n = re.subn(
        pattern,
        rf"threadgroup=(min({size}, \1), 1, 1)",
        source,
        count=1,
    )
    if n == 0:
        raise ValueError("could not find a threadgroup=(min(...), 1, 1) call")
    return updated


def _replace_items_per_thread(
    source: str,
    items_per_thread: int,
    *,
    prox_kind: str,
) -> str:
    if items_per_thread not in {2, 4}:
        raise ValueError(
            "items_per_thread structured vectorization supports only 2 or 4; "
            f"got {items_per_thread}"
        )
    if "grid_n =" in source:
        raise ValueError("source already appears to use grid_n vectorization")

    source = _replace_metal_source(
        source,
        _vectorized_metal_source(items_per_thread, prox_kind=prox_kind),
    )
    source = source.replace(
        "    x_next, y_next = _FUSED_STEP_KERNEL(\n",
        (
            "    n = state.x.shape[0]\n"
            f"    items_per_thread = {items_per_thread}\n"
            "    grid_n = (n + items_per_thread - 1) // items_per_thread\n\n"
            "    x_next, y_next = _FUSED_STEP_KERNEL(\n"
        ),
        1,
    )
    source = source.replace(
        "        grid=(state.x.shape[0], 1, 1),",
        "        grid=(grid_n, 1, 1),",
        1,
    )
    source = source.replace(
        "        threadgroup=(min(256, state.x.shape[0]), 1, 1),",
        "        threadgroup=(min(256, grid_n), 1, 1),",
        1,
    )
    return source


def _remove_bounds_check(source: str) -> str:
    if "grid_n = (n + items_per_thread - 1) // items_per_thread" in source:
        raise ValueError(
            "remove_bounds_check is unsafe after items_per_thread vectorization"
        )
    updated, n = re.subn(
        r"\n\s*if\s*\(i\s*>=\s*y_shape\[0\]\)\s*return;\s*",
        "\n",
        source,
        count=1,
    )
    if n == 0:
        updated, n = re.subn(
            r"\n\s*if\s*\(i\s*>=\s*n\)\s*return;\s*",
            "\n",
            source,
            count=1,
        )
    if n == 0:
        raise ValueError("could not find the y_shape bounds check")
    return updated


def _replace_branchless_soft_threshold(source: str, *, prox_kind: str) -> str:
    if prox_kind == "nonnegative_lasso":
        if "T xi_new = metal::max(zi - thresh, T(0));" not in source:
            raise ValueError("could not find the nonnegative LASSO prox block")
        return source

    pattern = re.compile(
        r"(?P<indent>[ \t]*)T abs_zi = metal::abs\(zi\);\n"
        r"[ \t]*T sign_zi = \(zi > T\(0\)\) \? T\(1\) : "
        r"\(\(zi < T\(0\)\) \? T\(-1\) : T\(0\)\);\n"
        r"[ \t]*T xi_new = \(abs_zi > thresh\)\n"
        r"[ \t]*\? sign_zi \* \(abs_zi - thresh\)\n"
        r"[ \t]*: T\(0\);\n",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}T xi_pos = metal::max(zi - thresh, T(0));\n"
            f"{indent}T xi_neg = metal::max(-zi - thresh, T(0));\n"
            f"{indent}T xi_new = xi_pos - xi_neg;\n"
        )

    updated, n = pattern.subn(repl, source, count=1)
    if n == 0:
        if (
            "T xi_pos = metal::max(zi - thresh, T(0));" in source
            and "T xi_neg = metal::max(-zi - thresh, T(0));" in source
            and "T xi_new = xi_pos - xi_neg;" in source
        ):
            return source
        raise ValueError("could not find the branchy soft-threshold block")
    return updated


def _rename_kernel(source: str, suffix: str) -> str:
    safe_suffix = re.sub(r"[^0-9A-Za-z_]+", "_", suffix).strip("_")
    if not safe_suffix:
        raise ValueError("kernel_name_suffix must contain an identifier character")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.endswith(f"_{safe_suffix}"):
            return match.group(0)
        return f'name="{name}_{safe_suffix}"'

    updated, n = re.subn(r'name="([A-Za-z_][0-9A-Za-z_]*)"', repl, source, count=1)
    if n == 0:
        raise ValueError("could not find Metal kernel name")
    return updated


def _replace_metal_source(source: str, new_kernel_source: str) -> str:
    marker = 'source="""'
    start = source.find(marker)
    if start < 0:
        raise ValueError("could not find mx.fast.metal_kernel source string")
    body_start = start + len(marker)
    end = source.find('"""', body_start)
    if end < 0:
        raise ValueError("could not find end of mx.fast.metal_kernel source string")
    return source[:body_start] + "\n" + new_kernel_source.rstrip() + "\n    " + source[end:]


def _vectorized_metal_source(items_per_thread: int, *, prox_kind: str) -> str:
    prox_source = _prox_source(prox_kind, indent="            ")
    return f"""
        uint base = thread_position_in_grid.x * {items_per_thread};
        uint n = y_shape[0];

        T t      = scalars[0];
        T lam    = scalars[1];
        T mom    = scalars[2];
        T thresh = t * lam;

        for (uint offset = 0; offset < {items_per_thread}; ++offset) {{
            uint i = base + offset;
            if (i >= n) return;

            T zi = y[i] - t * g[i];
{prox_source}

            x_next[i] = xi_new;
            y_next[i] = xi_new + mom * (xi_new - x_prev[i]);
        }}
"""


def _prox_source(prox_kind: str, *, indent: str) -> str:
    if prox_kind == "lasso":
        return (
            f"{indent}T abs_zi = metal::abs(zi);\n"
            f"{indent}T sign_zi = (zi > T(0)) ? T(1) : "
            "((zi < T(0)) ? T(-1) : T(0));\n"
            f"{indent}T xi_new = (abs_zi > thresh)\n"
            f"{indent}    ? sign_zi * (abs_zi - thresh)\n"
            f"{indent}    : T(0);"
        )
    if prox_kind == "nonnegative_lasso":
        return f"{indent}T xi_new = metal::max(zi - thresh, T(0));"
    raise ValueError(f"unsupported prox kind: {prox_kind}")
