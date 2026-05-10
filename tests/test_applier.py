"""Tests for structured edit application."""

from __future__ import annotations

from pathlib import Path

import pytest

from convexkernels.synth.applier import (
    apply_config_edit,
    apply_edit,
    has_applicable_payload,
    has_config_payload,
)
from convexkernels.synth.lineage import Edit


def _edit(payload: dict) -> Edit:
    return Edit(
        type="tile_change",
        payload=payload,
        rationale="test",
        proposer_role="impl",
        proposer_model="test",
        source="manual",
    )


def test_has_applicable_payload_distinguishes_stub_payload():
    assert has_applicable_payload(_edit({"threadgroup_size": 128}))
    assert has_applicable_payload(_edit({"full_source": "def init_state(): pass"}))
    assert not has_applicable_payload(_edit({"variant_index": 1}))
    assert not has_applicable_payload(_edit({"gradient_strategy": "gram"}))
    assert has_config_payload(_edit({"gradient_strategy": "gram"}))
    assert has_config_payload(_edit({"dtype_strategy": "fp16_storage"}))


def test_structured_threadgroup_edit_from_seed(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info = apply_edit(
        _edit({"threadgroup_size": 128, "kernel_name_suffix": "tg128"}),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "threadgroup=(min(128, state.x.shape[0]), 1, 1)" in source
    assert 'name="fista_fused_zsoft_momentum_tg128"' in source
    assert "from convexkernels.kernels.mlx.lib import LassoMLX" in source
    assert info.hash.startswith("sha256:")


def test_structured_branchless_and_no_bounds_edit(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info = apply_edit(
        _edit(
            {
                "remove_bounds_check": True,
                "branchless_soft_threshold": True,
            }
        ),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "if (i >= y_shape[0]) return;" not in source
    assert "T xi_pos = metal::max(zi - thresh, T(0));" in source
    assert "T xi_new = xi_pos - xi_neg;" in source


def test_structured_items_per_thread_vectorization(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info = apply_edit(
        _edit({"items_per_thread": 2, "kernel_name_suffix": "vec2"}),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "uint base = thread_position_in_grid.x * 2;" in source
    assert "for (uint offset = 0; offset < 2; ++offset)" in source
    assert "grid_n = (n + items_per_thread - 1) // items_per_thread" in source
    assert "grid=(grid_n, 1, 1)" in source
    assert "threadgroup=(min(256, grid_n), 1, 1)" in source
    assert 'name="fista_fused_zsoft_momentum_vec2"' in source


def test_structured_items_per_thread_preserves_nonnegative_prox(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/nonnegative_fista_step_v0.py")
    info = apply_edit(
        _edit({"items_per_thread": 2, "kernel_name_suffix": "nn_vec2"}),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "uint base = thread_position_in_grid.x * 2;" in source
    assert "T xi_new = metal::max(zi - thresh, T(0));" in source
    assert "T abs_zi = metal::abs(zi);" not in source
    assert "T sign_zi" not in source
    assert "from convexkernels.kernels.mlx.lib import NonnegativeLassoMLX" in source
    assert 'name="nn_lasso_fused_zpos_momentum_nn_vec2"' in source


def test_structured_vectorization_composes_with_branchless(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info = apply_edit(
        _edit(
            {
                "items_per_thread": 2,
                "branchless_soft_threshold": True,
                "kernel_name_suffix": "vec2_branchless",
            }
        ),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "uint base = thread_position_in_grid.x * 2;" in source
    assert "T xi_pos = metal::max(zi - thresh, T(0));" in source
    assert "T xi_new = xi_pos - xi_neg;" in source
    assert "T abs_zi = metal::abs(zi);" not in source


def test_structured_branchless_is_noop_for_nonnegative_prox(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/nonnegative_fista_step_v0.py")
    info = apply_edit(
        _edit(
            {
                "branchless_soft_threshold": True,
                "kernel_name_suffix": "branchless_transfer",
            }
        ),
        tmp_path,
        parent_source_path=seed,
    )
    source = Path(info.path).read_text()

    assert "T xi_new = metal::max(zi - thresh, T(0));" in source
    assert "T xi_pos = metal::max(zi - thresh, T(0));" not in source
    assert 'name="nn_lasso_fused_zpos_momentum_branchless_transfer"' in source


def test_structured_remove_bounds_rejects_vectorized_kernel(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/nonnegative_fista_step_v0.py")
    with pytest.raises(ValueError, match="unsafe after items_per_thread"):
        apply_edit(
            _edit({"items_per_thread": 2, "remove_bounds_check": True}),
            tmp_path,
            parent_source_path=seed,
        )


def test_structured_items_per_thread_rejects_unsupported_value(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    with pytest.raises(ValueError, match="supports only 2 or 4"):
        apply_edit(
            _edit({"items_per_thread": 3}),
            tmp_path,
            parent_source_path=seed,
        )


def test_structured_edit_requires_parent_source(tmp_path: Path):
    with pytest.raises(ValueError, match="parent_source_path"):
        apply_edit(_edit({"threadgroup_size": 128}), tmp_path)


def test_config_edit_materializes_gram_wrapper(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info, kernel_module, problem_dtype, dtype_strategy = apply_config_edit(
        _edit(
            {
                "gradient_strategy": "gram",
                "dtype_strategy": "mixed_gram",
                "kernel_name_suffix": "gram_mixed",
            }
        ),
        tmp_path,
        parent_source_path=seed,
        default_kernel_module="convexkernels.kernels.mlx.seeds.fista_step_v0",
        default_problem_dtype="fp32",
        default_dtype_strategy="fp32",
    )
    source = Path(info.path).read_text()

    assert kernel_module == str(tmp_path / "source.py")
    assert problem_dtype == "fp32"
    assert dtype_strategy == "mixed_gram"
    assert 'GRADIENT_STRATEGY = "gram"' in source
    assert 'DTYPE_STRATEGY = "mixed_gram"' in source
    assert "LassoGramMLX.from_lasso_mlx" in source
    assert "gradient_dtype = mx.float16" in source
    assert "def prepare_problem(problem, config=None):" in source
    assert info.hash.startswith("sha256:")


def test_config_edit_materializes_fp16_storage_cast(tmp_path: Path):
    seed = Path("convexkernels/kernels/mlx/seeds/fista_step_v0.py")
    info, _, problem_dtype, dtype_strategy = apply_config_edit(
        _edit({"dtype_strategy": "fp16_storage"}),
        tmp_path,
        parent_source_path=seed,
        default_kernel_module="convexkernels.kernels.mlx.seeds.fista_step_v0",
        default_problem_dtype="fp32",
        default_dtype_strategy="fp32",
    )
    source = Path(info.path).read_text()

    assert problem_dtype == "fp16"
    assert dtype_strategy == "fp16_storage"
    assert 'GRADIENT_STRATEGY = "direct"' in source
    assert 'DTYPE_STRATEGY = "fp16_storage"' in source
    assert "return LassoMLX(" in source
