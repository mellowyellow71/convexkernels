import importlib


def test_top_level_import():
    mod = importlib.import_module("convexkernels")
    assert hasattr(mod, "__version__")


def test_subpackages_import():
    for name in [
        "convexkernels.frontend",
        "convexkernels.algorithms",
        "convexkernels.kernels",
        "convexkernels.kernels.mlx",
        "convexkernels.synth",
        "convexkernels.bench",
    ]:
        importlib.import_module(name)
