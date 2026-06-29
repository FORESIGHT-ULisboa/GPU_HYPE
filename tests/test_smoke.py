"""Smoke tests for GPU_HYPE.

These exercise the import chain and the bundled pre-trained model. They skip
themselves when no OpenCL platform is available (importing ``gpu_pso`` builds an
OpenCL context) or when the bundled artifacts are absent, so the suite is safe
to run in CI without a GPU.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = REPO_ROOT / "examples" / "GPU_HYPE_2026-03-24_13h15.pkl"
HYPE_FOLDER = REPO_ROOT / "examples" / "set7_germany_tuerkheim"


def _opencl_available() -> bool:
    try:
        import pyopencl as cl

        return len(cl.get_platforms()) > 0
    except Exception:
        return False


requires_opencl = pytest.mark.skipif(
    not _opencl_available(), reason="no OpenCL platform available"
)


@requires_opencl
def test_core_imports():
    import gpu  # noqa: F401
    import gpu_pso
    from conceptual.HYPE import HYPE  # noqa: F401
    from error.errorOpenCL import Error  # noqa: F401
    from new_error_model import NewErrorModel  # noqa: F401

    assert hasattr(gpu_pso, "GPU_PSO")


@requires_opencl
def test_load_pretrained_model():
    if not MODEL_PKL.exists():
        pytest.skip("bundled pre-trained model not present")

    from gpu import GPU

    model = GPU.load(str(MODEL_PKL))
    assert model.trained is True
    assert model.population.shape[0] == model.fitted.shape[0]
    assert model.fitted.shape[1] == 2  # (non_exceedance, error)
    assert str(model.modelObject.outputfile).endswith("0050675.txt")


def test_hype_executable_present():
    """The runnable HYPE path needs the bundled Windows executable."""
    exe = HYPE_FOLDER / "HYPEwithoutPopup4All.exe"
    if not exe.exists():
        pytest.skip("bundled HYPE executable not present")
    assert exe.stat().st_size > 0
