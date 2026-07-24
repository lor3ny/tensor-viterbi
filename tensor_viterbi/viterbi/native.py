from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM

import importlib as _importlib
import pathlib as _pathlib
import sys as _sys

_HERE = _pathlib.Path(__file__).parent.resolve()

_native  = None
_sys_name: str = ""


def _module_name(system: str, toolchain: str) -> str:
    """Mirror CMakeLists.txt's NATIVE_MODULE_NAME derivation.

    Each system/toolchain build is compiled as a distinctly named extension
    (e.g. `_native_xeon8480_gnu`) rather than a generic `_native`, so that
    loading two different builds in the same interpreter can never collide
    in Python's by-name module cache (sys.modules) — the bug that let one
    already-imported build silently shadow another.
    """
    sanitized = f"{system}_{toolchain}".replace("/", "_").replace("-", "_")
    return f"_native_{sanitized}"


def configure(system: str, toolchain: str) -> None:
    """Load the native extension for the given system/toolchain pair.

    Must be called before any decode_* function is used.
    Calling it again with a different pair loads that pair's own distinctly
    named module, so multiple system/toolchain builds can coexist in the
    same process without contention.
    """
    global _native, _sys_name
    _sys_name = f"{system}/{toolchain}"
    _so_dir = str(_HERE / system / toolchain)
    if _so_dir not in _sys.path:
        _sys.path.insert(0, _so_dir)
    mod_name = _module_name(system, toolchain)
    try:
        _native = _importlib.import_module(mod_name)
    except ImportError as e:
        raise RuntimeError(
            f"[native] Could not load {mod_name} extension from '{_so_dir}'.\n"
            f"Run: ./bench run --system {system} --toolchain {toolchain} --pack <pack> "
            f"(it compiles automatically before dispatching jobs)\n"
            f"Original error: {e}"
        ) from e


def _ensure_configured() -> None:
    if _native is None:
        raise RuntimeError(
            "[native] Native extension not loaded.\n"
            "Pass --system and --toolchain to bench, or call "
            "tensor_viterbi.viterbi.native.configure(system, toolchain) directly."
        )


def decode_tensor_viterbi_cpp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_configured()
    return _native.decode_tensor_viterbi_cpp(
        n_states, trans_mat, emission_probs,
        duration_probs_linear, start_probs, duration_probs, obs_seq,
    )


def decode_tensor_viterbi_cuda(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_configured()
    fn = getattr(_native, "decode_tensor_viterbi_cuda", None)
    if fn is None:
        raise RuntimeError("CUDA backend not available in this build.")
    return fn(n_states, trans_mat, emission_probs,
              duration_probs_linear, start_probs, duration_probs, obs_seq)


def decode_tensor_viterbi_omp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_configured()
    fn = getattr(_native, "decode_tensor_viterbi_omp", None)
    if fn is None:
        raise RuntimeError("OMP backend not available in this build.")
    return fn(n_states, trans_mat, emission_probs,
              duration_probs_linear, start_probs, duration_probs, obs_seq)

