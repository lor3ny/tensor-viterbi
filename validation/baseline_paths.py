"""validation/baseline_paths.py — selects the system/toolchain-specific
hsmmlearn / hsmmlearn_omp baseline build.

compile.py installs each baseline into an isolated per-system/toolchain
directory (<pkg>/build/<system>/<toolchain>) instead of the shared
environment site-packages, so recompiling for a different system/toolchain
never overwrites another build. configure_hsmmlearn()/configure_hsmmlearn_omp()
must be called before importing hsmmlearn/hsmmlearn_omp (or anything that
imports them) so the correct build is the one actually found on sys.path.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _configure(pkg_name: str, system: str, toolchain: str) -> None:
    build_dir = _ROOT / pkg_name / "build" / system / toolchain
    if not build_dir.is_dir():
        raise RuntimeError(
            f"[{pkg_name}] No build found for system/toolchain '{system}/{toolchain}' "
            f"at '{build_dir}'.\n"
            f"Run: ./bench run --system {system} --toolchain {toolchain} --pack <pack> "
            f"(it compiles automatically before dispatching jobs)"
        )

    so_dir = str(build_dir)

    existing = sys.modules.get(pkg_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if not existing_file.startswith(so_dir + "/"):
            raise RuntimeError(
                f"[{pkg_name}] Already loaded from '{existing_file}', which does not "
                f"belong to the requested system/toolchain '{system}/{toolchain}' "
                f"(expected under '{so_dir}'). Python cannot reload a module under a "
                f"different path within the same process — run this system/toolchain "
                f"combination in its own process."
            )
        return

    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)


def configure_hsmmlearn(system: str, toolchain: str) -> None:
    _configure("hsmmlearn", system, toolchain)


def configure_hsmmlearn_omp(system: str, toolchain: str) -> None:
    _configure("hsmmlearn_omp", system, toolchain)
