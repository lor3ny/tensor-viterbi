#!/usr/bin/env python3
"""
compile.py — builds the native extension for a given system/toolchain pair.

Library module only: it has no CLI of its own. `compile_system()` is called
by run_benchmark.py, which always compiles before dispatching benchmark jobs.
There is no standalone "compile only" entry point.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _compiler_for_toolchain(toolchain: str) -> tuple[str, str]:
    return {
        "cray":    ("cc",    "CC"),
        "intel":   ("icx",   "icpx"),
        "llvm":    ("clang", "clang++"),
        "fujitsu": ("fcc",   "FCC"),
        "amd":     ("gcc",   "g++"),
    }.get(toolchain, ("gcc", "g++"))


def _run_compile_slurm(system: str, toolchain: str, sys_conf: dict, tc_conf: dict, script: str) -> None:
    """Submit a blocking srun compilation job, streaming output directly to terminal."""
    sys_type = sys_conf["type"]
    uenv     = tc_conf.get("uenv", "")

    uenv_block = ""
    if uenv:
        uenv_block = f"""\
if [[ -z "${{_UENV_ACTIVE:-}}" ]]; then
    exec uenv run --view=modules "{uenv}" -- env _UENV_ACTIVE=1 bash "$0" "$@"
fi
"""

    tmp = SCRIPT_DIR / f".tmp_compile_{system}_{toolchain}.sh"
    try:
        tmp.write_text(f"#!/bin/bash\n{uenv_block}\n{script}")
        tmp.chmod(0o755)

        srun_flags = [
            "--nodes=1", "--ntasks=1",
            f"--job-name=tv_compile_{system}_{toolchain}",
            "--time=01:00:00",
        ]
        if "account" in sys_conf:
            srun_flags.append(f"--account={sys_conf['account']}")
        if sys_conf.get("partition"):
            srun_flags.append(f"--partition={sys_conf['partition']}")
        if "qos" in sys_conf:
            srun_flags.append(f"--qos={sys_conf['qos']}")
        if sys_type == "gpu":
            srun_flags += ["--cpus-per-task=1", "--gres=gpu:1"]
        else:
            srun_flags.append("--cpus-per-task=1")

        print(f"Submitting compilation job via srun ({system}/{toolchain}) ...")
        result = subprocess.run(["srun", *srun_flags, "bash", str(tmp)], cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            sys.exit(result.returncode)
    finally:
        tmp.unlink(missing_ok=True)


def compile_system(system: str, toolchain: str, sys_conf: dict, tc_conf: dict, likwid: bool = False) -> None:
    sys_type      = sys_conf["type"]
    scheduler     = sys_conf.get("scheduler", "slurm")
    gpu_arch      = sys_conf.get("gpu_arch", "")
    modules_build = tc_conf.get("modules_build", "")
    uenv          = tc_conf.get("uenv", "")
    system_dir    = f"{system}_likwid" if likwid else system
    sys_name      = f"{system_dir}/{toolchain}"
    build_dir     = SCRIPT_DIR / "build" / system_dir / toolchain

    # For local systems only: if a uenv is required and we are not already inside
    # it, re-exec under it. Slurm systems handle uenv inside the srun script.
    if scheduler != "slurm" and uenv and not os.environ.get("_UENV_ACTIVE"):
        env = {**os.environ, "_UENV_ACTIVE": "1"}
        os.execvpe(
            "uenv",
            ["uenv", "run", "--view=modules", uenv, "--", sys.executable] + sys.argv,
            env,
        )

    default_cc, default_cxx = _compiler_for_toolchain(toolchain)
    cc  = tc_conf.get("cc",  default_cc)
    cxx = tc_conf.get("cxx", default_cxx)

    # On Slurm, Python is resolved at runtime after modules load so that cmake
    # and pip target the same interpreter the benchmark will use. On local
    # systems compile.py already runs under the correct interpreter.
    if scheduler == "slurm":
        python_decl = "PYTHON_EXE=$(which python3 2>/dev/null || which python)"
    else:
        python_decl = f'PYTHON_EXE="{sys.executable}"'

    # CMake flags
    if sys_type == "gpu":
        if gpu_arch.startswith("gfx"):
            cmake_flags = (
                f"-DBUILD_GPU=ON -DGPU_PLATFORM=ROCM "
                f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch} "
                f"-DSYSTEM_NAME={sys_name}"
            )
        else:
            cmake_flags = (
                f"-DBUILD_GPU=ON -DGPU_PLATFORM=CUDA "
                f"-DCMAKE_CUDA_ARCHITECTURES={gpu_arch} "
                f"-DSYSTEM_NAME={sys_name}"
            )
    else:
        cmake_flags = f"-DBUILD_GPU=OFF -DSYSTEM_NAME={sys_name}"

    if likwid:
        cmake_flags += " -DUSE_LIKWID=ON"

    # Module-load commands (HPC only; empty string on local/bare systems)
    module_cmds = "\n".join(
        f"module load {m}" for m in modules_build.split(":") if m
    )

    # hsmmlearn baselines: CPU only, compiled with the same toolchain to avoid
    # OMP runtime mismatches between the baseline and the native extension.
    hsmmlearn_build = ""
    if sys_type == "cpu":
        hsmmlearn_build = textwrap.dedent(f"""\
            echo "Building hsmmlearn with CC={cc} CXX={cxx} ..."
            "$PYTHON_EXE" -m pip install --quiet wheel setuptools

            if [ "{int(likwid)}" -eq 1 ]; then
                LIKWID_INC=$(cmake -LA -N "{build_dir}" | grep '^LIKWID_INCLUDE_DIR:' | cut -d= -f2)
                LIKWID_LIB=$(cmake -LA -N "{build_dir}" | grep '^LIKWID_LIB_DIR:' | cut -d= -f2)

                if [ -z "$LIKWID_INC" ] || [ -z "$LIKWID_LIB" ]; then
                    echo "ERROR: LIKWID requested but paths are empty"
                    exit 1
                fi

                USE_LIKWID=1
            else
                USE_LIKWID=0
                LIKWID_INC=""
                LIKWID_LIB=""
            fi

            CC={cc} CXX={cxx} \\
            USE_LIKWID=$USE_LIKWID \\
            LIKWID_INCLUDE_DIR=$LIKWID_INC \\
            LIKWID_LIB_DIR=$LIKWID_LIB \\
            "$PYTHON_EXE" -m pip install --no-build-isolation --force-reinstall --no-deps "{SCRIPT_DIR}/hsmmlearn"

            CC={cc} CXX={cxx} \\
            USE_LIKWID=$USE_LIKWID \\
            LIKWID_INCLUDE_DIR=$LIKWID_INC \\
            LIKWID_LIB_DIR=$LIKWID_LIB \\
            "$PYTHON_EXE" -m pip install --no-build-isolation --force-reinstall --no-deps "{SCRIPT_DIR}/hsmmlearn_omp"
        """)

    script = f"""\
set -e
{module_cmds}
        {python_decl}

export CC={cc} CXX={cxx}

rm -rf "{build_dir}"
cmake -B "{build_dir}" -DPYTHON_EXECUTABLE="$PYTHON_EXE" {cmake_flags}
cmake --build "{build_dir}" -j 1

{hsmmlearn_build}"""

    if scheduler == "slurm":
        _run_compile_slurm(system, toolchain, sys_conf, tc_conf, script)
    else:
        result = subprocess.run(["bash", "-c", script], cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            sys.exit(result.returncode)
