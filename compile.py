#!/usr/bin/env python3
"""
compile.py — build the native extension for a given system/toolchain pair.
Reads system configuration from systems.json.

Activate your virtual environment and install dependencies before running:
    pip install -r requirements.txt
    python compile.py --system <system> --toolchain <toolchain>

Usage:
    python compile.py --system <system> --toolchain <toolchain>
    python compile.py --system <system> --toolchain all
"""

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[.+]", v) if x.isdigit())


def check_requirements() -> None:
    """Verify requirements.txt against the active Python environment. Exit on failure."""
    ops = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        "<":  lambda a, b: a < b,
    }
    req_file = SCRIPT_DIR / "requirements.txt"
    missing: list[str] = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*([><=!]{1,2})\s*([0-9][0-9.]*)", line)
        pkg     = m.group(1) if m else line
        op      = m.group(2) if m else None
        req_ver = m.group(3) if m else None
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"  {pkg}: not installed")
            continue
        if op and req_ver and not ops[op](_ver(installed), _ver(req_ver)):
            missing.append(f"  {pkg}: need {op}{req_ver}, got {installed}")

    if missing:
        print("Error: missing or outdated requirements:")
        print("\n".join(missing))
        print(f"Install them with: pip install -r {req_file}")
        sys.exit(1)


def load_systems() -> dict:
    with open(SCRIPT_DIR / "systems.json") as f:
        return json.load(f)


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
            srun_flags += ["--cpus-per-task=8", "--gres=gpu:1"]
        else:
            srun_flags.append("--cpus-per-task=8")

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
    sys_name      = f"{system}/{toolchain}"
    build_dir     = SCRIPT_DIR / "build" / system / toolchain

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

        hsmmlearn_build = f"""\
            echo "Building hsmmlearn with CC={cc} CXX={cxx} ..."
            "{sys.executable}" -m pip install --quiet wheel setuptools

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

            CC={cc} CXX={cxx} \
            USE_LIKWID=$USE_LIKWID \
            LIKWID_INCLUDE_DIR=$LIKWID_INC \
            LIKWID_LIB_DIR=$LIKWID_LIB \
            "{sys.executable}" -m pip install --quiet --no-build-isolation "{SCRIPT_DIR}/hsmmlearn"

            CC={cc} CXX={cxx} \
            USE_LIKWID=$USE_LIKWID \
            LIKWID_INCLUDE_DIR=$LIKWID_INC \
            LIKWID_LIB_DIR=$LIKWID_LIB \
            "{sys.executable}" -m pip install --quiet --no-build-isolation "{SCRIPT_DIR}/hsmmlearn_omp"
        """

    script = f"""\
        set -e
        {module_cmds}

        export CC={cc} CXX={cxx}

        rm -rf "{build_dir}"
        cmake -B "{build_dir}" -DPYTHON_EXECUTABLE="{sys.executable}" {cmake_flags}
        cmake --build "{build_dir}" -j 8

        {hsmmlearn_build}
    """

    if scheduler == "slurm":
        _run_compile_slurm(system, toolchain, sys_conf, tc_conf, script)
    else:
        result = subprocess.run(["bash", "-c", script], cwd=str(SCRIPT_DIR))
        if result.returncode != 0:
            sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the tensor-viterbi native extension for a given system/toolchain.\n"
            "Activate your virtual environment and install requirements.txt first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system",    "-s", required=True,
                        help="System key from systems.json")
    parser.add_argument("--toolchain", "-t", required=True,
                        help="Toolchain key, or 'all' to build every toolchain for the system")
    parser.add_argument("--likwid", action="store_true", default=False,
                        help="Enable LIKWID marker API instrumentation (-DUSE_LIKWID=ON)")
    args = parser.parse_args()

    # Python version gate
    if sys.version_info < (3, 10):
        print(f"Error: Python >= 3.10 required, found {sys.version.split()[0]}.")
        print("Activate a Python 3.10+ environment before running compile.py.")
        sys.exit(1)

    check_requirements()

    if Path.cwd().resolve() != SCRIPT_DIR:
        print("Error: must be run from the repository root.")
        print(f"  cd {SCRIPT_DIR} && python {Path(__file__).name} {' '.join(sys.argv[1:])}")
        sys.exit(1)

    systems = load_systems()

    if args.system not in systems:
        print(f"Error: Unknown system '{args.system}'.")
        print(f"Available systems: {', '.join(systems)}")
        sys.exit(1)

    sys_conf   = systems[args.system]
    toolchains = sys_conf.get("toolchains", {})

    if args.toolchain == "all":
        if not toolchains:
            print(f"Error: No toolchains defined for system '{args.system}'.")
            sys.exit(1)
        for tc in sorted(toolchains):
            print(f"=== Compiling {args.system} / {tc} ===")
            compile_system(args.system, tc, sys_conf, toolchains[tc], args.likwid)
        return

    if args.toolchain not in toolchains:
        print(f"Error: Toolchain '{args.toolchain}' not defined for system '{args.system}'.")
        print(f"Known toolchains: {', '.join(toolchains)}")
        sys.exit(1)

    compile_system(args.system, args.toolchain, sys_conf, toolchains[args.toolchain], args.likwid)


if __name__ == "__main__":
    main()
