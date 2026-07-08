"""bench check — validate a system YAML and probe the environment. Runs no
jobs. Exit policy:
  - YAML validation errors                       -> hard failure (exit 1)
  - scheduler: slurm but `sbatch` missing         -> hard failure (exit 1)
  - compiler / nvcc / hipcc / likwid-perfctr      -> informational only
    (these are often provided by `module load` at run time, which this
    check cannot simulate without actually loading modules, so a missing
    binary is only a hard error when the system has no modules configured
    at all to plausibly supply it)
"""

from __future__ import annotations

import shutil

from .systemsconf import SystemConfigError, load_system, select_toolchains

try:
    from compile import _compiler_for_toolchain
except ImportError:  # pragma: no cover - defensive only
    def _compiler_for_toolchain(_toolchain: str) -> tuple[str, str]:
        return "gcc", "g++"


def run_check(system_arg: str, toolchain_arg: str | None) -> int:
    try:
        conf, warnings = load_system(system_arg)
    except SystemConfigError as e:
        print(str(e))
        return 1

    for w in warnings:
        print(f"Warning in {conf['path']}: {w}")

    print(f"System '{conf['name']}': type={conf['type']} scheduler={conf['scheduler']} "
          f"toolchains={','.join(sorted(conf['toolchains']))}")

    hard_failed = False

    if conf["scheduler"] == "slurm":
        if shutil.which("sbatch") is None:
            print("[✗] sbatch not found in PATH, but scheduler is 'slurm'.")
            hard_failed = True
        else:
            print("[✓] sbatch found")

    try:
        toolchains = select_toolchains(conf, toolchain_arg or "all")
    except SystemConfigError as e:
        print(str(e))
        return 1

    for tc in toolchains:
        tc_conf = conf["toolchains"][tc]
        default_cc, default_cxx = _compiler_for_toolchain(tc)
        cc = tc_conf.get("cc") or default_cc
        has_modules = bool(tc_conf.get("modules"))

        if shutil.which(cc):
            print(f"[✓] {tc}: compiler '{cc}' found")
        elif has_modules:
            print(f"[i] {tc}: compiler '{cc}' not found yet "
                  f"(should be provided by modules: {tc_conf['modules']})")
        else:
            print(f"[!] {tc}: compiler '{cc}' not found in PATH")

        if conf["type"] == "gpu":
            gpu_tool = "hipcc" if conf["gpu_arch"].startswith("gfx") else "nvcc"
            if shutil.which(gpu_tool):
                print(f"[✓] {tc}: {gpu_tool} found")
            elif has_modules:
                print(f"[i] {tc}: {gpu_tool} not found yet "
                      f"(should be provided by modules: {tc_conf['modules']})")
            else:
                print(f"[!] {tc}: {gpu_tool} not found in PATH")

    if conf["type"] == "cpu":
        if shutil.which("likwid-perfctr"):
            print("[✓] likwid-perfctr found (bench likwid available)")
        else:
            print("[i] likwid-perfctr not found (bench likwid will not work until it's installed)")

    return 1 if hard_failed else 0
