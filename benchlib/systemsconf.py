"""Load and validate a single system YAML file (systems/<name>.yaml or an
explicit path). There is no scanning/auto-detection: the caller always
supplies a name or path explicitly via --system.

Normalized shape returned by load_system():

    {
      "name": str, "path": Path, "type": "cpu"|"gpu", "scheduler": "local"|"slurm",
      "cpus": int, "omp_bind": str, "omp_places": str, "gpu_arch": str,
      "toolchains": {
          tc_name: {"modules": str, "modules_build": str, "uenv": str,
                     "metrics_backend": str, "cc": str, "cxx": str},
          ...
      },
      "slurm": {"account": str|None, "partition": str|None, "qos": str|None},
    }

"modules"/"modules_build" are normalized to the colon-joined string form
that run_one.sh / compile.py already expect, regardless of whether the YAML
author wrote a list or a string.
"""


import os
from pathlib import Path

import yaml

from .paths import SYSTEMS_DIR

TOP_KEYS = {
    "name", "type", "toolchain", "toolchains", "scheduler",
    "cpus", "omp_bind", "omp_places", "metrics_backend", "gpu_arch",
    "cc", "cxx", "slurm",
}
SLURM_KEYS = {"account", "partition", "qos", "modules", "modules_build", "uenv"}
TOOLCHAIN_KEYS = {"modules", "modules_build", "uenv", "metrics_backend", "cc", "cxx"}

# Toolchain names that unambiguously imply GPU code (see cross-field check
# below). Everything else (gnu, intel, amd, cray, llvm, fujitsu, aocc, ...)
# is just a host-compiler label and is used on both cpu and gpu systems in
# this repo (e.g. toolchain "gnu" builds both gh200-grace [cpu] and
# gh200-hopper [gpu]) so it cannot be used to infer `type`.
GPU_ONLY_TOOLCHAINS = {"cuda", "hip"}


class SystemConfigError(Exception):
    """Raised with a fully-formatted, multi-line, actionable message."""


def resolve_system_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    return SYSTEMS_DIR / f"{name_or_path}.yaml"


def _normalize_modules(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ":".join(str(v) for v in value if v)
    return str(value)


def _check_unknown_keys(d: dict, allowed: set[str], where: str, warnings: list[str]) -> None:
    for key in d:
        if key not in allowed:
            warnings.append(f"unknown key '{key}' in {where} (ignored)")


def load_system(name_or_path: str) -> tuple[dict, list[str]]:
    """Load + validate one system file.

    Returns (normalized_config, warnings). Raises SystemConfigError with a
    complete, field-level message (all problems at once) on hard failures.
    """
    path = resolve_system_path(name_or_path)
    if not path.exists():
        raise SystemConfigError(
            f"Error: system file not found: {path}\n"
            f"  Pass a name (resolved to systems/<name>.yaml) or a path to a YAML file.\n"
            f"  Available systems: {', '.join(sorted(p.stem for p in SYSTEMS_DIR.glob('*.yaml') if p.stem != 'TEMPLATE'))}"
        )

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise SystemConfigError(f"Error in {path}:\n  invalid YAML: {e}")

    errors: list[str] = []
    warnings: list[str] = []

    _check_unknown_keys(raw, TOP_KEYS, str(path), warnings)

    name = raw.get("name")
    if not name:
        errors.append("`name` is required.")

    sys_type = raw.get("type")
    if sys_type not in ("cpu", "gpu"):
        errors.append(f"`type` must be 'cpu' or 'gpu', got {sys_type!r}.")

    scheduler = raw.get("scheduler")
    if scheduler not in ("local", "slurm"):
        errors.append(f"`scheduler` must be 'local' or 'slurm', got {scheduler!r}.")

    has_flat_tc  = "toolchain" in raw
    has_map_tc   = "toolchains" in raw
    if has_flat_tc and has_map_tc:
        errors.append("specify either `toolchain` or `toolchains`, not both.")
    elif not has_flat_tc and not has_map_tc:
        errors.append("one of `toolchain` (single) or `toolchains` (map) is required.")

    toolchains: dict = {}
    if has_flat_tc:
        tc_name = raw["toolchain"]
        slurm_raw = raw.get("slurm") or {}
        toolchains[tc_name] = {
            "modules":         _normalize_modules(slurm_raw.get("modules")),
            "modules_build":   _normalize_modules(slurm_raw.get("modules_build")) or _normalize_modules(slurm_raw.get("modules")),
            "uenv":            slurm_raw.get("uenv", "") or "",
            "metrics_backend": raw.get("metrics_backend", "") or "",
            "cc":              raw.get("cc", "") or "",
            "cxx":             raw.get("cxx", "") or "",
        }
    elif has_map_tc:
        tc_map = raw.get("toolchains") or {}
        if not tc_map:
            errors.append("`toolchains` map is empty.")
        for tc_name, tc_conf in tc_map.items():
            tc_conf = tc_conf or {}
            _check_unknown_keys(tc_conf, TOOLCHAIN_KEYS, f"toolchains.{tc_name}", warnings)
            modules = _normalize_modules(tc_conf.get("modules"))
            toolchains[tc_name] = {
                "modules":         modules,
                "modules_build":   _normalize_modules(tc_conf.get("modules_build")) or modules,
                "uenv":            tc_conf.get("uenv", "") or "",
                "metrics_backend": tc_conf.get("metrics_backend", "") or "",
                "cc":              tc_conf.get("cc", "") or "",
                "cxx":             tc_conf.get("cxx", "") or "",
            }

    for tc_name in toolchains:
        if tc_name in GPU_ONLY_TOOLCHAINS and sys_type != "gpu":
            errors.append(
                f"toolchain '{tc_name}' implies GPU code but `type` is {sys_type!r}. "
                f"Set `type: gpu`."
            )

    slurm_raw = raw.get("slurm")
    slurm_conf = {"account": None, "partition": None, "qos": None}
    if scheduler == "slurm":
        if slurm_raw is None:
            errors.append(
                "scheduler is 'slurm' but the `slurm:` block is missing.\n"
                "    Add a `slurm:` block with at least `account:`."
            )
            slurm_raw = {}
        else:
            _check_unknown_keys(slurm_raw, SLURM_KEYS, "slurm", warnings)
        slurm_conf["account"]   = slurm_raw.get("account")
        slurm_conf["partition"] = slurm_raw.get("partition")
        slurm_conf["qos"]       = slurm_raw.get("qos")
        if not slurm_conf["account"]:
            errors.append(
                "scheduler is 'slurm' but slurm.account is missing.\n"
                "    → Find your account with: sacctmgr show user $USER format=account"
            )
        if not slurm_conf["partition"] and not slurm_conf["qos"]:
            warnings.append(
                "scheduler is 'slurm' but neither slurm.partition nor slurm.qos is set "
                "— sbatch will use the cluster default partition"
            )
    else:
        if slurm_raw:
            warnings.append("`slurm:` block is set but scheduler is 'local' (ignored)")

    if sys_type == "gpu" and not raw.get("gpu_arch"):
        errors.append("`type: gpu` requires `gpu_arch` (CUDA compute cap like \"90\" or ROCm gfx code like \"gfx90a\").")

    if errors:
        lines = "\n".join(f"  {e}" for e in errors)
        raise SystemConfigError(f"Error in {path}:\n{lines}")

    conf = {
        "name":            name,
        "path":            path,
        "type":            sys_type,
        "scheduler":       scheduler,
        "cpus":            raw.get("cpus") or os.cpu_count() or 1,
        "omp_bind":        raw.get("omp_bind", "close") or "close",
        "omp_places":      raw.get("omp_places", "cores") or "cores",
        "gpu_arch":        raw.get("gpu_arch", "") or "",
        "toolchains":      toolchains,
        "slurm":           slurm_conf,
    }
    return conf, warnings


def list_available_systems() -> list[str]:
    return sorted(p.stem for p in SYSTEMS_DIR.glob("*.yaml") if p.stem != "TEMPLATE")


def select_toolchains(conf: dict, requested: str | None, allow_all: bool = True) -> list[str]:
    """Resolve --toolchain <tc>|all|None against a loaded system config."""
    available = conf["toolchains"]
    if requested is None or requested == "":
        if len(available) == 1:
            return list(available)
        raise SystemConfigError(
            f"System '{conf['name']}' defines multiple toolchains "
            f"({', '.join(sorted(available))}); pass --toolchain "
            f"{'<tc>|all' if allow_all else '<tc>'}."
        )
    if requested == "all":
        if not allow_all:
            raise SystemConfigError(
                "'all' is not supported for --toolchain here; pass one of "
                f"{', '.join(sorted(available))}."
            )
        if not available:
            raise SystemConfigError(f"No toolchains defined for system '{conf['name']}'.")
        return sorted(available)
    if requested not in available:
        raise SystemConfigError(
            f"Toolchain '{requested}' not defined for system '{conf['name']}'. "
            f"Known toolchains: {', '.join(sorted(available))}"
        )
    return [requested]
