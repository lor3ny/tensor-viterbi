"""benchmark_params.cfg grid + walltimes.yaml lookup + walltime packs."""


import sys
from pathlib import Path

import yaml

from . import flags as flagslib
from .paths import PARAMS_FILE, STRESS_PARAMS_FILE, WALLTIMES_FILE

# A job's total walltime is never requested below this floor (scheduling
# overhead noise) or above this ceiling (top of the "extra" pack) even if
# the underlying per-version estimates would sum to more.
MIN_WALLTIME_SECONDS = 60
MAX_WALLTIME_SECONDS = 20 * 3600

# Walltime buckets (30m/1h/2h/4h/5h/6h/8h/10h/14h/16h/20h grid values, with a
# natural gap between 8h and 20h/10h so "large" and "extra" don't overlap).
PACKS: dict[str, tuple[int, int]] = {
    "small":  (0,             1 * 3600),
    "medium": (1 * 3600 + 1,  2 * 3600),
    "large":  (2 * 3600 + 1,  8 * 3600),
    "extra":  (8 * 3600 + 1, 20 * 3600),
}

# The GPU-only stress-test pack: unlike PACKS above, it isn't a walltime
# bucket over benchmark_params.cfg — it has its own dedicated grid
# (benchmark_params_stress.cfg, loaded in full, no walltime filtering) and
# is only ever run with --gpu. Kept out of PACKS so the normal
# walltime-bucketing logic (pack_of_walltime, validate_grid_covered_by_packs)
# stays untouched by it.
STRESS_PACK = "stress"


def resolve_pack_name(pack: str) -> str:
    if pack == STRESS_PACK:
        return pack
    if pack not in PACKS:
        valid = ", ".join(list(PACKS) + [STRESS_PACK])
        print(f"Error: unknown pack '{pack}'. Valid packs: {valid}")
        sys.exit(1)
    return pack


def load_benchmark_params(params_file: Path = PARAMS_FILE) -> tuple[list[int], list[int], list[int]]:
    params: dict[str, list[int]] = {}
    for line in params_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.partition("=")
        params[key.strip()] = [int(v) for v in raw.split(",") if v.strip()]
    try:
        return params["states"], params["durations"], params["timesteps"]
    except KeyError as e:
        print(f"Error: {params_file.name} is missing key {e}")
        sys.exit(1)


def hms_to_seconds(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_hms(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_walltimes() -> tuple[dict[str, dict[tuple[int, int, int], int]], int]:
    """Returns ({version: {(states, duration, timesteps): per-iteration seconds}},
    default_per_iteration_seconds)."""
    raw = yaml.safe_load(WALLTIMES_FILE.read_text()) or {}
    versions: dict[str, dict[tuple[int, int, int], int]] = {}
    for version, table in (raw.get("versions") or {}).items():
        entries: dict[tuple[int, int, int], int] = {}
        for key, wt in (table or {}).items():
            s, d, t = (int(x) for x in key.split("_"))
            entries[(s, d, t)] = hms_to_seconds(wt)
        versions[version] = entries
    default = hms_to_seconds(raw.get("default", "01:00:00"))
    return versions, default


_VERSION_ENTRIES, _DEFAULT_SECONDS = load_walltimes()


def effective_iterations(timesteps: int, iterations: int) -> int:
    """Mirrors manifest.build_jobs's iteration cap for large timestep counts."""
    return min(iterations, 2) if timesteps >= 1_000_000 else iterations


def get_walltime(states: int, duration: int, timesteps: int, viterbi_flags: str,
                  iterations: int) -> str:
    """Sums the per-iteration walltime of every version selected by
    `viterbi_flags`, multiplies by the job's iteration count, and clamps to
    [MIN_WALLTIME_SECONDS, MAX_WALLTIME_SECONDS]."""
    per_iteration = sum(
        _VERSION_ENTRIES.get(v, {}).get((states, duration, timesteps), _DEFAULT_SECONDS)
        for v in flagslib.expand_versions(viterbi_flags)
    )
    total = per_iteration * iterations
    total = max(MIN_WALLTIME_SECONDS, min(total, MAX_WALLTIME_SECONDS))
    return seconds_to_hms(total)


def pack_of_walltime(walltime: str) -> str | None:
    seconds = hms_to_seconds(walltime)
    for name, (lo, hi) in PACKS.items():
        if lo <= seconds <= hi:
            return name
    return None


def validate_grid_covered_by_packs(viterbi_flags: str, iterations: int) -> list[tuple[int, int, int, str]]:
    """Returns the list of (states, duration, timesteps, walltime) grid points
    whose walltime falls outside every pack — should be empty for the shipped
    grid; used by `bench plan` to fail loudly if walltimes.yaml/PACKS drift."""
    states, durations, timesteps = load_benchmark_params()
    orphans = []
    for s in states:
        for d in durations:
            for t in timesteps:
                iters = effective_iterations(t, iterations)
                wt = get_walltime(s, d, t, viterbi_flags, iters)
                if pack_of_walltime(wt) is None:
                    orphans.append((s, d, t, wt))
    return orphans
