"""
metrics.py — Per-iteration hardware metrics scaffolding.

Usage
-----
    collector = get_collector(backend)   # once, before the benchmark loop
    for i in range(iterations):
        collector.start()
        # ... timed region ...
        elapsed = stop_timer()
        extra = collector.stop()         # dict[str, float], may be empty
        row = [..., elapsed, *[extra.get(k) for k in collector.column_names()]]

Adding a new backend
--------------------
1. Subclass Collector and implement start(), stop(), column_names().
2. Register the token string in get_collector().
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod


class Collector(ABC):
    """Base class for per-iteration hardware metric collectors."""

    @abstractmethod
    def start(self) -> None:
        """Called immediately before the timed region begins."""

    @abstractmethod
    def stop(self) -> dict[str, float]:
        """Called immediately after the timed region ends.

        Returns a dict whose keys match column_names().  Missing keys are
        written as empty strings in the CSV.
        """

    @abstractmethod
    def column_names(self) -> list[str]:
        """Ordered list of extra CSV column names this collector emits.

        Must be stable across start()/stop() calls for a single run.
        """


class NullCollector(Collector):
    """No-op collector used when no metrics backend is configured."""

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, float]:
        return {}

    def column_names(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Cray power management counters  (epyc-7763 / partition C on LUMI)
# ---------------------------------------------------------------------------

class CrayPMCollector(Collector):
    """Read Cray pm_counters energy files before and after each decode call.

    Each counter file has the format:
        <joules> J <microseconds> us

    Base CSV columns (always present when the node exposes them):
        energy_j,        energy_us
        cpu_energy_j,    cpu_energy_us
        memory_energy_j, memory_energy_us

    Additional columns are added automatically for each accelN_energy file
    found in the pm_counters directory (accel0 … accel3):
        accel0_energy_j, accel0_energy_us
        accel1_energy_j, accel1_energy_us
        ...
    """

    _BASE_COUNTERS = ("energy", "cpu_energy", "memory_energy")
    _ACCEL_RANGE   = range(4)          # accel0_energy … accel3_energy
    _BASE = "/sys/cray/pm_counters"

    def __init__(self) -> None:
        counters = list(self._BASE_COUNTERS)
        for i in self._ACCEL_RANGE:
            name = f"accel{i}_energy"
            if os.path.exists(f"{self._BASE}/{name}"):
                counters.append(name)
        self._counters: tuple[str, ...] = tuple(counters)
        self._start: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _read(name: str) -> tuple[int, int]:
        """Return (joules, microseconds) from a pm_counters file."""
        path = f"{CrayPMCollector._BASE}/{name}"
        with open(path) as fh:
            parts = fh.read().split()
        # format: "<J> J <us> us"
        return int(parts[0]), int(parts[2])

    def start(self) -> None:
        self._start = {name: self._read(name) for name in self._counters}

    def stop(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in self._counters:
            j1, us1 = self._start[name]
            j2, us2 = self._read(name)
            result[f"{name}_j"]  = j2  - j1
            result[f"{name}_us"] = us2 - us1
        return result

    def column_names(self) -> list[str]:
        cols = []
        for name in self._counters:
            cols.append(f"{name}_j")
            cols.append(f"{name}_us")
        return cols


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Collector]] = {
    "cray_pm": CrayPMCollector,
}


def get_collector(backend: str | None) -> Collector:
    """Return the Collector for *backend* token.

    ``backend`` comes from the ``SYS_METRICS_BACKEND`` environment variable
    (set per system/toolchain in systems.conf and exported by run.slrm).
    An absent or empty token returns a NullCollector.

    Raises ValueError for unknown non-empty tokens so mis-configuration is
    caught at startup rather than silently dropped.
    """
    if not backend:
        return NullCollector()
    cls = _REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown metrics backend {backend!r}. "
            f"Available: {sorted(_REGISTRY) or ['(none yet)']}"
        )
    return cls()
