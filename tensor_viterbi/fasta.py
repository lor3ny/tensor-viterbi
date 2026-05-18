from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class FastaReader:
    """Read a FASTA file and map nucleotides to observation indices.

    Symbols must be defined before reading so the mapping is fixed.
    Characters not in the symbol set are skipped silently.

    Parameters
    ----------
    path:
        Path to the .fa / .fasta file.
    symbols:
        Ordered list of valid characters (e.g. ["A","C","G","T","N"]).
        Position in the list is the observation index passed to the model.
        Matching is case-insensitive.
    """

    def __init__(self, path: str | Path, symbols: list[str]):
        self._path = Path(path)
        self.symbols = symbols
        self._index: dict[str, int] = {s.upper(): i for i, s in enumerate(symbols)}

    # ------------------------------------------------------------------
    # Core generator — skips headers and unknown characters
    # ------------------------------------------------------------------

    def _iter_indices(self):
        with open(self._path) as f:
            for line in f:
                if line.startswith(">"):
                    continue
                for ch in line.rstrip():
                    idx = self._index.get(ch.upper())
                    if idx is not None:
                        yield idx

    # ------------------------------------------------------------------
    # Public reading API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Load the entire sequence as a numpy array of observation indices."""
        return np.fromiter(self._iter_indices(), dtype=np.int64)

    def iter_chars(self):
        """Yield one observation index at a time (memory-efficient)."""
        yield from self._iter_indices()

    def iter_windows(self, k: int):
        """Yield sliding windows of length k (step=1) as int64 arrays."""
        if k < 1:
            raise ValueError(f"Window size must be >= 1, got {k}")
        buf = deque()
        for idx in self._iter_indices():
            buf.append(idx)
            if len(buf) == k:
                yield np.array(buf, dtype=np.int64)
                buf.popleft()
