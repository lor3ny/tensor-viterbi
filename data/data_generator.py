"""
generate_hmm_input.py
=====================
Generic Hidden Semi-Markov Model (HSMM) input-JSON generator.

The function `generate_hmm_input` is fully application-neutral:
it accepts explicit numerical arrays for all model parameters and
an observation sequence, then validates, normalises, and serialises
everything to a JSON file compatible with ViterbiImpl.

Usage example (mimicking the original sleep script) is at the bottom
of this file under  if __name__ == "__main__".
"""

from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np


# =============================================================================
# Core generator
# =============================================================================

def generate_hmm_input(
    *,
    # ── model dimensions ──────────────────────────────────────────────────────
    n_states:   int,
    n_emissions: int,
    n_steps:    int,
    max_duration: int,
    # ── required probability arrays ──────────────────────────────────────────
    trans_mat:       Sequence[Sequence[float]],   # (n_states × n_states)
    emission_probs:  Sequence[Sequence[float]],   # (n_emissions × n_states)  ← col = state
    start_probs:     Sequence[float],             # (n_states,)
    duration_probs:  Sequence[Sequence[float]],   # (n_states × max_duration)
    obs_seq:         Sequence[int],               # (n_steps,)  1-indexed values
    # ── optional metadata ─────────────────────────────────────────────────────
    state_names:     Sequence[str] | None = None,
    emission_type:   str = "nonp",
    duration_type:   str = "nonp",
    seed:            int | None = None,
    # ── output ────────────────────────────────────────────────────────────────
    out_path:        str | None = None,
) -> dict:
    """
    Build and (optionally) write a JSON config for ViterbiImpl.

    Parameters
    ----------
    n_states        : Number of hidden states  (J in ViterbiImpl).
    n_emissions     : Number of discrete emission symbols / bins.
    n_steps         : Length of the observation sequence  (tau).
    max_duration    : Maximum state-sojourn duration  (M).
    trans_mat       : Row-stochastic transition matrix; shape (n_states, n_states).
                      Diagonal entries are ignored / treated as 0 by the
                      semi-Markov model but are accepted here for convenience.
    emission_probs  : Raw (un-normalised) emission probabilities.
                      Shape (n_emissions, n_states): each *column* corresponds
                      to one state and will be normalised to sum to 1.
    start_probs     : Initial state distribution; will be normalised to sum to 1.
    duration_probs  : Sojourn-duration distributions.
                      Shape (n_states, max_duration): each *row* is the PMF
                      over durations 1 … max_duration for that state and will
                      be normalised to sum to 1.
    obs_seq         : Observed symbol indices (1-indexed, i.e. 1 … n_emissions).
                      Must have length == n_steps.
    state_names     : Optional list of human-readable state labels.
                      Defaults to ["s0", "s1", …].
    emission_type   : Tag stored in JSON ("nonp", "gaussian", …).
    duration_type   : Tag stored in JSON ("nonp", "gaussian", …).
    seed            : Optional RNG seed recorded in JSON for reproducibility.
    out_path        : If given, write the JSON to this path (directories are
                      created automatically).

    Returns
    -------
    dict  – the complete config dictionary (also written to *out_path* when set).
    """

    # ── 1. coerce to numpy ───────────────────────────────────────────────────
    trans      = np.array(trans_mat,      dtype=float)
    emis_raw   = np.array(emission_probs, dtype=float)   # (n_emissions, n_states)
    pi_raw     = np.array(start_probs,    dtype=float)
    dur_raw    = np.array(duration_probs, dtype=float)   # (n_states, max_duration)
    obs        = list(obs_seq)

    # ── 2. dimension checks ──────────────────────────────────────────────────
    J, M, E, T = n_states, max_duration, n_emissions, n_steps

    if trans.shape != (J, J):
        raise ValueError(f"trans_mat must be ({J},{J}), got {trans.shape}")
    if emis_raw.shape != (E, J):
        raise ValueError(f"emission_probs must be ({E},{J}), got {emis_raw.shape}")
    if pi_raw.shape != (J,):
        raise ValueError(f"start_probs must have length {J}, got {pi_raw.shape}")
    if dur_raw.shape != (J, M):
        raise ValueError(f"duration_probs must be ({J},{M}), got {dur_raw.shape}")
    if len(obs) != T:
        raise ValueError(f"obs_seq length must be {T}, got {len(obs)}")
    if any(o < 1 or o > E for o in obs):
        raise ValueError(f"obs_seq values must be in [1, {E}]")

    # ── 3. normalise ─────────────────────────────────────────────────────────
    # Transition rows
    row_sums = trans.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("trans_mat has a row that sums to zero")
    trans = trans / row_sums

    # Emission columns (one column per state)
    col_sums = emis_raw.sum(axis=0, keepdims=True)
    if np.any(col_sums == 0):
        raise ValueError("emission_probs has a column (state) that sums to zero")
    emis_norm = emis_raw / col_sums                        # (n_emissions, n_states)

    # emission_by_state[s] = list of length n_emissions
    emission_by_state = [
        [round(float(emis_norm[b, s]), 6) for b in range(E)]
        for s in range(J)
    ]

    # Start probabilities
    pi_sum = pi_raw.sum()
    if pi_sum == 0:
        raise ValueError("start_probs sums to zero")
    pi_norm = pi_raw / pi_sum

    # Duration rows
    dur_sums = dur_raw.sum(axis=1, keepdims=True)
    if np.any(dur_sums == 0):
        raise ValueError("duration_probs has a row (state) that sums to zero")
    dur_norm = dur_raw / dur_sums

    # ── 4. default state names ───────────────────────────────────────────────
    if state_names is None:
        names = [f"s{i}" for i in range(J)]
    else:
        names = list(state_names)
        if len(names) != J:
            raise ValueError(f"state_names must have length {J}, got {len(names)}")

    # ── 5. build states block ────────────────────────────────────────────────
    states_block = [
        {
            "id":             s,
            "name":           names[s],
            "emission_probs": emission_by_state[s],
            "duration_probs": [round(float(v), 8) for v in dur_norm[s]],
        }
        for s in range(J)
    ]

    # ── 6. assemble config ───────────────────────────────────────────────────
    config: dict = {
        "n_steps":   T,
        "M":         M,
        "emission":  emission_type,
        "duration":  duration_type,
        "obs_seq":   obs,
        "pi":        [round(float(v), 6) for v in pi_norm],
        "trans_mat": [[round(float(v), 6) for v in row] for row in trans.tolist()],
        "n_bins":    E,
        "states":    states_block,
    }
    if seed is not None:
        config["seed"] = seed

    # ── 7. optional write ────────────────────────────────────────────────────
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(config, fh, indent=2)
        print(f"Config written to: {out_path}")

    # ── 8. validation summary ────────────────────────────────────────────────
    _print_summary(config, names, trans, emis_norm, pi_norm)

    return config


# =============================================================================
# Helpers
# =============================================================================

def gaussian_duration_pmf(max_duration: int, mean: float, std: float) -> list[float]:
    """Return a normalised Gaussian PMF over durations 1 … max_duration."""
    x = np.arange(max_duration)
    g = np.exp(-0.5 * ((x - mean) / std) ** 2)
    g /= g.sum()
    return g.tolist()


def _print_summary(config, names, trans, emis_norm, pi_norm) -> None:
    J = len(names)
    E = config["n_bins"]
    M = config["M"]
    print(f"\n{'─'*60}")
    print(f"States ({J}): {names}")
    print(f"Emissions    : {E} bins")
    print(f"M (max dur)  : {M}")
    print(f"Timesteps    : {config['n_steps']}")
    print(f"emission type: {config['emission']}   "
          f"duration type: {config['duration']}")
    print(f"pi           : {[round(float(v),4) for v in pi_norm]}")
    print("trans_mat rows (rounded to 3dp):")
    for name, row in zip(names, trans):
        print(f"  {name:<18}: {[round(float(v),3) for v in row]}")
    print("Row-sum checks (trans_mat, should all be 1.0):")
    for name, row in zip(names, trans):
        print(f"  {name:<18}: {row.sum():.6f}")
    print("Emission column-sum checks (should all be 1.0):")
    for s, name in enumerate(names):
        print(f"  {name:<18}: {emis_norm[:, s].sum():.6f}")
    print(f"{'─'*60}\n")


# =============================================================================
# Random parameter generators (scale to any N)
# =============================================================================

def _random_trans(n_states: int, rng: np.random.Generator) -> np.ndarray:
    """
    Row-stochastic transition matrix with zero diagonal (no self-loops),
    sparse enough to look realistic: each state connects strongly to 1-3
    neighbours and weakly to the rest.
    """
    raw = np.zeros((n_states, n_states))
    for i in range(n_states):
        # pick 1–3 preferred next states (excluding self)
        others = [j for j in range(n_states) if j != i]
        n_pref = min(rng.integers(1, 4), len(others))
        preferred = rng.choice(others, size=n_pref, replace=False)
        for j in others:
            raw[i, j] = 0.5 if j in preferred else 0.05
    np.fill_diagonal(raw, 0.0)
    return raw                     # normalised inside generate_hmm_input


def _random_emission(n_emissions: int, n_states: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Each state gets a peaked emission profile: high weight on a contiguous
    region of bins, low weight everywhere else.  Shape: (n_emissions, n_states).
    """
    raw = np.full((n_emissions, n_states), 0.01)
    for s in range(n_states):
        # centre the peak uniformly across the emission range
        centre = int(round(s * (n_emissions - 1) / max(n_states - 1, 1)))
        width  = max(2, n_emissions // n_states)
        lo = max(0, centre - width)
        hi = min(n_emissions, centre + width + 1)
        raw[lo:hi, s] += rng.uniform(0.5, 1.5, size=(hi - lo))
    return raw                     # normalised inside generate_hmm_input


def _random_duration(n_states: int, max_duration: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Each state gets a Gaussian duration PMF whose mean is spread evenly
    across [2, max_duration/4] and whose std is randomised slightly.
    """
    means = np.linspace(2, max(2, max_duration // 4), n_states)
    rows  = []
    for s in range(n_states):
        std  = float(rng.uniform(1.0, max(1.1, means[s] / 2)))
        rows.append(gaussian_duration_pmf(max_duration, float(means[s]), std))
    return np.array(rows)          # normalised inside generate_hmm_input


def _random_obs(n_steps: int, n_emissions: int,
                rng: np.random.Generator) -> list[int]:
    """Uniform random observations, 1-indexed."""
    return (rng.integers(0, n_emissions, size=n_steps) + 1).tolist()


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a random HSMM input JSON for ViterbiImpl."
    )
    parser.add_argument("--n-states",    type=int, default=5,
                        help="Number of hidden states (default: 5)")
    parser.add_argument("--n-emissions", type=int, default=10,
                        help="Number of discrete emission bins (default: 10)")
    parser.add_argument("--n-steps",     type=int, default=500,
                        help="Observation sequence length (default: 500)")
    parser.add_argument("--max-duration",type=int, default=50,
                        help="Maximum state-sojourn duration M (default: 50)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--out",         type=str, default=None,
                        help="Output JSON path (auto-named if omitted)")
    args = parser.parse_args()

    J = args.n_states
    E = args.n_emissions
    T = args.n_steps
    M = args.max_duration
    S = args.seed

    rng = np.random.default_rng(S)

    trans = _random_trans(J, rng)
    emis  = _random_emission(E, J, rng)
    pi    = rng.uniform(0, 1, size=J)          # normalised inside generator
    dur   = _random_duration(J, M, rng)
    obs   = _random_obs(T, E, rng)

    out_path = args.out or f"data/{J}states_{T}steps_{M}dur.json"

    generate_hmm_input(
        n_states       = J,
        n_emissions    = E,
        n_steps        = T,
        max_duration   = M,
        trans_mat      = trans,
        emission_probs = emis,
        start_probs    = pi,
        duration_probs = dur,
        obs_seq        = obs,
        seed           = S,
        out_path       = out_path,
    )


# python data_generator.py --n-states 7 --n-emissions 10 --n-steps 500 --max-duration 50