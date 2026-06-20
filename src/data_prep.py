"""
data_prep.py
============
SENTINELA — data loading and chronological split.

Exposes a single public function:

    train, val, test = load_and_split(
        centrality_path=...,
        modelling_ready_path=...,
    )

The split logic is a **faithful extraction** of the logic in
``train_models.py`` (Steps 3-A and 3-B).  Any behaviour change here
should be mirrored there, and vice versa.

Design notes
------------
* ``split_timestamp`` is used as the **sole** ordering key.  Row order in
  the CSV files is intentionally NOT trusted — it was verified during the
  Step 1→Step 2 handoff investigation to be non-chronological.
* If ``split_timestamp`` is absent from the centrality file (a known
  pipeline defect from the Step 1→Step 2 handoff), this module recovers
  it from ``astram_modelling_ready.csv`` via a **verified positional
  join**: every shared column must match row-for-row (NaN-aware,
  float-tolerant) before the join is accepted.  Any mismatch raises.
* The stable sort preserves original row order for ties, matching
  ``train_models.py``'s ``kind="stable"`` argument.
* TRAIN, VAL, TEST fractions are 70 / 15 / 15, implemented via
  ``round()`` (not ``floor()`` or ``ceil()``) so rounding errors average
  out across the three splits.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

def _configure_root_if_needed() -> None:
    """Attach a StreamHandler to the root logger if nothing is configured yet.

    Callers that already configure logging (e.g. train_models.py) are
    unaffected.  Callers that don't (e.g. a notebook or a one-off script)
    get readable output automatically.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )


# ── default file locations ────────────────────────────────────────────────────
# These match the paths used in train_models.py.  Override via environment
# variables or by passing explicit arguments to load_and_split().

_HERE = Path(__file__).resolve().parent  # src/
_PROJECT_ROOT = _HERE.parent

def _env_path(var: str, fallback: Path) -> Path:
    """Return ``Path(os.environ[var])`` if the variable is set and non-empty,
    otherwise return ``fallback``.

    ``Path("")`` evaluates to ``Path(".")`` (the current directory) and is
    truthy, so a plain ``Path(os.getenv(var, "")) or fallback`` expression
    silently resolves to ``.`` whenever the variable is unset.  This helper
    avoids that pitfall.
    """
    val = os.environ.get(var, "").strip()
    return Path(val) if val else fallback


# Evaluated lazily at call time (not at import time) so that env vars set
# after import — e.g. by dashboard.py — are always picked up correctly.
def _default_centrality_path() -> Path:
    return _env_path(
        "SENTINELA_CENTRALITY_PATH",
        _PROJECT_ROOT / "data" / "processed" / "astram_with_centrality.csv",
    )

def _default_modelling_ready_path() -> Path:
    return _env_path(
        "SENTINELA_MODELLING_READY_PATH",
        _PROJECT_ROOT / "data" / "processed" / "astram_modelling_ready.csv",
    )


# ── constants (mirrors train_models.py) ──────────────────────────────────────

SPLIT_COL: str = "split_timestamp"
TARGET_CLF: str = "requires_road_closure"
TARGET_REG: str = "resolution_minutes"

TRAIN_FRAC: float = 0.70
VAL_FRAC:   float = 0.15
# remaining 0.15 → TEST (implicit)


# ── public API ────────────────────────────────────────────────────────────────


def load_and_split(
    centrality_path: Optional[os.PathLike | str] = None,
    modelling_ready_path: Optional[os.PathLike | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the centrality-augmented incident dataset and return the
    chronological 70/15/15 split used by the SENTINELA model pipeline.

    Parameters
    ----------
    centrality_path:
        Path to ``astram_with_centrality.csv`` (Step 2 output).
        Defaults to ``$SENTINELA_CENTRALITY_PATH`` or
        ``<project_root>/data/processed/astram_with_centrality.csv``.
    modelling_ready_path:
        Path to ``astram_modelling_ready.csv`` (Step 1 output).
        Only needed when ``split_timestamp`` is absent from the centrality
        file.  Defaults to ``$SENTINELA_MODELLING_READY_PATH`` or
        ``<project_root>/data/processed/astram_modelling_ready.csv``.

    Returns
    -------
    (train, val, test) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three DataFrames in chronological order.  All original columns are
        preserved, including ``split_timestamp`` and the two target columns
        (``requires_road_closure``, ``resolution_minutes``).  Indices are
        reset (0-based, contiguous).

    Raises
    ------
    FileNotFoundError
        If ``centrality_path`` does not exist.
    ValueError
        If ``split_timestamp`` cannot be recovered, cannot be parsed, or if
        the row-alignment verification between the two source files fails.
    """
    _configure_root_if_needed()

    centrality_path      = Path(centrality_path      or _default_centrality_path())
    modelling_ready_path = Path(modelling_ready_path or _default_modelling_ready_path())

    df = _load_dataset(centrality_path, modelling_ready_path)
    return _chronological_split(df)


# ── internal steps ────────────────────────────────────────────────────────────


def _load_dataset(
    centrality_path: Path,
    modelling_ready_path: Path,
) -> pd.DataFrame:
    """Load the centrality CSV and guarantee ``split_timestamp`` is present
    and parseable as datetime64[ns, UTC-naive ISO8601].

    Mirrors ``train_models.py::load_dataset()`` exactly.
    """
    if not centrality_path.exists():
        raise FileNotFoundError(
            f"Expected Step 2 output at {centrality_path}. "
            "Run src/centrality.py first, or set SENTINELA_CENTRALITY_PATH."
        )

    log.info("Loading centrality-augmented dataset from %s", centrality_path)
    df = pd.read_csv(centrality_path)
    log.info("Loaded %d incidents × %d columns", *df.shape)

    # ── recover split_timestamp if it was dropped during Step 1→Step 2 ──
    if SPLIT_COL not in df.columns:
        log.warning(
            "'%s' is MISSING from %s — attempting recovery from %s …",
            SPLIT_COL, centrality_path.name, modelling_ready_path.name,
        )
        df = _recover_split_timestamp(df, centrality_path, modelling_ready_path)

    if SPLIT_COL not in df.columns:
        raise ValueError(
            f"'{SPLIT_COL}' could not be recovered. Cannot perform a "
            "chronological split without it. Re-run the Step 1→Step 2 "
            f"pipeline so that '{SPLIT_COL}' is carried through to "
            f"{centrality_path.name}, or supply astram_modelling_ready.csv "
            "at the path given by modelling_ready_path."
        )

    # ── parse timestamps — pin to ISO8601 to avoid silent NaT on mixed ──
    # ── formats (some rows have microseconds, some don't)               ──
    parsed = pd.to_datetime(df[SPLIT_COL], format="ISO8601", errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} rows have an unparseable '{SPLIT_COL}' value after "
            "ISO8601 parsing. Inspect these rows before proceeding — silently "
            "dropping or imputing timestamps would compromise the chronological "
            "split."
        )
    df[SPLIT_COL] = parsed

    # ── report duplicate timestamps (ties broken by stable sort) ────────
    n_dupe_ts = int(df[SPLIT_COL].duplicated().sum())
    if n_dupe_ts:
        log.info(
            "%d rows share an exact-duplicate timestamp with another row "
            "(ties broken by original row order during the stable sort).",
            n_dupe_ts,
        )

    return df


def _recover_split_timestamp(
    df: pd.DataFrame,
    centrality_path: Path,
    modelling_ready_path: Path,
) -> pd.DataFrame:
    """Best-effort, verified recovery of ``split_timestamp``.

    Loads ``modelling_ready_path``, verifies exact row alignment against
    ``df`` across every shared column (NaN-aware, float-tolerant), and
    only then attaches the timestamp column by row position.

    Returns ``df`` **unchanged** (still missing the column) if recovery
    cannot be safely performed.  The caller checks for the column's
    presence and raises if it is still absent.

    Mirrors ``train_models.py::_recover_split_timestamp()`` exactly.
    """
    if not modelling_ready_path.exists():
        log.error(
            "Recovery source %s not found — cannot recover '%s'.",
            modelling_ready_path, SPLIT_COL,
        )
        return df

    ref = pd.read_csv(modelling_ready_path)

    if SPLIT_COL not in ref.columns:
        log.error(
            "%s does not contain '%s' either — cannot recover.",
            modelling_ready_path.name, SPLIT_COL,
        )
        return df

    if len(ref) != len(df):
        log.error(
            "Row count mismatch: %s has %d rows, %s has %d rows. "
            "Refusing a positional join — row alignment cannot be assumed.",
            centrality_path.name, len(df),
            modelling_ready_path.name, len(ref),
        )
        return df

    # ── verify every shared column matches row-for-row ───────────────────
    shared_cols = [c for c in ref.columns if c in df.columns and c != SPLIT_COL]
    log.info(
        "Verifying row alignment across %d shared columns …", len(shared_cols)
    )
    mismatches = 0
    for col in shared_cols:
        a, b = df[col], ref[col]
        both_na = a.isna() & b.isna()
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            close = np.isclose(
                a.fillna(0).to_numpy(dtype=float),
                b.fillna(0).to_numpy(dtype=float),
                rtol=1e-6, atol=1e-6,
            )
            mismatch_mask = ~(close | both_na)
        else:
            mismatch_mask = (a.astype(str) != b.astype(str)) & ~both_na
        n_col_mismatch = int(mismatch_mask.sum())
        if n_col_mismatch:
            log.debug("  column '%s': %d mismatching cells", col, n_col_mismatch)
        mismatches += n_col_mismatch

    if mismatches:
        raise ValueError(
            f"Found {mismatches} mismatched cells across {len(shared_cols)} "
            f"shared columns between {centrality_path.name} and "
            f"{modelling_ready_path.name}. Refusing to join — files do not "
            "appear to be row-aligned. Re-run the Step 1→Step 2 pipeline to "
            "produce a fresh centrality file that includes split_timestamp."
        )

    log.info(
        "Row alignment verified (0 mismatches across %d shared columns). "
        "Recovering '%s' from %s by row position.",
        len(shared_cols), SPLIT_COL, modelling_ready_path.name,
    )
    df = df.copy()
    df[SPLIT_COL] = ref[SPLIT_COL].values
    return df


def _chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by ``split_timestamp`` (stable) and cut at 70/15/15.

    Mirrors ``train_models.py::chronological_split()`` exactly.
    """
    df_sorted = df.sort_values(SPLIT_COL, kind="stable").reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(round(n * TRAIN_FRAC))
    n_val   = int(round(n * VAL_FRAC))

    train = df_sorted.iloc[:n_train].reset_index(drop=True)
    val   = df_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test  = df_sorted.iloc[n_train + n_val :].reset_index(drop=True)

    log.info(
        "Chronological split → train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
        len(train), 100 * len(train) / n,
        len(val),   100 * len(val)   / n,
        len(test),  100 * len(test)  / n,
    )
    log.info(
        "  train range : %s  →  %s",
        train[SPLIT_COL].min(), train[SPLIT_COL].max(),
    )
    log.info(
        "  val   range : %s  →  %s",
        val[SPLIT_COL].min(), val[SPLIT_COL].max(),
    )
    log.info(
        "  test  range : %s  →  %s",
        test[SPLIT_COL].min(), test[SPLIT_COL].max(),
    )

    # ── warn if duplicate timestamps straddle a boundary ────────────────
    if train[SPLIT_COL].max() > val[SPLIT_COL].min() or \
       val[SPLIT_COL].max() > test[SPLIT_COL].min():
        log.warning(
            "Duplicate-timestamp rows straddle a split boundary — a small "
            "number of rows may share an identical timestamp across adjacent "
            "splits. This does not affect ordering correctness, only "
            "boundary ties."
        )

    # ── class-balance summary ────────────────────────────────────────────
    for name, split in (("train", train), ("val", val), ("test", test)):
        if TARGET_CLF in split.columns:
            pos = int(split[TARGET_CLF].astype(bool).sum())
            rate = pos / len(split) if len(split) else float("nan")
            log.info(
                "  %-5s  n=%-5d  positives=%-4d  (%.1f%%)",
                name, len(split), pos, 100 * rate,
            )

    return train, val, test


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow overriding paths from the command line for quick testing:
    #   python src/data_prep.py path/to/centrality.csv path/to/modelling_ready.csv
    _c = sys.argv[1] if len(sys.argv) > 1 else None
    _m = sys.argv[2] if len(sys.argv) > 2 else None

    train, val, test = load_and_split(centrality_path=_c, modelling_ready_path=_m)

    print(f"\ntrain : {len(train):>5d} rows | {train[SPLIT_COL].min()} → {train[SPLIT_COL].max()}")
    print(f"val   : {len(val):>5d} rows | {val[SPLIT_COL].min()} → {val[SPLIT_COL].max()}")
    print(f"test  : {len(test):>5d} rows | {test[SPLIT_COL].min()} → {test[SPLIT_COL].max()}")