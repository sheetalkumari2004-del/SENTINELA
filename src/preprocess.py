"""
SENTINELA – Step 1: Preprocessing Pipeline
===========================================
Astram Traffic Incident Dataset · Bengaluru
Production-ready · generates modelling-ready CSV + data_audit.md
"""

import os
import sys
import logging
import textwrap
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# PATHS  (resolve relative to this file's parent)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH     = PROJECT_ROOT / "data" / "raw"  / "astram_raw.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "astram_modelling_ready.csv"
AUDIT_PATH   = PROJECT_ROOT / "reports" / "data_audit.md"

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinela.preprocess")


# ══════════════════════════════════════════════
# STEP 1-A  Load
# ══════════════════════════════════════════════
def load_raw(path: Path) -> pd.DataFrame:
    """
    Load the raw CSV.
    Decision: no dtype coercion at read-time – we inspect and convert
    deliberately in later steps so failures are explicit, not silent.
    """
    log.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Raw shape: %d rows × %d cols", *df.shape)
    return df


# ══════════════════════════════════════════════
# STEP 1-B  Timestamp parsing
# ══════════════════════════════════════════════
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset stores timestamps in ISO-8601 with sub-second precision
    and a UTC offset (+00).  Two distinct sub-formats appear:

        2024-03-07 17:01:48.111+00          ← milliseconds only
        2024-01-30 04:17:46.828979+00       ← microseconds
        2024-02-12 02:05:46+00              ← no fractional seconds

    pandas format='mixed' handles all three.  utc=True normalises
    everything to a single UTC-aware dtype, eliminating offset ambiguity.

    We parse four columns:
        start_datetime    – incident open time     (0 nulls in raw)
        resolved_datetime – set when status=resolved (74 non-null)
        closed_datetime   – set when status=closed (3,141 non-null)
        created_date      – record-creation time   (0 nulls; kept for audit)

    end_datetime is NOT parsed: it has 94 % missingness and its semantics
    overlap with closed_datetime.  It is dropped downstream.
    """
    log.info("Parsing timestamps …")
    for col in ["start_datetime", "resolved_datetime", "closed_datetime", "created_date"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed", errors="coerce")
    return df


# ══════════════════════════════════════════════
# STEP 1-C  Resolution time
# ══════════════════════════════════════════════
def compute_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolution time = elapsed minutes from incident open to incident close.

    End-of-incident timestamp strategy (priority order):
        1. resolved_datetime   – explicit operator resolution (highest signal)
        2. closed_datetime     – system close (most populated: 3,141 records)

    combine_first() picks resolved_datetime where available and falls
    back to closed_datetime otherwise.  Records with neither are
    assigned NaT and will be removed in the filter step.

    Resolution minutes is then:
        (end_ts − start_ts).total_seconds() / 60

    Negative values (4 records) arise from data-entry errors where the
    close timestamp predates the open timestamp.  They are physically
    meaningless and removed in the filter step alongside out-of-range
    values.
    """
    log.info("Computing resolution_minutes …")
    df["end_ts"] = df["resolved_datetime"].combine_first(df["closed_datetime"])
    df["resolution_minutes"] = (
        (df["end_ts"] - df["start_datetime"]).dt.total_seconds() / 60
    )
    return df


# ══════════════════════════════════════════════
# STEP 1-D  Filter resolution window 0–1440 min
# ══════════════════════════════════════════════
def filter_resolution_window(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep only records with a valid, plausible resolution time.

    0 min lower bound  – an incident cannot be resolved before it opens.
    1440 min (24 h) upper bound – established project requirement; incidents
        logged open for more than a day are either never-closed tickets or
        administrative artefacts that would bias any regression target.

    Removal breakdown (approximate from EDA):
        No end timestamp :  4,964 records  (status=active, or missing close)
        Negative          :      4 records
        > 1440 minutes    :    672 records
    """
    log.info("Filtering to resolution window 0–1440 minutes …")
    n_before = len(df)

    no_end      = df["resolution_minutes"].isna().sum()
    negative    = (df["resolution_minutes"] < 0).sum()
    over_1440   = (df["resolution_minutes"] > 1440).sum()

    df = df[
        df["resolution_minutes"].notna() &
        (df["resolution_minutes"] >= 0) &
        (df["resolution_minutes"] <= 1440)
    ].copy()

    n_after = len(df)
    stats = {
        "n_before_filter" : n_before,
        "removed_no_end"  : int(no_end),
        "removed_negative": int(negative),
        "removed_over1440": int(over_1440),
        "n_after_filter"  : n_after,
    }
    log.info(
        "Filter removed %d rows → %d records remain",
        n_before - n_after, n_after,
    )
    return df, stats


# ══════════════════════════════════════════════
# STEP 1-E  Planned / unplanned flag
# ══════════════════════════════════════════════
def create_planned_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    event_type contains exactly two values: 'planned' and 'unplanned'.
    We encode this as a binary integer:
        is_planned = 1  →  planned incident (road-works, events, VIP movement)
        is_planned = 0  →  unplanned incident (breakdown, accident, etc.)

    The original event_type string column is retained for interpretability
    and then dropped in the feature-selection step.

    Decision to use integer (not bool): scikit-learn estimators and most
    gradient-boosted libraries accept 0/1 without additional encoding.
    """
    log.info("Creating is_planned flag …")
    df["is_planned"] = (df["event_type"] == "planned").astype(int)
    return df


# ══════════════════════════════════════════════
# STEP 1-F  Temporal feature extraction
# ══════════════════════════════════════════════
def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three cyclically meaningful features are extracted from start_datetime
    (all timestamps are UTC; Bengaluru is UTC+5:30, so IST = UTC + 5.5 h).

    We convert to IST before extracting hour/weekday/month so that
    'hour_of_day=08' genuinely means 8 AM Bengaluru time, not 2:30 AM UTC.

    hour_of_day  [0–23]  – captures peak/off-peak traffic dynamics
    day_of_week  [0–6]   – Monday=0, Sunday=6  (weekly patterns)
    month        [1–12]  – seasonal effects (monsoon, festivals)

    These are stored as integers.  Sine/cosine cyclical encoding is left
    to the feature-engineering step (Step 2), not Step 1.

    split_timestamp – the full IST-converted datetime (date + time),
    preserved verbatim alongside the cyclical features above.  This is
    NOT a model feature: hour/day/month already capture the cyclical
    signal a tree model can use, and a raw timestamp would either add
    nothing useful or be wrongly treated as an ordinal feature. Its sole
    purpose is to allow a chronological train/test split downstream
    (sort by split_timestamp, hold out the most recent slice) instead of
    a random split, which would leak future incidents into training. It
    is carried through Steps 1 and 2 untouched and must be dropped (or
    excluded from the feature list) immediately before fitting any
    model in Step 3.
    """
    log.info("Extracting temporal features (IST) …")
    ist = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour_of_day"]     = ist.dt.hour
    df["day_of_week"]     = ist.dt.dayofweek
    df["month"]           = ist.dt.month
    df["split_timestamp"] = ist
    return df


# ══════════════════════════════════════════════
# STEP 1-G  Normalise event_cause
# ══════════════════════════════════════════════
def normalise_event_cause(df: pd.DataFrame) -> pd.DataFrame:
    """
    event_cause has minor case/capitalisation inconsistencies:
        'Debris' and 'debris' are the same category.
        'Fog / Low Visibility' appears twice (only in the raw set; absent
         after the resolution-time filter).

    We lower-case and strip so downstream one-hot encoding produces
    consistent dummies regardless of entry capitalisation.
    """
    log.info("Normalising event_cause …")
    df["event_cause"] = df["event_cause"].str.strip().str.lower()
    return df


# ══════════════════════════════════════════════
# STEP 1-H  Drop unusable / leaky / near-empty columns
# ══════════════════════════════════════════════
# Columns removed and the rationale for each:
DROP_COLUMNS = [
    # ── Target-leaky (contain post-resolution information) ──────────────
    "end_ts",               # intermediate helper column
    "resolved_datetime",    # used to build resolution_minutes; would leak target
    "closed_datetime",      # same
    "end_datetime",         # 94 % missing; redundant with closed_datetime
    "modified_datetime",    # reflects last admin edit, not incident dynamics
    "closed_by_id",         # post-resolution actor – leaks target
    "resolved_by_id",       # same
    "resolved_at_address",  # post-resolution location – leaks target
    "resolved_at_latitude", # same
    "resolved_at_longitude",# same

    # ── Priority: explicitly non-informative per project brief ──────────
    "priority",             # all non-null values are 'High'; no variance

    # ── Identifiers (no predictive signal, just keys) ───────────────────
    "id",                   # record key
    "veh_no",               # individual vehicle registration plate
    "kgid",                 # internal KGIS identifier
    "gba_identifier",       # internal GBA zone code (57 % missing; zone covers it)
    "client_id",            # single-value column (all 1)
    "created_by_id",        # user ID of operator who filed the report
    "last_modified_by_id",  # user ID of last editor
    "assigned_to_police_id",# 98 % missing
    "citizen_accident_id",  # 98 % missing
    "closed_by_id",         # already listed above; dedupe safe

    # ── Near-completely missing (> 90 %) ────────────────────────────────
    "map_file",             # 100 % missing
    "direction",            # 99.5 % missing
    "end_address",          # 91.6 % missing
    "cargo_material",       # 96.6 % missing
    "reason_breakdown",     # 96.6 % missing
    "age_of_truck",         # 96.6 % missing
    "route_path",           # 98.3 % missing
    "comment",              # 100 % missing
    "meta_data",            # 100 % missing

    # ── Redundant / derived ─────────────────────────────────────────────
    "event_type",           # encoded as is_planned
    "start_datetime",       # raw string dropped; full value preserved separately as split_timestamp (not a feature)
    "created_date",         # collinear with start_datetime; not needed
    "status",               # reflects final state (closed/resolved) – leaks target
    "authenticated",        # administrative flag; single dominant value ('yes')
    "endlatitude",          # destination coords; 2 % missing; mostly zero (unplanned)
    "endlongitude",         # same
    "police_station",       # collinear with zone/corridor at a coarser grain
]

# Columns that are intentionally present in the Step 1 output but are NOT
# model features. Step 3 (LightGBM training) must exclude these from X
# before fitting — they exist purely for chronological train/test
# splitting (split_timestamp) or other non-modelling purposes.
NON_FEATURE_COLUMNS = [
    "split_timestamp",      # full IST datetime; used only to sort/split train vs test
]


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all columns listed in DROP_COLUMNS.
    We use a set intersection so the function is safe if a column was
    already removed upstream or never existed.

    Note: split_timestamp is deliberately NOT in DROP_COLUMNS — it must
    survive into astram_modelling_ready.csv and astram_with_centrality.csv
    for chronological splitting in Step 3. See NON_FEATURE_COLUMNS above.
    """
    log.info("Dropping unusable columns …")
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info("Retained %d columns: %s", len(df.columns), df.columns.tolist())
    return df


# ══════════════════════════════════════════════
# STEP 1-I  Missing value handling
# ══════════════════════════════════════════════
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    After column drops, remaining missingness is modest and categorical:

    veh_type   ~27 % missing in the filtered set (695/2533).
               Vehicles not classified are predominantly non-vehicle events
               (accidents with no truck involved, tree falls, water logging).
               Strategy: fill with 'unknown'.  Do NOT impute with mode – the
               modal class (lcv) would be a fabrication for non-vehicle events.

    corridor   ~0.2 % missing (4 rows in filtered set).
               Strategy: fill with 'unknown'.  Too few to warrant model imputation.

    zone       ~56 % missing.  High missingness but the column is retained
               as a feature because zone is spatially informative where
               present.  Strategy: fill with 'unknown'.  Downstream
               models should use a learned embedding or treat 'unknown' as
               its own category.

    junction   ~65 % missing.  Same reasoning as zone.
               Strategy: fill with 'unknown'.

    description  ~8 % missing in filtered set.  Free text; not used in
               tabular modelling.  Retained as-is for potential NLP step.
               No imputation.

    address    0 % missing after filter.  No action needed.

    Numerical columns (latitude, longitude):  0 % missing.  No action.
    """
    log.info("Handling missing values …")

    categorical_fills = {
        "veh_type" : "unknown",
        "corridor" : "unknown",
        "zone"     : "unknown",
        "junction" : "unknown",
    }
    for col, fill in categorical_fills.items():
        if col in df.columns:
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(fill)
            log.info("  %-12s  filled %d NaN → '%s'", col, n_filled, fill)

    return df


# ══════════════════════════════════════════════
# STEP 1-J  Save processed dataset
# ══════════════════════════════════════════════
def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved modelling-ready dataset → %s  (%d rows × %d cols)",
             path, *df.shape)


# ══════════════════════════════════════════════
# STEP 1-K  Auto-generate data_audit.md
# ══════════════════════════════════════════════
def write_audit(df_raw: pd.DataFrame,
                df_final: pd.DataFrame,
                filter_stats: dict,
                path: Path) -> None:
    """
    Writes a structured Markdown audit report to reports/data_audit.md.
    All figures are computed from the actual DataFrames – no hard-coded numbers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Missing value table for raw ──────────────────────────────────────
    miss_raw = (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
    miss_rows = "\n".join(
        f"| `{col}` | {pct:.2f} % |"
        for col, pct in miss_raw[miss_raw > 0].sort_values(ascending=False).items()
    )

    # ── Missing value table for final ────────────────────────────────────
    miss_final = (df_final.isnull().sum() / len(df_final) * 100).round(2)
    miss_final_rows = "\n".join(
        f"| `{col}` | {pct:.2f} % |"
        for col, pct in miss_final[miss_final > 0].sort_values(ascending=False).items()
    ) or "| *(none)* | – |"

    # ── Planned / unplanned ──────────────────────────────────────────────
    pu = df_final["is_planned"].value_counts()
    n_planned   = int(pu.get(1, 0))
    n_unplanned = int(pu.get(0, 0))

    # ── Resolution time stats ─────────────────────────────────────────────
    res = df_final["resolution_minutes"]

    # ── Column list ───────────────────────────────────────────────────────
    col_list = "\n".join(f"- `{c}`" for c in df_final.columns)

    report = textwrap.dedent(f"""\
    # SENTINELA · Data Audit Report
    **Generated:** {now}
    **Pipeline:** `src/preprocess.py`

    ---

    ## 1. Record Counts

    | Stage | Records |
    |---|---|
    | Raw dataset loaded | {filter_stats['n_before_filter']:,} |
    | Removed – no end timestamp | {filter_stats['removed_no_end']:,} |
    | Removed – negative resolution time | {filter_stats['removed_negative']:,} |
    | Removed – resolution > 1440 min | {filter_stats['removed_over1440']:,} |
    | **Final modelling dataset** | **{filter_stats['n_after_filter']:,}** |

    Retention rate: **{filter_stats['n_after_filter']/filter_stats['n_before_filter']*100:.1f} %**

    ---

    ## 2. Planned vs Unplanned

    | event_type | Count | Share |
    |---|---|---|
    | Unplanned | {n_unplanned:,} | {n_unplanned/len(df_final)*100:.1f} % |
    | Planned   | {n_planned:,}   | {n_planned/len(df_final)*100:.1f} % |
    | **Total** | **{len(df_final):,}** | 100 % |

    ---

    ## 3. Resolution Time Distribution (minutes)

    | Statistic | Value |
    |---|---|
    | Min    | {res.min():.1f} |
    | 25th % | {res.quantile(0.25):.1f} |
    | Median | {res.median():.1f} |
    | Mean   | {res.mean():.1f} |
    | 75th % | {res.quantile(0.75):.1f} |
    | Max    | {res.max():.1f} |
    | Std    | {res.std():.1f} |

    ---

    ## 4. Missing Values – Raw Dataset (columns with any missingness)

    | Column | Missing % |
    |---|---|
    {miss_rows}

    ---

    ## 5. Missing Values – Final Modelling Dataset

    | Column | Missing % |
    |---|---|
    {miss_final_rows}

    ---

    ## 6. Rows Removed During Cleaning

    | Reason | Rows removed |
    |---|---|
    | No end timestamp (active/never-closed incidents) | {filter_stats['removed_no_end']:,} |
    | Negative resolution time (timestamp errors) | {filter_stats['removed_negative']:,} |
    | Resolution time > 1440 minutes | {filter_stats['removed_over1440']:,} |
    | **Total removed** | **{filter_stats['n_before_filter'] - filter_stats['n_after_filter']:,}** |

    ---

    ## 7. Final Modelling Dataset

    - **Rows:** {len(df_final):,}
    - **Columns:** {len(df_final.columns)}
    - **File:** `data/processed/astram_modelling_ready.csv`

    ### Retained columns

    {col_list}

    ---

    ## 8. Notes

    - `veh_type`, `corridor`, `zone`, `junction` NaNs filled with `"unknown"`.
    - `description` retained as free text; not imputed.
    - All timestamps converted to UTC then shifted to IST (Asia/Kolkata) for
      temporal feature extraction.
    - `priority` dropped: only two non-null values exist and both are `"High"` –
      zero variance.
    """)

    path.write_text(report, encoding="utf-8")
    log.info("Audit report written → %s", path)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def run_pipeline() -> None:
    log.info("═" * 60)
    log.info("SENTINELA – Step 1: Preprocessing Pipeline")
    log.info("═" * 60)

    # Load
    df = load_raw(RAW_PATH)
    df_raw_snapshot = df.copy()   # keep for audit missingness table

    # Transform
    df = parse_timestamps(df)
    df = compute_resolution_time(df)
    df, filter_stats = filter_resolution_window(df)
    df = create_planned_flag(df)
    df = extract_temporal_features(df)
    df = normalise_event_cause(df)
    df = drop_unusable_columns(df)
    df = handle_missing_values(df)

    # Output
    save_processed(df, OUT_PATH)
    write_audit(df_raw_snapshot, df, filter_stats, AUDIT_PATH)

    log.info("═" * 60)
    log.info("Step 1 complete.  Modelling-ready dataset: %d rows × %d cols",
             *df.shape)
    log.info("═" * 60)


if __name__ == "__main__":
    run_pipeline()
