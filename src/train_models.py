"""
SENTINELA · Step 3 — Resolution Time & Road Closure Models
============================================================
Pipeline stage: src/train_models.py

Trains two independent LightGBM models on the centrality-augmented,
chronologically-split incident dataset:

    1. Regressor  : target = resolution_minutes
    2. Classifier : target = requires_road_closure

Both models are evaluated ONLY on the held-out, chronologically-latest
TEST slice. The VALIDATION slice is used exclusively for early stopping,
classification threshold tuning, and sanity-checking class balance.

Outputs
-------
- models/resolution_model.pkl
- models/closure_model.pkl
- models/feature_metadata.json
- reports/model_report.md
- reports/regression_feature_importance.csv
- reports/classification_feature_importance.csv

Run
---
    python src/train_models.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CENTRALITY_PATH = PROJECT_ROOT / "data" / "processed" / "astram_with_centrality.csv"
# Fallback source used ONLY to recover `split_timestamp` if it is missing from
# CENTRALITY_PATH (see `load_dataset` — this guards against the exact pipeline
# bug discovered during the Step 1→Step 2 handoff, where split_timestamp was
# dropped between the modelling-ready file and the centrality-augmented file).
MODELLING_READY_PATH = PROJECT_ROOT / "data" / "processed" / "astram_modelling_ready.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_REG = "resolution_minutes"
TARGET_CLF = "requires_road_closure"
SPLIT_COL = "split_timestamp"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> TEST

RANDOM_SEED = 42

# ── Feature selection ──────────────────────────────────────────────────
# Numeric features available at incident-report time (no outcome leakage).
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_planned",
    "centrality_score",
    "snap_distance_m",
    "low_confidence_snap",
]

# Categorical features (native LightGBM categorical handling via pandas
# `category` dtype — categories fixed from TRAIN only, see `fit_categories`).
CATEGORICAL_FEATURES = [
    "event_cause",
    "veh_type",
    "corridor",
    "zone",
    "junction",
]

BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns deliberately EXCLUDED from every model, with rationale:
#   split_timestamp     - split key only; trivially encodes row's chronological
#                          position, must never be a feature (explicit requirement).
#   address              - free-text, near-unique per row on this dataset; an
#                          identifier in practice, not a generalisable feature.
#                          Location is already captured via latitude/longitude
#                          and corridor/zone/junction.
#   description          - free text, ~17% missing, no NLP feature-extraction
#                          pipeline in scope for Step 3 (flagged as future work
#                          in the report).
#   nearest_u/v/key      - raw OSM graph node/edge identifiers (~1,600 unique
#                          values). Not ordinal/semantic on their own; the
#                          predictive signal they carry is already captured by
#                          centrality_score (derived FROM these IDs) and
#                          snap_distance_m/low_confidence_snap (snap quality).
#                          Using the raw IDs as categoricals risks the model
#                          memorising specific intersections rather than
#                          learning generalisable structure.
#
# Cross-target exclusions (see `get_feature_columns`):
#   requires_road_closure is excluded from the REGRESSOR's features.
#   resolution_minutes    is excluded from the CLASSIFIER's features.
# Rationale: both are downstream incident OUTCOMES, observed only once the
# incident is underway/resolved. In the project architecture, road-closure
# need and resolution time are both things being PREDICTED — not inputs to
# each other — so neither may appear as a feature for the other's model
# without leaking information that would not be available at decision time.

# ── Classifier-only feature/training overrides ─────────────────────────
# These apply ONLY to the classifier (`requires_road_closure`). The
# regressor's pipeline (NUMERIC_FEATURES/CATEGORICAL_FEATURES/prepare_features)
# is deliberately left unchanged — this investigation was scoped to
# improving classification ROC-AUC only. Full rationale, ablations, and
# validation numbers for every choice below are in `reports/experiment_log.md`.

# `junction` (230 categories, many singleton) and `zone` are replaced with
# smoothed, out-of-fold target-encoded numeric columns rather than used as
# native high-cardinality categoricals — this was the single best-performing
# encoding choice found, and notably better than adding frequency/aggregate
# features alongside the native categorical.
CLF_TARGET_ENCODE_COLS = ["junction", "zone"]
CLF_TARGET_ENCODE_SMOOTHING = 20
CLF_TARGET_ENCODE_FOLDS = 5

# TRAIN-only random oversampling of the minority class to this
# positive:negative ratio, replacing `scale_pos_weight`. VALIDATION/TEST are
# never resampled. Found to give a small but consistent and lower-variance
# ROC-AUC improvement over scale_pos_weight across repeated trials.
CLF_OVERSAMPLE_RATIO = 0.5

# LightGBM hyperparameter overrides found, via a grid search re-confirmed at
# higher seed counts (to guard against picking a high-variance fluke), to
# regularise more aggressively than the defaults below — sensible given only
# ~110 positive examples in TRAIN spread across sparse categoricals.
CLF_TUNED_LGBM_PARAMS = dict(
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=40,
    reg_lambda=0.5,
)

# Approximate Bengaluru city-centre (Vidhana Soudha), used only to derive an
# engineered `dist_from_center` numeric feature (Euclidean, in lat/lon
# degree-space — consistent with how it was validated in the experiment log).
BENGALURU_CENTER = (12.9716, 77.5946)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinela.train_models")


# ══════════════════════════════════════════════
# STEP 3-A  Load + validate dataset
# ══════════════════════════════════════════════
def load_dataset() -> pd.DataFrame:
    """
    Loads the Step 2 output (centrality-augmented incidents) and guarantees
    `split_timestamp` is present and parseable.

    If `split_timestamp` is missing from CENTRALITY_PATH (a known pipeline
    defect observed during handoff — Step 1's modelling-ready file had it,
    but it was lost by the time the centrality-augmented file was produced),
    this function attempts a SAFE recovery: it loads MODELLING_READY_PATH and
    re-attaches `split_timestamp` by row position, but ONLY after verifying
    every shared column matches row-for-row (NaN-aware, float-tolerant). If
    that verification fails, it raises rather than silently joining
    mismatched rows.
    """
    if not CENTRALITY_PATH.exists():
        raise FileNotFoundError(
            f"Expected Step 2 output at {CENTRALITY_PATH}. Run src/centrality.py first."
        )

    log.info("Loading centrality-augmented dataset from %s", CENTRALITY_PATH)
    df = pd.read_csv(CENTRALITY_PATH)
    log.info("Loaded %d incidents × %d cols", *df.shape)

    if SPLIT_COL not in df.columns:
        log.warning(
            "'%s' is MISSING from %s. Attempting recovery from %s …",
            SPLIT_COL, CENTRALITY_PATH.name, MODELLING_READY_PATH.name,
        )
        df = _recover_split_timestamp(df)

    if SPLIT_COL not in df.columns:
        raise ValueError(
            f"'{SPLIT_COL}' could not be recovered. Cannot perform a chronological "
            f"split without it. Re-run the Step 1→Step 2 pipeline so that "
            f"'{SPLIT_COL}' is carried through to {CENTRALITY_PATH.name}."
        )

    # Parse explicitly with ISO8601 — the raw column has a mix of timestamps
    # with and without a microsecond component, which trips up pandas' format
    # auto-inference (silent NaT) if not pinned to ISO8601.
    parsed = pd.to_datetime(df[SPLIT_COL], format="ISO8601", errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} rows have an unparseable '{SPLIT_COL}' value after ISO8601 "
            f"parsing. Inspect these rows before proceeding — silently dropping "
            f"or imputing timestamps would compromise the chronological split."
        )
    df[SPLIT_COL] = parsed

    n_dupe_ts = int(df[SPLIT_COL].duplicated().sum())
    if n_dupe_ts:
        log.info(
            "%d rows share an exact-duplicate timestamp with another row "
            "(negligible at this volume); ties broken by original row order "
            "during the stable sort.", n_dupe_ts,
        )

    return df


def _recover_split_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort, verified recovery of split_timestamp. Returns df unchanged
    (still missing the column) if recovery cannot be safely performed."""
    if not MODELLING_READY_PATH.exists():
        log.error(
            "Recovery source %s not found. Cannot recover '%s'.",
            MODELLING_READY_PATH, SPLIT_COL,
        )
        return df

    ref = pd.read_csv(MODELLING_READY_PATH)
    if SPLIT_COL not in ref.columns:
        log.error(
            "%s does not contain '%s' either. Cannot recover.",
            MODELLING_READY_PATH.name, SPLIT_COL,
        )
        return df

    if len(ref) != len(df):
        log.error(
            "Row count mismatch: %s has %d rows, %s has %d rows. "
            "Refusing a positional join — row alignment cannot be assumed.",
            CENTRALITY_PATH.name, len(df), MODELLING_READY_PATH.name, len(ref),
        )
        return df

    shared_cols = [c for c in ref.columns if c in df.columns and c != SPLIT_COL]
    mismatches = 0
    for c in shared_cols:
        a, b = df[c], ref[c]
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
        mismatches += int(mismatch_mask.sum())

    if mismatches:
        log.error(
            "Found %d mismatched cells across %d shared columns between %s and %s. "
            "Refusing to join — files do not appear to be row-aligned.",
            mismatches, len(shared_cols), CENTRALITY_PATH.name, MODELLING_READY_PATH.name,
        )
        return df

    log.info(
        "Verified exact row alignment across %d shared columns (0 mismatches). "
        "Recovering '%s' from %s by row position.",
        len(shared_cols), SPLIT_COL, MODELLING_READY_PATH.name,
    )
    df = df.copy()
    df[SPLIT_COL] = ref[SPLIT_COL].values
    return df


# ══════════════════════════════════════════════
# STEP 3-B  Chronological split
# ══════════════════════════════════════════════
def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(SPLIT_COL, kind="stable").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))

    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train : n_train + n_val]
    test = df_sorted.iloc[n_train + n_val :]

    log.info(
        "Chronological split → train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
        len(train), 100 * len(train) / n,
        len(val), 100 * len(val) / n,
        len(test), 100 * len(test) / n,
    )
    log.info("  train range : %s  →  %s", train[SPLIT_COL].min(), train[SPLIT_COL].max())
    log.info("  val   range : %s  →  %s", val[SPLIT_COL].min(), val[SPLIT_COL].max())
    log.info("  test  range : %s  →  %s", test[SPLIT_COL].min(), test[SPLIT_COL].max())

    overlap_train_val = train[SPLIT_COL].max() > val[SPLIT_COL].min()
    overlap_val_test = val[SPLIT_COL].max() > test[SPLIT_COL].min()
    if overlap_train_val or overlap_val_test:
        log.warning(
            "Duplicate-timestamp rows straddle a split boundary — a handful of "
            "rows may share an identical timestamp across adjacent splits. "
            "This does not affect ordering correctness, only boundary ties."
        )

    return train, val, test


# ══════════════════════════════════════════════
# STEP 3-C  Feature preparation
# ══════════════════════════════════════════════
def get_feature_columns(target: str) -> list[str]:
    """Returns BASE_FEATURES, additionally excluding whichever outcome
    column is NOT the current target (cross-target leakage guard)."""
    other_outcome = TARGET_CLF if target == TARGET_REG else TARGET_REG
    assert other_outcome not in BASE_FEATURES, "outcome column leaked into BASE_FEATURES"
    return list(BASE_FEATURES)


def fit_categories(train_col: pd.Series) -> pd.CategoricalDtype:
    """Category vocabulary is fixed from TRAIN ONLY. Categories seen in
    val/test but never in train become NaN (LightGBM treats NaN as a
    legitimate 'missing' branch) rather than silently being included as
    if the model had ever observed them historically."""
    cats = sorted(train_col.dropna().astype(str).unique().tolist())
    return pd.CategoricalDtype(categories=cats)


def apply_categories(col: pd.Series, dtype: pd.CategoricalDtype) -> pd.Series:
    return col.astype(str).astype(dtype)


def prepare_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Builds model-ready X frames. Categorical dtype categories are fit on
    TRAIN only and applied identically to val/test. Numeric/boolean columns
    are cast to a consistent numeric dtype."""
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols_present = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    for c in cat_cols_present:
        dtype = fit_categories(X_train[c])
        X_train[c] = apply_categories(X_train[c], dtype)
        X_val[c] = apply_categories(X_val[c], dtype)
        X_test[c] = apply_categories(X_test[c], dtype)
        n_unseen_val = X_val[c].isna().sum() - val[c].isna().sum()
        n_unseen_test = X_test[c].isna().sum() - test[c].isna().sum()
        if n_unseen_val > 0 or n_unseen_test > 0:
            log.info(
                "  categorical '%s': %d unseen-in-train categories in val, "
                "%d in test (mapped to missing).", c, n_unseen_val, n_unseen_test,
            )

    num_cols_present = [c for c in NUMERIC_FEATURES if c in feature_cols]
    for frame in (X_train, X_val, X_test):
        for c in num_cols_present:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)

    return X_train, X_val, X_test, cat_cols_present


# ══════════════════════════════════════════════
# STEP 3-C′  Classifier-only feature engineering
# (see CLF_* constants above and reports/experiment_log.md for rationale)
# ══════════════════════════════════════════════
def target_encode_oof(train_col: pd.Series, y_train: pd.Series, smoothing: float, n_folds: int, seed: int):
    """Smoothed mean-target encoding. TRAIN rows are encoded out-of-fold
    (a row never sees a statistic computed using its own label) to avoid
    target leakage; the returned `encoder` dict is fit on ALL of TRAIN and
    is what must be applied to VAL/TEST (and persisted for future inference
    on new categories, which fall back to `global_mean`)."""
    global_mean = float(y_train.mean())
    train_col = train_col.astype(str).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    train_encoded = np.full(len(train_col), global_mean, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in kf.split(train_col):
        stats = y_train.iloc[fit_idx].groupby(train_col.iloc[fit_idx]).agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        train_encoded[hold_idx] = train_col.iloc[hold_idx].map(smoothed).fillna(global_mean).values

    full_stats = y_train.groupby(train_col).agg(["mean", "count"])
    full_smoothed = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
    encoder = {
        "mapping": {str(k): float(v) for k, v in full_smoothed.items()},
        "global_mean": global_mean,
        "smoothing": smoothing,
    }
    return train_encoded, encoder


def apply_target_encoding(col: pd.Series, encoder: dict) -> np.ndarray:
    """Applies a TRAIN-fit target-encoding map to VAL/TEST (or future
    production rows). Categories never seen in TRAIN fall back to
    `global_mean`, matching the OOF fallback behaviour used during fitting."""
    mapping = encoder["mapping"]
    return col.astype(str).map(mapping).fillna(encoder["global_mean"]).astype(float).values


def add_distance_from_center(df: pd.DataFrame) -> pd.Series:
    """Euclidean distance (lat/lon degree-space) from BENGALURU_CENTER —
    a simple, strong positive feature in the experiment log (Δ+0.0152 AUC
    on its own; ~Δ+0.02 contribution within the final combined feature set)."""
    lat0, lon0 = BENGALURU_CENTER
    return np.sqrt((df["latitude"] - lat0) ** 2 + (df["longitude"] - lon0) ** 2)


def random_oversample(X_train: pd.DataFrame, y_train: pd.Series, ratio: float, seed: int):
    """Duplicates minority rows (sampling with replacement) until
    n_minority / n_majority == ratio. TRAIN-only; never applied to VAL/TEST."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_train.values == 1)[0]
    neg_idx = np.where(y_train.values == 0)[0]
    n_target_pos = int(round(len(neg_idx) * ratio))
    extra = n_target_pos - len(pos_idx)
    if extra > 0:
        extra_idx = rng.choice(pos_idx, size=extra, replace=True)
        all_idx = np.concatenate([np.arange(len(y_train)), extra_idx])
    else:
        all_idx = np.arange(len(y_train))
    rng.shuffle(all_idx)
    return X_train.iloc[all_idx].reset_index(drop=True), y_train.iloc[all_idx].reset_index(drop=True)


def prepare_classifier_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Classifier-specific feature pipeline — distinct from the generic
    `prepare_features` used by the regressor, which is left unchanged.

    Differences from the base 14-feature set, each validated in
    reports/experiment_log.md:
      1. `junction`/`zone` -> smoothed OOF target-encoded numeric columns
         (`junction_te`/`zone_te`), instead of native categoricals.
      2. `dist_from_center` numeric feature added.
      3. `event_cause`, `veh_type`, `corridor` remain native categoricals
         (TRAIN-only vocabulary, same mechanism as `prepare_features`).
      4. All original numeric features (incl. `centrality_score`) are kept
         unchanged/raw — transforms tested showed no benefit, and dropping
         centrality_score hurt AUC.

    Returns (X_train, X_val, X_test, cat_cols_present, encoders, feature_cols).
    """
    clf_cat_native = [c for c in CATEGORICAL_FEATURES if c not in CLF_TARGET_ENCODE_COLS]
    base_numeric = list(NUMERIC_FEATURES)

    X_train = train[base_numeric + clf_cat_native].copy()
    X_val = val[base_numeric + clf_cat_native].copy()
    X_test = test[base_numeric + clf_cat_native].copy()

    for c in clf_cat_native:
        dtype = fit_categories(train[c])
        X_train[c] = apply_categories(X_train[c], dtype)
        X_val[c] = apply_categories(X_val[c], dtype)
        X_test[c] = apply_categories(X_test[c], dtype)
        n_unseen_val = X_val[c].isna().sum() - val[c].isna().sum()
        n_unseen_test = X_test[c].isna().sum() - test[c].isna().sum()
        if n_unseen_val > 0 or n_unseen_test > 0:
            log.info(
                "  categorical '%s': %d unseen-in-train categories in val, "
                "%d in test (mapped to missing).", c, n_unseen_val, n_unseen_test,
            )

    for frame in (X_train, X_val, X_test):
        for c in base_numeric:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)

    y_train_clf = train[TARGET_CLF].astype(bool).astype(int)
    encoders = {}
    te_cols = []
    for col in CLF_TARGET_ENCODE_COLS:
        train_enc, encoder = target_encode_oof(
            train[col], y_train_clf,
            smoothing=CLF_TARGET_ENCODE_SMOOTHING, n_folds=CLF_TARGET_ENCODE_FOLDS, seed=RANDOM_SEED,
        )
        col_name = f"{col}_te"
        X_train[col_name] = train_enc
        X_val[col_name] = apply_target_encoding(val[col], encoder)
        X_test[col_name] = apply_target_encoding(test[col], encoder)
        encoders[col_name] = encoder
        te_cols.append(col_name)
        log.info("  target-encoded '%s' -> '%s' (smoothing=%d, folds=%d)",
                  col, col_name, CLF_TARGET_ENCODE_SMOOTHING, CLF_TARGET_ENCODE_FOLDS)

    X_train["dist_from_center"] = add_distance_from_center(train).values
    X_val["dist_from_center"] = add_distance_from_center(val).values
    X_test["dist_from_center"] = add_distance_from_center(test).values

    feature_cols = base_numeric + clf_cat_native + te_cols + ["dist_from_center"]
    X_train = X_train[feature_cols]
    X_val = X_val[feature_cols]
    X_test = X_test[feature_cols]

    return X_train, X_val, X_test, clf_cat_native, encoders, feature_cols


# ══════════════════════════════════════════════
# STEP 3-D  Class balance analysis
# ══════════════════════════════════════════════
def class_balance_report(train, val, test) -> dict:
    rows = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        vc = split[TARGET_CLF].astype(bool).value_counts()
        pos = int(vc.get(True, 0))
        neg = int(vc.get(False, 0))
        rows[name] = {
            "n": int(len(split)),
            "positive": pos,
            "negative": neg,
            "positive_rate": pos / len(split) if len(split) else float("nan"),
        }
        log.info(
            "Class balance (%s) — requires_road_closure: positive=%d (%.2f%%)  negative=%d",
            name, pos, 100 * rows[name]["positive_rate"], neg,
        )
        if pos < 10:
            log.warning(
                "  '%s' split has only %d positive examples — metrics computed on/"
                "from this split (esp. AUC/threshold tuning) will have high variance.",
                name, pos,
            )
    return rows


# ══════════════════════════════════════════════
# STEP 3-E  Regression model
# ══════════════════════════════════════════════
def train_regressor(X_train, y_train, X_val, y_val, cat_cols):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["mae", "rmse"],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info("Regressor best iteration: %s", model.best_iteration_)
    return model


def evaluate_regressor(model, X_test, y_test, train_y_for_baseline) -> dict:
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
        "medae": median_absolute_error(y_test, preds),
    }

    baseline_median = np.full(len(y_test), train_y_for_baseline.median())
    baseline_mean = np.full(len(y_test), train_y_for_baseline.mean())
    baselines = {
        "median_baseline": {
            "mae": mean_absolute_error(y_test, baseline_median),
            "rmse": mean_squared_error(y_test, baseline_median) ** 0.5,
            "r2": r2_score(y_test, baseline_median),
            "medae": median_absolute_error(y_test, baseline_median),
        },
        "mean_baseline": {
            "mae": mean_absolute_error(y_test, baseline_mean),
            "rmse": mean_squared_error(y_test, baseline_mean) ** 0.5,
            "r2": r2_score(y_test, baseline_mean),
            "medae": median_absolute_error(y_test, baseline_mean),
        },
    }

    log.info("Regression TEST metrics: %s", metrics)
    log.info("Regression baselines: %s", baselines)
    return {"model": metrics, "baselines": baselines}


# ══════════════════════════════════════════════
# STEP 3-F  Classification model
# ══════════════════════════════════════════════
def train_classifier(X_train, y_train, X_val, y_val, cat_cols):
    n_pos_before = int(y_train.sum())
    n_neg_before = int(len(y_train) - n_pos_before)
    log.info(
        "Train class counts before oversampling — positive=%d negative=%d (%.2f%%)",
        n_pos_before, n_neg_before, 100 * n_pos_before / len(y_train),
    )

    X_train_res, y_train_res = random_oversample(
        X_train, y_train, ratio=CLF_OVERSAMPLE_RATIO, seed=RANDOM_SEED
    )
    n_pos_after = int(y_train_res.sum())
    n_neg_after = int(len(y_train_res) - n_pos_after)
    log.info(
        "Train class counts AFTER oversampling (target ratio=%.2f) — "
        "positive=%d negative=%d (%.2f%%)",
        CLF_OVERSAMPLE_RATIO, n_pos_after, n_neg_after, 100 * n_pos_after / len(y_train_res),
    )

    params = dict(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    params.update(CLF_TUNED_LGBM_PARAMS)
    log.info("Classifier LightGBM params (after tuned overrides): %s", params)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],  # un-resampled VAL — early stopping/AUC tracked on true class balance
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    log.info("Classifier best iteration: %s", model.best_iteration_)
    return model


def tune_threshold(model, X_val, y_val) -> tuple[float, dict]:
    """Sweeps decision thresholds on VALIDATION ONLY and selects the
    threshold maximising F1 for the positive (closure-required) class."""
    val_proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        preds = (val_proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    val_preds_at_best = (val_proba >= best_t).astype(int)
    val_summary = {
        "threshold": float(best_t),
        "val_f1": float(best_f1),
        "val_precision": float(precision_score(y_val, val_preds_at_best, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds_at_best, zero_division=0)),
    }
    log.info(
        "Threshold tuned on VALIDATION: t=%.2f  →  F1=%.3f  P=%.3f  R=%.3f",
        best_t, val_summary["val_f1"], val_summary["val_precision"], val_summary["val_recall"],
    )
    return best_t, val_summary


def evaluate_classifier(model, X_test, y_test, threshold, train_y_for_baseline) -> dict:
    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "recall": recall_score(y_test, test_preds, zero_division=0),
        "f1": f1_score(y_test, test_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix": cm.tolist(),
        "threshold_used": threshold,
    }

    majority_class = int(train_y_for_baseline.mode().iloc[0])
    baseline_preds = np.full(len(y_test), majority_class)
    baseline_cm = confusion_matrix(y_test, baseline_preds, labels=[0, 1])
    baseline = {
        "accuracy": accuracy_score(y_test, baseline_preds),
        "precision": precision_score(y_test, baseline_preds, zero_division=0),
        "recall": recall_score(y_test, baseline_preds, zero_division=0),
        "f1": f1_score(y_test, baseline_preds, zero_division=0),
        "roc_auc": float("nan"),  # constant predictor has no discrimination — undefined, not 0.5
        "confusion_matrix": baseline_cm.tolist(),
        "note": f"majority-class baseline (always predicts {bool(majority_class)})",
    }

    log.info("Classification TEST metrics: %s", metrics)
    log.info("Classification baseline: %s", baseline)
    return {"model": metrics, "baseline": baseline}


# ══════════════════════════════════════════════
# STEP 3-G  Feature importance export
# ══════════════════════════════════════════════
def export_feature_importance(model, feature_cols: list[str], path: Path) -> pd.DataFrame:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    imp = pd.DataFrame(
        {"feature": feature_cols, "gain_importance": gain, "split_importance": split}
    ).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    imp.to_csv(path, index=False)
    log.info("Feature importance written to %s", path)
    return imp


# ══════════════════════════════════════════════
# STEP 3-H  Report generation
# ══════════════════════════════════════════════
def generate_report(
    df, train, val, test, class_balance, reg_results, clf_results,
    threshold, threshold_val_summary, reg_importance, clf_importance,
    reg_feature_cols, clf_feature_cols, clf_cat_cols, recovered_split_timestamp,
) -> str:
    skew = train[TARGET_REG].skew()
    r2 = reg_results["model"]["r2"]
    auc = clf_results["model"]["roc_auc"]
    clf_te_cols = [c for c in clf_feature_cols if c.endswith("_te")]
    clf_numeric_cols = [c for c in clf_feature_cols if c not in clf_cat_cols and c not in clf_te_cols]

    diagnostics = []
    diagnostics.append(
        "- This classifier configuration (junction/zone target encoding, "
        "`dist_from_center`, TRAIN-only oversampling to a "
        f"{CLF_OVERSAMPLE_RATIO:.2f} ratio, and tuned LightGBM regularisation) "
        "follows a dedicated ROC-AUC improvement investigation. See "
        "`reports/experiment_log.md` for full methodology, every candidate "
        "feature/strategy tried — including ones that did **not** help and "
        "were rejected — and the validation evidence behind this specific "
        "configuration. The regressor's pipeline was left untouched, since "
        "the investigation was scoped to classification ROC-AUC only."
    )
    if r2 < 0.3:
        diagnostics.append(
            f"- Regression R² on TEST is {r2:.3f} (low). `resolution_minutes` has a "
            f"heavily right-skewed distribution in TRAIN (skew={skew:.2f}), which "
            f"makes raw-scale MAE/RMSE dominated by a small number of very long "
            f"incidents. A `log1p(resolution_minutes)` target transform (with "
            f"`expm1` back-transform before scoring) is a recommended follow-up "
            f"experiment, along with checking whether outlier incidents (near the "
            f"1437-minute cap) belong to a distinct causal regime worth modelling "
            f"separately."
        )
    if not np.isnan(auc) and auc < 0.75:
        diagnostics.append(
            f"- Classification ROC-AUC on TEST is {auc:.3f}. With only "
            f"{class_balance['train']['positive']} positive examples in TRAIN, the "
            f"classifier has limited signal to learn from for the minority class "
            f"even after the feature/imbalance/hyperparameter changes above. The "
            f"most promising further lever is collecting more closure-required "
            f"incidents over time — at this sample size the ceiling looks "
            f"data-limited rather than feature- or hyperparameter-limited."
        )
    diagnostics.append(
        "- `month` is a temporal feature whose value range shifts systematically "
        "across the chronological split (TRAIN covers earlier months than TEST by "
        "construction). It is kept because it is genuinely available at incident-"
        "report time and may capture real seasonality, but its importance score "
        "should be interpreted cautiously — some of its apparent predictive value "
        "may reflect the split boundary rather than a stable seasonal effect."
    )
    diagnostics.append(
        "- `description` (free text, ~17% missing) and `address` (near-unique "
        "free text) were excluded from both models. A dedicated text pipeline "
        "(e.g. language detection + embeddings, given the corpus mixes English "
        "and other scripts) is a natural Step 4 extension, not in scope here."
    )
    diagnostics.append(
        f"- Dataset size (n={len(df)}) is modest relative to the categorical "
        f"cardinality of `junction` ({df['junction'].nunique()} levels) and "
        f"`corridor` ({df['corridor'].nunique()} levels). Watch for overfitting "
        f"to rare categories; a grouped/stratified cross-validation pass (rather "
        f"than a single chronological holdout) would tighten the confidence in "
        f"these test metrics before production deployment."
    )

    def fmt_imp(imp_df: pd.DataFrame, n=15) -> str:
        lines = ["| Feature | Gain Importance | Split Importance |", "|---|---|---|"]
        for _, r in imp_df.head(n).iterrows():
            lines.append(f"| `{r['feature']}` | {r['gain_importance']:.1f} | {int(r['split_importance'])} |")
        return "\n".join(lines)

    cm = clf_results["model"]["confusion_matrix"]
    bcm = clf_results["baseline"]["confusion_matrix"]

    report = f"""# SENTINELA · Step 3 Model Report
**Pipeline:** `src/train_models.py`

---

## 1. Data Lineage

- Input: `{CENTRALITY_PATH.relative_to(PROJECT_ROOT)}` ({len(df)} rows × {df.shape[1]} cols)
- `split_timestamp` recovery from `{MODELLING_READY_PATH.name}` was {"TRIGGERED and verified (0 mismatches across shared columns)" if recovered_split_timestamp else "NOT needed — column was already present"}.
- Date range: {df[SPLIT_COL].min()} → {df[SPLIT_COL].max()}

## 2. Chronological Split

| Split | Rows | Share | Date range |
|---|---|---|---|
| Train | {len(train)} | {100*len(train)/len(df):.1f}% | {train[SPLIT_COL].min()} → {train[SPLIT_COL].max()} |
| Validation | {len(val)} | {100*len(val)/len(df):.1f}% | {val[SPLIT_COL].min()} → {val[SPLIT_COL].max()} |
| Test | {len(test)} | {100*len(test)/len(df):.1f}% | {test[SPLIT_COL].min()} → {test[SPLIT_COL].max()} |

`{SPLIT_COL}` is used only to sort and split the data — it is excluded from every model's feature set.

## 3. Feature Selection

**Regressor** ({len(reg_feature_cols)} features) — target = `resolution_minutes` — uses the original base feature set, unchanged:
Numeric — {", ".join(f"`{c}`" for c in NUMERIC_FEATURES)}
Categorical — {", ".join(f"`{c}`" for c in CATEGORICAL_FEATURES)} (native LightGBM categorical handling; vocabulary fixed from TRAIN only)

**Classifier** ({len(clf_feature_cols)} features) — target = `requires_road_closure` — uses a classifier-specific set, the result of a dedicated ROC-AUC improvement investigation (`reports/experiment_log.md`):
Numeric — {", ".join(f"`{c}`" for c in clf_numeric_cols)} (all unchanged/raw from the base set, plus an engineered `dist_from_center`: Euclidean distance from the approximate Bengaluru city centre)
Categorical (native) — {", ".join(f"`{c}`" for c in clf_cat_cols)} (TRAIN-only vocabulary)
Target-encoded — {", ".join(f"`{c}`" for c in CLF_TARGET_ENCODE_COLS)} → {", ".join(f"`{c}`" for c in clf_te_cols)} (smoothed, out-of-fold target encoding, replacing native high-cardinality categoricals for these two columns specifically)

**Excluded from all models:** `{SPLIT_COL}` (split key), `address` (near-unique free text / identifier), `description` (free text, {df['description'].isna().mean()*100:.1f}% missing, no NLP pipeline in scope), `nearest_u`/`nearest_v`/`nearest_key` (raw OSM graph IDs — predictive content already captured by `centrality_score` and `snap_distance_m`/`low_confidence_snap`).

**Cross-target exclusions:**
- Regressor features exclude `{TARGET_CLF}` — closure status is itself a predicted outcome, not an input available ahead of resolution time.
- Classifier features exclude `{TARGET_REG}` — resolution time is observed only after/during the incident and is not available at the point a closure decision would be predicted.

## 4. Class Balance Analysis — `requires_road_closure`

| Split | N | Positive | Negative | Positive rate |
|---|---|---|---|---|
| Train | {class_balance['train']['n']} | {class_balance['train']['positive']} | {class_balance['train']['negative']} | {class_balance['train']['positive_rate']*100:.2f}% |
| Validation | {class_balance['val']['n']} | {class_balance['val']['positive']} | {class_balance['val']['negative']} | {class_balance['val']['positive_rate']*100:.2f}% |
| Test | {class_balance['test']['n']} | {class_balance['test']['positive']} | {class_balance['test']['negative']} | {class_balance['test']['positive_rate']*100:.2f}% |

Imbalance is handled via TRAIN-only random oversampling of the minority class to a positive:negative ratio of **{CLF_OVERSAMPLE_RATIO:.2f}**, replacing the original `scale_pos_weight` approach. VALIDATION and TEST are **never** resampled — both retain the true class balance shown above. A dedicated ablation (`reports/experiment_log.md`) found this gives a small but consistent and lower-variance ROC-AUC improvement over `scale_pos_weight`; the decision threshold (below) is tuned on the un-resampled VALIDATION set.

## 5. Regression Results — TEST set only (target = `resolution_minutes`)

| Metric | Model | Median baseline | Mean baseline |
|---|---|---|---|
| MAE | {reg_results['model']['mae']:.3f} | {reg_results['baselines']['median_baseline']['mae']:.3f} | {reg_results['baselines']['mean_baseline']['mae']:.3f} |
| RMSE | {reg_results['model']['rmse']:.3f} | {reg_results['baselines']['median_baseline']['rmse']:.3f} | {reg_results['baselines']['mean_baseline']['rmse']:.3f} |
| R² | {reg_results['model']['r2']:.4f} | {reg_results['baselines']['median_baseline']['r2']:.4f} | {reg_results['baselines']['mean_baseline']['r2']:.4f} |
| Median Abs. Error | {reg_results['model']['medae']:.3f} | {reg_results['baselines']['median_baseline']['medae']:.3f} | {reg_results['baselines']['mean_baseline']['medae']:.3f} |

Baselines predict a single constant (TRAIN median / TRAIN mean of `resolution_minutes`) for every TEST row.

### Top feature importances (regressor)
{fmt_imp(reg_importance)}

## 6. Classification Results — TEST set only (target = `requires_road_closure`)

Threshold tuned on VALIDATION: **{threshold:.2f}** (val F1={threshold_val_summary['val_f1']:.3f}, val P={threshold_val_summary['val_precision']:.3f}, val R={threshold_val_summary['val_recall']:.3f})

| Metric | Model | Majority-class baseline |
|---|---|---|
| Accuracy | {clf_results['model']['accuracy']:.4f} | {clf_results['baseline']['accuracy']:.4f} |
| Precision | {clf_results['model']['precision']:.4f} | {clf_results['baseline']['precision']:.4f} |
| Recall | {clf_results['model']['recall']:.4f} | {clf_results['baseline']['recall']:.4f} |
| F1 | {clf_results['model']['f1']:.4f} | {clf_results['baseline']['f1']:.4f} |
| ROC-AUC | {clf_results['model']['roc_auc']:.4f} | N/A (constant predictor has no discrimination) |

**Model confusion matrix (TEST)** — rows=actual, cols=predicted, order=[False, True]:
```
{cm}
```

**Baseline confusion matrix (TEST)** — {clf_results['baseline']['note']}:
```
{bcm}
```

### Top feature importances (classifier)
{fmt_imp(clf_importance)}

## 7. Diagnostics & Recommendations

{chr(10).join(diagnostics)}

## 8. Artifacts

- `models/resolution_model.pkl`
- `models/closure_model.pkl`
- `models/feature_metadata.json`
- `reports/regression_feature_importance.csv`
- `reports/classification_feature_importance.csv`
"""
    return report


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    had_split_ts_before = False
    if CENTRALITY_PATH.exists():
        had_split_ts_before = SPLIT_COL in pd.read_csv(CENTRALITY_PATH, nrows=1).columns

    df = load_dataset()
    recovered = not had_split_ts_before

    train, val, test = chronological_split(df)
    class_balance = class_balance_report(train, val, test)

    # ── Regressor ──────────────────────────────────────────────
    reg_feature_cols = get_feature_columns(TARGET_REG)
    Xr_train, Xr_val, Xr_test, reg_cat_cols = prepare_features(train, val, test, reg_feature_cols)
    yr_train, yr_val, yr_test = train[TARGET_REG], val[TARGET_REG], test[TARGET_REG]

    reg_model = train_regressor(Xr_train, yr_train, Xr_val, yr_val, reg_cat_cols)
    reg_results = evaluate_regressor(reg_model, Xr_test, yr_test, yr_train)
    reg_importance = export_feature_importance(
        reg_model, reg_feature_cols, REPORTS_DIR / "regression_feature_importance.csv"
    )

    # ── Classifier ─────────────────────────────────────────────
    Xc_train, Xc_val, Xc_test, clf_cat_cols, clf_encoders, clf_feature_cols = prepare_classifier_features(
        train, val, test
    )
    yc_train = train[TARGET_CLF].astype(bool).astype(int)
    yc_val = val[TARGET_CLF].astype(bool).astype(int)
    yc_test = test[TARGET_CLF].astype(bool).astype(int)

    clf_model = train_classifier(Xc_train, yc_train, Xc_val, yc_val, clf_cat_cols)
    threshold, threshold_val_summary = tune_threshold(clf_model, Xc_val, yc_val)
    clf_results = evaluate_classifier(clf_model, Xc_test, yc_test, threshold, yc_train)
    clf_importance = export_feature_importance(
        clf_model, clf_feature_cols, REPORTS_DIR / "classification_feature_importance.csv"
    )

    # ── Save models + metadata ────────────────────────────────
    joblib.dump(reg_model, MODELS_DIR / "resolution_model.pkl")
    joblib.dump(clf_model, MODELS_DIR / "closure_model.pkl")
    log.info("Saved models to %s", MODELS_DIR)

    metadata = {
        "regression": {
            "target": TARGET_REG,
            "feature_columns": reg_feature_cols,
            "categorical_features": reg_cat_cols,
            "categories": {c: Xr_train[c].cat.categories.tolist() for c in reg_cat_cols},
        },
        "classification": {
            "target": TARGET_CLF,
            "feature_columns": clf_feature_cols,
            "categorical_features": clf_cat_cols,
            "categories": {c: Xc_train[c].cat.categories.tolist() for c in clf_cat_cols},
            # Target-encoding maps for junction_te/zone_te, persisted so a future
            # production inference pipeline can replicate the encoding on new
            # incoming rows. Unseen categories fall back to "global_mean", matching
            # the OOF fallback behaviour used during training (see apply_target_encoding).
            "target_encoders": clf_encoders,
            "decision_threshold": threshold,
            "oversample_ratio": CLF_OVERSAMPLE_RATIO,
            "tuned_lgbm_params": CLF_TUNED_LGBM_PARAMS,
        },
        "split": {
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": round(1 - TRAIN_FRAC - VAL_FRAC, 4),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("Saved feature metadata to %s", MODELS_DIR / "feature_metadata.json")

    # ── Report ──────────────────────────────────────────────────
    report_text = generate_report(
        df, train, val, test, class_balance, reg_results, clf_results,
        threshold, threshold_val_summary, reg_importance, clf_importance,
        reg_feature_cols, clf_feature_cols, clf_cat_cols, recovered,
    )
    report_path = REPORTS_DIR / "model_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Wrote report to %s", report_path)
    log.info("Done.")


if __name__ == "__main__":
    main()