"""
cascade_risk_score.py
======================
SENTINELA — Cascade Risk Score (CRS)

Combines the three available model outputs

    P  = closure_probability              (classifier, already in [0, 1])
    T  = predicted_resolution_minutes      (regressor, minutes, right-skewed)
    C  = centrality_score                  (graph metric, right-skewed)

into a single 0–100 score representing the EXPECTED cascading impact of an
incident: how *likely* it is to require a closure, multiplied by how
*severe* the consequences would be if it does (duration + network reach).

    P_norm = P                                          (already bounded)
    T_norm = QuantileRank_TRAIN(T)                       in [0, 1]
    C_norm = QuantileRank_TRAIN(C)                       in [0, 1]
    Impact = w_T * T_norm + w_C * C_norm                 in [0, 1]
    CRS    = 100 * sqrt( P_norm * Impact )               in [0, 100]

See `reports/cascade_risk_score_rationale.md` (or the chat explanation
this file accompanied) for the full scientific rationale behind each
design choice. In short:

  - Risk = Likelihood x Impact is the standard FMEA / ISO-31000 /
    actuarial risk framework — it is not specific to this project.
  - The sqrt (geometric mean of P_norm and Impact) follows the same logic
    the UN Human Development Index adopted in 2010: it still forces the
    score to 0 if either factor is 0, but avoids the harsh compression of
    a flat product.
  - T and C are quantile-rank transformed (not min-max) because both are
    heavily right-skewed in the real training data (resolution_minutes:
    median 45 min but max 1437 min; centrality_score: median 0.0034 but
    max 0.066) — a raw min-max would squeeze almost every real incident
    into the bottom few percent of the scale.

This module has no LightGBM / project-specific dependency — it only
needs numpy / pandas / scikit-learn, and can be unit-tested or reused
independently of the rest of the SENTINELA pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# ----------------------------------------------------------------------
# Configuration — tune here, not buried in the math.
#
# Defaults are an unbiased 50/50 split between "how long" and "how central"
# because we have no historical ground-truth on *actual* downstream
# cascade magnitude to calibrate against. If/when such data exists (e.g.
# measured downstream delivery-delay minutes per incident), refit these
# weights via a regression of observed cascade magnitude on T_norm and
# C_norm and use the standardized coefficients as w_T / w_C instead.
# ----------------------------------------------------------------------
WEIGHT_TIME = 0.5          # contribution of resolution-time severity to Impact
WEIGHT_CENTRALITY = 0.5    # contribution of network-centrality severity to Impact
assert abs(WEIGHT_TIME + WEIGHT_CENTRALITY - 1.0) < 1e-9, "w_T + w_C must equal 1"

# Operational priority bands. These are starting points, not a law of
# nature — once the score is running in production, recalibrate the cut
# points to the *actual* observed score distribution (e.g. top 5% =
# Critical, next 15% = High, ...) rather than leaving them as fixed
# numbers forever.
RISK_BANDS = [
    (0,   20, "Low",      "Log only — no action needed"),
    (20,  40, "Moderate", "Flag for monitoring; suggest alternate routes to affected drivers"),
    (40,  60, "Elevated", "Dispatch traffic management to the corridor"),
    (60,  80, "High",     "Proactive rerouting + traveler alerts across the zone"),
    (80, 101, "Critical", "Immediate multi-agency response; treat as a network-wide event"),
]


def risk_band(score: float) -> tuple[str, str]:
    """Map a 0-100 score to (label, recommended_action)."""
    for lo, hi, label, action in RISK_BANDS:
        if lo <= score < hi:
            return label, action
    return RISK_BANDS[-1][2], RISK_BANDS[-1][3]


class CascadeRiskScorer:
    """
    Fit the two quantile-rank transformers once on TRAIN data; reuse the
    fitted object for every future prediction (val/test/production).
    Never refit on val/test/production data — that would let the
    distribution of *future* incidents leak into what "high resolution
    time" or "high centrality" means today.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.qt_time: Optional[QuantileTransformer] = None
        self.qt_centrality: Optional[QuantileTransformer] = None

    def fit(self, train_resolution_minutes: np.ndarray, train_centrality_score: np.ndarray) -> "CascadeRiskScorer":
        train_resolution_minutes = np.asarray(train_resolution_minutes, dtype=float).reshape(-1, 1)
        train_centrality_score = np.asarray(train_centrality_score, dtype=float).reshape(-1, 1)

        n_q = min(self.n_quantiles, len(train_resolution_minutes))
        self.qt_time = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform", random_state=42)
        self.qt_centrality = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform", random_state=42)

        self.qt_time.fit(train_resolution_minutes)
        self.qt_centrality.fit(train_centrality_score)
        return self

    def score(
        self,
        closure_probability: Union[float, np.ndarray],
        predicted_resolution_minutes: Union[float, np.ndarray],
        centrality_score: Union[float, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compute the Cascade Risk Score for one or many incidents.
        Returns a DataFrame with every intermediate quantity so the
        result is auditable, not a black box.
        """
        if self.qt_time is None or self.qt_centrality is None:
            raise RuntimeError("Call .fit(train_resolution_minutes, train_centrality_score) before .score().")

        p = np.atleast_1d(np.asarray(closure_probability, dtype=float))
        t = np.atleast_1d(np.asarray(predicted_resolution_minutes, dtype=float))
        c = np.atleast_1d(np.asarray(centrality_score, dtype=float))

        p_norm = np.clip(p, 0.0, 1.0)
        t_norm = self.qt_time.transform(t.reshape(-1, 1)).ravel()
        c_norm = self.qt_centrality.transform(c.reshape(-1, 1)).ravel()

        impact = WEIGHT_TIME * t_norm + WEIGHT_CENTRALITY * c_norm
        cascade_risk_score = 100.0 * np.sqrt(p_norm * impact)

        out = pd.DataFrame({
            "closure_probability": p,
            "predicted_resolution_minutes": t,
            "centrality_score": c,
            "p_norm": np.round(p_norm, 4),
            "t_norm": np.round(t_norm, 4),
            "c_norm": np.round(c_norm, 4),
            "impact": np.round(impact, 4),
            "cascade_risk_score": np.round(cascade_risk_score, 2),
        })
        bands = out["cascade_risk_score"].apply(risk_band)
        out["priority"] = bands.apply(lambda b: b[0])
        out["recommended_action"] = bands.apply(lambda b: b[1])
        return out

    def save(self, path: Union[str, Path]) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> "CascadeRiskScorer":
        import joblib
        return joblib.load(path)


# ----------------------------------------------------------------------
# Demo / worked examples
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_prep import load_and_split  # SENTINELA's own chronological split

    train, val, test = load_and_split()

    scorer = CascadeRiskScorer().fit(
        train_resolution_minutes=train["resolution_minutes"].values,
        train_centrality_score=train["centrality_score"].values,
    )

    # Six illustrative incidents. centrality_score / resolution_minutes are
    # REAL values pulled from actual rows in the dataset (so two of the
    # three inputs are grounded in real incidents); closure_probability is
    # a hypothetical model output for demo purposes, since no live
    # classifier is running in this environment. Labelled accordingly.
    pool = pd.concat([val, test], ignore_index=True)
    examples = pd.DataFrame({
        "incident": [
            "A: certain, long, central   (major arterial pile-up)",
            "B: certain, short, peripheral (minor residential fender-bender)",
            "C: uncertain, long, central (forecasted festival congestion)",
            "D: certain, short, central  (quick clearance at a key junction)",
            "E: uncertain, short, peripheral (low-stakes routine report)",
            "F: certain, long, peripheral (multi-hour rural road closure)",
        ],
        "closure_probability": [0.93, 0.91, 0.35, 0.88, 0.22, 0.90],
        "predicted_resolution_minutes": [
            pool["resolution_minutes"].quantile(0.92),
            pool["resolution_minutes"].quantile(0.10),
            pool["resolution_minutes"].quantile(0.90),
            pool["resolution_minutes"].quantile(0.15),
            pool["resolution_minutes"].quantile(0.20),
            pool["resolution_minutes"].quantile(0.93),
        ],
        "centrality_score": [
            pool["centrality_score"].quantile(0.95),
            pool["centrality_score"].quantile(0.08),
            pool["centrality_score"].quantile(0.93),
            pool["centrality_score"].quantile(0.90),
            pool["centrality_score"].quantile(0.10),
            pool["centrality_score"].quantile(0.07),
        ],
    })

    result = scorer.score(
        examples["closure_probability"],
        examples["predicted_resolution_minutes"],
        examples["centrality_score"],
    )
    result.insert(0, "incident", examples["incident"])
    result = result.sort_values("cascade_risk_score", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print(result[[
        "incident", "closure_probability", "predicted_resolution_minutes", "centrality_score",
        "p_norm", "t_norm", "c_norm", "cascade_risk_score", "priority",
    ]].to_string(index=False))
