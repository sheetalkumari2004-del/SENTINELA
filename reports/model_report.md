# SENTINELA · Step 3 Model Report
**Pipeline:** `src/train_models.py`

---

## 1. Data Lineage

- Input: `data\processed\astram_with_centrality.csv` (2533 rows × 22 cols)
- `split_timestamp` recovery from `astram_modelling_ready.csv` was TRIGGERED and verified (0 mismatches across shared columns).
- Date range: 2023-11-10 00:54:48.154000+05:30 → 2024-04-08 16:48:02.081000+05:30

## 2. Chronological Split

| Split | Rows | Share | Date range |
|---|---|---|---|
| Train | 1773 | 70.0% | 2023-11-10 00:54:48.154000+05:30 → 2024-03-02 07:31:13.778000+05:30 |
| Validation | 380 | 15.0% | 2024-03-02 09:00:15.995000+05:30 → 2024-03-24 11:20:20.839000+05:30 |
| Test | 380 | 15.0% | 2024-03-24 13:19:41.393000+05:30 → 2024-04-08 16:48:02.081000+05:30 |

`split_timestamp` is used only to sort and split the data — it is excluded from every model's feature set.

## 3. Feature Selection

**Regressor** (14 features) — target = `resolution_minutes` — uses the original base feature set, unchanged:
Numeric — `latitude`, `longitude`, `hour_of_day`, `day_of_week`, `month`, `is_planned`, `centrality_score`, `snap_distance_m`, `low_confidence_snap`
Categorical — `event_cause`, `veh_type`, `corridor`, `zone`, `junction` (native LightGBM categorical handling; vocabulary fixed from TRAIN only)

**Classifier** (15 features) — target = `requires_road_closure` — uses a classifier-specific set, the result of a dedicated ROC-AUC improvement investigation (`reports/experiment_log.md`):
Numeric — `latitude`, `longitude`, `hour_of_day`, `day_of_week`, `month`, `is_planned`, `centrality_score`, `snap_distance_m`, `low_confidence_snap`, `dist_from_center` (all unchanged/raw from the base set, plus an engineered `dist_from_center`: Euclidean distance from the approximate Bengaluru city centre)
Categorical (native) — `event_cause`, `veh_type`, `corridor` (TRAIN-only vocabulary)
Target-encoded — `junction`, `zone` → `junction_te`, `zone_te` (smoothed, out-of-fold target encoding, replacing native high-cardinality categoricals for these two columns specifically)

**Excluded from all models:** `split_timestamp` (split key), `address` (near-unique free text / identifier), `description` (free text, 17.4% missing, no NLP pipeline in scope), `nearest_u`/`nearest_v`/`nearest_key` (raw OSM graph IDs — predictive content already captured by `centrality_score` and `snap_distance_m`/`low_confidence_snap`).

**Cross-target exclusions:**
- Regressor features exclude `requires_road_closure` — closure status is itself a predicted outcome, not an input available ahead of resolution time.
- Classifier features exclude `resolution_minutes` — resolution time is observed only after/during the incident and is not available at the point a closure decision would be predicted.

## 4. Class Balance Analysis — `requires_road_closure`

| Split | N | Positive | Negative | Positive rate |
|---|---|---|---|---|
| Train | 1773 | 112 | 1661 | 6.32% |
| Validation | 380 | 33 | 347 | 8.68% |
| Test | 380 | 44 | 336 | 11.58% |

Imbalance is handled via TRAIN-only random oversampling of the minority class to a positive:negative ratio of **0.50**, replacing the original `scale_pos_weight` approach. VALIDATION and TEST are **never** resampled — both retain the true class balance shown above. A dedicated ablation (`reports/experiment_log.md`) found this gives a small but consistent and lower-variance ROC-AUC improvement over `scale_pos_weight`; the decision threshold (below) is tuned on the un-resampled VALIDATION set.

## 5. Regression Results — TEST set only (target = `resolution_minutes`)

| Metric | Model | Median baseline | Mean baseline |
|---|---|---|---|
| MAE | 121.677 | 110.213 | 118.970 |
| RMSE | 270.671 | 287.614 | 277.752 |
| R² | 0.0206 | -0.1059 | -0.0314 |
| Median Abs. Error | 49.549 | 28.875 | 53.447 |

Baselines predict a single constant (TRAIN median / TRAIN mean of `resolution_minutes`) for every TEST row.

### Top feature importances (regressor)
| Feature | Gain Importance | Split Importance |
|---|---|---|
| `event_cause` | 120644539.0 | 26 |
| `latitude` | 27360243.4 | 138 |
| `veh_type` | 24059833.7 | 25 |
| `snap_distance_m` | 22472175.1 | 147 |
| `centrality_score` | 20430201.0 | 129 |
| `longitude` | 18650076.1 | 106 |
| `day_of_week` | 10112518.2 | 39 |
| `hour_of_day` | 8811294.0 | 94 |
| `month` | 2003882.4 | 32 |
| `zone` | 1997696.0 | 4 |
| `corridor` | 263949.9 | 10 |
| `is_planned` | 0.0 | 0 |
| `low_confidence_snap` | 0.0 | 0 |
| `junction` | 0.0 | 0 |

## 6. Classification Results — TEST set only (target = `requires_road_closure`)

Threshold tuned on VALIDATION: **0.35** (val F1=0.468, val P=0.361, val R=0.667)

| Metric | Model | Majority-class baseline |
|---|---|---|
| Accuracy | 0.8079 | 0.8842 |
| Precision | 0.3294 | 0.0000 |
| Recall | 0.6364 | 0.0000 |
| F1 | 0.4341 | 0.0000 |
| ROC-AUC | 0.7970 | N/A (constant predictor has no discrimination) |

**Model confusion matrix (TEST)** — rows=actual, cols=predicted, order=[False, True]:
```
[[279, 57], [16, 28]]
```

**Baseline confusion matrix (TEST)** — majority-class baseline (always predicts False):
```
[[336, 0], [44, 0]]
```

### Top feature importances (classifier)
| Feature | Gain Importance | Split Importance |
|---|---|---|
| `dist_from_center` | 7258.7 | 741 |
| `snap_distance_m` | 7227.0 | 844 |
| `junction_te` | 6942.5 | 644 |
| `longitude` | 6619.7 | 667 |
| `centrality_score` | 6111.9 | 796 |
| `corridor` | 5520.8 | 275 |
| `latitude` | 5448.0 | 609 |
| `hour_of_day` | 4690.7 | 438 |
| `zone_te` | 4269.8 | 480 |
| `event_cause` | 3928.1 | 191 |
| `veh_type` | 2928.9 | 203 |
| `day_of_week` | 2582.7 | 285 |
| `month` | 1483.6 | 176 |
| `is_planned` | 62.8 | 21 |
| `low_confidence_snap` | 0.0 | 0 |

## 7. Diagnostics & Recommendations

- This classifier configuration (junction/zone target encoding, `dist_from_center`, TRAIN-only oversampling to a 0.50 ratio, and tuned LightGBM regularisation) follows a dedicated ROC-AUC improvement investigation. See `reports/experiment_log.md` for full methodology, every candidate feature/strategy tried — including ones that did **not** help and were rejected — and the validation evidence behind this specific configuration. The regressor's pipeline was left untouched, since the investigation was scoped to classification ROC-AUC only.
- Regression R² on TEST is 0.021 (low). `resolution_minutes` has a heavily right-skewed distribution in TRAIN (skew=5.57), which makes raw-scale MAE/RMSE dominated by a small number of very long incidents. A `log1p(resolution_minutes)` target transform (with `expm1` back-transform before scoring) is a recommended follow-up experiment, along with checking whether outlier incidents (near the 1437-minute cap) belong to a distinct causal regime worth modelling separately.
- `month` is a temporal feature whose value range shifts systematically across the chronological split (TRAIN covers earlier months than TEST by construction). It is kept because it is genuinely available at incident-report time and may capture real seasonality, but its importance score should be interpreted cautiously — some of its apparent predictive value may reflect the split boundary rather than a stable seasonal effect.
- `description` (free text, ~17% missing) and `address` (near-unique free text) were excluded from both models. A dedicated text pipeline (e.g. language detection + embeddings, given the corpus mixes English and other scripts) is a natural Step 4 extension, not in scope here.
- Dataset size (n=2533) is modest relative to the categorical cardinality of `junction` (230 levels) and `corridor` (23 levels). Watch for overfitting to rare categories; a grouped/stratified cross-validation pass (rather than a single chronological holdout) would tighten the confidence in these test metrics before production deployment.

## 8. Artifacts

- `models/resolution_model.pkl`
- `models/closure_model.pkl`
- `models/feature_metadata.json`
- `reports/regression_feature_importance.csv`
- `reports/classification_feature_importance.csv`
