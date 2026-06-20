    # SENTINELA · Data Audit Report
    **Generated:** 2026-06-17 14:41 UTC
    **Pipeline:** `src/preprocess.py`

    ---

    ## 1. Record Counts

    | Stage | Records |
    |---|---|
    | Raw dataset loaded | 8,173 |
    | Removed – no end timestamp | 4,964 |
    | Removed – negative resolution time | 4 |
    | Removed – resolution > 1440 min | 672 |
    | **Final modelling dataset** | **2,533** |

    Retention rate: **31.0 %**

    ---

    ## 2. Planned vs Unplanned

    | event_type | Count | Share |
    |---|---|---|
    | Unplanned | 2,509 | 99.1 % |
    | Planned   | 24   | 0.9 % |
    | **Total** | **2,533** | 100 % |

    ---

    ## 3. Resolution Time Distribution (minutes)

    | Statistic | Value |
    |---|---|
    | Min    | 0.7 |
    | 25th % | 22.2 |
    | Median | 46.1 |
    | Mean   | 98.7 |
    | 75th % | 85.3 |
    | Max    | 1437.1 |
    | Std    | 205.1 |

    ---

    ## 4. Missing Values – Raw Dataset (columns with any missingness)

    | Column | Missing % |
    |---|---|
    | `map_file` | 100.00 % |
| `meta_data` | 100.00 % |
| `comment` | 100.00 % |
| `direction` | 99.47 % |
| `resolved_at_longitude` | 99.09 % |
| `resolved_at_latitude` | 99.09 % |
| `resolved_at_address` | 99.09 % |
| `resolved_datetime` | 99.09 % |
| `resolved_by_id` | 99.09 % |
| `citizen_accident_id` | 98.43 % |
| `assigned_to_police_id` | 98.43 % |
| `route_path` | 98.32 % |
| `age_of_truck` | 96.62 % |
| `cargo_material` | 96.62 % |
| `reason_breakdown` | 96.62 % |
| `end_datetime` | 94.00 % |
| `end_address` | 91.59 % |
| `junction` | 69.29 % |
| `closed_datetime` | 61.57 % |
| `closed_by_id` | 61.57 % |
| `gba_identifier` | 57.86 % |
| `zone` | 57.86 % |
| `veh_no` | 40.22 % |
| `veh_type` | 40.21 % |
| `description` | 16.64 % |
| `kgid` | 3.17 % |
| `endlatitude` | 2.07 % |
| `endlongitude` | 2.07 % |
| `corridor` | 0.24 % |
| `address` | 0.04 % |
| `last_modified_by_id` | 0.04 % |
| `created_by_id` | 0.02 % |
| `priority` | 0.02 % |

    ---

    ## 5. Missing Values – Final Modelling Dataset

    | Column | Missing % |
    |---|---|
    | `description` | 17.45 % |

    ---

    ## 6. Rows Removed During Cleaning

    | Reason | Rows removed |
    |---|---|
    | No end timestamp (active/never-closed incidents) | 4,964 |
    | Negative resolution time (timestamp errors) | 4 |
    | Resolution time > 1440 minutes | 672 |
    | **Total removed** | **5,640** |

    ---

    ## 7. Final Modelling Dataset

    - **Rows:** 2,533
    - **Columns:** 16
    - **File:** `data/processed/astram_modelling_ready.csv`

    ### Retained columns

    - `latitude`
- `longitude`
- `address`
- `event_cause`
- `requires_road_closure`
- `description`
- `veh_type`
- `corridor`
- `zone`
- `junction`
- `resolution_minutes`
- `is_planned`
- `hour_of_day`
- `day_of_week`
- `month`
- `split_timestamp`

    ---

    ## 8. Notes

    - `veh_type`, `corridor`, `zone`, `junction` NaNs filled with `"unknown"`.
    - `description` retained as free text; not imputed.
    - All timestamps converted to UTC then shifted to IST (Asia/Kolkata) for
      temporal feature extraction.
    - `priority` dropped: only two non-null values exist and both are `"High"` –
      zero variance.
