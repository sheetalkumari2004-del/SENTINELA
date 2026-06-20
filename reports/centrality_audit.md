    # SENTINELA · Road Network Centrality Audit
    **Generated:** 2026-06-16 19:08 UTC
    **Pipeline:** `src/centrality.py`

    ---

    ## 1. Road Network

    | Property | Value |
    |---|---|
    | Place queried | Bengaluru, Karnataka, India |
    | Network type | drive |
    | Nodes (raw graph) | 155,359 |
    | Edges (raw graph) | 393,717 |
    | Nodes (simple undirected, used for centrality) | 155,359 |
    | Edges (simple undirected, used for centrality) | 204,732 |
    | Projected CRS | EPSG:32643 |
    | Source | OpenStreetMap via OSMnx |

    ---

    ## 2. Betweenness Centrality Computation

    | Property | Value |
    |---|---|
    | Method | approximate |
    | Pivot nodes (k) | 500 |
    | Random seed | 42 |
    | Edge weight | length (metres) |
    | Normalized | Yes |
    | Computation time | 608.8 s |

    **Note:** approximate (k-pivot) betweenness was used because the graph exceeds the exact-computation threshold (3,000 nodes). This is the standard production approximation (Brandes & Pich, 2007) and converges close to exact values for city-scale graphs.

    ---

    ## 3. Incident Snapping

    | Metric | Value |
    |---|---|
    | Total incidents | 2,533 |
    | Successfully snapped & scored | 2,533 (100.0 %) |
    | Min snap distance | 0.0 m |
    | Median snap distance | 4.0 m |
    | Mean snap distance | 163.2 m |
    | 95th percentile snap distance | 64.1 m |
    | Max snap distance | 18433.3 m |
    | Low-confidence snaps (> 750 m) | 76 (3.0 %) |

    Low-confidence snaps are **retained** (not dropped) with a `low_confidence_snap`
    flag column, since this is a feature-engineering step rather than a filtering
    step — the decision to exclude them is left to the modelling stage.

    ---

    ## 4. Centrality Score Distribution

    | Statistic | Value |
    |---|---|
    | Min    | 0.000000 |
    | 25th % | 0.000176 |
    | Median | 0.003696 |
    | Mean   | 0.008523 |
    | 75th % | 0.011164 |
    | Max    | 0.066421 |
    | Std    | 0.012228 |
    | Missing | 0 |

    ---

    ## 5. Runtime Breakdown

    | Stage | Duration |
    |---|---|
    | Load incidents | 0.1 s |
| Pull/load road network | 156.5 s |
| Project network | 8.5 s |
| Collapse to simple undirected | 1.4 s |
| Compute betweenness centrality | 609.0 s |
| Snap incidents to edges | 5.2 s |
| Assign centrality scores | 0.0 s |
    | **Total** | **780.7 s** |

    ---

    ## 6. Output Dataset

    - **Rows:** 2,533
    - **Columns:** 21
    - **File:** `data/processed/astram_with_centrality.csv`
    - **New columns added:** `nearest_u`, `nearest_v`, `nearest_key`,
      `snap_distance_m`, `low_confidence_snap`, `centrality_score`

    ---

    ## 7. Notes

    - Road network was collapsed from a directed MultiDiGraph (OSMnx's native
      format) to a simple undirected graph before computing centrality, since
      congestion-propagation relevance is not direction-sensitive and the
      MultiDiGraph's parallel edges (e.g. divided carriageways) would otherwise
      double-count the same physical road segment.
    - Centrality scores are **edge** betweenness centrality (not node), matching
      the requirement to snap incidents to the nearest road edge rather than
      the nearest junction.
    - Edge weighting uses real-world length in metres (via UTM projection,
      EPSG:32643), so centrality reflects actual travel-distance importance,
      not hop count.
    - The road network is cached to `data/interim/bengaluru_drive_graph.graphml`
      so repeated pipeline runs do not re-query the Overpass API. Delete this
      file or set `FORCE_REFRESH = True` to pull a fresh network.
