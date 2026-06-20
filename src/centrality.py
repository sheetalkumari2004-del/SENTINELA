"""
SENTINELA – Step 2: Road Network Centrality
=============================================
Bengaluru drivable road network (OSMnx) · Betweenness Centrality (NetworkX)
Production-ready · snaps incidents to nearest road edge, assigns a centrality
score, generates astram_with_centrality.csv + reports/centrality_audit.md

Pipeline position (fixed architecture):
    Astram Data → Preprocessing ✅ → Road Network Centrality ← THIS STEP
                → LightGBM Models → Cascade Risk Score → Dashboard

Input : data/processed/astram_modelling_ready.csv   (from Step 1)
Output: data/processed/astram_with_centrality.csv
        reports/centrality_audit.md
"""

import sys
import time
import logging
import textwrap
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pyproj import Transformer

# ──────────────────────────────────────────────
# PATHS  (mirrors src/preprocess.py convention)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH      = PROJECT_ROOT / "data" / "processed" / "astram_modelling_ready.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "astram_with_centrality.csv"
AUDIT_PATH   = PROJECT_ROOT / "reports" / "centrality_audit.md"
GRAPH_CACHE  = PROJECT_ROOT / "data" / "interim" / "bengaluru_drive_graph.graphml"

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
PLACE_NAME    = "Bengaluru, Karnataka, India"
NETWORK_TYPE  = "drive"
FORCE_REFRESH = False     # set True to re-download the graph even if a cache exists

# Exact betweenness centrality costs O(V · E) — infeasible in production for a
# city-scale graph (Bengaluru's drivable network typically has tens of
# thousands of nodes). We use NetworkX's k-pivot approximation (Brandes &
# Pich, 2007): betweenness is estimated from a random sample of `K_PIVOTS`
# source nodes instead of all nodes, with a fixed seed for reproducibility.
# Below EXACT_THRESHOLD_NODES the graph is small enough to compute exactly.
EXACT_THRESHOLD_NODES = 3000
K_PIVOTS              = 500
RANDOM_SEED           = 42

# Incidents whose nearest road edge is farther than this are flagged as
# low-confidence snaps (likely bad geocoding) but are NOT dropped — this is
# a feature-engineering step, not a filtering step.
MAX_SNAP_DISTANCE_M = 750

# Bengaluru's approximate metro bounding box, used only as a sanity check
# on incoming coordinates (not used for graph extraction).
BLR_LAT_RANGE = (12.7, 13.3)
BLR_LON_RANGE = (77.3, 77.9)

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinela.centrality")


# ══════════════════════════════════════════════
# STEP 2-A  Load incidents
# ══════════════════════════════════════════════
def load_incidents(path: Path) -> pd.DataFrame:
    """
    Load the Step 1 modelling-ready dataset.

    Validates that latitude/longitude exist and flags (without dropping)
    coordinates outside Bengaluru's approximate metro bounding box, since
    those would later receive an unreliable or missing centrality score.
    """
    if not path.exists():
        log.error("Input file not found: %s", path)
        raise FileNotFoundError(
            f"Expected Step 1 output at {path}. Run src/preprocess.py first."
        )

    log.info("Loading incidents from %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d incidents × %d cols", *df.shape)

    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from input dataset.")

    n_missing_coords = df[["latitude", "longitude"]].isna().any(axis=1).sum()
    if n_missing_coords:
        log.warning("%d incidents have missing latitude/longitude.", n_missing_coords)

    out_of_bounds = (
        ~df["latitude"].between(*BLR_LAT_RANGE)
        | ~df["longitude"].between(*BLR_LON_RANGE)
    ) & df["latitude"].notna() & df["longitude"].notna()
    n_oob = int(out_of_bounds.sum())
    if n_oob:
        log.warning(
            "%d incidents fall outside the expected Bengaluru bounding box "
            "(lat %.1f–%.1f, lon %.1f–%.1f). They will still be snapped if "
            "possible, but flagged as low-confidence.",
            n_oob, *BLR_LAT_RANGE, *BLR_LON_RANGE,
        )

    return df


# ══════════════════════════════════════════════
# STEP 2-B  Pull / cache the road network
# ══════════════════════════════════════════════
def get_road_network(place: str, network_type: str, cache_path: Path,
                      force_refresh: bool = False) -> "ox.graph":
    """
    Pull Bengaluru's drivable road network from OpenStreetMap via OSMnx.

    A GraphML cache is used so repeated pipeline runs (e.g. during model
    iteration in later steps) don't re-hit the Overpass API every time —
    this is the single slowest and least reliable part of the pipeline
    (Overpass servers are rate-limited and occasionally time out).

    Falls back to retrying once on transient network errors before raising,
    since Overpass timeouts are common for large city queries.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        log.info("Loading cached road network from %s", cache_path)
        G = ox.load_graphml(cache_path)
        log.info("Cached graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    log.info("Pulling '%s' (%s network) from OpenStreetMap via OSMnx …", place, network_type)
    last_err = None
    for attempt in (1, 2):
        try:
            G = ox.graph_from_place(place, network_type=network_type)
            break
        except Exception as exc:  # noqa: BLE001 — Overpass can raise several distinct exception types
            last_err = exc
            log.warning("Attempt %d to pull road network failed: %s", attempt, exc)
            time.sleep(5)
    else:
        log.error("Failed to pull road network for '%s' after 2 attempts.", place)
        raise RuntimeError(
            f"Could not retrieve OSM data for '{place}'. Last error: {last_err}"
        )

    log.info("Pulled graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    ox.save_graphml(G, cache_path)
    log.info("Cached raw graph → %s", cache_path)
    return G


def project_network(G) -> tuple:
    """
    Project the graph to its local UTM zone (EPSG:32643 for Bengaluru) so
    that edge lengths and snap distances are in metres rather than degrees.
    Returns the projected graph and its CRS.
    """
    log.info("Projecting graph to local UTM …")
    G_proj = ox.project_graph(G)
    crs = G_proj.graph["crs"]
    log.info("Projected CRS: %s", crs)
    return G_proj, crs


# ══════════════════════════════════════════════
# STEP 2-C  Collapse to a simple undirected graph for centrality
# ══════════════════════════════════════════════
def to_simple_undirected(G) -> nx.Graph:
    """
    OSMnx graphs are MultiDiGraphs: directed (one edge per direction of
    travel) and allow parallel edges between the same node pair (e.g. a
    divided carriageway represented as two OSM ways).

    Betweenness centrality for this project is a measure of structural
    importance in the road network, not of one-way-restricted routing —
    almost all roads are traversable in both directions for the purposes
    of congestion propagation. We therefore collapse the graph to a simple
    undirected Graph, keeping the *shortest* parallel edge between any
    node pair (the path a vehicle would actually prefer).
    """
    log.info("Collapsing MultiDiGraph → simple undirected Graph for centrality computation …")
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G.nodes(data=True))

    n_parallel_collapsed = 0
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        if G_simple.has_edge(u, v):
            n_parallel_collapsed += 1
            if length < G_simple[u][v]["length"]:
                G_simple[u][v]["length"] = length
        else:
            G_simple.add_edge(u, v, length=length)

    log.info(
        "Simple graph: %d nodes, %d edges (%d parallel/duplicate edges collapsed)",
        G_simple.number_of_nodes(), G_simple.number_of_edges(), n_parallel_collapsed,
    )
    return G_simple


# ══════════════════════════════════════════════
# STEP 2-D  Betweenness centrality
# ══════════════════════════════════════════════
def compute_edge_betweenness(G_simple: nx.Graph) -> tuple:
    """
    Computes edge betweenness centrality, weighted by edge length (so a
    long bypass segment and a short junction link are compared on actual
    travel distance, not hop count).

    Uses exact computation for small graphs and the k-pivot approximation
    for large ones (see EXACT_THRESHOLD_NODES / K_PIVOTS in CONFIG).

    Returns:
        ebc_lookup : dict {frozenset({u, v}): centrality_score}
                     keyed by unordered node pair so it can be looked up
                     regardless of the original MultiDiGraph's edge direction.
        meta       : dict of method/runtime metadata for the audit report.
    """
    n_nodes = G_simple.number_of_nodes()
    use_exact = n_nodes <= EXACT_THRESHOLD_NODES
    k = None if use_exact else min(K_PIVOTS, n_nodes - 1)

    log.info(
        "Computing edge betweenness centrality on %d nodes (%s, %s) …",
        n_nodes,
        "exact" if use_exact else f"approximate, k={k} pivots",
        "weighted by length",
    )

    t0 = time.perf_counter()
    ebc_raw = nx.edge_betweenness_centrality(
        G_simple, k=k, weight="length", seed=RANDOM_SEED, normalized=True
    )
    elapsed = time.perf_counter() - t0
    log.info("Betweenness centrality computed in %.1f s", elapsed)

    ebc_lookup = {frozenset((u, v)): score for (u, v), score in ebc_raw.items()}

    meta = {
        "n_nodes": n_nodes,
        "n_edges": G_simple.number_of_edges(),
        "method": "exact" if use_exact else "approximate",
        "k_pivots": k if k is not None else n_nodes,
        "seed": RANDOM_SEED,
        "runtime_s": elapsed,
    }
    return ebc_lookup, meta


# ══════════════════════════════════════════════
# STEP 2-E  Snap incidents to nearest road edge
# ══════════════════════════════════════════════
def snap_incidents_to_edges(df: pd.DataFrame, G_proj, crs: str) -> pd.DataFrame:
    """
    Snaps every incident's (latitude, longitude) to the nearest edge in the
    projected road network using a single vectorised OSMnx call (avoids a
    slow per-row Python loop over thousands of incidents).

    Incidents with missing coordinates are skipped (left as NaN) rather
    than dropped, so row count is preserved for a clean join back onto the
    full dataset.
    """
    log.info("Snapping %d incidents to nearest road edge …", len(df))

    has_coords = df["latitude"].notna() & df["longitude"].notna()
    n_skipped = int((~has_coords).sum())
    if n_skipped:
        log.warning("%d incidents skipped (missing coordinates) — will have NaN centrality.", n_skipped)

    result = pd.DataFrame(
        index=df.index,
        columns=["nearest_u", "nearest_v", "nearest_key", "snap_distance_m"],
    )

    if has_coords.any():
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = transformer.transform(
            df.loc[has_coords, "longitude"].values,
            df.loc[has_coords, "latitude"].values,
        )

        edges, dists = ox.distance.nearest_edges(G_proj, X=xs, Y=ys, return_dist=True)

        result.loc[has_coords, "nearest_u"] = [e[0] for e in edges]
        result.loc[has_coords, "nearest_v"] = [e[1] for e in edges]
        result.loc[has_coords, "nearest_key"] = [e[2] for e in edges]
        result.loc[has_coords, "snap_distance_m"] = dists

    df = df.join(result)
    df["snap_distance_m"] = pd.to_numeric(df["snap_distance_m"], errors="coerce")

    n_low_conf = int((df["snap_distance_m"] > MAX_SNAP_DISTANCE_M).sum())
    df["low_confidence_snap"] = df["snap_distance_m"] > MAX_SNAP_DISTANCE_M
    if n_low_conf:
        log.warning(
            "%d incidents snapped > %dm from the road network — flagged as low_confidence_snap.",
            n_low_conf, MAX_SNAP_DISTANCE_M,
        )

    return df


# ══════════════════════════════════════════════
# STEP 2-F  Assign centrality score
# ══════════════════════════════════════════════
def assign_centrality(df: pd.DataFrame, ebc_lookup: dict) -> pd.DataFrame:
    """
    Looks up each incident's snapped edge in the betweenness centrality
    table and assigns the score as `centrality_score`.

    Lookup is by unordered node pair (frozenset) since edge betweenness was
    computed on the undirected collapsed graph, while incidents were
    snapped against the original directed MultiDiGraph edges.
    """
    log.info("Assigning centrality scores …")

    def _lookup(row):
        if pd.isna(row["nearest_u"]) or pd.isna(row["nearest_v"]):
            return np.nan
        key = frozenset((row["nearest_u"], row["nearest_v"]))
        return ebc_lookup.get(key, np.nan)

    df["centrality_score"] = df.apply(_lookup, axis=1)

    n_unscored = int(df["centrality_score"].isna().sum())
    if n_unscored:
        log.warning(
            "%d incidents have no centrality score (missing coords, or snapped "
            "edge absent from the betweenness lookup).", n_unscored,
        )

    return df


# ══════════════════════════════════════════════
# STEP 2-G  Save output
# ══════════════════════════════════════════════
def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved centrality-augmented dataset → %s  (%d rows × %d cols)", path, *df.shape)


# ══════════════════════════════════════════════
# STEP 2-H  Audit report
# ══════════════════════════════════════════════
def write_audit(df: pd.DataFrame, graph_meta: dict, centrality_meta: dict,
                 timings: dict, path: Path) -> None:
    """
    Writes reports/centrality_audit.md. All figures are computed from the
    actual run — no hard-coded numbers — so the report stays accurate if
    the pipeline is re-run after an OSM data refresh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cs = df["centrality_score"]
    sd = df["snap_distance_m"]
    n_total = len(df)
    n_scored = int(cs.notna().sum())
    n_low_conf = int(df["low_confidence_snap"].sum())

    total_runtime = sum(timings.values())
    timing_rows = "\n".join(
        f"| {stage} | {seconds:.1f} s |" for stage, seconds in timings.items()
    )

    report = textwrap.dedent(f"""\
    # SENTINELA · Road Network Centrality Audit
    **Generated:** {now}
    **Pipeline:** `src/centrality.py`

    ---

    ## 1. Road Network

    | Property | Value |
    |---|---|
    | Place queried | {graph_meta['place']} |
    | Network type | {graph_meta['network_type']} |
    | Nodes (raw graph) | {graph_meta['n_nodes_raw']:,} |
    | Edges (raw graph) | {graph_meta['n_edges_raw']:,} |
    | Nodes (simple undirected, used for centrality) | {centrality_meta['n_nodes']:,} |
    | Edges (simple undirected, used for centrality) | {centrality_meta['n_edges']:,} |
    | Projected CRS | {graph_meta['crs']} |
    | Source | {graph_meta['source']} |

    ---

    ## 2. Betweenness Centrality Computation

    | Property | Value |
    |---|---|
    | Method | {centrality_meta['method']} |
    | Pivot nodes (k) | {centrality_meta['k_pivots']:,} |
    | Random seed | {centrality_meta['seed']} |
    | Edge weight | length (metres) |
    | Normalized | Yes |
    | Computation time | {centrality_meta['runtime_s']:.1f} s |

    {"**Note:** approximate (k-pivot) betweenness was used because the graph "
     "exceeds the exact-computation threshold (" + f"{EXACT_THRESHOLD_NODES:,}" + " nodes). "
     "This is the standard production approximation (Brandes & Pich, 2007) and "
     "converges close to exact values for city-scale graphs." if centrality_meta['method'] == 'approximate'
     else "**Note:** graph was small enough for exact betweenness centrality — no "
          "approximation was needed."}

    ---

    ## 3. Incident Snapping

    | Metric | Value |
    |---|---|
    | Total incidents | {n_total:,} |
    | Successfully snapped & scored | {n_scored:,} ({n_scored/n_total*100:.1f} %) |
    | Min snap distance | {sd.min():.1f} m |
    | Median snap distance | {sd.median():.1f} m |
    | Mean snap distance | {sd.mean():.1f} m |
    | 95th percentile snap distance | {sd.quantile(0.95):.1f} m |
    | Max snap distance | {sd.max():.1f} m |
    | Low-confidence snaps (> {MAX_SNAP_DISTANCE_M} m) | {n_low_conf:,} ({n_low_conf/n_total*100:.1f} %) |

    Low-confidence snaps are **retained** (not dropped) with a `low_confidence_snap`
    flag column, since this is a feature-engineering step rather than a filtering
    step — the decision to exclude them is left to the modelling stage.

    ---

    ## 4. Centrality Score Distribution

    | Statistic | Value |
    |---|---|
    | Min    | {cs.min():.6f} |
    | 25th % | {cs.quantile(0.25):.6f} |
    | Median | {cs.median():.6f} |
    | Mean   | {cs.mean():.6f} |
    | 75th % | {cs.quantile(0.75):.6f} |
    | Max    | {cs.max():.6f} |
    | Std    | {cs.std():.6f} |
    | Missing | {int(cs.isna().sum()):,} |

    ---

    ## 5. Runtime Breakdown

    | Stage | Duration |
    |---|---|
    {timing_rows}
    | **Total** | **{total_runtime:.1f} s** |

    ---

    ## 6. Output Dataset

    - **Rows:** {n_total:,}
    - **Columns:** {len(df.columns)}
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
    """)

    path.write_text(report, encoding="utf-8")
    log.info("Audit report written → %s", path)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def run_pipeline() -> None:
    log.info("═" * 60)
    log.info("SENTINELA – Step 2: Road Network Centrality")
    log.info("═" * 60)

    timings = {}

    t0 = time.perf_counter()
    df = load_incidents(IN_PATH)
    timings["Load incidents"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = get_road_network(PLACE_NAME, NETWORK_TYPE, GRAPH_CACHE, FORCE_REFRESH)
    timings["Pull/load road network"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G_proj, crs = project_network(G)
    timings["Project network"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G_simple = to_simple_undirected(G_proj)
    timings["Collapse to simple undirected"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ebc_lookup, centrality_meta = compute_edge_betweenness(G_simple)
    timings["Compute betweenness centrality"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = snap_incidents_to_edges(df, G_proj, crs)
    timings["Snap incidents to edges"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = assign_centrality(df, ebc_lookup)
    timings["Assign centrality scores"] = time.perf_counter() - t0

    save_output(df, OUT_PATH)

    graph_meta = {
        "place": PLACE_NAME,
        "network_type": NETWORK_TYPE,
        "n_nodes_raw": G.number_of_nodes(),
        "n_edges_raw": G.number_of_edges(),
        "crs": str(crs),
        "source": "OpenStreetMap via OSMnx",
    }
    write_audit(df, graph_meta, centrality_meta, timings, AUDIT_PATH)

    log.info("═" * 60)
    log.info(
        "Step 2 complete. %d incidents scored, %d flagged low-confidence.",
        int(df["centrality_score"].notna().sum()),
        int(df["low_confidence_snap"].sum()),
    )
    log.info("═" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        log.exception("Step 2 pipeline failed.")
        sys.exit(1)