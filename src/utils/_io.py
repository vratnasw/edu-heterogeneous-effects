"""Shared I/O + data-prep helpers for Phase B modules.

Centralises:
  - .env loading (R2 credentials live in `edu-data-pipeline/.env`)
  - master-panel download with simple disk cache
  - GNN district-embedding + metadata loading
  - top-N feature selection by variance
  - canonical district / year alignment
  - KMeans archetype assignment over the 128-dim embeddings
  - JSON-safe dumping (numpy / torch scalars and arrays).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# .env + R2
# --------------------------------------------------------------------------- #

def load_dotenv() -> None:
    """Look for .env in the usual places, populate os.environ."""
    candidates = [
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ]
    for c in candidates:
        if c.exists():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            return


def _r2_client():
    sys.path.insert(0, str(REPO / "config"))
    import r2_client  # noqa: WPS433 — local import to avoid circular load
    return r2_client


# --------------------------------------------------------------------------- #
# Master panel
# --------------------------------------------------------------------------- #

OUTCOMES: tuple[str, ...] = (
    "caaspp_math_met_pct",
    "caaspp_ela_met_pct",
    "chronic_absenteeism_rate",
    "suspension_rate_pct",
    "graduation_rate_pct",
)

# Non-numeric / identifier columns that must NOT enter X.
EXCLUDED_COVARIATES: set[str] = {
    "cds", "year_num", "county_code", "county_name",
    *OUTCOMES,
}


def _cache_path() -> Path:
    cache = REPO / "results" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "master_panel.parquet"


def load_master_panel(use_cache: bool = True) -> pd.DataFrame:
    """Download (or read from cache) the master panel parquet.

    The panel is 10,100 rows × 537 cols (1,010 districts × 10 years).
    """
    cp = _cache_path()
    if use_cache and cp.exists():
        log.info("master_panel: cache hit (%s)", cp)
        return pd.read_parquet(cp)
    load_dotenv()
    r2_client = _r2_client()
    df = r2_client.download("processed/joined/master_panel.parquet")
    df.to_parquet(cp, index=False)
    log.info("master_panel: %s -> cached at %s", df.shape, cp)
    return df


def make_lcff_treatment(df: pd.DataFrame) -> pd.Series:
    """Construct binary LCFF concentration-grant treatment.

    `pct_frl` is *raw* (not a fraction) in the master panel, so we divide by
    enrollment to recover the 0-1 ratio, then threshold at 0.55. Districts
    with missing enrollment are dropped from T (returned as NaN).
    """
    ratio = df["pct_frl"] / df["enrollment"].replace(0, np.nan)
    ratio = ratio.clip(lower=0, upper=2)
    treat = (ratio > 0.55).astype("float")
    treat[ratio.isna() | df["pct_frl"].isna() | df["enrollment"].isna()] = np.nan
    return treat


def make_per_pupil_proxy(df: pd.DataFrame) -> pd.Series:
    """log(per-pupil expenditure proxy) using bea_gdp_total / enrollment.

    Documented in spec — no real per-pupil column ships in the master panel.
    Returns log-transformed series with NaN where either input is missing or
    enrollment is 0.
    """
    pp = df["bea_gdp_total"] / df["enrollment"].replace(0, np.nan)
    pp = pp.where(pp > 0, np.nan)
    return np.log(pp)


def top_n_covariates(df: pd.DataFrame, n: int = 30,
                     exclude: Sequence[str] | None = None) -> list[str]:
    """Return the top-N numeric columns by variance, ex-outcomes/IDs.

    Selecting on variance keeps memory bounded for econml on a 537-col panel.
    Constant or all-NaN columns are dropped.
    """
    excluded = set(EXCLUDED_COVARIATES) | set(exclude or ())
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric if c not in excluded]
    var = df[candidates].var(numeric_only=True).fillna(0.0)
    var = var[var > 0].sort_values(ascending=False)
    return var.head(n).index.tolist()


# --------------------------------------------------------------------------- #
# GNN embeddings
# --------------------------------------------------------------------------- #

GNN_EMBED_PATH = (REPO.parent / "edu-gnn" / "results" / "embeddings"
                  / "district_embeddings.pt")
GNN_META_PATH = (REPO.parent / "edu-gnn" / "results" / "embeddings"
                 / "embedding_metadata.json")


def load_district_embeddings() -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Load district embeddings as (tensor, cds->idx, year->idx).

    Tensor shape: (N_districts=202, T=10, hidden_dim=128).
    """
    import torch
    if not GNN_EMBED_PATH.exists():
        raise FileNotFoundError(f"missing Layer-4 embeddings: {GNN_EMBED_PATH}")
    if not GNN_META_PATH.exists():
        raise FileNotFoundError(f"missing Layer-4 metadata: {GNN_META_PATH}")
    emb = torch.load(GNN_EMBED_PATH, map_location="cpu", weights_only=False)
    if hasattr(emb, "numpy"):
        emb = emb.detach().cpu().numpy()
    meta = json.loads(GNN_META_PATH.read_text(encoding="utf-8"))
    cds_idx: dict[str, int] = dict(meta["district_cds_to_index"])
    yr_idx: dict[str, int] = {str(k): int(v) for k, v in meta["year_to_index"].items()}
    return np.asarray(emb), cds_idx, yr_idx


def last_snapshot_embeddings(emb: np.ndarray) -> np.ndarray:
    """Take the last-year slice. Returns (N_districts, hidden_dim)."""
    if emb.ndim != 3:
        raise ValueError(f"expected 3-d embeddings, got shape {emb.shape}")
    return emb[:, -1, :]


def archetype_labels(emb_last: np.ndarray, k: int = 5,
                     random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """KMeans with k=5 over the (N, 128) last-year embeddings.

    Returns (labels, centroids).
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(emb_last)
    return labels, km.cluster_centers_


def near_boundary_districts(emb_last: np.ndarray, labels: np.ndarray,
                            centroids: np.ndarray,
                            bottom_frac: float = 0.25) -> np.ndarray:
    """Layer-7 fallback: districts in the bottom 25% by distance-to-centroid.

    The lowest-distance districts are NOT the ambiguous ones. We want the
    bottom 25% of (max - dist) → equivalently, the *largest* distances to
    their assigned centroid, because districts far from any centroid are the
    most "ambiguous" in the embedding space.
    """
    n = emb_last.shape[0]
    dists = np.linalg.norm(
        emb_last - centroids[labels], axis=1,
    )
    # "ambiguous" = furthest from own cluster centroid → top distance quartile
    cutoff = np.quantile(dists, 1.0 - bottom_frac)
    mask = dists >= cutoff
    return np.where(mask)[0]


# --------------------------------------------------------------------------- #
# Panel ↔ embedding alignment
# --------------------------------------------------------------------------- #

def align_panel_to_gnn(df: pd.DataFrame, cds_idx: dict[str, int],
                       year: int | None = None) -> pd.DataFrame:
    """Filter the panel to the 202 districts that have GNN embeddings.

    If `year` is given, pull only that year (matched on `year_num`).
    Otherwise returns all years for those districts.
    """
    sub = df[df["cds"].astype(str).isin(cds_idx)].copy()
    if year is not None:
        sub = sub[sub["year_num"] == year]
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def _to_native(o):  # noqa: PLR0911
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_to_native(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_to_native(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_native(v) for k, v in o.items()}
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    return o


def save_json(path: Path | str, payload: dict | list) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_to_native(payload), fh, indent=2)
    return p


__all__ = [
    "OUTCOMES",
    "EXCLUDED_COVARIATES",
    "load_master_panel",
    "make_lcff_treatment",
    "make_per_pupil_proxy",
    "top_n_covariates",
    "load_district_embeddings",
    "last_snapshot_embeddings",
    "archetype_labels",
    "near_boundary_districts",
    "align_panel_to_gnn",
    "save_json",
]
