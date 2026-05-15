"""Module 13: heterogeneous LCFF concentration-grant effects via causal forest.

Spec (Phase B):
  - T (binary): pct_frl / enrollment > 0.55 — i.e. district-years above the
    LCFF concentration-grant threshold.
  - Y: each of the 5 CAASPP outcomes, run separately.
  - X: top-30 numeric covariates by variance, ex-outcomes/IDs.
  - Estimator: econml.grf.CausalForest (n_estimators=200).
  - Heterogeneity test: best-linear-predictor — regress predicted CATEs on X
    via OLS, report F-stat and R² for the joint significance of the slopes.
  - Archetype cross-tab: KMeans (k=5) on the 202×128 GNN district embeddings
    (last-snapshot slice), cross-tabbed by CATE quartile.

Outputs `results/causal_forest/<outcome>.json` per outcome.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils._io import (
    OUTCOMES,
    align_panel_to_gnn,
    archetype_labels,
    last_snapshot_embeddings,
    load_district_embeddings,
    load_master_panel,
    make_lcff_treatment,
    save_json,
    top_n_covariates,
)

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "causal_forest"


# --------------------------------------------------------------------------- #
# Best-linear-predictor heterogeneity test
# --------------------------------------------------------------------------- #

def best_linear_predictor(cate: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Regress predicted CATE on X via OLS.

    Returns (F-statistic for joint significance of slopes, R²).
    """
    # Drop NaN rows
    mask = np.isfinite(cate) & np.all(np.isfinite(X), axis=1)
    y = cate[mask]
    Xm = X[mask]
    if len(y) < 10 or Xm.shape[1] == 0:
        return float("nan"), float("nan")
    # Standardise X to keep matrix conditioning sane on a 30-col regression
    Xs = (Xm - Xm.mean(axis=0)) / (Xm.std(axis=0) + 1e-9)
    Xd = np.column_stack([np.ones(len(y)), Xs])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, k = Xd.shape  # k includes intercept
    p = k - 1  # number of slopes
    df_res = max(1, n - k)
    if p < 1 or ss_res <= 0 or ss_tot <= 0:
        return float("nan"), r2
    f_stat = ((ss_tot - ss_res) / p) / (ss_res / df_res)
    return float(f_stat), float(r2)


# --------------------------------------------------------------------------- #
# Single-outcome runner
# --------------------------------------------------------------------------- #

def _run_outcome(panel: pd.DataFrame, outcome: str,
                 X_cols: list[str],
                 archetype_by_cds: dict[str, int],
                 fast: bool = False) -> dict:
    sub = panel.copy()
    T = make_lcff_treatment(sub)
    sub["__T__"] = T
    sub = sub.dropna(subset=["__T__", outcome, *X_cols])
    if len(sub) < 50:
        log.warning("causal_forest %s: only %d obs — skipping", outcome, len(sub))
        return {"outcome": outcome, "skipped": True, "n": int(len(sub))}

    Y = sub[outcome].to_numpy(dtype=float)
    Tv = sub["__T__"].to_numpy(dtype=float).astype(int)
    X = sub[X_cols].to_numpy(dtype=float)

    from econml.grf import CausalForest
    n_est = 80 if fast else 200
    cf = CausalForest(n_estimators=n_est, min_samples_leaf=5,
                       max_depth=None, random_state=0, n_jobs=-1)
    t0 = time.time()
    cf.fit(X, Tv, Y)
    cate = cf.predict(X).reshape(-1)
    dur = time.time() - t0
    log.info("causal_forest %s: fit on n=%d in %.1fs (n_est=%d)",
              outcome, len(Y), dur, n_est)

    # Per-district mean CATE (over years), used for the archetype × quartile cross-tab.
    sub = sub.assign(__cate__=cate)
    district_cate = (sub.groupby("cds")["__cate__"].mean()
                       .reset_index().rename(columns={"__cate__": "cate"}))

    quartiles = pd.qcut(district_cate["cate"], q=4,
                         labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    district_cate["quartile"] = quartiles.astype(str)
    district_cate["archetype"] = district_cate["cds"].map(archetype_by_cds)

    # Archetype × quartile cross-tab — keys are str so save_json behaves.
    cross: dict[str, dict[str, int]] = {}
    for q in ["q1", "q2", "q3", "q4"]:
        cross[q] = {}
        rows_q = district_cate[district_cate["quartile"] == q]
        for arch, group in rows_q.groupby("archetype"):
            if pd.isna(arch):
                continue
            cross[q][f"a{int(arch)}"] = int(len(group))

    blp_f, blp_r2 = best_linear_predictor(cate, X)

    return {
        "outcome": outcome,
        "n_observations": int(len(Y)),
        "n_districts": int(district_cate["cds"].nunique()),
        "mean_cate": float(np.nanmean(cate)),
        "cate_std": float(np.nanstd(cate)),
        "cate_quartiles": {
            "q1": float(np.quantile(cate, 0.25)),
            "q2": float(np.quantile(cate, 0.50)),
            "q3": float(np.quantile(cate, 0.75)),
            "q4": float(np.max(cate)),
        },
        "blp_f_stat": blp_f,
        "blp_r2": blp_r2,
        "archetype_by_quartile": cross,
        "n_estimators": int(n_est),
        "fit_seconds": round(dur, 1),
        "x_cols": X_cols,
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run(fast: bool = False) -> dict:
    """Run causal forest on all 5 outcomes; write per-outcome JSON files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_master_panel()
    # Use only districts with GNN embeddings → keeps the archetype cross-tab valid.
    emb, cds_idx, _ = load_district_embeddings()
    emb_last = last_snapshot_embeddings(emb)
    labels, _ = archetype_labels(emb_last, k=5)
    cds_list = sorted(cds_idx, key=lambda c: cds_idx[c])
    archetype_by_cds = {cds: int(labels[cds_idx[cds]]) for cds in cds_list}
    panel = align_panel_to_gnn(panel, cds_idx)

    X_cols = top_n_covariates(panel, n=30)
    log.info("causal_forest: n_districts=%d top-30 X cols selected (e.g. %s)",
              panel["cds"].nunique(), X_cols[:3])

    outcomes = OUTCOMES if not fast else OUTCOMES[:2]
    out: dict = {"outcomes": {}, "x_cols": X_cols,
                 "n_districts_with_gnn": int(panel["cds"].nunique())}
    for o in outcomes:
        res = _run_outcome(panel, o, X_cols, archetype_by_cds, fast=fast)
        save_json(RESULTS_DIR / f"{o}.json", res)
        out["outcomes"][o] = res
    save_json(RESULTS_DIR / "summary.json", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print(run(fast=True))
