"""Module 22: SHAP feature attributions per archetype.

Spec (Phase B):
  - Layer-4 GNN architecture lives in `edu-gnn/src/models/htgnn.py`, but its
    forward pass requires a list of `HeteroData` snapshots which is not
    re-derivable in this repo. As anticipated in the spec's fallback clause,
    we use the *saved* district embeddings (202 × 10 × 128) directly and
    attribute district-panel features to outcomes-via-embeddings.
  - For each of 5 KMeans archetypes (k=5 on last-year embeddings):
      * fit a LightGBM regressor f(X) → outcome on the districts in that
        archetype
      * run SHAP KernelExplainer with a 50-row background sample
      * rank features by mean(|SHAP|)
  - Save top-10 per archetype + heatmap (rows=archetypes, cols=top-20 features).

Output: `results/shap/by_archetype.json` + `figures/shap/archetype_feature_importance.pdf`.
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils._io import (
    align_panel_to_gnn,
    archetype_labels,
    last_snapshot_embeddings,
    load_district_embeddings,
    load_master_panel,
    save_json,
    top_n_covariates,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "shap"
FIGURES_DIR = REPO_ROOT / "figures" / "shap"

OUTCOME = "caaspp_math_met_pct"  # SHAP attribution target outcome
SHAP_TARGET_N_FEATURES_FAST = 20  # spec: if KernelExplainer is too slow, sample to 20


def _fit_lgbm(X: np.ndarray, y: np.ndarray, fast: bool):
    """Light, deterministic regressor for SHAP attribution."""
    from lightgbm import LGBMRegressor
    n_est = 60 if fast else 150
    model = LGBMRegressor(n_estimators=n_est, max_depth=-1, learning_rate=0.05,
                          random_state=0, n_jobs=1, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _kernel_shap(model, X_explain: np.ndarray, X_background: np.ndarray,
                 fast: bool) -> np.ndarray:
    """Run SHAP KernelExplainer; returns SHAP values of shape X_explain.

    Wraps `model.predict` in a plain `lambda` because shap.convert_to_model
    tries to set `feature_names_in_` on the bound method's `__self__`, which
    raises AttributeError on LightGBM ≥ 4.0 (property without setter).
    """
    import shap
    # shap.KernelExplainer logs per-row phi arrays at INFO via the root logger;
    # silence its loggers so a fast SHAP run doesn't spam thousands of lines.
    for ln in ("shap", "shap.explainers", "shap.explainers._kernel",
                "shap.utils._legacy", "shap.utils.transformers"):
        logging.getLogger(ln).setLevel(logging.WARNING)
    nb = min(50, X_background.shape[0])
    bg = shap.sample(X_background, nb, random_state=0)
    # plain callable — defeats shap's feature_names_in_ injection
    predict_fn = lambda x: model.predict(x)  # noqa: E731
    explainer = shap.KernelExplainer(predict_fn, bg, link="identity")
    nsamples = 64 if fast else 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    return np.asarray(sv)


def _archetype_panel(panel: pd.DataFrame,
                      cds_idx: dict[str, int],
                      labels: np.ndarray) -> pd.DataFrame:
    """Add an `archetype` column to the panel (one row per district-year)."""
    archetype_by_cds = {cds: int(labels[cds_idx[cds]]) for cds in cds_idx}
    p = panel.copy()
    p["archetype"] = p["cds"].astype(str).map(archetype_by_cds)
    return p


def _heatmap(top_per_arch: dict[int, list[dict]], out_path: Path) -> None:
    """Render archetype × top-20 feature heatmap (mean|SHAP|)."""
    import matplotlib.pyplot as plt

    # Build union of features across archetypes, capped at 20 highest by max-row-mean.
    all_feats: dict[str, float] = {}
    for arch, rows in top_per_arch.items():
        for r in rows:
            all_feats[r["feature"]] = max(all_feats.get(r["feature"], 0.0),
                                            r["mean_abs_shap"])
    top_feats = sorted(all_feats, key=lambda f: -all_feats[f])[:20]
    if not top_feats:
        return
    # Build matrix
    arch_keys = sorted(top_per_arch)
    M = np.zeros((len(arch_keys), len(top_feats)))
    for i, a in enumerate(arch_keys):
        for j, f in enumerate(top_feats):
            for r in top_per_arch[a]:
                if r["feature"] == f:
                    M[i, j] = r["mean_abs_shap"]
                    break

    fig, ax = plt.subplots(figsize=(11, 4 + 0.4 * len(arch_keys)))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(arch_keys)), [f"archetype {a}" for a in arch_keys])
    ax.set_xticks(range(len(top_feats)), top_feats, rotation=70, ha="right",
                   fontsize=7)
    ax.set_title(f"Mean |SHAP| over top-20 features  (target: {OUTCOME})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean(|SHAP|)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(fast: bool = False) -> dict:
    """Run SHAP attribution per archetype. Writes per-archetype top-10 + heatmap."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_master_panel()
    emb, cds_idx, _ = load_district_embeddings()
    emb_last = last_snapshot_embeddings(emb)
    labels, _ = archetype_labels(emb_last, k=5)

    panel = align_panel_to_gnn(panel, cds_idx)
    panel = _archetype_panel(panel, cds_idx, labels)

    # Feature pool: top-30 covariates by variance, plus the documented fallback
    # downsample to 20 features when fast/slow.
    n_feat = SHAP_TARGET_N_FEATURES_FAST if fast else 30
    X_cols = top_n_covariates(panel, n=n_feat)
    sub = panel.dropna(subset=[OUTCOME, *X_cols])

    by_arch: dict[int, list[dict]] = {}
    archetype_summary: dict[str, dict] = {}
    for a in sorted(sub["archetype"].dropna().unique().astype(int)):
        arch_rows = sub[sub["archetype"] == a]
        if len(arch_rows) < 30:
            log.warning("shap: archetype %d only has %d obs — skipping",
                         a, len(arch_rows))
            continue
        Xa = arch_rows[X_cols].to_numpy(dtype=float)
        ya = arch_rows[OUTCOME].to_numpy(dtype=float)
        model = _fit_lgbm(Xa, ya, fast=fast)

        # Explain on a random subsample for speed; KernelExplainer is O(n_samples^3 * f).
        n_explain = min(40 if fast else 100, len(Xa))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Xa), size=n_explain, replace=False)
        t0 = time.time()
        sv = _kernel_shap(model, Xa[idx], Xa, fast=fast)
        dur = time.time() - t0

        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        top_rows = [
            {"feature": X_cols[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order[:10]
        ]
        by_arch[a] = top_rows
        archetype_summary[f"archetype_{a}"] = {
            "n_districts": int(arch_rows["cds"].nunique()),
            "n_observations": int(len(arch_rows)),
            "shap_seconds": round(dur, 1),
            "top10": top_rows,
        }
        log.info("shap archetype %d: n=%d top=%s (%.1fs)",
                  a, len(arch_rows), top_rows[0]["feature"], dur)

    payload = {
        "target_outcome": OUTCOME,
        "explainer": "shap.KernelExplainer",
        "features_used": X_cols,
        "n_features": int(len(X_cols)),
        "n_archetypes_explained": len(by_arch),
        "fallback_note": ("Used saved district embeddings + LightGBM regressor "
                           "(not raw GNN forward) because reloading the Layer-4 "
                           "HTGNN requires HeteroData snapshots not exported."),
        "by_archetype": {f"archetype_{a}": rows for a, rows in by_arch.items()},
        "summary": archetype_summary,
    }
    save_json(RESULTS_DIR / "by_archetype.json", payload)

    try:
        _heatmap(by_arch, FIGURES_DIR / "archetype_feature_importance.pdf")
    except Exception as e:  # noqa: BLE001
        log.warning("shap heatmap failed: %s", e)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print(run(fast=True))
