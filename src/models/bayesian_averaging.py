"""Module 35: PyMC hierarchical BMA over RD / DiD / DML / CF LCFF estimates.

Spec (Phase B):
  - Estimates to combine:
      RD  : Layer-2 anchor from `edu-causal-rl/.../interventional/summary.json`
      DiD : if present in that file
      DML : this repo's `results/double_ml/results.json`
      CF  : this repo's `results/causal_forest/caaspp_math_met_pct.json`
        (mean_cate)
  - Model: true_effect ~ Normal(0, 5); each estimate ~ Normal(true_effect, se).
  - 1000 draws / 500 tune / 2 chains (fast: 500 / 200 / 2).
  - Posterior mean + 95% HDI for `true_effect`. If posterior mean differs from
    the current RD anchor by >10%, write a revised
    `results/bma/updated_anchor_effects.json`.

Output: `results/bma/results.json`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.utils._io import save_json

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "bma"
CAUSAL_RL_SUMMARY = (REPO_ROOT.parent / "edu-causal-rl" / "results"
                      / "interventional" / "summary.json")
DOUBLE_ML_RESULT = REPO_ROOT / "results" / "double_ml" / "results.json"
CAUSAL_FOREST_MATH = (REPO_ROOT / "results" / "causal_forest"
                       / "caaspp_math_met_pct.json")

OUTCOME = "caaspp_math_met_pct"


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        log.warning("BMA: missing %s — that estimate will be skipped", p)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("BMA: could not read %s: %s", p, e)
        return None


def _gather_estimates() -> tuple[list[dict], dict]:
    """Collect (mean, se, label) tuples from upstream results."""
    estimates: list[dict] = []
    diagnostics: dict = {"sources": {}}

    rd = _read_json(CAUSAL_RL_SUMMARY)
    rd_anchor = float("nan")
    if rd:
        rec = (rd.get("outcomes", {}) or {}).get(OUTCOME, {}) or {}
        anchor = rec.get("anchor")
        se = rec.get("ate_se")
        if anchor is not None and se is not None and np.isfinite(anchor) and np.isfinite(se) and se > 0:
            estimates.append({"label": "RD", "mean": float(anchor), "se": float(se)})
            rd_anchor = float(anchor)
        diagnostics["sources"]["RD"] = {"path": str(CAUSAL_RL_SUMMARY), "loaded": True,
                                          "anchor": anchor, "ate_se": se}
        # The summary.json doesn't currently store a DiD anchor distinct from the RD anchor;
        # the spec permits skipping it when absent.
        if "did_anchor" in rec and "did_se" in rec:
            estimates.append({"label": "DiD", "mean": float(rec["did_anchor"]),
                                "se": float(rec["did_se"])})
            diagnostics["sources"]["DiD"] = {"loaded": True}
        else:
            diagnostics["sources"]["DiD"] = {"loaded": False,
                                                "note": "no DiD anchor in Layer-2 summary"}

    dml = _read_json(DOUBLE_ML_RESULT)
    if dml and np.isfinite(dml.get("dml_coef", float("nan"))) and \
        np.isfinite(dml.get("dml_se", float("nan"))) and dml["dml_se"] > 0:
        estimates.append({"label": "DML", "mean": float(dml["dml_coef"]),
                            "se": float(dml["dml_se"])})
        diagnostics["sources"]["DML"] = {"path": str(DOUBLE_ML_RESULT), "loaded": True,
                                          "coef": dml.get("dml_coef"), "se": dml.get("dml_se")}
    else:
        diagnostics["sources"]["DML"] = {"path": str(DOUBLE_ML_RESULT), "loaded": False}

    cf = _read_json(CAUSAL_FOREST_MATH)
    if cf and np.isfinite(cf.get("mean_cate", float("nan"))):
        # CausalForest doesn't expose an SE for the mean directly; use cate_std / sqrt(n)
        n = float(cf.get("n_districts", 1))
        cf_std = float(cf.get("cate_std", float("nan")))
        cf_se = cf_std / max(np.sqrt(n), 1.0) if np.isfinite(cf_std) and cf_std > 0 else 1.0
        estimates.append({"label": "CausalForest", "mean": float(cf["mean_cate"]),
                            "se": cf_se})
        diagnostics["sources"]["CausalForest"] = {
            "path": str(CAUSAL_FOREST_MATH), "loaded": True,
            "mean_cate": cf.get("mean_cate"), "se_derived": cf_se,
        }
    else:
        diagnostics["sources"]["CausalForest"] = {
            "path": str(CAUSAL_FOREST_MATH), "loaded": False}

    diagnostics["rd_anchor_current"] = rd_anchor
    return estimates, diagnostics


def _hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Highest-density interval (simple sorted-slice implementation)."""
    s = np.sort(np.asarray(samples))
    n = len(s)
    if n == 0:
        return float("nan"), float("nan")
    k = int(np.floor(prob * n))
    if k <= 0:
        return float(s[0]), float(s[-1])
    widths = s[k:] - s[: n - k]
    if widths.size == 0:
        return float(s[0]), float(s[-1])
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k])


def run(fast: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    estimates, diag = _gather_estimates()
    log.info("BMA: %d estimates available: %s", len(estimates),
              [e["label"] for e in estimates])
    if not estimates:
        payload = {"error": "no upstream estimates available",
                    "diagnostics": diag}
        save_json(RESULTS_DIR / "results.json", payload)
        return payload

    means = np.array([e["mean"] for e in estimates], dtype=float)
    ses = np.array([e["se"] for e in estimates], dtype=float)

    import pymc as pm
    draws = 500 if fast else 1000
    tune = 200 if fast else 500
    with pm.Model():
        true_effect = pm.Normal("true_effect", mu=0.0, sigma=5.0)
        pm.Normal("obs", mu=true_effect, sigma=ses, observed=means)
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                            progressbar=False, random_seed=0, target_accept=0.9)

    post = idata.posterior["true_effect"].values.reshape(-1)
    posterior_mean = float(np.mean(post))
    hdi_lo, hdi_hi = _hdi(post, prob=0.95)

    current_anchor = diag.get("rd_anchor_current")
    if current_anchor is not None and np.isfinite(current_anchor) and current_anchor != 0:
        rel_diff = abs(posterior_mean - current_anchor) / max(abs(current_anchor), 1e-9)
        should_update = bool(rel_diff > 0.10)
    else:
        rel_diff = float("nan")
        should_update = False

    payload = {
        "outcome": OUTCOME,
        "posterior_mean": posterior_mean,
        "posterior_hdi": [hdi_lo, hdi_hi],
        "posterior_n_draws": int(post.size),
        "individual_estimates": {e["label"]: {"mean": e["mean"], "se": e["se"]}
                                    for e in estimates},
        "current_anchor": current_anchor if np.isfinite(current_anchor) else None,
        "rel_diff_vs_anchor": rel_diff if np.isfinite(rel_diff) else None,
        "anchor_should_update": should_update,
        "sampling": {"draws": draws, "tune": tune, "chains": 2},
        "diagnostics": diag,
    }
    save_json(RESULTS_DIR / "results.json", payload)

    if should_update:
        save_json(RESULTS_DIR / "updated_anchor_effects.json", {
            "outcome": OUTCOME,
            "bma_anchor": posterior_mean,
            "bma_hdi_95": [hdi_lo, hdi_hi],
            "replaces": current_anchor,
            "reason": ("BMA posterior mean differs from current Layer-2 anchor "
                        "by >10% (relative)."),
        })
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print(json.dumps(run(fast=True), indent=2))
