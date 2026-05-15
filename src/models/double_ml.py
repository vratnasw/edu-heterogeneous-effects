"""Module 14: per-pupil spending effect via econml LinearDML.

Spec (Phase B):
  - T: log(per-pupil expenditure proxy) = log(bea_gdp_total / enrollment).
    The master panel ships no real per-pupil column; this is a documented
    economic-activity proxy and is flagged in the output JSON.
  - Y: caaspp_math_met_pct.
  - W: same top-30 covariates used by causal forest.
  - Estimator: econml.dml.LinearDML with LGBMRegressor nuisance learners.
  - Cross-check: compare DML coef to the Layer-2 RD anchor for math
    (`edu-causal-rl/results/interventional/summary.json` → outcomes[math].anchor).
    Report a z-score on the difference and a within-2σ boolean.

Output: `results/double_ml/results.json`.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils._io import (
    load_master_panel,
    make_per_pupil_proxy,
    save_json,
    top_n_covariates,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "double_ml"
CAUSAL_RL_SUMMARY = (REPO_ROOT.parent / "edu-causal-rl" / "results"
                      / "interventional" / "summary.json")

PRIMARY_OUTCOME = "caaspp_math_met_pct"


def _load_rd_anchor(outcome: str = PRIMARY_OUTCOME) -> tuple[float, float]:
    """Pull Layer-2 anchor + SE for the requested outcome.

    The summary.json doesn't include a `ate_se` for the anchor specifically;
    we use `ate_se` from the same outcome as the conservative SE since the
    anchor is derived from the same RD estimation pipeline.
    """
    if not CAUSAL_RL_SUMMARY.exists():
        log.warning("layer-2 summary not found at %s — anchor unavailable",
                     CAUSAL_RL_SUMMARY)
        return float("nan"), float("nan")
    data = json.loads(CAUSAL_RL_SUMMARY.read_text(encoding="utf-8"))
    rec = data.get("outcomes", {}).get(outcome, {})
    return float(rec.get("anchor", "nan")), float(rec.get("ate_se", "nan"))


def run(fast: bool = False) -> dict:
    """Fit LinearDML; compare to RD anchor; write results JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_master_panel()

    W_cols = top_n_covariates(panel, n=30)
    T = make_per_pupil_proxy(panel)
    panel = panel.assign(__T__=T)

    needed = ["__T__", PRIMARY_OUTCOME, *W_cols]
    sub = panel.dropna(subset=needed)
    if len(sub) < 50:
        raise RuntimeError(f"double_ml: too few obs after dropna ({len(sub)})")

    Y = sub[PRIMARY_OUTCOME].to_numpy(dtype=float)
    Tv = sub["__T__"].to_numpy(dtype=float)
    W = sub[W_cols].to_numpy(dtype=float)

    from econml.dml import LinearDML
    from lightgbm import LGBMRegressor

    cv = 3 if fast else 5
    n_est = 80 if fast else 200
    base = dict(n_estimators=n_est, max_depth=-1, learning_rate=0.05,
                  random_state=0, n_jobs=1, verbose=-1)
    dml = LinearDML(
        model_y=LGBMRegressor(**base),
        model_t=LGBMRegressor(**base),
        discrete_treatment=False,
        cv=cv,
        random_state=0,
    )

    t0 = time.time()
    # LinearDML treats W (controls) as nuisance; pass X=None to get a scalar
    # constant marginal effect for the continuous treatment.
    dml.fit(Y=Y, T=Tv, X=None, W=W)
    # `const_marginal_ate()` returns the scalar ATE; `_inference()` exposes the SE.
    cm_ate = dml.const_marginal_ate()
    dml_coef = float(np.atleast_1d(cm_ate).flatten()[0])
    try:
        inf = dml.const_marginal_ate_inference()
        dml_se = float(np.atleast_1d(inf.stderr).flatten()[0])
    except Exception as e:  # noqa: BLE001
        log.warning("double_ml: const_marginal_ate_inference unavailable (%s); "
                     "falling back to ate_interval / 2*1.96 width", e)
        try:
            lo, hi = dml.const_marginal_ate_interval(alpha=0.05)
            dml_se = float((np.atleast_1d(hi).flatten()[0] -
                              np.atleast_1d(lo).flatten()[0]) / (2 * 1.96))
        except Exception:
            dml_se = float("nan")
    log.info("double_ml: dml_coef=%.4f se=%.4f n=%d  (%.1fs)",
              dml_coef, dml_se, len(Y), time.time() - t0)

    rd_anchor, rd_se = _load_rd_anchor(PRIMARY_OUTCOME)

    # z-score on the difference. SE of diff = sqrt(se_dml^2 + se_rd^2).
    if np.isfinite(dml_se) and np.isfinite(rd_se) and (dml_se + rd_se) > 0:
        se_diff = float(np.sqrt(dml_se ** 2 + rd_se ** 2))
        z = (dml_coef - rd_anchor) / se_diff if se_diff > 0 else float("nan")
        within = bool(abs(z) <= 2.0)
    else:
        z = float("nan")
        within = False

    payload = {
        "outcome": PRIMARY_OUTCOME,
        "treatment_definition": "log(bea_gdp_total / enrollment) — proxy "
                                  "for log per-pupil expenditure (no "
                                  "per_pupil column in master panel)",
        "n_observations": int(len(Y)),
        "n_controls": int(W.shape[1]),
        "cv_folds": int(cv),
        "lgbm_n_estimators": int(n_est),
        "dml_coef": dml_coef,
        "dml_se": dml_se,
        "rd_anchor": rd_anchor,
        "rd_se": rd_se,
        "agreement_z_score": z,
        "agreement_within_2se": within,
        "w_cols": W_cols,
    }
    save_json(RESULTS_DIR / "results.json", payload)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print(run(fast=True))
