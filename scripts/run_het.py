"""Phase-B orchestrator: run all 5 heterogeneous-effects modules in sequence,
then assemble `results/heterogeneous_paper_summary.json`.

Order matters: BMA depends on the JSON outputs of causal_forest + double_ml,
so they must run first.

Usage:
    python scripts/run_het.py            # full run (~10–15 min)
    python scripts/run_het.py --fast     # fewer estimators / draws (~3–5 min)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _run_module(name: str, fn) -> tuple[dict | None, str | None]:
    log.info("=" * 72)
    log.info("MODULE: %s", name)
    log.info("=" * 72)
    t0 = time.time()
    try:
        res = fn()
        log.info("%s OK (%.1fs)", name, time.time() - t0)
        return res, None
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log.error("%s FAILED: %s\n%s", name, e, tb)
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                      help="Smaller n_estimators / SHAP samples / MCMC draws.")
    ap.add_argument("--skip", nargs="*", default=[],
                      help="Module names to skip (causal_forest, double_ml, "
                           "shap, counterfactual, bma).")
    args = ap.parse_args()

    fast = args.fast
    skip = set(args.skip)
    log.info("run_het.py — fast=%s skip=%s", fast, sorted(skip))

    from src.models import (bayesian_averaging, causal_forest,
                              counterfactual_generator, double_ml,
                              shap_decomposition)

    pipeline = [
        ("causal_forest", lambda: causal_forest.run(fast=fast)),
        ("double_ml",    lambda: double_ml.run(fast=fast)),
        ("shap",         lambda: shap_decomposition.run(fast=fast)),
        ("counterfactual", lambda: counterfactual_generator.run(fast=fast)),
        ("bma",          lambda: bayesian_averaging.run(fast=fast)),
    ]

    outputs: dict[str, dict | None] = {}
    errors: dict[str, str] = {}
    for name, fn in pipeline:
        if name in skip:
            log.info("MODULE %s skipped via --skip", name)
            continue
        res, err = _run_module(name, fn)
        outputs[name] = res
        if err:
            errors[name] = err

    # ---------------------------------------------------------------- #
    # Assemble paper-headline summary
    # ---------------------------------------------------------------- #
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fast": bool(fast),
        "errors": errors,
    }
    cf = outputs.get("causal_forest") or {}
    cf_math = (cf.get("outcomes") or {}).get("caaspp_math_met_pct") or {}
    summary["causal_forest"] = {
        "math_mean_cate": cf_math.get("mean_cate"),
        "math_blp_f": cf_math.get("blp_f_stat"),
        "math_blp_r2": cf_math.get("blp_r2"),
    }
    dml = outputs.get("double_ml") or {}
    summary["double_ml"] = {
        "dml_coef": dml.get("dml_coef"),
        "rd_anchor": dml.get("rd_anchor"),
        "agreement_z": dml.get("agreement_z_score"),
        "within_2se": dml.get("agreement_within_2se"),
    }
    shap_r = outputs.get("shap") or {}
    by_arch = shap_r.get("by_archetype") or {}
    top_per = {}
    for k, rows in by_arch.items():
        if rows:
            top_per[k] = rows[0]["feature"]
    summary["shap"] = top_per
    cfac = outputs.get("counterfactual") or {}
    if cfac.get("districts"):
        action_norms = [d["action_norm"] for d in cfac["districts"]]
        gains = [d["embedding_shift_norm"] for d in cfac["districts"]]
        summary["counterfactuals"] = {
            "n_near_boundary_analyzed": cfac.get("n_optimised"),
            "mean_action_norm": float(sum(action_norms) / len(action_norms))
                                  if action_norms else None,
            "mean_predicted_gain": float(sum(gains) / len(gains)) if gains else None,
        }
    else:
        summary["counterfactuals"] = {
            "n_near_boundary_analyzed": 0,
            "mean_action_norm": None,
            "mean_predicted_gain": None,
        }
    bma = outputs.get("bma") or {}
    summary["bma"] = {
        "posterior_mean": bma.get("posterior_mean"),
        "posterior_hdi": bma.get("posterior_hdi"),
        "anchor_should_update": bma.get("anchor_should_update"),
    }

    out_path = REPO / "results" / "heterogeneous_paper_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("wrote %s", out_path)
    log.info("modules ok: %d ; failed: %d",
              sum(1 for v in outputs.values() if v is not None), len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
