"""Phase-B smoke tests.

  - Module imports + `run` callable surface (carried over from Phase A).
  - Output-shape / type checks per spec:
      * CATE quartiles dict has 4 keys
      * SHAP top features are strings
      * BMA HDI is a 2-tuple
"""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_modules_import():
    from src.models import bayesian_averaging, causal_forest
    from src.models import counterfactual_generator, double_ml
    from src.models import shap_decomposition
    for mod in (causal_forest, double_ml, shap_decomposition,
                  counterfactual_generator, bayesian_averaging):
        assert hasattr(mod, "run"), f"{mod.__name__}.run missing"


def test_causal_forest_output_shape():
    p = REPO / "results" / "causal_forest" / "caaspp_math_met_pct.json"
    if not p.exists():
        return  # smoke: skip if not yet run
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "n_districts" in data
    assert "cate_quartiles" in data
    assert len(data["cate_quartiles"]) == 4
    assert "blp_f_stat" in data and "blp_r2" in data


def test_shap_top_features_are_strings():
    p = REPO / "results" / "shap" / "by_archetype.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    by_arch = data.get("by_archetype", {})
    assert by_arch, "expected at least one archetype's top features"
    for _, rows in by_arch.items():
        for r in rows:
            assert isinstance(r["feature"], str)


def test_bma_hdi_is_two_tuple():
    p = REPO / "results" / "bma" / "results.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    hdi = data.get("posterior_hdi")
    assert hdi is not None and len(hdi) == 2
    lo, hi = hdi
    assert lo <= hi


def test_counterfactual_uncertainty_shape():
    p = REPO / "results" / "counterfactuals" / "near_boundary_interventions.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("districts"):
        for d in data["districts"]:
            assert "recommended_action" in d
            assert isinstance(d["recommended_action"], list)
            assert "uncertainty" in d
