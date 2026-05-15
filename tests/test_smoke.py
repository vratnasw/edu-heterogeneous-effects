"""Smoke test: every model stub must import + expose run().

Phase A only verifies the scaffold parses; Phase B will add real tests.
"""
from __future__ import annotations


def test_modules_import():
    from src.models import causal_forest
    from src.models import double_ml
    from src.models import shap_decomposition
    from src.models import counterfactual_generator
    from src.models import bayesian_averaging
    assert hasattr(causal_forest, 'run'), 'causal_forest.run missing'
    assert hasattr(double_ml, 'run'), 'double_ml.run missing'
    assert hasattr(shap_decomposition, 'run'), 'shap_decomposition.run missing'
    assert hasattr(counterfactual_generator, 'run'), 'counterfactual_generator.run missing'
    assert hasattr(bayesian_averaging, 'run'), 'bayesian_averaging.run missing'
