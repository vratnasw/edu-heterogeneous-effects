"""SHAP DeepExplainer on Layer-4 GNN embeddings.

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - processed/joined/master_panel.parquet
  - ../edu-gnn/results/embeddings/layer4_embeddings.parquet
Outputs: results/shap/feature_attributions.json
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
