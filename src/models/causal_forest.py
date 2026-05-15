"""econml GRFForestRegressor for heterogeneous LCFF CATEs.

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - processed/joined/master_panel.parquet
Outputs: results/causal_forest/cates.json
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
