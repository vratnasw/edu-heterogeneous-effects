"""Orchestrator stub. Phase B will wire all modules + write
results/het_paper_summary.json."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.parse_args()
    print("SCAFFOLD ONLY -- Phase B pending. Run `python scripts/preflight.py` to verify R2 access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
