"""Preflight check: R2 reachability + required source parquets exist.

Exits 0 on PASS (GO), 1 on any FAIL (NO-GO).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REQUIRED_R2_KEYS = ['processed/joined/master_panel.parquet']
REQUIRED_PKGS = ('econml', 'shap', 'torch', 'torch-geometric', 'pymc', 'pandas', 'pyarrow', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'boto3', 'pyyaml')


def _load_dotenv() -> None:
    """Look for .env at repo root, then teamspace root, then edu-data-pipeline."""
    candidates = [
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ]
    for c in candidates:
        if c.exists():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            log.info("loaded .env from %s", c)
            return
    log.info("no .env found (will rely on already-exported env vars)")


def _result(name: str, ok: bool, msg: str = "") -> dict:
    sym = "PASS" if ok else "FAIL"
    log.info("  %-32s %s  %s", name, sym, msg)
    return {"name": name, "pass": ok, "msg": msg}


def check_env_vars() -> dict:
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT_URL")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        return _result("env_vars", False, f"missing: {missing}")
    return _result("env_vars", True, "all 5 R2 vars set")


def check_packages() -> dict:
    """Warning only -- a missing package does not fail preflight (Phase B will
    require them, but the R2 essentials we check here only need boto3 + yaml)."""
    pkg_import_map = {
        "pyyaml": "yaml",
        "scikit-learn": "sklearn",
        "torch-geometric": "torch_geometric",
    }
    missing = []
    for pkg in REQUIRED_PKGS:
        mod = pkg_import_map.get(pkg, pkg.replace("-", "_"))
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.warning("  python_packages (advisory)        WARN  not yet installed: %s", missing)
    return _result("python_packages_essentials", True,
                   f"{len(REQUIRED_PKGS) - len(missing)}/{len(REQUIRED_PKGS)} importable; Phase B will pin remaining")


def check_r2_bucket() -> dict:
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client
        info = r2_client.smoke_check()
        if not info.get("ok"):
            return _result("r2_bucket", False, str(info))
        return _result("r2_bucket", True,
                       f"bucket={info.get('bucket')}")
    except Exception as e:  # noqa: BLE001
        return _result("r2_bucket", False, f"exception: {e}")


def check_master_panel() -> dict:
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client
        info = r2_client.exists("processed/joined/master_panel.parquet")
        if not info:
            return _result("master_panel", False,
                           "key not readable: processed/joined/master_panel.parquet")
        size_mb = info.get("size", 0) / 1e6
        return _result("master_panel", True,
                       f"exists ({size_mb:.1f} MB)")
    except Exception as e:  # noqa: BLE001
        return _result("master_panel", False, f"exception: {e}")


def check_required_keys() -> dict:
    try:
        sys.path.insert(0, str(REPO / "config"))
        import r2_client
        missing = []
        for key in REQUIRED_R2_KEYS:
            if r2_client.exists(key) is None:
                missing.append(key)
        if missing:
            return _result("required_keys", False, f"missing: {missing}")
        return _result("required_keys", True,
                       f"all {len(REQUIRED_R2_KEYS)} keys present")
    except Exception as e:  # noqa: BLE001
        return _result("required_keys", False, f"exception: {e}")


def main() -> int:
    log.info("=== preflight: %s ===", REPO.name)
    _load_dotenv()
    results = [
        check_env_vars(),
        check_packages(),
        check_r2_bucket(),
        check_master_panel(),
        check_required_keys(),
    ]
    n_fail = sum(1 for r in results if not r["pass"])
    log.info("=== %d/%d checks passed ===",
             len(results) - n_fail, len(results))
    if n_fail:
        log.error("preflight FAILED -- NO-GO")
        return 1
    log.info("preflight PASS -- GO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
