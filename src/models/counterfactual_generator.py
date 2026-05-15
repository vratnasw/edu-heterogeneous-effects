"""Module 23: minimum-norm phase-boundary interventions via gradient search.

Spec (Phase B):
  - Load Layer-5 world-model ensemble (state_dim=128, action_dim=5, 5 members)
    from `edu-world-model/results/checkpoints/best.pt`.
  - Identify near-boundary districts: districts furthest from their assigned
    KMeans (k=5) centroid in the 128-dim embedding space — the "ambiguous"
    quartile by the Layer-7 fallback defined in spec.
  - For each near-boundary district:
      * initial state s_0 = its 128-dim embedding (last snapshot)
      * Adam optimise action a ∈ R^5 over 200 steps (lr=0.01) to MAXIMIZE
        post-rollout distance from its current cluster centroid
      * 3-step lookahead via deterministic world-model rollout
      * L2 penalty on |a| ⇒ minimum-norm action.
  - Record predicted final embedding shift + 5-member ensemble uncertainty.

Output: `results/counterfactuals/near_boundary_interventions.json` — list of
top-20 near-boundary districts with their recommended action vectors.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils._io import (
    archetype_labels,
    last_snapshot_embeddings,
    load_district_embeddings,
    near_boundary_districts,
    save_json,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "counterfactuals"
WORLD_MODEL_REPO = REPO_ROOT.parent / "edu-world-model"
WORLD_MODEL_CKPT = WORLD_MODEL_REPO / "results" / "checkpoints" / "best.pt"

L2_PENALTY = 0.01
LEARNING_RATE = 0.01
N_STEPS = 200
HORIZON = 3


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _ensure_paths() -> None:
    """Put `edu-world-model/src` + its `src/models` on sys.path because the
    world-model package uses `from models.ensemble_member import …` internally.
    """
    wm_src = WORLD_MODEL_REPO / "src"
    wm_models = wm_src / "models"
    for p in (str(wm_src), str(wm_models)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_world_model() -> Any:
    """Instantiate WorldModelEnsemble + load state_dict from best.pt."""
    import torch
    _ensure_paths()
    from models.ensemble_member import MemberConfig  # noqa: WPS433
    from models.world_model_ensemble import WorldModelEnsemble  # noqa: WPS433

    if not WORLD_MODEL_CKPT.exists():
        raise FileNotFoundError(f"world model checkpoint missing: {WORLD_MODEL_CKPT}")
    ckpt = torch.load(WORLD_MODEL_CKPT, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", {})
    cfg = MemberConfig(
        state_dim=int(arch.get("state_dim", 128)),
        action_dim=int(arch.get("action_dim", 5)),
        hidden_dim=int(arch.get("hidden_dim", 256)),
        num_layers=int(arch.get("num_layers", 3)),
    )
    model = WorldModelEnsemble(member_cfg=cfg,
                                ensemble_size=int(arch.get("ensemble_size", 5)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- #
# Differentiable 3-step rollout
# --------------------------------------------------------------------------- #

def _rollout_grad(model, state: "torch.Tensor", action: "torch.Tensor",
                  horizon: int = HORIZON) -> "torch.Tensor":
    """Differentiable open-loop rollout. Uses ensemble *mean* of member means
    at each step so gradients flow. Action is held constant across the horizon.
    """
    s = state
    for _ in range(horizon):
        means, _ = model(s, action)  # (E, B, D), (E, B, D)
        s = means.mean(dim=0)
    return s


def _ensemble_uncertainty(model, state: "torch.Tensor", action: "torch.Tensor",
                          horizon: int = HORIZON) -> "torch.Tensor":
    """Run the same rollout but return per-member trajectories at the final
    step. Returns std across members in embedding space — a (B, D) tensor.
    """
    import torch
    # Carry per-member chains. Start with E identical copies of state.
    E = len(model.members)
    s_e = state.unsqueeze(0).expand(E, *state.shape).contiguous()  # (E, B, D)
    a_e = action.unsqueeze(0).expand(E, *action.shape).contiguous()
    for _ in range(horizon):
        new_s = []
        for i, member in enumerate(model.members):
            mu, _ = member(s_e[i], a_e[i])
            new_s.append(mu)
        s_e = torch.stack(new_s, dim=0)
    return s_e.std(dim=0, unbiased=False)  # (B, D)


# --------------------------------------------------------------------------- #
# Optimise the per-district action
# --------------------------------------------------------------------------- #

def _optimise_action(model, s0: "torch.Tensor", centroid: "torch.Tensor",
                     action_dim: int, n_steps: int = N_STEPS,
                     lr: float = LEARNING_RATE,
                     l2_penalty: float = L2_PENALTY) -> dict:
    """Optimise one action vector for one district.

    Loss = -‖rollout(s0, a) - centroid‖₂  +  l2_penalty * ‖a‖²
    """
    import torch
    a = torch.zeros(1, action_dim, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    s0_b = s0.unsqueeze(0)               # (1, D)
    c_b = centroid.unsqueeze(0)          # (1, D)

    for _step in range(n_steps):
        opt.zero_grad()
        s_final = _rollout_grad(model, s0_b, a, horizon=HORIZON)
        dist = (s_final - c_b).norm(dim=-1)        # (1,)
        loss = -dist + l2_penalty * (a ** 2).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_final = _rollout_grad(model, s0_b, a, horizon=HORIZON)
        embedding_shift = (s_final - s0_b).squeeze(0).cpu().numpy()
        unc = _ensemble_uncertainty(model, s0_b, a, horizon=HORIZON)
        unc_vec = unc.squeeze(0).cpu().numpy()

    a_np = a.detach().squeeze(0).cpu().numpy()
    return {
        "action": a_np,
        "action_norm": float(np.linalg.norm(a_np)),
        "predicted_outcome_change": embedding_shift,  # (D,) shift in embedding
        "embedding_shift_norm": float(np.linalg.norm(embedding_shift)),
        "uncertainty": unc_vec,
        "uncertainty_mean": float(unc_vec.mean()),
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run(fast: bool = False) -> dict:
    """Per spec: optimise minimum-norm interventions for top-20 near-boundary
    districts; write results JSON."""
    import torch
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    emb, cds_idx, _ = load_district_embeddings()
    emb_last = last_snapshot_embeddings(emb)            # (202, 128)
    labels, centroids = archetype_labels(emb_last, k=5)
    near_idx = near_boundary_districts(emb_last, labels, centroids,
                                        bottom_frac=0.25)
    n_to_run = min(20, len(near_idx)) if not fast else min(5, len(near_idx))
    near_idx = near_idx[:n_to_run]

    model, cfg = _load_world_model()
    log.info("counterfactuals: world model state_dim=%d action_dim=%d ensemble=%d",
              cfg.state_dim, cfg.action_dim, len(model.members))

    n_steps = 60 if fast else N_STEPS
    log.info("counterfactuals: optimising %d districts × %d steps each",
              n_to_run, n_steps)

    idx_to_cds = {v: k for k, v in cds_idx.items()}
    out: list[dict] = []
    t0 = time.time()
    for j, i in enumerate(near_idx):
        s0 = torch.from_numpy(emb_last[i]).float()
        c = torch.from_numpy(centroids[labels[i]]).float()
        res = _optimise_action(model, s0, c, action_dim=cfg.action_dim,
                                n_steps=n_steps)
        out.append({
            "cds": idx_to_cds[int(i)],
            "current_archetype": int(labels[i]),
            "recommended_action": res["action"].tolist(),
            "action_norm": res["action_norm"],
            "predicted_outcome_change": res["predicted_outcome_change"].tolist(),
            "embedding_shift_norm": res["embedding_shift_norm"],
            "uncertainty": res["uncertainty"].tolist(),
            "uncertainty_mean": res["uncertainty_mean"],
        })
        if (j + 1) % 5 == 0:
            log.info("  %d/%d done (elapsed %.1fs)", j + 1, n_to_run,
                      time.time() - t0)
    dur = time.time() - t0

    payload = {
        "n_near_boundary_total": int(len(near_idx)),
        "n_optimised": len(out),
        "world_model": {
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "ensemble_size": len(model.members),
        },
        "optimisation": {
            "horizon": HORIZON,
            "lr": LEARNING_RATE,
            "n_steps": n_steps,
            "l2_penalty": L2_PENALTY,
            "objective": "maximise post-rollout L2 from current centroid; "
                          "minimise ‖a‖² as a soft constraint",
        },
        "elapsed_seconds": round(dur, 1),
        "districts": out,
    }
    save_json(RESULTS_DIR / "near_boundary_interventions.json", payload)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    print(json.dumps(run(fast=True), indent=2)[:1000])
