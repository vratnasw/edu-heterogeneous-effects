# edu-heterogeneous-effects

**Status: Phase A scaffold.** Module stubs in place; implementation (Phase B) pending.

Heterogeneous treatment effect + interpretability models — causal forest, double ML, SHAP, counterfactuals, BMA. Models 13/14/22/23/35.

## Research questions

- Which districts have the largest heterogeneous CATE for LCFF dollars?
- What is the partial effect of per-pupil spending after double-ML orthogonalization?
- Which Layer-4 GNN-embedded features are most predictive (SHAP) of outcomes?
- What minimum-norm counterfactual moves a district across a phase boundary?
- What is the Bayesian-model-averaged LCFF effect across RD / DiD / DML / CF?

## Models / modules

- `src/models/causal_forest.py` -- econml GRFForestRegressor for heterogeneous LCFF CATEs. (-> `results/causal_forest/cates.json`)
- `src/models/double_ml.py` -- econml LinearDML for per-pupil spending effect. (-> `results/double_ml/dml_estimates.json`)
- `src/models/shap_decomposition.py` -- SHAP DeepExplainer on Layer-4 GNN embeddings. (-> `results/shap/feature_attributions.json`)
- `src/models/counterfactual_generator.py` -- Gradient search for minimum-norm phase-boundary-crossing interventions. (-> `results/counterfactual/min_norm_interventions.json`)
- `src/models/bayesian_averaging.py` -- PyMC BMA over RD / DiD / DML / CF LCFF estimates. (-> `results/bma/posterior_averages.json`)

## Data sources

Master panel (canonical join, 10,100 rows x 537 cols) at R2 key `processed/joined/master_panel.parquet`. Repo-specific parquets are listed in `config/config.yaml` under `data:`.

## Target journal

Econometrica or QJE

## Quick start

```
pip install -r requirements.txt
cp .env.example .env   # fill in 5 R2 vars
python scripts/preflight.py        # GO/NO-GO on R2 + parquets
python scripts/run_het.py --fast  # scaffold stub
```

## Layout

```
config/        # config.yaml + r2_client.py (canonical copy)
src/models/    # one stub .py per model in this paper
src/causal/    # (Phase B)
src/utils/     # config_loader.py
scripts/       # preflight.py + run_het.py orchestrator
tests/         # smoke tests for stub parsing
notebooks/     # exploration.ipynb (empty)
figures/       # paper figures (Phase B)
results/       # JSON outputs (Phase B)
```

