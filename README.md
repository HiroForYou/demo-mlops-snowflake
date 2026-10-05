# MLOps migration: Databricks → Snowflake

**English** · [Español](README.es.md)

Migration of a complete MLOps workflow from Databricks to Snowflake: data validation, feature store, hyperparameter search, many-model training, partitioned batch inference and model observability (drift and performance monitoring).

## Project structure

| Folder | Content |
|---|---|
| `migration/` | **Main project**: the full MLOps workflow migrated to Snowflake |
| `databricks/` | Original Databricks code (training, inference, monitoring) |
| `demo-original/` | Original demos in notebook format |
| `demo-fine/` | Refined versions of the demos |
| `arca_deployment_demo/` | Standalone demo: data setup, feature store, segmentation, many-model training and deployment options |
| `docs-js/` | Generators for the project documents (BBP and KT) |

## Main project: `migration/`

Sequential scripts, run in numerical order. Each `.py` has a matching notebook in `migration/notebooks/`.

### Training

| Script | Step |
|---|---|
| `01_data_validation_and_cleaning.py` | Validate and clean the training and inference datasets |
| `02_feature_store_setup.py` | Build and materialize the feature dataset |
| `03_hyperparameter_search.py` | Per-group hyperparameter search (LGBM / XGB) with `RandomSearch` |
| `03b_hyperparameter_search_bayesian.py` | Same search using Bayesian optimization (`BayesOpt`) |
| `04_many_model_training.py` | Train one model per group (16 models) and register them in the Model Registry |
| `05_create_partitioned_model.py` | Wrap the 16 models into a single partitioned model |

### Baselines and promotion to production

| Script | Step |
|---|---|
| `06a_setup_baselines.py` | Create support tables and run inference on training data |
| `06b_data_drift_baseline.py` | Baseline histograms of input features |
| `06c_prediction_drift_baseline.py` | Baseline histograms of predictions per segment |
| `06d_performance_drift_baseline.py` | Baseline performance metrics (WAPE, RMSE, MAE, F1) |
| `07a_copy_baselines.py` | Copy baselines from the development schema to production |
| `07b_copy_models.py` | Copy the `PRODUCTION` model version to the production registry and tag it |

### Inference and observability

| Script | Step |
|---|---|
| `08_partitioned_inference_batch.py` | Partitioned batch inference on production data for missing (version, week) pairs |
| `09a_setup_observability.py` | Landing tables for drift and performance metrics |
| `09b_data_drift.py` | Input data drift against the baseline |
| `09c_prediction_drift.py` | Prediction drift (Jensen-Shannon) against the baseline |
| `09d_performance_drift.py` | Performance degradation against the baseline |
| `10_alertas.py` | Report of records with warning or critical alerts |

`cleanup_all_resources*.sql` removes the Snowflake objects created by the workflow.

## Documentation

Located in `migration/docs/` (in Spanish): business blueprint (`documento-bbp.md`), knowledge transfer (`documento-kt.md`), naming convention and table dictionary.

## Utilities

| File | Use |
|---|---|
| `environment.yml` | Conda environment |
| `convert_to_notebooks.py` | Convert `.py` scripts (`# %%` cells) to `.ipynb` |
| `convert_from_notebooks.py` | Convert `.ipynb` notebooks back to `.py` |
