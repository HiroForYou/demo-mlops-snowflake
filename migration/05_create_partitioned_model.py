# %% [markdown]
# # Modelo particionado (16 modelos de 04)
# Carga los modelos por grupo del Registry, crea CustomModel con API particionada y lo registra.

# %%
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.model import custom_model, task
import pandas as pd
import numpy as np
from datetime import datetime

session = get_active_session()
session.sql("USE DATABASE BD_AA_DEV").collect()
session.sql("USE SCHEMA SC_STORAGE_BMX_PS").collect()
registry = Registry(session=session, database_name="BD_AA_DEV", schema_name="SC_MODELS_BMX")
print(f"✅ {session.get_current_database()}.{session.get_current_schema()}")

# %% [markdown]
# ## 1. Cargar los 16 modelos (PRODUCTION)

# %%

groups_df = session.sql(
    """
    SELECT DISTINCT stats_ntile_group AS GROUP_NAME
    FROM BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED
    WHERE stats_ntile_group IS NOT NULL
    ORDER BY stats_ntile_group
"""
)
groups_list = [row["GROUP_NAME"] for row in groups_df.collect()]
print(f"\n📊 Grupos: {len(groups_list)}")

loaded_models = {}
feature_cols = None

for group_name in groups_list:
    model_name = f"uni_box_regression_{group_name.lower()}"
    try:
        model_ref = registry.get_model(model_name)
        model_version = model_ref.version("PRODUCTION")
        native_model = model_version.load()
        if feature_cols is None:
            feature_cols = getattr(native_model, "feature_cols", None)
            if not feature_cols:
                raise ValueError("Model has no feature_cols; run 04 first.")
        loaded_models[group_name] = {"model": native_model, "model_version": model_version, "model_name": model_name}
        ver_name = getattr(model_version, "name", str(model_version))
        print(f"✅ {group_name} (versión: {ver_name})")
    except Exception as e:
        print(f"❌ {group_name}: {str(e)[:100]}")

if not loaded_models:
    raise ValueError("No models loaded. Run 04_many_model_training.py first.")
if len(loaded_models) < len(groups_list):
    print(f"⚠️  {len(loaded_models)}/{len(groups_list)} modelos cargados")
versions_loaded = {getattr(m["model_version"], "name", None) for m in loaded_models.values()}
print(f"\n✅ {len(loaded_models)} modelos, {len(feature_cols)} features (alias PRODUCTION → {versions_loaded or 'N/A'})")

# %% [markdown]
# ## 2. Define Partitioned Model Class

# %%
print("\n" + "=" * 80)
print("🔧 DEFINING PARTITIONED MODEL CLASS")
print("=" * 80)


class PartitionedUniBoxModel(custom_model.CustomModel):
    """Modelo particionado: enruta predicciones por stats_ntile_group."""

    def __init__(self, model_context):
        super().__init__(model_context)
        # Usamos las mismas columnas de features detectadas al cargar los modelos base
        self.feature_cols = feature_cols

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Predicción por grupo; enruta según stats_ntile_group."""
        if len(input_df) == 0:
            return pd.DataFrame(columns=["customer_id", "stats_ntile_group", "predicted_uni_box_week"])

        part_col = "stats_ntile_group" if "stats_ntile_group" in input_df.columns else "STATS_NTILE_GROUP"
        group_name = input_df[part_col].iloc[0]

        model_key = group_name.lower()
        try:
            model = self.context.model_ref(model_key)
        except Exception:
            try:
                model = self.context.model_ref(group_name.lower())
            except Exception:
                raise ValueError(f"Model not found for group: {group_name}. Keys: {list(self.context.models.keys())}")

        # Seleccionamos únicamente las columnas de features ya conocidas
        X = input_df[self.feature_cols].fillna(0).astype(np.float64, errors="ignore")
        pred_out = model.predict(X)

        if isinstance(pred_out, pd.DataFrame):
            pred_col = next((c for c in pred_out.columns if "PREDICT" in c.upper() or "OUTPUT" in c.upper()), pred_out.columns[0])
            predictions = np.asarray(pred_out[pred_col], dtype=np.float64).ravel()
        elif hasattr(pred_out, "flatten"):
            predictions = np.asarray(pred_out).flatten()
        else:
            predictions = np.asarray(pred_out).ravel()

        cust = input_df["customer_id"].values if "customer_id" in input_df.columns else np.arange(len(predictions))
        return pd.DataFrame({
            "customer_id": cust,
            "stats_ntile_group": group_name,
            "predicted_uni_box_week": predictions,
        })


print("✅ PartitionedUniBoxModel class defined")

# %% [markdown]
# ## 3. Create Model Context and Partitioned Model

# %%
print("\n" + "=" * 80)
print("📦 CREATING PARTITIONED MODEL")
print("=" * 80)

models_dict = {}
for group_name, model_info in loaded_models.items():
    model_key = group_name.lower()
    models_dict[model_key] = model_info["model"]
    print(f"   Added {group_name} → {model_key}")
print(f"\n✅ ModelContext created with {len(models_dict)} models")

model_context = custom_model.ModelContext(models=models_dict)
partitioned_model = PartitionedUniBoxModel(model_context=model_context)
print("✅ Partitioned model created")

# %% [markdown]
# ## 4. Sample input (opcional, para inspección)

# %%
training_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
sample_input_sp = (
    training_df.select("customer_id", "stats_ntile_group", *feature_cols)
    .filter(training_df["stats_ntile_group"].isin(list(loaded_models.keys())))
    .group_by("stats_ntile_group")
    .agg(*[F.min(F.col(c)).alias(c) for c in ["customer_id"] + feature_cols])
    .select("customer_id", "stats_ntile_group", *feature_cols)
    .limit(min(16, len(loaded_models)))
)
sample_input = sample_input_sp.to_pandas()
print(f"✅ Sample input prepared: {len(sample_input)} rows (one per group)")

# %% [markdown]
# ## 5. Register Partitioned Model

# %%
print("\n" + "=" * 80)
print("📝 REGISTERING PARTITIONED MODEL")
print("=" * 80)

version_date = datetime.now().strftime("%Y%m%d_%H%M")

print(f"\n📝 Registering in Model Registry...")
print(f"   Name: UNI_BOX_REGRESSION_PARTITIONED")
print(f"   Version: v_{version_date}")

try:
    mv = registry.log_model(
        partitioned_model,
        model_name="UNI_BOX_REGRESSION_PARTITIONED",
        version_name=f"v_{version_date}",
        comment=f"Partitioned regression model for uni_box_week - Combines {len(loaded_models)} group-specific models (LGBM/XGB/SGD)",
        metrics={
            "num_groups": len(loaded_models),
            "num_features": len(feature_cols),
            "model_type": "mixed",
            "groups": ",".join(sorted(loaded_models.keys())),
        },
        sample_input_data=sample_input,
        task=task.Task.TABULAR_REGRESSION,
        options={"function_type": "TABLE_FUNCTION"},
    )

    print("\n✅ Partitioned model registered successfully!")
    mv.set_alias("PRODUCTION")
    print(f"🏷️  Alias 'PRODUCTION' configured")

except Exception as e:
    print(f"\n❌ Error registering model: {str(e)}")
    raise

# %% [markdown]
# ## 6. Verify Registration

# %%
print("\n" + "=" * 80)
print("🔍 VERIFYING REGISTRATION")
print("=" * 80)

result = session.sql(
    """
    SHOW MODELS LIKE 'UNI_BOX_REGRESSION_PARTITIONED' 
    IN SCHEMA BD_AA_DEV.SC_MODELS_BMX
"""
).collect()

if result:
    print("✅ Partitioned model found in registry")

    versions = session.sql(
        """
        SHOW VERSIONS IN MODEL BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED
    """
    ).collect()

    print(f"\n📊 Versions: {len(versions)}")
    for v in versions[-3:]:
        print(f"   - {v['name']}")
else:
    print("❌ Model not found in registry")

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "=" * 80)
print("✅ PARTITIONED MODEL CREATION COMPLETE!")
print("=" * 80)

print("\n📋 Summary:")
print(f"   ✅ Source models: {len(loaded_models)} group-specific models")
print(f"   ✅ Partitioned model: UNI_BOX_REGRESSION_PARTITIONED")
print(f"   ✅ Version: v_{version_date}")
print(f"   ✅ Alias: PRODUCTION")
print(f"   ✅ Features: {len(feature_cols)}")
print(
    f"   ✅ Groups: {', '.join(sorted(loaded_models.keys())[:5])}... ({len(loaded_models)} total)"
)

print("\n💡 Next Steps:")
print("   1. Review partitioned model registration")
print("   2. Run 06_partitioned_inference_batch.py for batch inference")
print(
    "   3. Use partitioned inference syntax: TABLE(model!PREDICT(...) OVER (PARTITION BY stats_ntile_group))"
)

print("\n🎯 Key Benefits:")
print("   - 16 models combined into one partitioned model")
print("   - Automatic routing by stats_ntile_group")
print("   - Consistent inference syntax")
print("   - Each group uses its optimized hyperparameters")

print("\n" + "=" * 80)
