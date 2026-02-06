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
            return pd.DataFrame(
                columns=[
                    "customer_id",
                    "stats_ntile_group",
                    "week",
                    "brand_pres_ret",
                    "predicted_uni_box_week",
                ]
            )

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

        # Propagar columnas de contexto para evitar JOINs ambiguos en inferencia
        cust = (
            input_df["customer_id"].values
            if "customer_id" in input_df.columns
            else np.arange(len(predictions))
        )
        week_vals = (
            input_df["week"].values if "week" in input_df.columns else [None] * len(predictions)
        )
        brand_vals = (
            input_df["brand_pres_ret"].values
            if "brand_pres_ret" in input_df.columns
            else [None] * len(predictions)
        )

        return pd.DataFrame(
            {
                "customer_id": cust,
                "stats_ntile_group": group_name,
                "week": week_vals,
                "brand_pres_ret": brand_vals,
                "predicted_uni_box_week": predictions,
            }
        )


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
# ## 4. Sample input (homologado con 06: incluye week y brand_pres_ret como contexto)

# %%
print("\n" + "=" * 80)
print("📋 PREPARING SAMPLE INPUT (HOMOLOGADO CON 06)")
print("=" * 80)

# Usar la misma lista de exclusión que 02/04/06 para asegurar que sample_input tenga la misma firma que inferencia
excluded_cols_sample = [
    "customer_id",
    "brand_pres_ret",
    "week",
    "group",
    "stats_group",
    "percentile_group",
    "stats_ntile_group",
    "uni_box_week",  # Target - no es feature
    "FEATURE_TIMESTAMP",
]
excluded_upper_sample = {c.upper() for c in excluded_cols_sample}

# Obtener esquema de training para identificar features numéricas (igual que 06 con inference)
training_schema = session.sql(
    "DESCRIBE TABLE BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED"
).collect()
col_type_dict_sample = {row["name"].upper(): str(row["type"]).upper() for row in training_schema}
all_cols_sample = [row["name"] for row in training_schema]

NUMERIC_PREFIXES_SAMPLE = ("FLOAT", "NUMBER", "INTEGER", "BIGINT", "DOUBLE")
feature_cols_sample = [
    c for c in all_cols_sample
    if c.upper() not in excluded_upper_sample
    and (col_type_dict_sample.get(c.upper()) or "").startswith(NUMERIC_PREFIXES_SAMPLE)
]

# Verificar que feature_cols_sample coincida con feature_cols de los modelos (deberían ser iguales)
if set(c.upper() for c in feature_cols_sample) != set(c.upper() for c in feature_cols):
    print(f"⚠️  Feature mismatch: sample tiene {len(feature_cols_sample)}, modelos tienen {len(feature_cols)}")
    print(f"   Usando feature_cols de modelos para mantener compatibilidad")
    feature_cols_for_sample = feature_cols
else:
    # Ordenar feature_cols_sample según el orden de feature_cols para mantener consistencia
    feature_cols_for_sample = [c for c in feature_cols if c.upper() in {x.upper() for x in feature_cols_sample}]
    if len(feature_cols_for_sample) != len(feature_cols):
        print(f"⚠️  Algunas features de modelos no están en sample, usando todas las de modelos")
        feature_cols_for_sample = feature_cols

training_df = session.table("BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_CLEANED")
sample_input_sp = (
    training_df.select("customer_id", "stats_ntile_group", "week", "brand_pres_ret", *feature_cols_for_sample)
    .filter(training_df["stats_ntile_group"].isin(list(loaded_models.keys())))
    .group_by("stats_ntile_group")
    .agg(
        # Asegurar NO-NULL en columnas de contexto para inferir firma (Snowflake ML requiere al menos un no-null)
        F.min(F.col("customer_id")).alias("customer_id"),
        F.coalesce(F.min(F.col("week")), F.lit("000000")).alias("week"),
        F.coalesce(F.min(F.col("brand_pres_ret")), F.lit("UNKNOWN")).alias("brand_pres_ret"),
        *[F.min(F.col(c)).alias(c) for c in feature_cols_for_sample],
    )
    .select("customer_id", "stats_ntile_group", "week", "brand_pres_ret", *feature_cols_for_sample)
    .limit(min(16, len(loaded_models)))
)
sample_input = sample_input_sp.to_pandas()
print(f"✅ Sample input prepared: {len(sample_input)} rows (one per group)")
print(f"   Columns: customer_id, stats_ntile_group, week, brand_pres_ret, {len(feature_cols_for_sample)} features")
if len(sample_input) == 0:
    raise ValueError("Sample input is empty. Verifica que TRAIN_DATASET_CLEANED tenga los mismos stats_ntile_group que los modelos cargados.")

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
    # Mover alias PRODUCTION de forma "limpia":
    # 1) quitar alias al version previo (si existe)
    # 2) asignar alias al nuevo version
    model_fqn = "BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED"
    new_version_name = f"V_{version_date}"
    try:
        session.sql(f"ALTER MODEL {model_fqn} VERSION PRODUCTION UNSET ALIAS").collect()
        print("🧹 Alias 'PRODUCTION' removido del version anterior")
    except Exception as alias_unset_err:
        print(f"ℹ️  No se pudo remover alias previo (puede no existir): {str(alias_unset_err)[:120]}")

    session.sql(
        f"ALTER MODEL {model_fqn} VERSION {new_version_name} SET ALIAS='PRODUCTION'"
    ).collect()
    print("🏷️  Alias 'PRODUCTION' movido al nuevo version")

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
