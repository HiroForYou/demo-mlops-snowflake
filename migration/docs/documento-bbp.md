# BUSINESS BLUEPRINT

## Migración del Modelo de Pronóstico de Ventas

## de Databricks a Snowflake y Evaluación de Capacidades MLOps

## End-to-End

```
Versión 3.0 · Marzo 2026 · Confidencial
```

##### BUSINESS BLUEPRINT — APROBACIÓN

**Documento Nombre Fecha de aprobación Firma**


## Guía de Lectura del Documento

Este documento está organizado en tres bloques estratégicos claramente diferenciados. Cada sección está
etiquetada con un indicador de alcance para facilitar la comprensión del estado actual frente a la evolución
futura:

```
[POC] Implementado y validado en la Prueba de Concepto. Componentes reales,
ejecutables y demostrados.
```
```
[PROD.
CERCANO]
```
```
Mejoras y componentes por formalizar en la siguiente fase hacia ambiente
productivo.
```
```
[ PROD.
LEJANO]
```
```
Roadmap estratégico: automatización avanzada, escalabilidad y observabilidad
ampliada.
```

## 1. Introducción

Arca Continental manifestó interés en evaluar de extremo a extremo las capacidades de Snowflake para
desplegar modelos de ML mediante MLOps (Model Registry, Feature Store, ML Observability) y abordar la
migración de su modelo de Pronóstico de Ventas Semanales, actualmente en ejecución en Databricks, hacia
Snowflake.

Este documento describe la arquitectura técnica propuesta, el alcance de la Prueba de Concepto (POC)
realizada y la evolución planificada hacia un ambiente productivo robusto, gobernado y escalable.

## 2. Objetivos

El objetivo principal es plantear y validar la migración del modelo de ML de Pronóstico de Ventas Semanales
(uni_box_week), el cual fue analizado mediante el Snowpark Migration Accelerator (SMA), hacia Snowflake,
aprovechando la oportunidad para evaluar el ciclo de vida completo de MLOps con las mejores prácticas de la
plataforma.

Los objetivos específicos incluyen:

- Validar la factibilidad técnica del ciclo completo MLOps en Snowflake (entrenamiento, registro,
    inferencia, monitoreo).
- Reproducir funcionalmente la arquitectura MLOps actualmente implementada en Databricks.
- Simplificar la operación mediante capacidades nativas de Snowflake ML, reduciendo la complejidad y
    la dependencia de componentes externos.
- Establecer las bases técnicas para la evolución hacia un ambiente productivo automatizado,
    gobernado y auditable.

## 3. Alcance

### 3.1 Estándar Arquitectónico Adoptado

La arquitectura propuesta homologa el uso de tablas físicas (tablas fijas) como estándar de almacenamiento en
lugar de Dynamic Tables o Snowflake Tasks. Esta decisión responde a las restricciones de permisos actuales
del entorno de desarrollo, y se mantendrá también en la estrategia de orquestación productiva.

```
Decisiones de Estandarización
```
- Tablas físicas (fijas) como estándar de almacenamiento para features y resultados.
- Cálculo de lags y ventanas temporales mediante SQL.
- Python (Snowpark) y SQL como únicos lenguajes en los notebooks.
- Sin uso de Snowflake Tasks ni tablas dinámicas en la POC por restricciones de permisos.
- Orquestación externa o manual hasta definición formal de la estrategia productiva.

El uso exclusivo de Python y SQL responde a los siguientes criterios técnicos:

- Compatibilidad nativa con las APIs de Snowflake ML (snowflake.ml.modeling, Feature Store, Registry).
- Estandarización del stack tecnológico, facilitando el mantenimiento y la transferencia de conocimiento.
- Gobernanza y auditabilidad: los notebooks en Python + SQL son inspeccionables, versionables y
    reproducibles dentro del ecosistema Snowflake.
- Eliminación de dependencias externas (frameworks adicionales, entornos virtuales complejos, librerías
    de orquestación de terceros).


### 3.2 Lo que Incluye la POC

##### ▶ POC

- Ejecución del entrenamiento de modelos sobre datasets previamente preparados y disponibles en
    Snowflake.
- Implementación del Feature Store como tablas físicas con features calculadas mediante SQL (lags,
    rolling aggregations).
- Búsqueda de hiperparámetros (HPO) por grupo utilizando Bayesian Search
    (snowflake.ml.modeling.tune).
- Entrenamiento de 16 sub-modelos en paralelo mediante Many Model Training (MMT).
- Registro y versionado de modelos en el Snowflake Model Registry.
- Inferencia batch particionada mediante sintaxis SQL nativa.
- Configuración inicial observabilidad para perfomance drift, data drift y prediction drift.

### 3.3 Lo que NO Incluye la POC

- Ingesta, limpieza o transformación de datos desde fuentes origen (los datasets son provistos ya
    curados por Arca Continental).
- Diseño o implementación de pipelines de datos bajo el modelo Medallón.
- Definición o ajuste de métricas de negocio, KPIs o umbrales finales de aceptación.
- Orquestación completamente automatizada end-to-end (Snowflake Tasks, ML Jobs).
- Parametrización dinámica mediante tablas de configuración centralizadas.
- CI/CD automatizado con validación en cada etapa.
- Real-time inference y endpoints de serving (SPCS).


## 4. Datasets

Los datasets utilizados en la POC fueron provistos por Arca Continental de forma curada y lista para su uso
como fuente de entrenamiento e inferencia:

```
Dataset Volumen Descripción
```
```
Training ~ 88 millones de filas
(2. 6 GB)
```
```
Ventas semanales por cliente. Información del
año 2024 y 2025 dividida por semanas.
Segmentado en 16 grupos
(STATS_NTILE_GROUP).
```
```
Inference ~24.7 millones de filas
(807.2 MB)
```
```
Misma granularidad que el dataset de
entrenamiento. Utilizado para validar
predicciones y evaluar el comportamiento
productivo del modelo.
```
```
Holdout ~9 millones de filas
(304.5 MB)
```
```
Misma granularidad que el dataset de
entrenamiento. Corresponde a una partición del
10% del training set inicial proporcionado por
Arca, utilizada para establecer el baseline de los
drifts en el monitoreo. Se seleccionó el 10% de
los registros más recientes (ordenados por
semana) por su adecuada representatividad de
la distribución de los datos. Este porcentaje es
parametrizable en el notebook 01.
```
#### Gestión del Test Set y Predicciones Productivas

##### ▶ POC

Una consideración crítica de diseño es la diferenciación entre el conjunto de prueba (test set) utilizado durante
el entrenamiento y las predicciones productivas semanales. Estas dos operaciones sirven propósitos distintos y
deben mantenerse separadas arquitectónicamente:

- **Test Set:** Subconjunto del dataset de entrenamiento reservado para evaluación de métricas. Se utiliza
    un split 80/20, donde el 80% se destina a entrenamiento y el 20% a prueba (validación). Esta
    proporción ofrece un balance práctico entre suficiente volumen de datos para que el modelo aprenda
    patrones robustos y un tamaño de muestra adecuado para obtener métricas estadísticamente estables
    y representativas. El split se realiza de forma temporal por grupo (STATS_NTILE_GROUP) para data
    leakage.
- **Predicciones productivas:** Se ejecutan únicamente sobre el dataset de inferencia correspondiente a
    la semana en curso, sin mezcla con el test set de entrenamiento.

```
Nota Importante de Sampling
En el proceso de HPO se usa un muestreo estratificado para que el subset tenga una distribución similar al
dataset completo, y dicho subset será el espacio de búsqueda donde el modelo deberá encontrar la mejor
configuración HPO como si estuviese viendo el dataset completo. Sin embargo, en el proceso de MMT se
usa un muestro basado en un split temporal para evitar data leakage (que el modelo pueda predecir datos
se semanas que anteriormente ya ha visto durante su entrenamiento).
```

## 5. Arquitectura Actual en Databricks

El siguiente diagrama representa la arquitectura MLOps en Databricks, compartida por Arca Continental como
referencia para el análisis y definición de la arquitectura objetivo en Snowflake.

La arquitectura actual implementa un enfoque de entrenamiento de múltiples modelos especializados. Los
puntos clave son:

1. Segmentación del dataset de entrenamiento en 16 grupos basados en patrones recientes de venta
    (últimas 4 semanas) y volumen histórico por cuartiles (0-25%, 25-50%, 50-75%, 75-100%).
2. Entrenamiento distribuido manual: 16 sub-modelos entrenados individualmente con AutoML de
    Databricks (60 min timeout por modelo), resultando en ~16 horas secuenciales de cómputo.
3. Ensemble Orchestration Custom: librería "Pandas Ensemble" personalizada que unifica los 16 sub-
    modelos en un objeto lógico único con ruteo automático por grupo en inferencia.
4. Gestión con MLflow: versionado por timestamp, aliases por fecha de ejecución (ej. 20250910), tags
    con execution_date y cutoff_date.
5. Monitoreo manual con alertas no proactivas: Data Drift (Jensen-Shannon >0.2 Warning, >0.4 Critical),
    Prediction Drift por distribución histórica vs actual, y Performance Drift con WAPE + F1 Score
    retroactivo.


## 6. Arquitectura Propuesta en Snowflake

### 6.1 Visión General de la POC

##### ▶ POC

La arquitectura propuesta para la POC en Snowflake tiene como objetivo reproducir el flujo MLOps de Databricks
manteniendo la equivalencia funcional del modelo de Pronóstico de Ventas, pero simplificando su operación
mediante capacidades nativas de Snowflake ML.

Procesos como el entrenamiento de múltiples sub-modelos, el ruteo de inferencias, el versionado de modelos y
el monitoreo del desempeño se resuelven mediante HPO manual + MMT, ML Experiments y ML Registry, en
lugar de AutoML propietario y componentes custom.

### 6.2 Comparativa de Capacidades MLOps

La siguiente tabla resume la equivalencia funcional entre Databricks y Snowflake para cada componente del
ciclo MLOps:

```
Componente Databricks (actual) Snowflake (propuesto)
```
```
Entrenamiento AutoML Databricks + loops
manuales de segmentación
```
```
HPO Bayesian Search + Many
Model Training (MMT), como una
alternativa parcial. Revisar sección
8.3.
```
```
Inferencia Pandas Ensemble custom
(PyFunc) con ruteo manual
```
```
Partitioned Model Inference con
ruteo automático nativo. Revisar
sección 1 1.
```

Gestión de Modelos MLflow externo con aliases y tags Snowflake Model Registry
integrado (versiones, alias, tags).
Revisar sección 9.

Orquestación Azure Data Factory POC: manual. Revisar sección 1 4.

Observabilidad Notebooks custom de monitoreo
desacoplados

```
Observabilidad con monitores
custom. Revisar sección 1 2.
```
Feature Management Tablas Delta en Databricks Tablas físicas + SQL (lags
calculados, sin FeatureView por
permisos). Revisar sección 7.

Seguridad Permisos a nivel de catálogo
Databricks

```
RBAC nativo Snowflake (roles
ML_DEV, ML_OPS,
ML_CONSUMER). Revisar sección
13 .4.
```

## 7. Feature Store

### 7.1 Implementación en la POC — Tablas Físicas con Cálculo Manual

##### ▶ POC

En la POC, el Feature Store se implementa mediante tablas físicas (tablas planas fijas) en el schema
SC_FEATURES_BMX. Esta decisión responde a restricciones de permisos del entorno de desarrollo que
impiden el uso de Dynamic Tables o Snowflake Tasks para actualizaciones automáticas.

La POC utiliza la tabla proporcionada de **FEAT_CUSTBPR_WEEKLY** para el uso común de las notebooks y
como fuente de ingesta principal de features.

Sin embargo, para la prueba de concepto de cómo es que se crea un flujo lineal sobre construcción de features
desde cero se tienen los siguientes pasos:

1. La tabla BD_AA_DEV.SC_STORAGE_BMX_PS.TRAIN_DATASET_STRUCTURED se toman como
    fuente única (o en su caso aquella tabla la cual represente la transaccionalidad a construir).
2. A través de esa tabla (TRAIN_DATASET_STRUCTURED) se crea un subset de datos los cuales
    contienen la entidad, granularidad y valor numérico necesario para la creación del feature store (si la
    tabla fuente ya tiene ese formato tomaríamos directamente esta).
3. Se ejecutan transformaciones SQL directamente en Snowflake mediante la compilación secuencial de
    stored procedures para re-calcular features de lag (ventas de las últimas N semanas), rolling
    aggregations (sumas, promedios, máximos en ventanas de 4, 12 y 24 semanas).
4. Los resultados se materializan en la tabla física, la cual emula la estructura de
    FEAT_CUSTBPR_WEEKLY dentro del schema de features, lista para ser consumida tanto por el
    proceso de entrenamiento como por inferencia.

Las features disponibles creados a partir de una simulación de procesos en la POC incluyen ventanas
temporales de ventas pasadas (W_M1_TOTAL, W_M2_TOTAL, W_M3_TOTAL, W_M4_TOTAL, AVG_PREV2,
MAX_PREV3, SUM_PAST_4_WEEKS,AVG_PAST_12_WEEKS).

```
Cálculo de Lags mediante SQL
Los lags y ventanas temporales se calculan con sentencias SQL estándar utilizando funciones de
ventana (WINDOW FUNCTIONS). Por ejemplo:
SUM(ventas) OVER (PARTITION BY cliente, producto ORDER BY semana ROWS BETWEEN 12
PRECEDING AND 1 PRECEDING)
Este enfoque es reproducible, auditable y compatible con el engine de Snowflake sin dependencias
adicionales.
```
### 7.2 Evolución y metodología del Feature Store

▶ **PROD. CERCANO**

En producción cercano, el Feature Store evoluciona hacia una arquitectura batch incremental con trazabilidad
completa por lote. Esta arquitectura responde a los requerimientos de Arca Continental: trazabilidad por batch
procesado, capacidad de auditoría y re-ejecución de lotes específicos/nuevos, y compatibilidad con el
esquema de cortes que ya manejan actualmente.

#### Componentes de la Arquitectura

```
Tabla Schema Descripción
```
```
TRAIN_TRANSACTIONS SC_FEATURES_BMX Tabla fuente con columnas técnicas de
control: BATCH_ID, PROCESS_STATUS
```

##### (PENDING / IN_PROGRESS /

##### PROCESSED / FAILED),

```
PROCESSED_AT. Define el corte de cada
ejecución.
```
```
FEATURE_BATCH_RUNS SC_FEATURES_BMX Control de auditoría por lote: BATCH_ID,
RUN_TS_START, RUN_TS_END,
STATUS, conteos de registros procesados y
notas de la ejecución.
```
```
FEAT_<entity>_<frequency>_BATCH SC_FEATURES_BMX Staging del batch actual: features
calculadas en esta ejecución (ENTITY_ID,
FEATURE_TS, BATCH_ID + columnas de
features). Se sobreescribe en cada corrida.
```
```
FEAT_<entity>_<frequency>_HIST SC_FEATURES_BMX Histórico append-only: acumula todas las
ejecuciones con su BATCH_ID. Fuente para
entrenamiento: permite reproducir el estado
de features de cualquier batch pasado.
```
```
FEAT_<entity>_<frequency>_LATEST SC_FEATURES_BMX Estado actual (DROP/RECREATE):
contiene las entidades activas del último
batch. Fuente optimizada para inferencia.
No mantiene entidades inactivas de
ejecuciones anteriores.
```
```
Decisión clave: DROP/RECREATE en FEATURES_UNI_BOX_LATEST (no MERGE)
```
```
La tabla LATEST no se actualiza mediante MERGE/UPSERT por ENTITY_ID. En cada ejecución de batch,
la tabla se elimina y se recrea con el contenido del batch actual, esto siempre y cuando el batch el proceso
sea parte del grano temporal más reciente (semana más alta, mes más alto, etc). Esto significa que
LATEST refleja exactamente las entidades transaccionalmente activas en el último lote, sin arrastrar
entidades inactivas de ejecuciones anteriores. El histórico completo siempre está disponible en
FEAT_<entity>_<frequency>_HIST (append-only), donde cada registro mantiene su BATCH_ID para
trazabilidad. Para añadir nuevas features: se necesita cambiar el proceso de compute features para los
siguientes batches, para la estructura histórica se aplica ALTER TABLE para agregar columnas (en una
primera instancia de prueba este proceso seria de forma manual, posteriormente se pude tener una tabla de
configuración la cual verifique las features existentes y a partir de esto tener un alter automatizado). Las
features no se eliminan a nivel estructural — solo se excluyen del cómputo si ya no se necesitan,
preservando así la trazabilidad del histórico.
```
#### Definición de Entity y Grain

El Feature Store se organiza siguiendo el principio de **entity + grain temporal consistente** , lo que garantiza
coherencia entre entrenamiento, inferencia y trazabilidad histórica.
La **entity** corresponde a la unidad de predicción del modelo, es decir, la combinación de identificadores sobre
la cual se calculan las features. El **grain** define el nivel temporal de agregación utilizado por el modelo.
Para el caso UNI_BOX, el grano lógico se define como:

- **ENTITY** : combinación de identificadores de negocio (por ejemplo, Cliente + Producto).
- **GRAIN TEMPORAL** : semana.
Cada registro del Feature Store representa entonces el estado de una entidad específica en una semana
determinada. Las columnas clave que permiten identificar de forma única cada registro son:
- ENTITY_ID
- SEMANA (o identificador temporal equivalente)
- BATCH_ID


Esta estructura asegura consistencia en la generación de features, facilita el versionamiento histórico y permite
reproducir datasets de entrenamiento de manera determinística.

#### Generación y Control del BATCH_ID

Cada ejecución del pipeline de features genera un identificador único denominado **BATCH_ID** , el cual
representa el lote de procesamiento asociado a una corrida específica del pipeline.

El BATCH_ID se genera al inicio del proceso de ejecución, típicamente mediante un identificador único (por
ejemplo, UUID, timestamp o identificador de ejecución del orquestador). Este valor se registra inmediatamente
en la tabla FEATURE_BATCH_RUNS, donde se almacena la metadata del proceso.

La columna BATCH_ID se define como **clave primaria** en FEATURE_BATCH_RUNS, garantizando que cada
ejecución sea única y evitando colisiones entre lotes.
Este identificador se propaga a todas las tablas involucradas en el pipeline:

##### • TRAIN_TRANSACTIONS

- FEAT_<entity>_<frequency>_BATCH
- FEAT_<entity>_<frequency>_HIST
- FEAT_<entity>_<frequency>_LATEST

De esta forma, todas las features generadas pueden asociarse inequívocamente al lote que las produjo,
permitiendo trazabilidad completa del procesamiento.

#### Manejo de Reprocesamiento de Features

En situaciones donde sea necesario reprocesar features para un periodo específico (por ejemplo, debido a
corrección de datos de origen o cambios en la lógica de cálculo), el sistema genera un **nuevo BATCH_ID** y
ejecuta nuevamente el pipeline para las entidades correspondientes.
El resultado de este reproceso se inserta en FEAT_<entity>_<frequency>_HIST sin eliminar registros históricos
previos. Esto permite mantener múltiples versiones de features para una misma entidad y periodo temporal,
cada una identificada por su BATCH_ID. El consumo para entrenamiento o auditoría se realiza filtrando
explícitamente por el BATCH_ID o por el rango temporal deseado, garantizando reproducibilidad y evitando
sobrescrituras destructivas del histórico.

#### Incorporación de Nuevas Features

La incorporación de nuevas features en el Feature Store se realiza mediante la actualización del proceso de
cálculo de features utilizado en el pipeline. Cuando se introduce una nueva feature, se añade la columna
correspondiente a la estructura de las tablas históricas mediante un cambio de esquema:
ALTER TABLE FEAT_<entity>_<frequency>_HIST
ADD COLUMN <feature_name>

Durante las primeras etapas de implementación, este proceso puede ejecutarse manualmente como parte del
despliegue del pipeline. En fases posteriores, este debe automatizarse y crear un query dinámico mediante una
tabla de configuración de features que permita validar y aplicar evoluciones de esquema de forma controlada.
Las features existentes no se eliminan a nivel estructural. En caso de que una feature deje de utilizarse en
modelos futuros, simplemente se excluye del cálculo o del consumo del modelo, preservando así la trazabilidad
del histórico.

#### Separación por Granularidad Temporal

El Feature Store mantiene consistencia en el **grain temporal** dentro de cada estructura de almacenamiento.
En caso de que se requieran modelos con diferentes granularidades temporales (por ejemplo, semanal y
mensual), se crearán estructuras independientes para cada grain. Por ejemplo:


##### • FEAT_<BPR>_<WEEK>_HIST

##### • FEAT_<BPR>_<WEEK>_LATEST

##### • FEAT_<BPR>_<MONTHLY>_HIST

##### • FEAT_<BPR>_<MONTHLY>_LATEST

Esta separación evita mezclar granularidades dentro de una misma tabla, simplifica las consultas y asegura
consistencia entre la lógica de cálculo de features y el modelo que las consume.

#### Compute de Features

El cálculo de features se ejecuta mediante la **notebook en las celdas de procesamiento** que forma parte del
pipeline del Feature Store. Esta notebook es responsable de construir las transformaciones necesarias sobre
las tablas base y generar las features correspondientes para el batch en ejecución.
El proceso inicia con la creación de los **stages intermedios** necesarios para calcular las features. Estos stages
contienen transformaciones parciales derivadas de las fuentes transaccionales, como agregaciones temporales,
lags y variables de comportamiento por entidad.

Cada grupo de features puede calcularse de manera independiente dentro de estos stages, lo que permite
modularizar el proceso de generación de variables y facilita la evolución futura del pipeline.
Una vez calculadas las features en sus respectivos stages, la notebook ejecuta una **unión final de todas las
features calculadas** , generando el dataset consolidado que se materializa en la tabla
FEAT_<entity>_<frequency>_BATCH. Este dataset incluye el identificador de la entidad, la referencia temporal
(FEATURE_TS o semana) y el BATCH_ID correspondiente al proceso de ejecución.

Como mejora futura del pipeline, se recomienda evolucionar el proceso de unión de features hacia un **modelo
dinámico basado en configuración** , donde la definición de las features a incluir se controle mediante una tabla
de metadata o configuración. Bajo este enfoque, el SQL encargado de consolidar las features se genera
dinámicamente a partir de esta configuración, lo que permite agregar nuevas features o modificar el pipeline sin
necesidad de alterar manualmente la lógica de unión.
Este enfoque facilita la **escalabilidad del Feature Store** , permitiendo incorporar nuevas variables de manera
controlada y manteniendo un pipeline flexible y mantenible a medida que se incorporen nuevos modelos o
fuentes de información.

#### Validaciones del Pipeline de Features (Proceso de QA sugerido)

Antes de actualizar la tabla FEAT_<entity>_<frequency>_LATEST, el pipeline debe ejecuta una serie de
validaciones para asegurar la integridad del batch generado.
Entre las validaciones consideradas se incluyen:

- Verificación de conteo de registros generados.
- Validación de consistencia del grain temporal.
- Revisión de valores nulos en features críticas.
- Confirmación de rango temporal esperado.
Solo si estas validaciones son exitosas, el estado del batch en FEATURE_BATCH_RUNS se actualiza a
SUCCESS y se procede con la reconstrucción de la tabla LATEST.

#### Optimización y Crecimiento del Feature Store

A medida que el volumen de datos del Feature Store crezca, se evaluarán estrategias de optimización orientadas
a mejorar el rendimiento de consultas de entrenamiento e inferencia.
Entre las estrategias consideradas se incluye la aplicación de **clustering keys** en tablas históricas basadas en
columnas de acceso frecuente, tales como:

- ENTITY_ID
- Grano de tiempo
Estas optimizaciones permiten mejorar la eficiencia de las consultas analíticas y reducir los costos de
procesamiento en entornos productivos de gran volumen.


#### Flujo del Pipeline Batch

1. Generación de BATCH_ID e inicio del registro en FEATURE_BATCH_RUNS (STATUS =
    IN_PROGRESS, RUN_TS_START).
2. Selección de transacciones con PROCESS_STATUS = PENDING desde TRAIN_TRANSACTIONS.
3. Marcado IN_PROGRESS con el BATCH_ID actual: congela el corte. Nuevas transacciones que lleguen
    en paralelo quedan en PENDING para el siguiente batch.
4. Cálculo de features para el batch actual (lags, rolling aggregations, ventanas temporales) mediante SQL.
    Resultado en FEAT_<entity>_<frequency>_BATCH.
5. INSERT en FEAT_<entity>_<frequency>_HIST (append). El histórico es inmutable y trazable por
    BATCH_ID.
6. DROP y RECREATE de FEAT_<entity>_<frequency>_LATEST con el contenido del batch actual. La
    tabla refleja solo las entidades activas del lote.
7. Marcado de transacciones como PROCESSED en TRAIN_TRANSACTIONS.
8. Cierre del batch en FEATURE_BATCH_RUNS: STATUS = SUCCESS o FAILED, RUN_TS_END y
    conteos finales.

#### Uso por Componente del Pipeline

```
 Entrenamiento (HPO + MMT): consume FEAT_<entity>_<frequency>_HIST filtrando por BATCH_ID o
rango temporal. Permite reproducir exactamente el estado de features de cualquier ejecución pasada.
 Inferencia batch: consume FEAT_<entity>_<frequency>_LATEST directamente. Al ser
DROP/RECREATE, el contenido es siempre el del último batch activo sin filtros adicionales.
```

#### Orquestación del Feature Store

▶ **PROD. CERCANO**

```
 Opción A — Snowflake Tasks (si se habilitan permisos): pipeline nativo con dependencias y logging
integrado. Actualmente no disponible en la POC por restricciones de permisos.
 Opción B — Snowflake ML Jobs: alternativa orientada a cargas ML. Requiere permisos similares a
Tasks pero puede habilitarse de forma independiente.
 Opción C — Orquestador externo (Dagster u otro): el pipeline se implementa en notebooks
orquestados desde Dagster, invocando las operaciones SQL en Snowflake vía conector. No requiere
permisos adicionales en Snowflake.
```
## 8. Entrenamiento: HPO y Many Model Training

### 8.1 HPO — Hyperparameter Optimization

##### ▶ POC

#### Qué es y qué resuelve


HPO (Hyperparameter Optimization) es el proceso de búsqueda sistemática de la combinación óptima de
hiperparámetros para cada algoritmo y segmento. HPO automatiza la exploración y selecciona la configuración
que minimiza el error de validación.

En la POC, HPO también puede considerarse una forma de orquestación manual, ya que es el componente que
controla qué algoritmo y qué configuración se asigna a cada uno de los 16 grupos, registrando los resultados
de forma trazable en ML Experiments.

#### Implementación y decisiones técnicas

El proceso de HPO sigue el siguiente flujo para cada uno de los 16 grupos de segmentación:

1. Se carga el dataset del grupo desde la tabla de features (FEAT_CUSTBPR_WEEKLY) o directamente
    desde la tabla de entrenamiento limpia como fallback.
2. Se ejecutan **15 trials** de Bayesian Search por grupo, valor que equilibra calidad de exploración y
    tiempo de cómputo (≈1 hora para los 16 grupos sobre el cluster de 5 nodos M). Este número es
    configurable: más trials amplían el espacio explorado a costa de mayor tiempo; menos trials aceleran
    el proceso, pero pueden derivar en soluciones subóptimas.
3. Se aplica un **_muestreo estratificado del 20%_** del dataset del grupo para la fase de búsqueda.
    Aunque en primera instancia pueda parecer una decisión arbitraria, tiene una justificación técnica
    sólida: con ~ 88 M de filas (2. 6 GB) distribuidos en 16 grupos, entrenar 15 trials con el 100% de los
    datos implicaría tiempos de exploración de muchas horas por grupo. El 20% mantiene suficiente
    representatividad estadística para discriminar configuraciones buenas de malas, que es el objetivo del
    HPO, sin incurrir en el costo de entrenamiento completo. El entrenamiento final con MMT sí utiliza el
    100% de los datos.
4. Se realiza un split 80/20 dentro del subconjunto **_muestreado temporalmente_** para train/validation.
5. Se ejecuta Bayesian Search (BayesOpt) con 15 trials y hasta 4 trials concurrentes por grupo. El
    proceso está estructurado en un bucle secuencial a nivel lógico (un grupo tras otro), pero internamente
    cada trial de HPO se ejecuta en paralelo sobre el cluster de 5 nodos M, al igual que MMT. Esto reduce
    significativamente el tiempo real de exploración respecto a una ejecución puramente secuencial.
6. Los hiperparámetros optimizados incluyen: número de estimadores, profundidad máxima, tasa de
    aprendizaje, parámetros de regularización (reg_alpha, reg_lambda), subsampling y colsample_bytree.
7. Los resultados (mejores parámetros, RMSE, MAE, WAPE por grupo) se registran en ML Experiments
    para trazabilidad, con fallback a tabla HPO_<model> si ML Experiments no está disponible.

```
¿Por qué Bayesian Search y no Random Search o Grid Search?
Grid Search evalúa todas las combinaciones posibles del espacio de hiperparámetros: con 9
parámetros y rangos continuos, el número de combinaciones es computacionalmente prohibitivo.
Random Search muestrea combinaciones al azar de forma ciega, sin aprender de las evaluaciones
anteriores. Es mejor que Grid, pero ineficiente en espacios de alta dimensión.
Bayesian Search (BayesOpt) construye un modelo probabilístico del espacio de hiperparámetros a
partir de los trials previos, y dirige los siguientes trials hacia las regiones con mayor probabilidad de
mejorar la métrica objetivo. Esto permite encontrar configuraciones óptimas con un número mucho
menor de evaluaciones, lo que se traduce directamente en menor tiempo de cómputo y menor costo
de warehouse.
```
El mapeo de algoritmo por grupo es el siguiente (resultado del análisis de la POC, el cual consistió en validar la
compatibilidad entre las librerías nativas de modelos ML en Snowflake vs las usadas en Databricks):

##### GROUP_MODEL = {

```
"group_stat_0_1": "LGBMRegressor",
"group_stat_0_2": "LGBMRegressor",
"group_stat_0_3": "LGBMRegressor",
"group_stat_0_4": "LGBMRegressor",
```

```
"group_stat_1_1": "LGBMRegressor",
"group_stat_1_2": "LGBMRegressor",
"group_stat_1_3": "XGBRegressor",
"group_stat_1_4": "XGBRegressor",
"group_stat_2_1": "LGBMRegressor",
"group_stat_2_2": "LGBMRegressor",
"group_stat_2_3": "XGBRegressor",
"group_stat_2_4": "XGBRegressor",
"group_stat_3_1": "LGBMRegressor",
"group_stat_3_2": "LGBMRegressor",
"group_stat_3_3": "LGBMRegressor",
"group_stat_3_4": "XGBRegressor",
}
```
En la POC, el algoritmo asignado a cada grupo está definido explícitamente en el código, tomando como
referencia los modelos ya utilizados por Arca Continental en Databricks. Esta decisión reduce el riesgo de la
migración al mantener una base conocida y validada por el negocio. A futuro, la sección 8.3 describe cómo
automatizar esta selección mediante una capa de orquestación sobre HPO, eliminando la necesidad de definir
el algoritmo manualmente por segmento.

#### Frecuencia de ejecución

- **En la POC:** HPO se ejecuta manualmente una vez antes del primer entrenamiento completo.
- **En producción cercana:** HPO se ejecutará de forma puntual bajo demanda, cuando el equipo ML
    detecte degradación de performance o un cambio significativo en la distribución de los datos.
- **En producción lejana:** HPO se ejecutará de forma automática, disparado por los monitores de ML
    Observability ante drift significativo, evaluando además múltiples algoritmos candidatos por segmento
    según la propuesta de AutoML descrita en la sección 8.3.

### 8.2 MMT — Many Model Training

##### ▶ POC

#### Qué es y qué resuelve

Many Model Training (MMT) es una funcionalidad nativa de Snowflake que permite entrenar múltiples modelos
de forma paralela, donde cada modelo se especializa en un subconjunto específico de los datos definido por
una columna de partición. En el contexto de ARCA, esta columna es STATS_NTILE_GROUP, que define los 16
segmentos del modelo.

Mientras que HPO optimiza la configuración para cada grupo, MMT entrena los 16 modelos finales de forma
simultánea usando esos parámetros validados. Esto simplifica radicalmente la orquestación: en Databricks se
requieren loops manuales, diccionarios de tracking (experiment_submodel_map, group_submodel_map) y
gestión manual de artefactos; MMT automatiza todo con una sola llamada especificando la columna de partición.

#### Implementación y tiempos de ejecución

El proceso de MMT sigue el siguiente flujo:

1. Se cargan los hiperparámetros óptimos por grupo desde ML Experiments o desde la tabla de
    resultados de HPO.
2. Se carga el dataset completo (100% de los datos, ~ 88 M filas / 2. 6 GB) desde la tabla de features.
3. Se define la función de entrenamiento por segmento, que recibe los hiperparámetros específicos del
    grupo e instancia el modelo correspondiente (XGBRegressor o LGBMRegressor según el mapeo por
    grupo).


4. Se ejecuta ManyModelTraining con particionamiento por STATS_NTILE_GROUP. Snowflake
    distribuye automáticamente los 16 grupos entre los nodos del cluster, ejecutándolos en paralelo sin
    intervención manual.
5. El entrenamiento completo de los 16 sub-modelos sobre el 100% del dataset (~96M filas / 2.9 GB)
    tarda aproximadamente 7 minutos en el cluster de 5 nodos M.
6. Cada modelo se registra en el Snowflake Model Registry con su versión, métricas (RMSE, MAE,
    WAPE) y timestamp de entrenamiento.

```
# Fragmento ilustrativo del flujo MMT
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining
```
```
mmt = ManyModelTraining(
session=session,
partition_column="STATS_NTILE_GROUP",
train_func=train_func_with_hpo_params, # función que carga params por grupo
)
mmt.run(train_df) # Snowflake paraleliza automáticamente los 16 grupos
```
#### Frecuencia de ejecución

- **En la POC:** MMT se ejecuta manualmente una vez por ciclo de experimentación, bajo demanda del
    equipo ML.
- **En producción cercana:** MMT se ejecutará de forma periódica (frecuencia semanal o mensual por
    definir con Arca Continental), disparado manualmente o mediante un orquestador externo al inicio de
    cada ciclo de negocio.
- **En producción lejana:** MMT se ejecutará de forma automática, disparado por trigger de drift
    detectado por ML Observability o por calendario, con validación de métricas integrada antes de
    promover el modelo resultante a producción.

#### Escalabilidad de Modelos

La estrategia de escalamiento adoptada es la segmentación por columna de partición (STATS_NTILE_GROUP).
Este enfoque fue seleccionado sobre alternativas como partición lógica por región o distribución computacional
genérica, porque:

- Preserva la lógica de negocio existente en Databricks (16 grupos basados en patrones de venta y
    volumen).
- Alinea el escalamiento con el patrón de MMT de Snowflake, que está optimizado para la paralelización
    por partición.
- Permite asignar algoritmos distintos por segmento (XGBoost para grupos de alto volumen, LightGBM
    para grupos de baja densidad) sin modificar la arquitectura base.
- El código escala automáticamente sin modificaciones: al aumentar el número de grupos o el tamaño
    del warehouse, MMT distribuye la carga entre nodos disponibles.

### 8.3 Propuesta Conceptual: AutoML en Snowflake

▶ **PROD. LEJANO**

```
Contexto: ¿por qué los modelos están fijos en el código hoy?
En la POC actual, el algoritmo asignado a cada grupo (XGBoost o LightGBM) está definido
explícitamente en el código de entrenamiento. Esto significa que para cambiar el modelo de un
segmento hay que modificar el código manualmente.
Arca Continental señaló la necesidad de automatizar esta selección, de modo que el sistema pueda
evaluar múltiples algoritmos por segmento y elegir el óptimo sin intervención humana.
```

```
La propuesta conceptual que se describe a continuación resuelve exactamente esta necesidad,
elevando el HPO actual a un AutoML completo dentro del ecosistema Snowflake.
```
Snowflake no dispone actualmente de una solución AutoML nativa equivalente a la de Databricks. No obstante,
es posible construir un enfoque AutoML utilizando las capacidades existentes de Hyperparameter Optimization
(HPO) y Many Model Training (MMT).

En la POC actual, el proceso de HPO optimiza hiperparámetros sobre un conjunto fijo de algoritmos definidos
en el código. La evolución propuesta consiste en ampliar esta lógica para que la exploración no solo ocurra
dentro del espacio de hiperparámetros, sino también a nivel de selección de modelo.

Conceptualmente, esto se implementaría mediante:

- Un diccionario o tabla de configuración que defina múltiples algoritmos candidatos (por ejemplo,
    XGBoost, LightGBM, modelos lineales, etc.).
- Para cada algoritmo, un espacio de búsqueda de hiperparámetros estructurado.
- Una capa de orquestación que itere sobre combinaciones de (modelo, espacio de búsqueda,
    segmento).
- Registro unificado de métricas en Model Registry para comparar desempeño entre algoritmos y
    configuraciones.

En este esquema, cada ejecución evaluaría múltiples combinaciones modelo-configuración, distribuidas sobre
el compute pool mediante HPO. Posteriormente, se seleccionaría automáticamente el mejor modelo por
segmento según la métrica objetivo definida.

Este enfoque no forma parte del alcance actual de la POC, pero puede implementarse como una capa adicional
de orquestación sobre HPO y MMT, manteniéndose completamente dentro del ecosistema Snowflake. De esta
manera, se simula un comportamiento AutoML sin depender de una solución externa propietaria.

A continuación, la representación gráfica de la solución propuesta:



## 9. Model Registry

##### ▶ POC

### 9.1 Función y Estructura

El Snowflake Model Registry es un repositorio integrado que centraliza el gobierno del ciclo de vida de los
modelos directamente dentro de Snowflake, reemplazando MLflow como componente externo. Permite
gestionar múltiples versiones, comparar performance entre ellas y controlar qué versión está activa en cada
ambiente.

### 9.2 Artefactos Almacenados por Versión

Cada versión registrada en el Model Registry almacena los siguientes artefactos y metadatos:

```
Artefacto / Metadato Descripción
```
```
Modelo serializado Objeto del modelo entrenado (XGBRegressor / LGBMRegressor)
serializado en formato compatible con Snowflake ML.
```
```
Métricas de performance RMSE, MAE, WAPE y MAPE por grupo (16 valores de cada métrica)
registradas en el momento del entrenamiento.
```
```
Hiperparámetros Configuración óptima encontrada por HPO para cada grupo
(n_estimators, max_depth, learning_rate, etc.).
```
```
Tags de versión activa (solo
en PROD)
```
```
Tags asignados al modelo para identificar qué versión está activa
por proyecto o grupo de consumo (ej. act_general, act_grupo1).
Controlados vía RBAC por ML_OPS_ROLE.
```
```
Autor y timestamp Usuario que registró el modelo y fecha/hora de registro para
trazabilidad y auditoría.
```
```
Dependencias Versiones de librerías utilizadas (XGBoost, LightGBM, Snowflake
ML) para reproducibilidad del entorno.
```
## 10. Estándares de Nomenclatura


Esta sección define las convenciones de nombres adoptadas para todos los objetos del pipeline MLOps en
Snowflake: modelos en el Model Registry, Feature Store, vistas de consumo y tablas de observabilidad. La
nomenclatura está diseñada para ser escalable a múltiples modelos, granularidades y ambientes, y debe
consultarse antes de registrar cualquier objeto nuevo.

### 10.1 Nomenclatura de Modelos

#### Patrón

##### <TARGET>_<ENTITY>_<FREQUENCY>_<METHOD>

Cada dimensión usa un código de 4 a 7 letras que identifica de forma única el aspecto del modelo que

representa. El vocabulario completo se mantiene en la sección 10. 7 (Vocabulary Registry) y debe consultarse
antes de registrar cualquier modelo nuevo.

```
Dimensión Qué describe Códigos actuales
```
```
TARGET Variable que se predice UNIBOX, PROB, OOS
```
```
ENTITY Nivel de granularidad de la entidad CUSTBPR, CUSTPROD, STORE, SKU
```
```
FREQUENCY Granularidad temporal WEEKLY, MONTHLY, DAILY
```
```
METHOD Tipo de modelo / metodología FORECAST, REGRESS, CLASSIF,
RANKING
```
**Ejemplos aplicados**

```
Databricks (actual) Snowflake (propuesto) Descripción
```
```
forecast_bpr_customer_week UNIBOX_CUSTBPR_WEEKLY_FORECAST Forecast de uni-box,
customer × BPR, semanal
```
```
(futuro) probability model PROB_CUSTPROD_MONTHLY_CLASSIF Probabilidad, customer ×
product, mensual
```
```
(futuro) regression model UNIBOX_CUSTBPR_DAILY_REGRESS Regresión de uni-box,
granularidad diaria
```
Los sub-modelos internos (los 16 modelos por grupo de STATS_NTILE_GROUP) usan doble underscore para
diferenciarse del modelo particionado público:

```
UNIBOX_CUSTBPR_WEEKLY_FORECAST__grp_0_1
UNIBOX_CUSTBPR_WEEKLY_FORECAST__grp_0_2
...
```
Solo el modelo particionado (UNIBOX_CUSTBPR_WEEKLY_FORECAST) se migra a producción y es el
artefacto público. Los sub-modelos son artefactos internos del entrenamiento.

### 10.2 Nomenclatura de Versionado y Ciclo de Vida del Modelo

#### Formato de versión


```
v_YYYYMMDD_HHMM (ej. v_20260311_1430)
```
El ciclo de vida se gestiona de forma distinta según el ambiente: aliases en DEV para marcar el estado dentro
del ciclo de experimentación, tags en PROD para controlar qué versión ejecutan los pipelines. Para mayor
detalle del flujo de promoción de un modelo a producción puede consultar la sección 13.3.

#### Ambiente de Desarrollo (BD_AA_DEV)

Los aliases no ejecutan inferencia directamente. Solo identifican qué versión será promovida al ambiente
productivo por el Notebook 07b de migración.

```
Alias Propósito
```
```
PRODUCTION Versión lista para promoción a PROD
```
```
CHALLENGER Versión candidata en evaluación (A/B,
shadow)
```
#### Ambiente de Producción (BD_AA_PROD)

En producción no se usan aliases. Los pipelines de inferencia y observabilidad leen los tags para determinar
qué versión ejecutar.

```
Tag Propósito Ejemplo
```
```
PRODUCTION_<USE_CASE> Versión activa para inferencia
batch
```
##### PRODUCTION_<USE_CASE> =

```
'v_20260311_1430'
```
```
ROLLBACK_VERSION
_<USE_CASE>
```
```
Versión anterior disponible para
rollback rápido
```
##### ROLLBACK_VERSION =

```
'v_20260301_0900'
```
<USE_CASE> identifica de forma estable el cliente o caso de uso.

El flujo es: en DEV se asigna el alias PRODUCTION a la versión lista → el script de migración la copia a
BD_AA_PROD → en PROD se asigna el tag PRODUCTION_<USE_CASE> apuntando a esa versión → los

pipelines leen el tag para resolver qué modelo ejecutar.

### 10.3 Layout de Schemas por Ambiente


En desarrollo se trabaja exclusivamente sobre BD_AA_DEV para entrenamiento, experimentación y registro
de modelos. En producción se trabaja exclusivamente sobre BD_AA_PROD para inferencia, feature store
productivo y observabilidad.

```
BD_AA_DEV (desarrollo)
├── SC_MODELS_BMX ← registry de desarrollo
│ ├── UNIBOX_CUSTBPR_WEEKLY_FORECAST ← modelo particionado (público)
│ └── unibox_custbpr_weekly_forecast__grp_* ← sub-modelos internos
├── SC_STORAGE_BMX_PS ← datos de entrenamiento
└── SC_FEATURES_BMX ← feature store
```
```
BD_AA_PROD (producción)
├── SC_MODELS_BMX_PROD ← registry de producción
│ └── UNIBOX_CUSTBPR_WEEKLY_FORECAST ← modelo migrado
├── SC_STORAGE_BMX_PS_PROD ← datos de inferencia + predicciones
└── SC_FEATURES_BMX_PROD ← feature store producción
```
### 10.4 Nomenclatura del Feature Store

#### Patrón

```
FEAT_<entity>_<frequency>
```
```
Descripción Nombre propuesto Ejemplos
```
```
entity Nivel/granularidad de la entidad
CUSTBPR (customer×BPR),
CUSTPROD, STORE
```
```
frequency
Granularidad temporal
WEEKLY, MONTHLY, DAILY
```
#### Layout de tablas y vistas

La feature table base FEAT_CUSTBPR_WEEKLY contiene las features materializadas, adicional a ello se
tienen otras tablas y vistas:

##### SC_FEATURES_BMX

```
├── FEAT_CUSTBPR_WEEKLY ← tabla materializada de features
├── FEAT_CUSTBPR_WEEKLY__TRAIN ← tabla: features + target (entrenamiento)
├── FEAT_CUSTBPR_WEEKLY__HOLDOUT ← tabla: holdout temporal (baselines)
├── FEAT_CUSTBPR_WEEKLY__INF ← tabla: features para inferencia
├── FEAT_CUSTBPR_WEEKLY__TRAIN_VW ← vista: sobre __TRAIN (opcional)
├── FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW ← vista: sobre __HOLDOUT (para baselines)
└── FEAT_CUSTBPR_WEEKLY__INF_VW ← vista: sobre __INF (para inferencia)
```
El pipeline de entrenamiento puede leer de FEAT_<entity>_<frequency>__TRAIN (tabla) o
FEAT_<entity>_<frequency>__TRAIN_VW (vista). Las vistas _VW se usan cuando se necesita aplicar
transformaciones adicionales o filtros sin modificar las tablas base. Cuando el Feature Store se conecte a


fuentes de producción, solo cambia la materialización upstream (cómo se llena la tabla). La interfaz mantiene
la misma estructura, garantizando cero cambios en los scripts de entrenamiento e inferencia.

### 10. 5 Nomenclatura de ML Experiment

#### Patrón

Los experimentos de ML (usados en hyperparameter search y model tracking) siguen una nomenclatura que

identifica el modelo, el tipo de búsqueda y la fecha:

EXP_<model>_<search_type>_<YYYYMMDD>

```
Dimensión ¿Qué describe? Códigos/Ejemplos
```
```
model Nombre del modelo (mismo que
en Model Registry)
```
##### UNIBOX_CUSTBPR_WEEKLY_FORECAST

```
search_type
Tipo de búsqueda de
hiperparámetros
```
##### RANDOM, BAYESIAN, GRID

```
YYYYMMDD Fecha de inicio del experimento 20260317
```
**Ejemplos:**

```
Nombre actual Nombre propuesto Descripció
n
```
```
hyperparameter_search_regressio
n_20260317
```
##### EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_RA

##### NDOM_20260317

```
Búsqueda
aleatoria de
hiperparám
etros
```
```
hyperparameter_search_bayesian
_20260317
```
##### EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BA

##### YESIAN_20260317

```
Búsqueda
bayesiana
de
hiperparám
etros
```
El sufijo de fecha permite múltiples ejecuciones del mismo experimento en diferentes días sin conflictos de
nombres. Los experimentos son inmutables una vez creados, por lo que cada nueva iteración requiere un

nuevo nombre.

### 10. 6 Nomenclatura de Tablas de Observabilidad

#### Patrón

```
OBS_<model>__<metric_type>
```

El doble underscore separa el nombre del modelo del tipo de métrica, manteniendo consistencia con la
convención de vistas del Feature Store.

### 10. 7 Vocabulary Registry

Tabla de referencia centralizada. Agregar nuevos códigos aquí antes de registrar cualquier modelo nuevo. Si

un modelo no encaja en las dimensiones existentes, se propone la extensión del vocabulario para aprobación
del equipo antes de registrarlo, para prevenir proliferación de nombres inconsistentes.

#### Targets

```
Código Nombre completo Descripción
```
```
UNIBOX Uni-box Unidades por caja
```
```
PROB Probability Probabilidad de evento binario
```
```
OOS Out-of-stock Indicador de desabasto
```
#### Entities

```
Código Nombre completo Descripción
```
```
CUSTBPR Customer × BPR Cliente cruzado con brand_pres_ret
```
```
CUSTPROD Customer × Product Cliente cruzado con producto
```
```
STORE Store Nivel tienda
```
```
SKU SKU Nivel SKU individual
```
#### Frequencies

```
Código Nombre completo
```
```
DAILY Daily (diario)
```
```
WEEKLY Weekly (semanal)
```
```
MONTHLY Monthly (mensual)
```
#### Methods

```
Código Nombre completo Descripción
```
```
FORECAST Forecast Pronóstico de series de tiempo
```
```
REGRESS Regression Regresión estadística / ML
```
```
CLASSIF Classification Clasificación binaria o multiclase
```
```
RANKING Ranking Modelos de recomendación
```
#### Search Types (tipo de búsqueda HPO)


```
Código Nombre completo Descripción
```
```
RANDOM Random Search Búsqueda aleatoria de hiperparámetros
```
```
BAYESIAN Bayesian Optimization Optimización bayesiana
```
```
GRID Grid Search Búsqueda exhaustiva en grilla
```
```
GENETIC Genetic Algorithm Algoritmo genético
```
#### Use Cases (cliente o caso de uso)

Para los tags de producción (por ejemplo, PRODUCTION_<USE_CASE>), se usa un token <USE_CASE>

que identifica de forma estable el cliente o caso de uso (no la versión del modelo).

**Reglas de formato:**

UPPER_SNAKE_CASE (solo A-Z, números y _) sin espacios ni caracteres especiales

**Ejemplos:**

##### • CLIENTA_DEFAULT

##### • CLIENTB_B2B

##### • PROMO_Q4

```
Importante: Si un nuevo modelo no encaja en las dimensiones existentes, primero se propone la extensión
del vocabulario para aprobación del equipo, y luego se registra. Esto previene proliferación de nombres
inconsistentes.
```

## 11. Inferencia Batch Particionada

##### ▶ POC

### 11 .1 Funcionamiento

Partitioned Model Inference es el complemento natural de MMT. Una vez registrado el modelo particionado en
el Registry, el motor de Snowflake resuelve automáticamente qué sub-modelo corresponde a cada registro en
función del valor de la columna STATS_NTILE_GROUP, sin necesidad de lógica condicional adicional ni
enrutador externo.

El modelo particionado encapsula los 16 sub-modelos de segmento como una sola unidad registrada en el
Model Registry, lo que simplifica su gestión: se versiona, se promueve entre ambientes y se audita como un
único objeto, independientemente del número de grupos.

#### Creación del Modelo Particionado

La construcción del modelo particionado sigue cinco pasos secuenciales:

```
1) Verificar modelos existentes: Se consulta el Model Registry para confirmar que los 16 sub-modelos
entrenados por MMT están disponibles y correctamente registrados, con sus versiones y métricas.
2) Cargar modelos por Registry: Se recuperan las referencias a cada sub-modelo desde el Registry,
resolviendo nombre, versión y columna de partición asociada (valor de STATS_NTILE_GROUP). Este
paso construye el mapeo entre cada segmento y su modelo correspondiente.
3) Definir el Modelo Particionado: Se define el wrapper del modelo particionado, especificando la
columna de partición (STATS_NTILE_GROUP) y la función de predicción que será invocada para cada
segmento.
4) Crear y Registrar el Modelo Particionado: Se crea el objeto particionado en Snowflake y se registra
en el Model Registry como una entidad unificada (UNIBOX_CUSTBPR_WEEKLY_FORECAST). En
este paso se asigna la versión y el alias PRODUCTION, que será el punto de referencia para los
procesos de inferencia y para la promoción entre ambientes.
5) Verificar Registry: Se valida que el modelo particionado quedó correctamente registrado, que la
columna de partición está bien configurada, y que es invocable desde SQL nativo antes de proceder a
la fase de inferencia.
```

### 11 .2 Flujo de Ejecución

El proceso de inferencia batch sigue este flujo:

1. Se carga el dataset de inferencia desde la fuente (FEAT_CUSTBPR_WEEKLY__INF_VW). Para la
    etapa de la POC se usa esta tabla, pero en Prod Cercano se usará el Feature Store para generar las
    tablas de entrenamiento e inferencia.
2. Se recupera la referencia al modelo particionado (por ejemplo, con el alias PRODUCTION) desde el
    Model Registry.


3. Se ejecuta la inferencia mediante sintaxis SQL nativa con la cláusula TABLE(MODEL(...)!PREDICT(...)
    OVER (PARTITION BY STATS_NTILE_GROUP)).
4. Los resultados se insertan en la tabla INFERENCE_PREDICTIONS con los campos: CUSTOMER_ID,
    STATS_NTILE_GROUP, WEEK, BRAND_PRES_RET, PROD_KEY, PREDICTED_UNI_BOX_WEEK,
    MODEL_VERSION y PREDICTION_TIMESTAMP.
5. Opcionalmente, si el volumen lo requiere, la inferencia puede ejecutarse en batches (BATCH_SIZE
    configurable) usando Snowpark order_by + limit/offset.

```
-- Sintaxis SQL nativa de inferencia particionada
INSERT INTO INFERENCE_PREDICTIONS (...)
SELECT p.CUSTOMER_ID, p.WEEK, p.predicted_uni_box_week, ...
FROM INFERENCE_INPUT t,
TABLE(
MODEL(BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST, TAG =>
'production_stable')
!PREDICT(t.CUSTOMER_ID, t.STATS_NTILE_GROUP, t.WEEK, ...)
OVER (PARTITION BY t.STATS_NTILE_GROUP)
) p
```
**Deployment en producción**

La estrategia de deployment está definida: Tags para gestión de versiones activas por proyecto, y
Export/Import de artefactos para la promoción entre ambientes DEV → QA → PROD. Ver sección 12
para el detalle completo.


## 12. ML Observability — Monitoreo de Datos y Modelos

### 12 .1 Estrategia de Monitoreo

##### ▶ POC

El monitoreo de ML en la POC se implementa mediante un enfoque completamente custom en Python/Snowpark
(notebooks 09a–09d), sin uso del módulo nativo de ML Observability de Snowflake. Esta decisión responde a
que el módulo nativo no cubre los requerimientos de granularidad por segmento (STATS_NTILE_GROUP) que
el modelo requiere. La estrategia se basa en tres pilares:

```
Tipo de Drift Qué detecta Métricas utilizadas
```
```
Data Drift Cambios en la distribución de
features de entrada entre
entrenamiento e inferencia.
```
```
Jensen-Shannon Distance (JSD) por
feature y segmento. Population
Stability Index (PSI) de composición
de segmentos.
```
```
Performance Drift Degradación de las métricas del
modelo respecto al baseline del set
de entrenamiento.
```
```
WAPE, RMSE, MAE, F1 Binary. Por
segmento y agregado ponderado
(full_model).
```
```
Prediction Drift Cambios en la distribución de
predicciones del modelo.
```
```
Jensen-Shannon Distance sobre la
distribución de
PREDICTED_UNI_BOX_WEEK, por
segmento.
```
Las métricas adicionales mencionadas en versiones anteriores del documento (KS, Wasserstein, Chi-square,
Cramér's V) **no están implementadas en la POC**. Su incorporación queda como punto de evolución para
producción cercana, según lo que Arca Continental priorice.

### 12 .2 Tablas de Soporte y Esquema

##### ▶ POC

El pipeline de observabilidad opera sobre nueve tablas físicas en SC_FEATURES_BMX (DEV) y
SC_FEATURES_BMX_PROD (PROD), siguiendo el patrón de nomenclatura OBS_<model>__<metric_type>
definido en la sección 10. 6. Tres son de baseline (read-only, generadas una vez en DEV y copiadas
explícitamente al promover el modelo) y seis son landing tables de producción con append incremental por
combinación de semana y versión de modelo.

```
Sección Nombre completo Descripción
```
```
Predicciones
OBS_PREDICTIONS Predicciones de producción con MODEL_NAME,
MODEL_VERSION y ENTITY_MAP
```
```
OBS_PREDICTIONS_VW Vista de predicciones con metadata
```
```
OBS_PREDICTIONS_BL Tabla de predicciones (baseline)
```
```
OBS_PREDICTIONS_BL_VW Vista de predicciones baseline
```
```
Data Drift
OBS_DATA_HIST Histogramas de features de producción (append
por semana/versión)
```
```
OBS_DATA_HIST_BL Histogramas de referencia de features (baseline)
```
```
OBS_DATA_DRIFT Métricas PSI y JSD por feature/segmento con
ALERT_LEVEL
```

```
Prediction Drift
OBS_PRED_HIST Histogramas de predicciones de producción
```
```
OBS_PRED_HIST_BL Histogramas de referencia de predicciones
(baseline)
```
```
OBS_PRED_DRIFT Métricas JSD de predicciones con ALERT_LEVEL
```
```
Performance
OBS_PERFORMANCE Métricas de performance vs baseline con
METRIC_DRIFT y ALERT_LEVEL
```
```
OBS_PERFORMANCE_BL Métricas de performance base por segmento y
agregado (WAPE, RMSE, MAE, F1)
```
El esquema de las tablas de histogramas incluye: RECORD_ID (SHA-256 de la combinación única),
MODEL_NAME, MODEL_VERSION, ENTITY_MAP como OBJECT con el contexto de la fila (feature_name,
week, data_date), AGGREGATED_COL y AGGREGATED_VALUE para la dimensión de segmentación,
METRIC_COL con el nombre de la métrica, METRIC_MAP como OBJECT con los datos del bin (bin_number,
bin_low, bin_high, bin_count), CALMONTH en formato YYYYMM y LDTS como timestamp de inserción.

Las tablas de métricas de drift extienden ese esquema con METRIC_VALUE (valor calculado),
METRIC_DRIFT (variación proporcional respecto al baseline, presente solo en __PERF),
WARNING_THRESHOLD, CRITICAL_THRESHOLD, ALERT_LEVEL (0/1/2) y BKCC como código de entidad
de negocio.

El RECORD_ID se genera con SHA-256 sobre la concatenación de MODEL_NAME, MODEL_VERSION,
WEEK, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL y, para JSD de features,
FEATURE_NAME. Esto garantiza idempotencia: insertar el mismo registro dos veces produce el mismo
RECORD_ID, lo que permite detectar duplicados sin queries adicionales.

### 12.3 Data Drift: PSI y Jensen-Shannon Distance

##### ▶ POC

El Data Drift detecta si los datos de entrada a inferencia han cambiado respecto al set de entrenamiento. El
notebook 09b implementa dos métricas complementarias con propósitos distintos.

#### PSI — Population Stability Index (estabilidad de composición de segmentos)

PSI mide si la distribución de entidades entre segmentos (STATS_NTILE_GROUP) ha cambiado respecto al
baseline. No analiza features individuales sino la composición relativa de los grupos: ¿sigue habiendo
aproximadamente el mismo porcentaje de clientes en cada cuartil de volumen?

El cálculo lee las proporciones de referencia desde OBS_DATA_HIST_BL, computa las proporciones del
dataset de inferencia actual, y aplica la fórmula:

```
PSI = Σ (P_inference_i − P_baseline_i) × ln(P_inference_i / P_baseline_i)
```
El PSI se calcula a nivel full_model (un valor por semana/versión, no por segmento individual). Los umbrales
son PSI > 0.1 para Warning y PSI > 0.2 para Critical. Los resultados se escriben en OBS_DATA_HIST.

Un PSI elevado puede indicar cambios estacionales, entrada masiva de nuevos clientes o reorganización
estructural de segmentos.


#### JSD — Jensen-Shannon Distance (drift por feature numérica individual)

JSD mide el drift en la distribución de cada feature numérica por separado, desagregado por segmento. A
diferencia del PSI, permite identificar qué variable específica está cambiando y en qué grupo concreto.

La implementación opera en dos fases. En la Fase 1 se construyen histogramas de producción: los valores de
cada feature se asignan a bins usando los mismos bin edges almacenados en OBS_DATA_HIST_BL (N_BINS
= 20 por feature y segmento). Los resultados se escriben en OBS_DATA_HIST. Reutilizar los bin edges del
baseline es una decisión de diseño deliberada: garantiza que siempre se compare contra la misma escala de
referencia, independientemente de cuándo se ejecute el monitoreo.

En la Fase 2 se calcula la distancia de Jensen-Shannon comparando la distribución del bin en producción (Q)
contra la del baseline (P):

##### M = (P + Q) / 2

```
JSD = sqrt( 0.5 × KL(P||M) + 0.5 × KL(Q||M) )
```
```
donde KL(P||M) = Σ P_i × ln(P_i / M_i)
```
Se aplica suavizado con EPSILON = 1e-10 para evitar log(0) en bins vacíos. El full outer join entre
histogramas de inferencia y baseline asegura que bins presentes solo en una distribución contribuyan
correctamente. Los resultados finales se escriben en OBS_DATA_HIST.

Los umbrales son JSD > 0.2 para Warning y JSD > 0.45 para Critical, calculados por segmento
(STATS_NTILE_GROUP y CUST_CATEGORY).

### 12.4 Prediction Drift

##### ▶ POC

El Prediction Drift detecta si la distribución de las predicciones del modelo (PREDICTED_UNI_BOX_WEEK)
ha cambiado respecto al baseline. Un drift en predicciones puede ocurrir incluso si los datos de entrada no
muestran drift significativo, por ejemplo ante cambios en estacionalidad o en patrones de comportamiento que
el modelo amplifica internamente.

El notebook 09c aplica el mismo mecanismo JSD de dos fases que el Data Drift pero sobre la columna de
predicción. En la Fase 1 se construyen histogramas de predicciones de producción usando los bin edges de
OBS_PREDICTIONS_HIST_BL, y los resultados se escriben en OBS_PREDICTIONS_HIST. En la Fase 2 se
calcula JSD comparando esa distribución contra el baseline, y los resultados finales se escriben en
OBS_PREDICTIONS_DRIFT.

Los umbrales son JSD > 0.2 para Warning y JSD > 0.45 para Critical, calculados por segmento
(STATS_NTILE_GROUP y CUST_CATEGORY).

### 12.5 Performance Drift

##### ▶ POC

El Performance Drift calcula métricas de calidad del modelo sobre datos de producción y las compara contra el
baseline para detectar degradación. El notebook 09d implementa cuatro métricas: WAPE, RMSE, MAE y F1
Binary.

#### Cálculo de métricas

El proceso une las predicciones desde OBS_PREDICTIONS con los valores reales de ACTUALS_TABLE_VW
mediante las claves CUSTOMER_ID, BRAND_PRES_RET y WEEK. La unión es left join desde predicciones:


los actuals faltantes se imputan como 0.0. El cálculo se realiza únicamente para semanas donde existen
actuals disponibles en ACTUALS_TABLE_VW; si una semana de inferencia no tiene actuals, la combinación
se omite automáticamente.

Las cuatro métricas se calculan en dos niveles de agregación simultáneamente: por segmento (agrupando por
STATS_NTILE_GROUP o CUST_CATEGORY) y a nivel full_model (sobre todos los grupos). El nivel
full_model usa suma ponderada por volumen, no promedio simple, para evitar que los segmentos de baja
rotación distorsionen la visión ejecutiva del comportamiento del modelo.

#### Cálculo del drift proporcional

El drift se expresa como variación proporcional relativa al baseline, leyendo los valores de referencia desde
OBS_PERFORMANCE_BL:

##### METRIC_DRIFT = (METRIC_VALUE − BASELINE_METRIC) / |BASELINE_METRIC|

El METRIC_DRIFT resultante queda almacenado en OBS_PERFORMANCE junto con METRIC_VALUE y

ALERT_LEVEL.

#### Umbrales y dirección del drift

```
Métrica Warning Critical Dirección
```
```
WAPE METRIC_DRIFT > +20% METRIC_DRIFT > +50% higher is worse
```
```
RMSE METRIC_DRIFT > +20% METRIC_DRIFT > +50% higher is worse
```
```
MAE METRIC_DRIFT > +20% METRIC_DRIFT > +50% higher is worse
```
```
F1 Binary METRIC_DRIFT < −15% METRIC_DRIFT < −30% lower is worse
```
Para F1, una caída por debajo de los umbrales negativos activa la alerta. Para las métricas de error, un
aumento por encima de los umbrales positivos activa la alerta.

### 12. 6 Idempotencia y Procesamiento Incremental

##### ▶ POC

Los cuatro notebooks (09a–09d) están diseñados para ser idempotentes y procesar únicamente
combinaciones nuevas. Esto los hace seguros para ejecutarse múltiples veces sin duplicar datos y eficientes
en coste de warehouse.

Al inicio de cada ejecución, el proceso consulta OBS_PREDICTIONS para obtener todas las combinaciones
(WEEK, MODEL_VERSION) presentes para el modelo. Luego consulta la landing table correspondiente para
identificar qué combinaciones ya fueron procesadas. La diferencia (left anti join) determina exactamente qué
hay que calcular. Si no hay combinaciones nuevas, el proceso se salta sin ejecutar ningún cálculo.

Este diseño tiene tres implicaciones operativas importantes. Primero, el notebook puede ejecutarse en
cualquier momento sin riesgo: si ya procesó la semana 202510 con la versión v_20260311_1430, no la
volverá a calcular aunque se ejecute de nuevo. Segundo, si una ejecución falla a mitad, la siguiente retoma
desde donde quedó: solo las combinaciones que no llegaron a escribirse en la landing table se recalculan.
Tercero, el procesamiento de nuevas semanas es automáticamente incremental: cada vez que llega una


nueva semana de inferencia, el notebook detecta que esa combinación no existe en la landing table y la
procesa.

### 12. 7 Visualización por Segmento y Agregada

##### ▶ POC

El módulo de Performance Drift produce resultados en dos niveles. A nivel de segmento
(STATS_NTILE_GROUP), las métricas se calculan individualmente para cada uno de los 16 grupos,
permitiendo identificar qué segmento específico está degradándose. A nivel agregado (full_model), las
métricas se consolidan mediante suma ponderada, no promedio simple, evitando que los grupos de baja
rotación distorsionen la visión ejecutiva.

Los dashboards de Snowsight se construyen directamente sobre las tablas OBS_DATA_HIST,
OBS_PRED_HIST y OBS_PERFORMANCE. La actualización de métricas depende de la frecuencia con que
se ejecute el notebook de monitoreo, que puede ser bajo demanda o periódico según lo que Arca Continental
defina.

### 12. 8 Configuración del Monitoreo

##### ▶ POC

El flujo de configuración y operación del monitoreo tiene cuatro fases:

- **Fase 1 — Setup de tablas (una vez, notebook 09a).** Se crean las nueve tablas de observabilidad
    con sus esquemas definidos. Esta fase no requiere repetirse salvo que se eliminen las tablas o se
    cambie el schema.
- **Fase 2 — Cálculo del baseline (una vez por versión de modelo, notebooks 06 y 07).** Se calculan
    los histogramas de referencia de features y predicciones, y las métricas de performance base, a partir
    del set de entrenamiento en DEV. Los resultados se almacenan en OBS_DATA_HIST_BL,
    OBS_PREDICTIONS_HIST_BL y OBS_PERFORMANCE_BL. Estas tres tablas deben copiarse
    explícitamente a producción al promover el modelo (ver sección 12.3), ya que no son artefactos del
    objeto modelo en el Registry y no se copian automáticamente con el CREATE MODEL.
- **Fase 3 — Cálculo incremental (por cada nueva semana de inferencia, notebooks 09b–09d).** El
    proceso detecta automáticamente qué combinaciones (WEEK, MODEL_VERSION) no han sido
    procesadas aún y calcula únicamente los nuevos registros. Los resultados se materializan en las
    tablas __DATA_DRIFT, __PRED_DRIFT y __PERF.
- **Fase 4 — Visualización.** Los dashboards de Snowsight se construyen sobre las tablas landing. La
    actualización de métricas depende de la frecuencia de ejecución de los notebooks. Solo requiere
    intervención manual ante cambio de umbrales o registro de una nueva versión de modelo.

```
Importancia de micropartitioning
```
```
El micropartitioning es el mecanismo de almacenamiento físico de Snowflake. Cuando se carga datos en
una tabla, Snowflake los divide automáticamente en particiones pequeñas e inmutables de entre 50 y 500
MB comprimidos, organizadas por orden de inserción y con metadatos de rango de valores por columna
almacenados por separado.
Esto es relevante para el monitoreo porque cuando el proceso Python/SQL consulta los logs de inferencia
filtrando por ventana temporal, grupo o versión de modelo, Snowflake usa esos metadatos para descartar
automáticamente las particiones que no contienen datos relevantes — sin necesidad de índices ni
particionamiento manual. El resultado es que los cálculos de drift y performance sobre subconjuntos
grandes de logs son significativamente más eficientes sin requerir configuración adicional.
```


## 13. Estrategia de Deployment y Promoción de Modelos

##### ▶ POC

Esta sección describe la estrategia completa de deployment: separación de ambientes, gestión de múltiples
versiones activas mediante Tags, y promoción de modelos entre ambientes mediante comandos SQL nativos
de Snowflake (CREATE MODEL ... WITH VERSION ... FROM MODEL ...) junto con la copia de tablas de soporte
para ML Observability.

### 13 .1 Separación por Ambientes

La estrategia define tres ambientes separados por base de datos en Snowflake, con roles de acceso
diferenciados:

```
Ambiente Base de Datos Propósito
```
```
DEV ML_DEV_DB Desarrollo, experimentación, entrenamiento y
validación inicial. Acceso a ML_DEV_ROLE.
```
```
QA / Staging ML_QA_DB Validación funcional y de calidad antes de
producción. Acceso a ML_OPS_ROLE.
```
```
PROD ML_PROD_DB Ambiente productivo. Solo ML_OPS_ROLE puede
realizar promociones. Los consumidores operan con
ML_CONSUMER_ROLE.
```
### 13 .2 Gestión de Versiones Activas mediante Tags

Arca Continental opera con un escenario donde un mismo modelo puede tener múltiples versiones activas
simultáneamente en producción, asociadas a distintos proyectos o grupos de consumo. La solución adoptada
es el uso de Tags en el Snowflake Model Registry, aplicados exclusivamente en BD_AA_PROD. En DEV el ciclo
de vida se gestiona mediante aliases (PRODUCTION, CHALLENGER), que no ejecutan inferencia y solo sirven
para identificar qué versión será promovida (ver sección 10.2).

Un Tag es una etiqueta con valor variable asignada directamente al objeto modelo en producción, manipulable
mediante SQL y con permisos controlables vía RBAC. El tag PRODUCTION_<USE_CASE> es el tag principal
que el notebook de inferencia (08_partitioned_inference_batch) lee para determinar qué versión ejecutar.
Adicionalmente, cada proyecto o grupo de consumo puede definir su propio tag, permitiendo que distintos
consumidores apunten a versiones diferentes del mismo modelo sin conflictos y sin modificar el código de
inferencia:

```
-- Tag principal de inferencia (asignado por 07_environment_change)
```
```
ALTER MODEL BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
PRODUCTION_<USE_CASE> = 'v_20260311_1430';
```
```
-- Tag de rollback disponible ante incidencias
```
```
ALTER MODEL BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
ROLLBACK_VERSION = 'v_20260301_0900';
```
```
-- Tags por proyecto o grupo de consumo (opcionales, gestionados por ML_OPS_ROLE)
```
```
ALTER MODEL BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
```

```
act_grupo1 = 'v_20260311_1430';
ALTER MODEL BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
act_grupo2 = 'v_20260301_0900';
```
```
-- Consultar versión activa en el proceso de inferencia
```
```
SET version = (SELECT SYSTEM$GET_TAG('PRODUCTION_<USE_CASE>',
'BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST', 'MODULE'));
```
Las ventajas de este mecanismo son el control de acceso granular (solo ML_OPS_ROLE puede modificar tags
en producción), trazabilidad completa en QUERY_HISTORY de cada cambio de versión activa, y cero cambios
de código en los procesos consumidores al actualizar una versión. El límite de 50 tags por modelo deja margen
amplio para los 16 grupos actuales con capacidad de crecimiento.

### 13 .3 Flujo de Promoción DEV → QA → PROD

Es importante señalar que en el contexto de la POC únicamente se han trabajado dos ambientes reales:
BD_AA_DEV para desarrollo y entrenamiento, y BD_AA_PROD para inferencia y monitoreo. El ambiente QA
descrito en esta sección es hipotético y se incluye con fines de entendimiento arquitectónico, para ilustrar cómo
debería verse un flujo de promoción completo y gobernado en un escenario productivo formal. La incorporación
de QA como ambiente real queda sujeta a la definición de la estrategia productiva con Arca Continental.

El modelo NO se reentrena en producción. Se entrena UNA VEZ en DEV, se valida en QA y se promueve a
BD_AA_PROD mediante el notebook 07_environment_change. La promoción involucra dos elementos: el
modelo particionado y las tablas de baseline requeridas por ML Observability.

#### Copia del Modelo Particionado entre Ambientes

El notebook resuelve primero el alias PRODUCTION en BD_AA_DEV para obtener el nombre exacto de la
versión a promover, sin necesidad de especificarla manualmente. Luego evalúa el estado del modelo en el
ambiente destino y actúa según corresponda:

- Si el modelo no existe en destino, lo crea con esa versión mediante CREATE MODEL ... WITH
    VERSION ... FROM MODEL.
- Si el modelo ya existe pero le falta esa versión específica, la añade mediante ALTER MODEL ADD
    VERSION sin afectar versiones previas.

Solo se copia el modelo particionado (UNIBOX_CUSTBPR_WEEKLY_FORECAST): este contiene internamente
los 16 sub-modelos de segmento y es el único artefacto público. Los sub-modelos individuales son artefactos
internos y no se promueven.

Una vez copiada la versión, el notebook asigna el tag PRODUCTION_<USE_CASE> apuntando a la versión
recién promovida. El notebook de inferencia (08_partitioned_inference_batch) lee este tag para determinar qué
versión ejecutar, sin cambios de código.

```
-- DEV → QA CREATE MODEL BD_AA_QA.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST WITH
VERSION v_20260311_1430 FROM MODEL BD_AA_DEV.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST
VERSION v_20260311_1430;
```
```
ALTER MODEL BD_AA_QA.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
PRODUCTION_<USE_CASE> = 'v_20260311_1430';
```

```
-- QA → PROD (mismo patrón) CREATE MODEL
BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST WITH VERSION v_20260311_1430
FROM MODEL BD_AA_QA.SC_MODELS_BMX.UNIBOX_CUSTBPR_WEEKLY_FORECAST VERSION v_20260311_1430;
```
```
ALTER MODEL BD_AA_PROD.SC_MODELS_BMX_PROD.UNIBOX_CUSTBPR_WEEKLY_FORECAST SET TAG
PRODUCTION_<USE_CASE> = 'v_20260311_1430';
```
#### Copia de Tablas de Soporte para ML Observability

Las tablas de baseline no son artefactos adjuntos al objeto modelo en el Registry y no se copian
automáticamente con el modelo. El notebook 07_environment_change las sincroniza en dos pasos: primero
crea las tablas en el schema destino si no existen mediante CREATE TABLE IF NOT EXISTS ... LIKE; luego
inserta únicamente las combinaciones (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL) que
existen en origen, pero faltan en destino usando un left anti join. Esto hace el proceso idempotente: ejecutarlo
múltiples veces no duplica datos.

Las tres tablas sincronizadas en cada promoción son:

- OBS_DATA_HIST_BL — histogramas de referencia de features. Sin esta tabla el proceso de
    monitoreo no puede calcular JSD ni PSI.
- OBS_PREDICTIONS_HIST_BL — histogramas de referencia de predicciones. Requerida para el
    cálculo de Prediction Drift.
- OBS_PERFORMANCE_BL — métricas de performance base (WAPE, RMSE, MAE, F1 por segmento
    y agregado). Sin esta tabla el cálculo de Performance Drift no puede determinar la variación
    proporcional respecto al estado original del modelo.

La sincronización se realiza una vez por promoción (DEV → QA y QA → PROD) ejecutando el notebook
07_environment_change con los schemas de origen y destino correspondientes. El proceso equivale a:

```
-- Crear tablas en destino si no existen (replica estructura)
CREATE TABLE IF NOT EXISTS BD_AA_QA.SC_FEATURES_BMX.OBS_DATA_HIST_BL
LIKE
BD_AA_DEV.SC_FEATURES_BMX.OBS_DATA_HIST_BL;
```
##### CREATE TABLE IF NOT EXISTS BD_AA_QA.SC_FEATURES_BMX.OBS_PREDICTIONS_HIST_BL

##### LIKE

```
BD_AA_DEV.SC_FEATURES_BMX.OBS_PREDICTIONS_HIST_BL;
```
##### CREATE TABLE IF NOT EXISTS BD_AA_QA.SC_FEATURES_BMX.OBS_PERFORMANCE_BL

##### LIKE

##### BD_AA_DEV.SC_FEATURES_BMX.OBS_PERFORMANCE_BL;

```
-- Insertar solo combinaciones faltantes (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL)
INSERT INTO
BD_AA_QA.SC_FEATURES_BMX.OBS_DATA_HIST_BL
SELECT src.* FROM BD_AA_DEV.SC_FEATURES_BMX.OBS_DATA_HIST_BL src
LEFT ANTI JOIN BD_AA_QA.SC_FEATURES_BMX.OBS_DATA_HIST_BL tgt
ON src.MODEL_NAME = tgt.MODEL_NAME
AND src.MODEL_VERSION = tgt.MODEL_VERSION
AND src.AGGREGATED_COL = tgt.AGGREGATED_COL;
```
```
-- Mismo patrón para __PRED_HIST_BL y __PERFORMANCE_BL
-- Al promover de QA a PROD se repite apuntando BD_AA_QA → BD_AA_PROD
```

#### Validación y Mecanismo de Promoción

La validación necesaria para aprobar cada promoción depende de lo que Arca Continental defina como criterios
de aceptación. El mecanismo de ejecución es flexible: el notebook puede ejecutarse manualmente por
ML_OPS_ROLE para la POC y producción cercana, o incorporarse como un nodo en un DAG de Dagster u otro
orquestador ejecutándose automáticamente tras la aprobación del gate de calidad.

El flujo de promoción completo es el siguiente:

```
1) Entrenamiento en DEV: HPO + MMT se ejecutan. Los modelos se registran en
BD_AA_DEV.SC_MODELS_BMX con versión, métricas y metadatos. Se asigna el alias PRODUCTION
a la versión lista.
2) Cálculo de baselines: los notebooks 06a–06d generan las tablas OBS_*__DATA_HIST_BL,
__PRED_HIST_BL y __PERFORMANCE_BL en DEV a partir del set de entrenamiento.
3) Validación en DEV: el equipo ML revisa métricas por grupo (RMSE, MAE, WAPE) como gate de calidad.
4) Promoción a QA: el notebook 07_environment_change se ejecuta apuntando de BD_AA_DEV a
BD_AA_QA. Copia el modelo resolviendo el alias PRODUCTION, sincroniza las tres tablas de baseline
y asigna el tag PRODUCTION_<USE_CASE> en QA.
5) Validación en QA: ejecución de inferencia funcional sobre datos de validación, verificación de
consistencia de predicciones y monitoreo. Gate de calidad final antes de promover a PROD.
6) Promoción a PROD: el notebook 07_environment_change se ejecuta nuevamente apuntando de
BD_AA_QA a BD_AA_PROD. Copia modelo, sincroniza baselines y asigna
PRODUCTION_<USE_CASE> en PROD.
7) Inferencia en PROD: el notebook 08_partitioned_inference_batch resuelve la versión activa leyendo
PRODUCTION_<USE_CASE> y ejecuta predicciones sobre FEAT_CUSTBPR_WEEKLY__INF_VW.
8) Monitoreo: los notebooks 09b–09d calculan drift y performance de forma incremental sobre las tablas
OBS_* en PROD.
```

### 13 .4 Control de Acceso RBAC

▶ **PROD. CERCANO**

El gobierno del ciclo de vida se implementa mediante cuatro roles diferenciados:


**Rol Permisos**

ML_DEV_ROLE Lectura/escritura en schemas DEV. Registro de modelos en Model Registry
de DEV. Sin acceso a QA ni PROD.

ML_OPS_ROLE Permisos completos para Export/Import de modelos y actualización de

```
Tags en todos los ambientes. OWNERSHIP en schemas de producción.
Único rol autorizado para modificar versiones activas en PROD.
```
ML_CONSUMER_ROLE Solo lectura y ejecución de modelos en producción (SELECT/USAGE

```
sobre funciones de inferencia). Sin permisos de modificación de modelos ni
Tags.
```
ML_MONITOR_ROLE Lectura de logs de inferencia, métricas y tablas de monitoreo. Sin permisos

```
de modificación sobre modelos o schemas de entrenamiento.
```

## 14. Orquestación del Pipeline

### 14 .1 Orquestación Manual

▶ **POC**

En la POC, el pipeline se ejecuta de forma manual mediante notebooks orquestados paso a paso. La
secuencia es la siguiente:

```
Paso Notebook Descripción Frecuencia
```
1. Validación
y limpieza de
datos

```
01_data_validation_and_cleaning
```
```
Verificación de estructura, calidad y
compatibilidad entre datasets de
entrenamiento e inferencia.
```
```
Una vez por
ciclo
```
##### 2.

```
Construcción
del Feature
Store
```
```
02_feature_store_setup
```
```
Cálculo de features (lags, rolling
aggregations) y materialización en
FEAT_CUSTBPR_WEEKLY. Vistas
__TRAIN_VW e __INF_VW listas para
consumo.
```
```
Una vez por
ciclo
```
3. HPO 03b_hyperparameter_search_bayesian

```
Búsqueda de hiperparámetros por grupo. El
03b implementa Bayesian Search con 15
trials sobre cluster de 5 nodos M.
```
```
Bajo demanda
ante
degradación o
cambio de
datos
```
4. MMT 04_many_model_training

```
Entrenamiento paralelo de los 16 sub-
modelos con hiperparámetros validados.
```
```
Una vez por
ciclo de
reentrenamiento
5.
Construcción
del Modelo
Particionado
```
```
05_create_partitioned_model
```
```
Verificar sub-modelos, definir particionado,
registrar
UNIBOX_CUSTBPR_WEEKLY_FORECAST
con alias PRODUCTION en BD_AA_DEV.
```
```
Una vez por
ciclo, tras MMT
```
##### 6.

```
Baselines de
observabilidad
```
```
06a_setup_baselines /
06b_data_drift_baseline /
06c_prediction_drift_baseline /
06d_performance_drift_baseline
```
```
Creación de tablas __DATA_HIST_BL,
__PRED_HIST_BL y
__PERFORMANCE_BL a partir del set de
entrenamiento en DEV.
```
```
Una vez por
versión de
modelo
```
7. Promoción
de baselines
drift entre
ambientes

```
07 a_copy_baselines
```
```
Copia de tablas de baseline de
observabilidad. Solo se ejecuta una vez por
versión de modelo, pero solo si el modelo
cuenta con monitoreo asociado.
```
```
Una vez por
versión de
modelo
```
8. Promoción
de modelos
entre
ambientes

```
07b_copy_models
```
```
Copia del modelo particionado de
BD_AA_DEV a BD_AA_PROD. Asignación
del tag PRODUCTION_<USE_CASE> en
BD_AA_PROD.
```
```
Una vez por
versión de
modelo
```
```
9. Inferencia
batch
```
```
08_partitioned_inference_batch
```
```
Predicciones sobre
FEAT_CUSTBPR_WEEKLY__INF_VW
consumiendo el modelo resuelto por tag
PRODUCTION_<USE_CASE>. Resultados
en __PRED.
```
```
Recurrente
(semanal)
```
```
10. Setup de
observabilidad
```
```
09a_setup_observability
```
```
Creación de las tablas OBS_* de landing.
Solo se ejecuta una vez por versión de
modelo o ante cambio de schema.
```
```
Una vez por
versión de
modelo
```

```
11. Data Drift 09b_data_drift
```
```
Cálculo incremental de PSI y JSD por
feature y segmento. Resultados en
__DATA_DRIFT.
```
```
Recurrente, tras
cada inferencia
```
```
12. Prediction
Drift
```
```
09c_prediction_drift
```
```
Cálculo incremental de JSD sobre
distribución de predicciones. Resultados en
__PRED_DRIFT.
```
```
Recurrente, tras
cada inferencia
```
```
13.
Performance
Drift
```
```
09d_performance_drift
```
```
Cálculo de WAPE, RMSE, MAE y F1 vs
baseline. Resultados en __PERF. Solo para
semanas con actuals disponibles.
```
```
Recurrente,
cuando hay
actualizaciones
```
### 14 .2 Orquestación Automatizada

▶ **PROD. CERCANO**

En producción cercano, el pipeline se automatiza mediante Snowflake Tasks, ML Jobs o un orquestador
externo (Dagster). El DAG de dependencias propuesto es:

```
1) Task/Job: Validación y limpieza de datos entrantes.
2) Task/Job dependiente: Actualización del Feature Store — pipeline batch incremental completo:
selección de PENDING, cálculo de features, INSERT en HIST, DROP/RECREATE de LATEST, cierre
de FEATURE_BATCH_RUNS.
3) Task/Job condicional: Re-entrenamiento (MMT) si se detecta drift significativo o se cumple la
periodicidad definida. Incluye construcción del modelo particionado y registro en Model Registry.
4) Task/Job de promoción: si el modelo supera el gate de calidad, copia automática o semi-automática
a QA/PROD via CREATE MODEL + copia de tablas de soporte, y actualización de Tags.
5) Task/Job programado: Inferencia batch sobre FEATURES_UNI_BOX_LATEST.
6) Task/Job de monitoreo: Generación de estadísticas de drift y performance, y disparo de alertas si se
superan umbrales definidos.
```

## 15. Comparativa: POC vs. Producción

La siguiente tabla diferencia tres horizontes: la POC ya implementada, la producción cercana (mejoras
inmediatas sin grandes cambios de arquitectura) y la producción lejana (evolución completa del sistema a
mediano/largo plazo):

```
Aspecto POC (actual) Prod. Cercano Prod. Lejano
```
```
Datos Históricos estáticos
curados por Arca
Continental
```
```
Mismos datos con proceso de
validación y limpieza
documentado
```
```
Actualización periódica
automatizada desde
fuentes origen
```
```
Inferencia Batch manual sobre
dataset de inferencia
curado
```
```
Batch periódico con trigger
definido, consumiendo
FEATURES_UNI_BOX_LATEST
```
```
Batch + Real-time (SPCS
con endpoints HTTPS)
```
```
Feature Store Tablas físicas estáticas,
cálculo manual de lags
mediante Window
Functions SQL
```
```
Batch incremental: HIST
(append-only) + LATEST
(DROP/RECREATE por lote),
trazabilidad por BATCH_ID
```
```
Tasks / ML Jobs nativos o
FeatureViews con
governance integrado
```
```
HPO Manual, puntual, una sola
ejecución
```
```
Bajo demanda ante degradación
de métricas detectada
```
```
Trigger automático por
drift
```
```
MMT —
Reentrenamiento
```
```
Manual, una vez por ciclo
de experimentación
```
```
Periódico (frecuencia por definir
con Arca Continental)
```
```
Automático con
aprobación basada en
umbrales de métricas
```
```
Modelo
Particionado
```
```
Construcción manual en 5
pasos: verificar modelos,
cargar por Registry,
definir particionado, crear
y registrar, verificar
Registry
```
```
Mismo proceso, incorporado en
script parametrizado ejecutado
por ML_OPS_ROLE
```
```
Construcción
automatizada como parte
del pipeline de
reentrenamiento
```
```
Orquestación Ejecución manual de
notebooks paso a paso
```
```
Orquestador externo (Dagster) o
scripts parametrizados
controlados por ML_OPS
```
```
Snowflake Tasks / ML
Jobs con DAG de
dependencias y logging
nativo
```
```
Deployment CREATE MODEL entre
ambientes + Tags por
proyecto, ejecución
manual con copia de
tablas de soporte para
Observability
```
```
Script parametrizado o nodo en
DAG; copia de baseline y
detalles de entrenamiento una
vez por versión
```
```
Pipeline CI/CD
automatizado con gates
de calidad por etapa
```
```
Escalamiento Warehouse X-
Small/Small, Compute
pool de 5 nodos M
```
```
Warehouse Medium/Large
Optimizado; ajuste manual
según carga. Compute pool de 5
nodos M o 2 nodos L de
acuerdo a la carga. Sugerencia
basada en análisis de
Snowflake.
```
```
Warehouse Large/X-
Large con multi-cluster y
auto-scaling dinámico
```
```
Versionado multi-
proyecto
```
```
Tags por proyecto en
Model Registry
(act_general, act_grupo1,
etc.)
```
```
Tags + copia nativa CREATE
MODEL con auditoría en
QUERY_HISTORY
```
```
Gobierno centralizado de
versiones con trazabilidad
completa
```

**Monitoreo** Monitoreo en Python/SQL
con Snowpark. Baseline
en tabla física generado
en DEV. Dashboards en
Snowsight con umbrales
Warning/Critical definidos.

```
Umbrales refinados con base en
la POC. Baseline y detalles de
entrenamiento copiados junto al
modelo en cada promoción.
```
```
Dashboards avanzados,
alertas proactivas y auto-
reentrenamiento por drift.
```
**Selección de
modelos**

```
Algoritmo fijo por
segmento (definido en
código, asignado por
HPO)
```
```
Selección validada
manualmente por el equipo ML.
Opcionalmente tambien se
puede implementar la propuesta
AutoML en una 1era versión.
```
```
Propuesta AutoML:
selección automática del
mejor modelo por
segmento
```

## 16. Consideraciones y Responsabilidades

### 16 .1 Responsabilidades de Arca Continental

1. Proveer accesos a Snowflake con los permisos necesarios para trabajar en el ambiente de desarrollo.
2. Participar en la transferencia de conocimiento sobre la funcionalidad actual del modelo en Databricks.
3. Entregar el dataset de entrenamiento y el de inferencia de forma curada y lista para usarse como
    fuente de la POC.
4. Acompañar en la fase de certificación de datos para la aprobación de la migración de los modelos
    creados.
5. Definir y validar los umbrales de aceptación de métricas (RMSE, WAPE) que determinarán la
    aprobación del modelo en cada ambiente.
6. Confirmar la estrategia de versionado multi-proyecto (aliases vs. Tags) y la frecuencia de
    reentrenamiento.

### 16 .2 Responsabilidades de Seidor Analytics

1. Desarrollar la migración técnica del modelo de Pronóstico de Ventas Semanales desde Databricks
    hacia Snowflake.
2. Implementar las mejores prácticas con las herramientas nativas de Snowflake para maximizar el
    aprovechamiento de la plataforma.
3. Entregar los modelos entrenados y las predicciones versionadas y certificadas para su aprobación.
4. Entregar documentación de desarrollo, certificación de datos y playbook de componentes
    desarrollados.
5. Definir y documentar formalmente la estrategia de deployment en producción una vez alineada con
    Arca Continental.


