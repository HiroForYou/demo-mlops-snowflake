const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    HeadingLevel, AlignmentType, LevelFormat, BorderStyle, WidthType,
    ShadingType, VerticalAlign, PageBreak, ImageRun, Header, Footer,
    NumberFormat, TabStopType, TabStopPosition
  } = require('docx');
  const fs = require('fs');
  
  // ─── helpers ───────────────────────────────────────────────────────────────
  const BLUE = "1F4E79";
  const LIGHT_BLUE = "2E75B6";
  const MEDIUM_BLUE = "D6E4F0";
  const LIGHT_GRAY = "F5F5F5";
  const DARK_GRAY = "595959";
  const WHITE = "FFFFFF";
  const CODE_BG = "2B2B2B";
  
  const border1 = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border1, bottom: border1, left: border1, right: border1 };
  const noBorders = {
    top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  };
  
  function h1(text) {
    return new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: BLUE })],
      spacing: { before: 360, after: 160 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LIGHT_BLUE, space: 1 } }
    });
  }
  
  function h2(text) {
    return new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: LIGHT_BLUE })],
      spacing: { before: 280, after: 120 }
    });
  }
  
  function h3(text) {
    return new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: DARK_GRAY })],
      spacing: { before: 200, after: 100 }
    });
  }
  
  function h4(text) {
    return new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: BLUE })],
      spacing: { before: 160, after: 80 }
    });
  }
  
  function para(text, opts = {}) {
    return new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, color: opts.color || "000000", bold: opts.bold || false, italics: opts.italic || false })],
      spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80 },
      alignment: opts.align || AlignmentType.JUSTIFIED
    });
  }
  
  function paraRuns(runs, opts = {}) {
    return new Paragraph({
      children: runs,
      spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80 },
      alignment: opts.align || AlignmentType.JUSTIFIED
    });
  }
  
  function run(text, opts = {}) {
    return new TextRun({ text, font: "Arial", size: 20, bold: opts.bold || false, italics: opts.italic || false, color: opts.color || "000000" });
  }
  
  function bullet(text, level = 0, opts = {}) {
    return new Paragraph({
      numbering: { reference: "bullets", level },
      children: [new TextRun({ text, font: "Arial", size: 20, bold: opts.bold || false, color: opts.color || "000000" })],
      spacing: { before: 40, after: 40 }
    });
  }
  
  function bulletRuns(runs, level = 0) {
    return new Paragraph({
      numbering: { reference: "bullets", level },
      children: runs,
      spacing: { before: 40, after: 40 }
    });
  }
  
  let _numGroupCounter = 0;
  let _numCurrentRef = null;
  function numberedGroup() {
    _numGroupCounter++;
    _numCurrentRef = "numbers_" + _numGroupCounter;
    return _numCurrentRef;
  }
  function numbered(text, level = 0, opts = {}) {
    if (!_numCurrentRef) { _numGroupCounter++; _numCurrentRef = "numbers_" + _numGroupCounter; }
    return new Paragraph({
      numbering: { reference: _numCurrentRef, level },
      children: [new TextRun({ text, font: "Arial", size: 20, bold: opts.bold || false })],
      spacing: { before: 40, after: 40 }
    });
  }
  function resetNum() { _numCurrentRef = null; }
  
  function pageBreak() {
    return new Paragraph({ children: [new PageBreak()] });
  }
  
  function spacer(n = 1) {
    return [...Array(n)].map(() => new Paragraph({ children: [new TextRun(" ")], spacing: { before: 0, after: 0 } }));
  }
  
  function infoBox(title, lines, bgColor = MEDIUM_BLUE) {
    const rows = [
      new TableRow({
        children: [
          new TableCell({
            borders: noBorders,
            shading: { fill: bgColor, type: ShadingType.CLEAR },
            margins: { top: 120, bottom: 120, left: 180, right: 180 },
            width: { size: 9360, type: WidthType.DXA },
            children: [
              new Paragraph({ children: [new TextRun({ text: title, font: "Arial", size: 20, bold: true, color: BLUE })], spacing: { after: 80 } }),
              ...lines.map(l => new Paragraph({ children: [new TextRun({ text: l, font: "Arial", size: 20, color: "333333" })], spacing: { after: 40 } }))
            ]
          })
        ]
      })
    ];
    return new Table({
      width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360], rows,
      borders: { top: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, bottom: border1, left: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, right: border1 }
    });
  }
  
  // Generic IO table (3-col: Tipo de Objeto | Nombre | Descripción)
  function ioTable(rows3) {
    const colW = [2000, 3000, 4360];
    const headerRow = new TableRow({
      children: ["Tipo de Objeto", "Nombre del Objeto", "Descripción"].map((h, i) =>
        new TableCell({
          borders,
          shading: { fill: BLUE, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          width: { size: colW[i], type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 19, bold: true, color: WHITE })] })]
        })
      )
    });
    const dataRows = rows3.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) =>
          new TableCell({
            borders,
            shading: { fill: ri % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 120, right: 120 },
            width: { size: colW[ci], type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 19, bold: ci === 0 })] })]
          })
        )
      })
    );
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: colW, rows: [headerRow, ...dataRows] });
  }
  
  // Two column table (label | value)
  function twoColTable(rows2, header1, header2, col1W = 3000, col2W = 6360) {
    const headerRow = new TableRow({
      children: [header1, header2].map((h, i) =>
        new TableCell({
          borders,
          shading: { fill: BLUE, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          width: { size: i === 0 ? col1W : col2W, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 19, bold: true, color: WHITE })] })]
        })
      )
    });
    const dataRows = rows2.map(([c1, c2], ri) =>
      new TableRow({
        children: [c1, c2].map((cell, ci) =>
          new TableCell({
            borders,
            shading: { fill: ri % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 120, right: 120 },
            width: { size: ci === 0 ? col1W : col2W, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 19, bold: ci === 0 })] })]
          })
        )
      })
    );
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [col1W, col2W], rows: [headerRow, ...dataRows] });
  }
  
  // Three column generic table
  function threeColTable(rows3, headers, colW = [2000, 3000, 4360]) {
    const headerRow = new TableRow({
      children: headers.map((h, i) =>
        new TableCell({
          borders,
          shading: { fill: BLUE, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          width: { size: colW[i], type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 19, bold: true, color: WHITE })] })]
        })
      )
    });
    const dataRows = rows3.map((row, ri) =>
      new TableRow({
        children: row.map((cell, ci) =>
          new TableCell({
            borders,
            shading: { fill: ri % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 120, right: 120 },
            width: { size: colW[ci], type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 19 })] })]
          })
        )
      })
    );
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: colW, rows: [headerRow, ...dataRows] });
  }
  
  // Generate enough numbered list references
  const numberedRefs = [...Array(40)].map((_, i) => ({
    reference: `numbers_${i + 1}`,
    levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
    ]
  }));
  
  // ─── document ──────────────────────────────────────────────────────────────
  const doc = new Document({
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [
            { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
            { level: 2, format: LevelFormat.BULLET, text: "\u25AA", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
          ]
        },
        ...numberedRefs
      ]
    },
    styles: {
      default: { document: { run: { font: "Arial", size: 20, color: "000000" } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Arial", color: BLUE }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: "Arial", color: LIGHT_BLUE }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, font: "Arial", color: DARK_GRAY }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1200, right: 1200, bottom: 1200, left: 1200 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            children: [
              new TextRun({ text: "KT — Migración del Modelo de Pronóstico de Ventas: Databricks → Snowflake", font: "Arial", size: 16, color: DARK_GRAY }),
              new TextRun({ text: "   |   Versión 1.0   |   Confidencial", font: "Arial", size: 16, color: DARK_GRAY, italics: true })
            ],
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE, space: 1 } },
            spacing: { after: 80 }
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            children: [
              new TextRun({ text: "Seidor Analytics  ·  Documento Confidencial", font: "Arial", size: 16, color: DARK_GRAY }),
            ],
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE, space: 1 } },
            spacing: { before: 80 }
          })]
        })
      },
      children: [
  
        // ══════════════════════════════════════════════════
        // PORTADA
        // ══════════════════════════════════════════════════
        new Paragraph({
          children: [new TextRun({ text: "KNOWLEDGE TRANSFER", font: "Arial", size: 48, bold: true, color: BLUE, allCaps: true })],
          alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }
        }),
        new Paragraph({
          children: [new TextRun({ text: "Migración del Modelo de Pronóstico de Ventas", font: "Arial", size: 32, bold: true, color: LIGHT_BLUE })],
          alignment: AlignmentType.CENTER, spacing: { after: 60 }
        }),
        new Paragraph({
          children: [new TextRun({ text: "Databricks → Snowflake | MLOps End-to-End", font: "Arial", size: 28, color: DARK_GRAY })],
          alignment: AlignmentType.CENTER, spacing: { after: 400 }
        }),
        new Paragraph({
          children: [new TextRun({ text: "Versión 1.0  ·  Marzo 2025  ·  Confidencial", font: "Arial", size: 20, color: DARK_GRAY, italics: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 600 }
        }),
  
        // Tabla de aprobación
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 2340, 2340, 2340],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, columnSpan: 4, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "KNOWLEDGE TRANSFER — APROBACIÓN", font: "Arial", size: 20, bold: true, color: WHITE, allCaps: true })] })] }),
            ]}),
            new TableRow({ children: ["Documento", "Nombre", "Fecha de aprobación", "Firma"].map(h => new TableCell({ borders, shading: { fill: MEDIUM_BLUE, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 18, bold: true })] })] })) }),
            ...[1, 2, 3].map(() => new TableRow({ children: ["", "", "", ""].map(() => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: " ", font: "Arial", size: 18 })] })] })) }))
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 1. INTRODUCCIÓN
        // ══════════════════════════════════════════════════
        h1("1. Introducción"),
        para("Este documento presenta la implementación integral de un pipeline de Machine Learning desarrollado como Prueba de Concepto (POC) para la migración de Databricks a Snowflake."),
        para("El desarrollo cubre el ciclo completo de MLOps para un caso de uso de pronóstico de ventas, utilizando capacidades nativas de Snowflake y un enfoque modular basado en notebooks desacoplados."),
        para("La arquitectura fue diseñada para ser ejecutada mediante un orquestador externo, priorizando control operacional, trazabilidad y escalabilidad, sin depender de automatizaciones internas como Tasks o Dynamic Tables."),
        ...spacer(1),
        paraRuns([
          run("Feature Engineering: ", { bold: true }),
          run("Se implementó un proceso estructurado de construcción de características a partir de datos históricos limpios. El enfoque se basa en la materialización explícita de features en tablas controladas y versionadas, permitiendo trazabilidad completa sobre qué snapshot de datos fue utilizado en cada entrenamiento. Este diseño garantiza reproducibilidad, auditoría y facilidad de integración con procesos de promoción entre ambientes."),
        ]),
        paraRuns([
          run("Training: ", { bold: true }),
          run("Se desarrolló el flujo completo de entrenamiento de modelos, incluyendo búsqueda de hiperparámetros, entrenamiento segmentado y registro formal en Snowflake Model Registry. La arquitectura contempla un modelo por segmento de negocio y posteriormente un modelo particionado que consolida todos los submodelos en un único artefacto productivo. El proceso permite escalabilidad, versionado controlado y capacidad de rollback por modelo o por grupo."),
        ]),
        paraRuns([
          run("Inference: ", { bold: true }),
          run("Se implementó un proceso de inferencia batch completamente ejecutado dentro de Snowflake. En ambiente de desarrollo, el sistema consume el modelo particionado desde el Model Registry mediante alias PRODUCTION. En ambientes productivos (QA/PROD), se utilizan tags personalizados (PRODUCTION_<USE_CASE>) que permiten gestionar múltiples casos de uso de forma independiente."),
        ]),
        paraRuns([
          run("Monitoring: ", { bold: true }),
          run("La arquitectura incluye un sistema completo de observabilidad MLOps que monitorea tres dimensiones críticas:"),
        ]),
        bullet("Data Drift: Detecta cambios en la distribución de las features de entrada mediante histogramas y divergencia KL"),
        bullet("Prediction Drift: Monitorea cambios en la distribución de predicciones del modelo"),
        bullet("Performance Drift: Evalúa degradación del desempeño mediante métricas comparadas contra baseline"),
        ...spacer(1),
        para("El sistema genera alertas automáticas (WARNING/CRITICAL) basadas en thresholds configurables y mantiene trazabilidad histórica completa de todas las métricas."),
        ...spacer(1),
        para("El desarrollo prioriza:", { bold: false }),
        bullet("Modularidad por etapas (cada notebook representa una capa del pipeline)"),
        bullet("Trazabilidad integral (features, hiperparámetros y modelos)"),
        bullet("Reproducibilidad mediante tablas materializadas y artefactos versionados"),
        bullet("Escalabilidad usando capacidades nativas de Snowflake ML"),
        bullet("Compatibilidad con orquestación externa empresarial"),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 2. ARQUITECTURA END-TO-END
        // ══════════════════════════════════════════════════
        h1("2. Arquitectura End-to-End"),
        para("El pipeline MLOps implementado se estructura en siete etapas principales, cada una ejecutada mediante scripts desacoplados diseñados para ser orquestados externamente:"),
        ...spacer(1),
  
        h2("Data Preparation"),
        bulletRuns([
          run("01_data_validation_and_cleaning — ", { bold: true }),
          run("Valida los datasets estructurados, aplica limpieza de datos (manejo de NULLs, filtrado opcional de outliers mediante umbral P99), realiza split temporal por grupo para generar conjuntos de entrenamiento y holdout, y crea tablas limpias versionadas para entrenamiento e inferencia."),
        ]),
  
        h2("Feature Engineering"),
        bulletRuns([
          run("02_feature_store_setup — ", { bold: true }),
          run("Construye el dataset de features y lo materializa en una tabla controlada (sin Feature Views / sin Dynamic Tables), registrando metadatos de versión para trazabilidad. La Feature Store se organiza por entidad y frecuencia temporal (FEAT_CUSTBPR_WEEKLY), no por modelo específico."),
        ]),
  
        h2("Model Training"),
        bulletRuns([
          run("03b_hyperparameter_search_bayesian — ", { bold: true }),
          run("Ejecuta búsqueda de hiperparámetros por grupo utilizando optimización bayesiana (BayesOpt) para convergencia más eficiente. Cada uno de los 16 grupos (STATS_NTILE_GROUP) se optimiza de forma independiente. Los resultados se almacenan en ML Experiments (fuente primaria) y tabla de respaldo HPO_{MODEL_NAME}."),
        ]),
        bulletRuns([
          run("04_many_model_training — ", { bold: true }),
          run("Entrena un modelo por segmento (16 grupos esperados) mediante Many Model Training (MMT). Utiliza los mejores hiperparámetros obtenidos en la etapa anterior, registra cada modelo individual en Snowflake Model Registry con métricas completas (RMSE, MAE, WAPE, MAPE) y asigna alias PRODUCTION a las versiones entrenadas."),
        ]),
        bulletRuns([
          run("05_create_partitioned_model — ", { bold: true }),
          run("Construye y registra un modelo particionado (PartitionedModel) que encapsula los 16 submodelos entrenados y enruta inferencias automáticamente al modelo correcto según el valor de STATS_NTILE_GROUP. Este artefacto unificado se registra en Model Registry como UNIBOX_CUSTBPR_WEEKLY_FORECAST."),
        ]),
  
        h2("Baseline Generation (Training Data)"),
        bulletRuns([run("06a_setup_baselines — ", { bold: true }), run("Inicializa la infraestructura de observabilidad, crea tablas de baseline, ejecuta inferencia batch sobre datos de holdout usando el modelo PRODUCTION y almacena predicciones de referencia en OBS_PREDICTIONS_BL.")]),
        bulletRuns([run("06b_data_drift_baseline — ", { bold: true }), run("Genera histogramas de referencia para todas las features del modelo utilizando el conjunto de holdout. Los histogramas se calculan por segmentos (STATS_NTILE_GROUP, CUST_CATEGORY) y se almacenan en OBS_DATA_HIST_BL.")]),
        bulletRuns([run("06c_prediction_drift_baseline — ", { bold: true }), run("Genera histogramas de referencia para la distribución de predicciones del modelo sobre el conjunto de holdout. Se almacenan en OBS_PRED_HIST_BL para posterior comparación.")]),
        bulletRuns([run("06d_performance_drift_baseline — ", { bold: true }), run("Calcula métricas de desempeño de referencia (WAPE, RMSE, MAE, F1_BINARY) sobre el conjunto de holdout. Las métricas base se almacenan en OBS_PERFORMANCE_BL por segmento.")]),
  
        h2("Environment Promotion"),
        bulletRuns([run("07a_copy_baselines — ", { bold: true }), run("Copia las tablas de baseline (histogramas de data drift, prediction drift y performance) desde el ambiente de desarrollo hacia el ambiente objetivo (QA/PROD). Sincroniza únicamente las combinaciones (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL) faltantes.")]),
        bulletRuns([run("07b_copy_models — ", { bold: true }), run("Copia los modelos registrados desde el Model Registry de desarrollo hacia el Model Registry del ambiente objetivo. Aplica tags de producción (PRODUCTION_<USE_CASE>) y rollback (ROLLBACK_VERSION_<USE_CASE>) para control de versiones operativas.")]),
  
        h2("Inference (Production Data)"),
        bulletRuns([run("08_partitioned_inference_batch — ", { bold: true }), run("Ejecuta inferencia batch sobre el dataset de inferencia utilizando el modelo particionado identificado mediante tags de producción (PRODUCTION_<USE_CASE>). Procesa datos por semana (WEEK), detecta combinaciones (versión, semana) faltantes y ejecuta MODEL()!PREDICT particionado por STATS_NTILE_GROUP.")]),
  
        h2("ML Observability (Production Data)"),
        bulletRuns([run("09a_setup_observability — ", { bold: true }), run("Inicializa la infraestructura de monitoreo para producción. Crea tablas de landing para histogramas de drift y métricas de desempeño.")]),
        bulletRuns([run("09b_data_drift — ", { bold: true }), run("Calcula histogramas de distribución de features sobre los datos de inferencia más recientes y compara contra baseline mediante divergencia de Kullback-Leibler (KL).")]),
        bulletRuns([run("09c_prediction_drift — ", { bold: true }), run("Calcula histogramas de distribución de predicciones sobre inferencia reciente y compara contra baseline mediante KL divergence.")]),
        bulletRuns([run("09d_performance_drift — ", { bold: true }), run("Calcula métricas de desempeño (WAPE, RMSE, MAE, F1_BINARY) sobre predicciones recientes, compara contra baseline y detecta degradación de desempeño.")]),
  
        h2("Alerting & Notifications"),
        bulletRuns([run("10_alertas — ", { bold: true }), run("Consolida las alertas generadas por las tablas de observabilidad (performance, data drift, prediction drift) para un modelo dado y construye un reporte unificado listo para envío por email (HTML) y/o publicación en cola (JSON).")]),
  
        ...spacer(1),
        infoBox("Resultado final del pipeline", [
          "Datos limpios → Features versionadas → HPO → Entrenamiento segmentado (MMT) → Modelo particionado en registry",
          "→ Baseline generation → Promoción entre ambientes → Inferencia batch productiva con trazabilidad completa",
          "→ Monitoreo de drift (data / prediction / performance) → Consolidación y publicación de alertas"
        ]),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 3. CONVENCIONES Y CONFIGURACIÓN
        // ══════════════════════════════════════════════════
        h1("3. Convenciones y Configuración Implementada"),
  
        h2("3.1 Base de Datos / Esquemas"),
        para("La arquitectura utiliza una convención de esquemas por propósito funcional:"),
        ...spacer(1),
        h3("Ambiente de Desarrollo (DEV)"),
        bullet("DATABASE = \"BD_AA_DEV\""),
        bullet("STORAGE_SCHEMA = \"SC_STORAGE_BMX_PS\" (datasets estructurados crudos)"),
        bullet("FEATURES_SCHEMA = \"SC_FEATURES_BMX\" (features materializadas, datasets limpios, tablas de observabilidad)"),
        bullet("MODELS_SCHEMA = \"SC_MODELS_BMX\" (Model Registry, resultados de HPO)"),
        ...spacer(1),
        h3("Promoción a Ambientes Superiores (QA/PROD)"),
        bullet("Los esquemas pueden variar según configuración del ambiente destino"),
        bullet("Los scripts 07a/07b manejan la sincronización de artefactos entre ambientes"),
        bullet("Se mantiene separación entre datos de desarrollo y producción"),
  
        ...spacer(1),
        h2("3.2 Nomenclatura de Objetos"),
        h3("Feature Store"),
        bullet("Nombre base: FEAT_CUSTBPR_WEEKLY (organizado por entidad + frecuencia temporal)"),
        bullet("Sufijos por propósito:"),
        bullet("__TRAIN: Dataset de entrenamiento limpio (~90% temporal)", 1),
        bullet("__HOLDOUT: Dataset de holdout para baseline (~10% temporal)", 1),
        bullet("__INF: Dataset de inferencia limpio", 1),
        bullet("__HOLDOUT_VW / __INF_VW: Vistas con enriquecimiento (categorías de clientes)", 1),
        ...spacer(1),
        h3("Modelo"),
        bullet("Nombre base: UNIBOX_CUSTBPR_WEEKLY_FORECAST"),
        bullet("Modelos por segmento: {model_name}__{stats_ntile_group} (ej: unibox_custbpr_weekly_forecast__group_stat_0_1)"),
        bullet("Modelo particionado: UNIBOX_CUSTBPR_WEEKLY_FORECAST (consolidado)"),
        ...spacer(1),
        h3("Tablas de Observabilidad"),
        bullet("Baseline (DEV): OBS_*_BL (ej: OBS_PREDICTIONS_BL, OBS_DATA_HIST_BL, OBS_PERFORMANCE_BL)"),
        bullet("Producción (QA/PROD): OBS_* (ej: OBS_PREDICTIONS, OBS_DATA_HIST, OBS_DATA_DRIFT)"),
  
        ...spacer(1),
        h2("3.3 Gestión de Versiones del Modelo"),
        h3("Ambiente de Desarrollo (DEV)"),
        bullet("Se utiliza el alias PRODUCTION para identificar la versión activa del modelo"),
        bullet("Sintaxis: registry.get_model(MODEL_NAME).version(\"PRODUCTION\")"),
        bullet("El alias se asigna automáticamente en el script 04 (Many Model Training)"),
        bullet("Simple y directo para un único caso de uso por ambiente"),
        ...spacer(1),
        h3("Ambientes Productivos (QA/PROD)"),
        bullet("Se utilizan tags personalizados para gestionar múltiples casos de uso de forma independiente"),
        bullet("Nomenclatura: PRODUCTION_{USE_CASE} (ej: PRODUCTION_CLIENTA_DEFAULT)"),
        bullet("Tag de rollback: ROLLBACK_VERSION_{USE_CASE}"),
        bullet("Permite múltiples versiones productivas coexistiendo para diferentes clientes/casos de uso"),
        bullet("Los tags se aplican en el script 07b (Copy Models) durante la promoción"),
  
        ...spacer(1),
        h2("3.4 Tablas Principales"),
        h3("Datasets Estructurados (Entrada)"),
        threeColTable([
          ["TRAIN_DATASET_STRUCTURED", "Dataset crudo de entrenamiento", "SC_STORAGE_BMX_PS"],
          ["INFERENCE_DATASET_STRUCTURED", "Dataset crudo de inferencia", "SC_STORAGE_BMX_PS"],
          ["GROUND_TRUTH_DATASET_STRUCTURED", "Valores reales para evaluación", "SC_STORAGE_BMX_PS"],
        ], ["Tabla", "Propósito", "Schema"], [3000, 4000, 2360]),
        ...spacer(1),
        h3("Feature Store (Limpieza + Transformación)"),
        threeColTable([
          ["FEAT_CUSTBPR_WEEKLY__TRAIN", "Features + labels de entrenamiento (~90% temporal)", "SC_FEATURES_BMX"],
          ["FEAT_CUSTBPR_WEEKLY__HOLDOUT", "Features + labels de holdout para baseline (~10% temporal)", "SC_FEATURES_BMX"],
          ["FEAT_CUSTBPR_WEEKLY__INF", "Features de inferencia limpias", "SC_FEATURES_BMX"],
          ["FEAT_CUSTBPR_WEEKLY", "Features materializadas completas (sin target)", "SC_FEATURES_BMX"],
        ], ["Tabla", "Propósito", "Schema"], [3000, 4000, 2360]),
        ...spacer(1),
        h3("Hiperparámetros y Entrenamiento"),
        threeColTable([
          ["HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST", "Mejores hiperparámetros por grupo (respaldo)", "SC_MODELS_BMX"],
          ["ML Experiments", "Runs de HPO con métricas (fuente primaria)", "Sistema Snowflake"],
          ["MMT_MODELS (Stage)", "Artefactos de Many Model Training", "SC_MODELS_BMX"],
        ], ["Tabla", "Propósito", "Schema"], [3000, 4000, 2360]),
        ...spacer(1),
        h3("Model Registry"),
        threeColTable([
          ["unibox_custbpr_weekly_forecast__group_stat_*", "16 modelos individuales por segmento", "SC_MODELS_BMX"],
          ["UNIBOX_CUSTBPR_WEEKLY_FORECAST", "Modelo particionado consolidado", "SC_MODELS_BMX"],
        ], ["Modelo", "Descripción", "Schema"], [3500, 3500, 2360]),
        ...spacer(1),
        h3("Tablas de Baseline (Scripts 06a-d)"),
        threeColTable([
          ["OBS_PREDICTIONS_BL", "Predicciones baseline sobre holdout", "SC_FEATURES_BMX"],
          ["OBS_DATA_HIST_BL", "Histogramas de features baseline", "SC_FEATURES_BMX"],
          ["OBS_PRED_HIST_BL", "Histogramas de predicciones baseline", "SC_FEATURES_BMX"],
          ["OBS_PERFORMANCE_BL", "Métricas de desempeño baseline", "SC_FEATURES_BMX"],
        ], ["Tabla", "Propósito", "Schema"], [3000, 4000, 2360]),
        ...spacer(1),
        h3("Tablas de Observabilidad (Producción, Scripts 09a-d)"),
        threeColTable([
          ["OBS_PREDICTIONS", "Predicciones de producción con metadatos", "SC_FEATURES_BMX"],
          ["OBS_DATA_HIST", "Histogramas de features en producción", "SC_FEATURES_BMX"],
          ["OBS_DATA_DRIFT", "Métricas de data drift con alertas (KL divergence)", "SC_FEATURES_BMX"],
          ["OBS_PRED_HIST", "Histogramas de predicciones en producción", "SC_FEATURES_BMX"],
          ["OBS_PRED_DRIFT", "Métricas de prediction drift con alertas", "SC_FEATURES_BMX"],
          ["OBS_PERFORMANCE", "Métricas de desempeño en producción con alertas", "SC_FEATURES_BMX"],
        ], ["Tabla", "Propósito", "Schema"], [3000, 4000, 2360]),
  
        ...spacer(1),
        h2("3.5 Parámetros Constantes"),
        ioTable([
          ["Parámetro", "DATABASE", "Base de datos utilizada para el proceso (BD_AA_DEV)"],
          ["Parámetro", "STORAGE_SCHEMA", "Schema de datasets estructurados (SC_STORAGE_BMX_PS)"],
          ["Parámetro", "FEATURES_SCHEMA", "Schema de Feature Store y observabilidad (SC_FEATURES_BMX)"],
          ["Parámetro", "MODELS_SCHEMA", "Schema de Model Registry y HPO (SC_MODELS_BMX)"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo (UNIBOX_CUSTBPR_WEEKLY_FORECAST)"],
          ["Parámetro", "FEATURE_STORE_NAME", "Nombre de Feature Store (FEAT_CUSTBPR_WEEKLY)"],
          ["Parámetro", "TARGET_COLUMN", "Variable objetivo (UNI_BOX_WEEK)"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna de segmentación (STATS_NTILE_GROUP)"],
          ["Parámetro", "USE_CASE", "Token de caso de uso para tags en PROD (ej: CLIENTA_DEFAULT)"],
        ]),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 4. DATA PREPARATION
        // ══════════════════════════════════════════════════
        h1("4. Data Preparation"),
        h2("4.1 Notebook: 01_data_validation_and_cleaning"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook tiene como objetivo validar y preparar los datasets iniciales que serán utilizados en el pipeline de entrenamiento e inferencia del modelo de Machine Learning."),
        para("El proceso realiza validaciones estructurales del dataset, limpieza de datos y generación de datasets derivados: dataset limpio de entrenamiento, dataset de inferencia limpio y conjunto de validación holdout para monitoreo de drift."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "TARGET_COLUMN", "Variable objetivo utilizada para entrenamiento del modelo"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna utilizada para segmentación de los datos"],
          ["Parámetro", "HOLDOUT_FRACTION", "Porcentaje del dataset reservado para validación holdout"],
          ["Parámetro", "EXCLUDED_COLS", "Columnas que no serán consideradas como features del modelo"],
          ["Tabla", "TRAIN_DATASET_STRUCTURED", "Dataset estructurado inicial utilizado para entrenamiento"],
          ["Tabla", "INFERENCE_DATASET_STRUCTURED", "Dataset estructurado utilizado para inferencia"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__TRAIN", "Dataset de entrenamiento limpio (~90% temporal) con nomenclatura basada en Feature Store"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__HOLDOUT", "Subconjunto de validación (~10% temporal) utilizado para monitoreo de drift"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__INF", "Dataset de inferencia limpio utilizado para generar predicciones"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark utilizando la función get_active_session."),
        numbered("Se configuran los parámetros de entorno como base de datos, esquema y variable objetivo."),
        numbered("Se consulta la tabla TRAIN_DATASET_STRUCTURED para validar la existencia del dataset y verificar su estructura."),
        numbered("Se valida la presencia de la variable objetivo definida en TARGET_COLUMN."),
        numbered("Se consulta el dataset de inferencia desde INFERENCE_DATASET_STRUCTURED."),
        numbered("Se identifican las columnas de metadata que no deben utilizarse como features del modelo."),
        numbered("Se aplica un proceso de limpieza de datos:"),
        bullet("Filtrado de valores NULL en columnas críticas (TARGET_COLUMN, CUSTOMER_ID, WEEK, STATS_NTILE_GROUP)", 1),
        bullet("Filtrado de valores negativos en la variable objetivo (TARGET_COLUMN >= 0)", 1),
        bullet("Aplicación opcional de filtro de outliers mediante umbral P99 (cuando APPLY_OUTLIER_FILTER_P99=True)", 1),
        numbered("Se divide el dataset de entrenamiento en dos subconjuntos mediante split temporal por grupo:"),
        bullet("FEAT_CUSTBPR_WEEKLY__TRAIN: contiene las semanas más antiguas (aproximadamente 90% temporal)", 1),
        bullet("FEAT_CUSTBPR_WEEKLY__HOLDOUT: contiene las semanas más recientes (aproximadamente 10% temporal)", 1),
        numbered("El split se calcula por STATS_NTILE_GROUP usando cutoff_week específico por grupo."),
        numbered("Los datasets generados se almacenan en: FEAT_CUSTBPR_WEEKLY__TRAIN, FEAT_CUSTBPR_WEEKLY__HOLDOUT, FEAT_CUSTBPR_WEEKLY__INF"),
  
        ...spacer(1),
        h3("Notas Técnicas Importantes"),
        h4("Outlier Handling — P99 Threshold (Justification)"),
        para("El pipeline utiliza una estrategia robusta de limpieza de etiquetas basada en un umbral P99 del target. Esta estrategia es opcional y se controla mediante el parámetro APPLY_OUTLIER_FILTER_P99 (default: False)."),
        para("Cuando APPLY_OUTLIER_FILTER_P99=True, el proceso garantiza que no haya filtración de información (label leakage) del conjunto de holdout hacia el conjunto de entrenamiento:", { bold: false }),
        resetNum(), numbered("Primero se calcula el cutoff_week temporal por grupo (basado en la distribución de registros por semana)"),
        numbered("Después se computa el umbral P99 utilizando únicamente los registros del conjunto TRAIN (WEEK <= cutoff_week)"),
        numbered("El umbral P99 se estima sin utilizar información del periodo holdout (WEEK > cutoff_week)"),
        numbered("Una vez fijado el umbral, se aplica el mismo filtro consistentemente a ambos conjuntos para eliminar valores extremos"),
        ...spacer(1),
        para("Justificación del umbral P99:", { bold: true }),
        bullet("Valores por encima del percentil 99 son considerados outliers extremos que pueden distorsionar el aprendizaje del modelo"),
        bullet("El filtrado se aplica globalmente (no por grupo) para mantener consistencia entre segmentos"),
        bullet("Cuando está desactivado, solo se aplican filtros básicos (NULL, negativos)"),
  
        ...spacer(1),
        h4("Audit Notes: Temporal Split + P99 Label Cleaning"),
        paraRuns([run("¿Por qué el split no es exactamente 10%? ", { bold: true }), run("El split implementado es temporal por grupo y se calcula utilizando periodos completos de WEEK. TRAIN utiliza WEEK <= cutoff_week y HOLDOUT utiliza WEEK > cutoff_week. El cutoff_week se define como la primera semana donde el share acumulativo alcanza o excede el umbral (1 - HOLDOUT_FRACTION). Dado que los volúmenes se agregan por semanas completas, el porcentaje resultante puede variar ligeramente del 10% configurado (por ejemplo, 9.23%). Esta variación es esperada y preserva la integridad del split temporal.")]),
        ...spacer(1),
        para("Riesgo de Label Leakage cuando APPLY_OUTLIER_FILTER_P99=True:"),
        bullet("El umbral P99 se computa únicamente desde la ventana TRAIN (WEEK <= cutoff_week)"),
        bullet("El periodo holdout NO se utiliza para estimar el umbral"),
        bullet("Por lo tanto, la regla de limpieza no depende de las etiquetas del holdout"),
        bullet("Una vez fijado el umbral, se aplica de forma consistente a ambos conjuntos para eliminar valores extremos del target"),
        ...spacer(1),
        infoBox("Garantía de Integridad", ["Esta implementación garantiza que no exista filtración de información entre los conjuntos de datos."]),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 5. FEATURE STORE
        // ══════════════════════════════════════════════════
        h1("5. Feature Store"),
        h2("5.1 Notebook: 02_feature_store_setup"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook construye la tabla de Feature Store que será utilizada para entrenar los modelos de Machine Learning."),
        para("El proceso identifica las columnas que representan features del modelo, excluyendo columnas de metadata y la variable objetivo, y crea una tabla consolidada que será utilizada en las etapas de entrenamiento."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "TARGET_COLUMN", "Variable objetivo utilizada en el modelo (UNI_BOX_WEEK)"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna de segmentación utilizada para entrenamiento"],
          ["Parámetro", "EXCLUDED_COLS", "Columnas excluidas del conjunto de features"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__TRAIN", "Dataset limpio utilizado para generar las features"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "FEAT_CUSTBPR_WEEKLY", "Tabla de Feature Store que contiene las features utilizadas por los modelos"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consulta la estructura de la tabla FEAT_CUSTBPR_WEEKLY__TRAIN utilizando DESCRIBE TABLE."),
        numbered("Se identifican todas las columnas disponibles en el dataset."),
        numbered("Se eliminan las columnas definidas en EXCLUDED_COLS (CUSTOMER_ID, BRAND_PRES_RET, PROD_KEY, WEEK, STATS_NTILE_GROUP)."),
        numbered("Las columnas restantes se consideran features del modelo."),
        numbered("Se construye un dataset que incluye columnas de identificación, columna temporal, columna de segmentación y features del modelo."),
        numbered("El dataset resultante se almacena en la tabla FEAT_CUSTBPR_WEEKLY."),
        numbered("Esta tabla se utiliza posteriormente en los scripts 03b y 04 para construir el dataset completo mediante join con las labels."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 6. TRAINING
        // ══════════════════════════════════════════════════
        h1("6. Training"),
        h2("6.1 Notebook: 03b_hyperparameter_search_bayesian"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook ejecuta el proceso de búsqueda de hiperparámetros utilizando optimización bayesiana para identificar las configuraciones óptimas de los modelos de Machine Learning."),
        para("El proceso evalúa múltiples combinaciones de hiperparámetros para cada segmento de datos y registra los resultados obtenidos."),
  
        h3("Código"),
        para("Python, Snowpark, Snowflake ML Tuner, Bayesian Optimization"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "TARGET_COLUMN", "Variable objetivo del modelo (UNI_BOX_WEEK)"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna utilizada para segmentar los modelos"],
          ["Parámetro", "NUM_TRIALS", "Número de pruebas de hiperparámetros (default: 15)"],
          ["Parámetro", "MAX_CONCURRENT_TRIALS", "Número máximo de ejecuciones paralelas (default: 4)"],
          ["Parámetro", "SAMPLE_RATE_PER_GROUP", "Fracción de datos utilizada durante el tuning (default: 0.2)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__TRAIN", "Dataset de entrenamiento limpio"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY", "Feature Store utilizada para entrenamiento"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST", "Tabla que almacena los resultados del proceso de optimización de hiperparámetros (respaldo)"],
          ["Experimento ML", "EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_{DATE}", "Experimento de Snowflake ML con runs de tuning por grupo (fuente primaria)"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se configuran los parámetros del experimento de optimización."),
        numbered("Se define el espacio de búsqueda de hiperparámetros. Todos los parámetros usan distribuciones continuas uniform() (requerido por BayesOpt). Parámetros enteros se convierten dentro de la función de entrenamiento."),
        numbered("Se escala el cluster de ejecución (CLUSTER_SIZE_HPO nodos)."),
        numbered("Para cada segmento de datos (STATS_NTILE_GROUP):"),
        bullet("Se carga un subconjunto de datos (SAMPLE_RATE_PER_GROUP, default 20%)", 1),
        bullet("Se utiliza el algoritmo de Bayesian Optimization para explorar el espacio de hiperparámetros", 1),
        bullet("Se ejecutan NUM_TRIALS pruebas con MAX_CONCURRENT_TRIALS ejecuciones en paralelo", 1),
        bullet("Se evalúa cada configuración mediante RMSE en un split temporal (80% train, 20% validation)", 1),
        bullet("Se registran las métricas obtenidas en ML Experiments (fuente primaria)", 1),
        numbered("Los mejores hiperparámetros por grupo se persisten en la tabla HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST como respaldo."),
        numbered("Se escala el cluster de vuelta a tamaño mínimo."),
  
        ...spacer(1),
        h3("Notas Técnicas: Modelos, BayesOpt y Search Space"),
        h4("¿Por qué se usan XGBRegressor y LGBMRegressor?"),
        para("Se seleccionan dos familias de Gradient Boosted Trees por su desempeño típico en regresión tabular con no-linealidades e interacciones complejas, su robustez ante escalas heterogéneas y su buen trade-off entre performance y costo computacional:"),
        bulletRuns([run("XGBRegressor: ", { bold: true }), run("suele ser estable y fuerte con regularización explícita (L1/L2) y control fino de complejidad por árbol.")]),
        bulletRuns([run("LGBMRegressor: ", { bold: true }), run("suele ser eficiente en entrenamiento y puede capturar relaciones complejas con num_leaves y controles de regularización.")]),
        para("En Snowflake se usan wrappers snowflake.ml.modeling.* para compatibilidad con el runtime (serialización/ejecución remota) y el ecosistema de Tuner/MMT."),
  
        ...spacer(1),
        h4("¿Cómo funciona la búsqueda bayesiana (BayesOpt)?"),
        para("BayesOpt es un método secuencial que en cada iteración propone el siguiente conjunto de hiperparámetros a evaluar usando un modelo sustituto (\"surrogate\") del rendimiento. En este script se usa: BayesOpt(utility_kwargs={\"kind\": \"ucb\", \"kappa\": 2.5, \"xi\": 0.0})"),
        para("La variante UCB (Upper Confidence Bound) balancea:"),
        bulletRuns([run("Explotación: ", { bold: true }), run("probar zonas con buen RMSE esperado.")]),
        bulletRuns([run("Exploración: ", { bold: true }), run("probar zonas con incertidumbre alta (controlado por kappa).")]),
  
        ...spacer(1),
        h4("Search Space (rangos) por modelo"),
        para("Todos los parámetros se definen como uniform(min, max) por requerimiento de BayesOpt (continuo). Los parámetros enteros se muestrean como float y se castean a int en la función de entrenamiento (INT_PARAMS)."),
        ...spacer(1),
        para("XGBRegressor (uniform):", { bold: true }),
        threeColTable([
          ["n_estimators", "50–300", "Nº de árboles; controla capacidad y costo"],
          ["max_depth", "3–10", "Profundidad por árbol; controla complejidad"],
          ["learning_rate", "0.01–0.3", "Tasa de aprendizaje; trade-off con nº de árboles"],
          ["subsample", "0.6–1.0", "Submuestreo de filas; regularización/robustez"],
          ["colsample_bytree", "0.6–1.0", "Submuestreo de columnas; reduce overfit"],
          ["min_child_weight", "1–7", "Controla splits por soporte de datos (regularización)"],
          ["gamma", "0–0.5", "Penaliza splits; reduce complejidad"],
          ["reg_alpha", "0–1", "Regularización L1; sparsity/robustez"],
          ["reg_lambda", "0–1", "Regularización L2; estabilidad"],
        ], ["Hiperparámetro", "Rango", "Rol técnico"], [2500, 1500, 5360]),
        ...spacer(1),
        para("LGBMRegressor (uniform):", { bold: true }),
        threeColTable([
          ["n_estimators", "50–300", "Nº de boosting iterations"],
          ["max_depth", "3–10", "Límite de profundidad; controla complejidad"],
          ["learning_rate", "0.01–0.3", "Tasa de aprendizaje"],
          ["num_leaves", "20–150", "Complejidad del árbol (hojas); controla capacidad"],
          ["subsample", "0.6–1.0", "Bagging de filas; regularización"],
          ["colsample_bytree", "0.6–1.0", "Bagging de columnas; regularización"],
          ["reg_alpha", "0–1", "Regularización L1"],
          ["reg_lambda", "0–1", "Regularización L2"],
          ["min_child_samples", "5–50", "Mín. muestras por hoja; evita sobreajuste en hojas pequeñas"],
        ], ["Hiperparámetro", "Rango", "Rol técnico"], [2500, 1500, 5360]),
  
        ...spacer(1),
        h2("6.2 Notebook: 04_many_model_training"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook ejecuta el proceso de entrenamiento de múltiples modelos utilizando la funcionalidad Many Model Training (MMT) de Snowflake. Se entrena un modelo independiente para cada segmento definido por la columna STATS_NTILE_GROUP."),
  
        h3("Código"),
        para("Python, Snowpark, Snowflake ML ManyModelTraining"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "TARGET_COLUMN", "Variable objetivo del modelo (UNI_BOX_WEEK)"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna utilizada para segmentar los modelos"],
          ["Parámetro", "GROUP_MODEL", "Mapeo entre segmento y algoritmo de ML (LGBM/XGB por grupo)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__TRAIN", "Dataset de entrenamiento limpio"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY", "Feature Store utilizada para entrenamiento"],
          ["Tabla", "HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST", "Resultados de optimización de hiperparámetros (respaldo)"],
          ["Experimento ML", "EXP_UNIBOX_CUSTBPR_WEEKLY_FORECAST_BAYESIAN_{DATE}", "ML Experiments con hiperparámetros (fuente primaria)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Stage Snowflake", "MMT_MODELS", "Stage donde se almacenan los modelos entrenados por segmento"],
          ["Model Registry", "unibox_custbpr_weekly_forecast__group_stat_*", "16 modelos registrados individualmente por grupo (con alias PRODUCTION)"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consulta la Feature Store FEAT_CUSTBPR_WEEKLY."),
        numbered("Se recuperan los mejores hiperparámetros desde HPO_UNIBOX_CUSTBPR_WEEKLY_FORECAST (respaldo) o desde ML Experiments (fuente primaria)."),
        numbered("Se cargan los hiperparámetros por grupo, utilizando valores por defecto cuando no están disponibles."),
        numbered("Se ejecuta el proceso Many Model Training (MMT):"),
        bullet("Configuración del cluster de ejecución con escalado dinámico (CLUSTER_SIZE_MMT nodos)", 1),
        bullet("Particionamiento de datos por STATS_NTILE_GROUP", 1),
        bullet("Ejecución distribuida mediante Ray/Snowpark-managed runtime", 1),
        bullet("Para cada grupo se entrena un modelo independiente utilizando el algoritmo asignado (LGBMRegressor o XGBRegressor según GROUP_MODEL)", 1),
        bullet("Cada función de entrenamiento ejecuta join FEAT_CUSTBPR_WEEKLY + FEAT_CUSTBPR_WEEKLY__TRAIN, split temporal interno, entrenamiento y evaluación", 1),
        numbered("Los modelos entrenados se almacenan en el stage MMT_MODELS."),
        numbered("Cada modelo se registra individualmente en Snowflake Model Registry con nombre, versión, métricas y task type TABULAR_REGRESSION."),
        numbered("Se asigna el alias PRODUCTION a cada versión recién entrenada (usado en DEV)."),
        numbered("Al finalizar, se escala el cluster de vuelta a tamaño mínimo."),
  
        ...spacer(1),
        h3("Nota técnica: Normalización y casteo de hiperparámetros"),
        para("Los hiperparámetros recuperados desde Experiments/tabla pueden llegar con tipos no nativos (por ejemplo, numpy.float64). Antes de instanciar el estimador, el script define un set de parámetros enteros esperados, castea a int los parámetros del set entero y a float los continuos. En caso de error de casteo, cae a defaults por modelo (DEFAULT_PARAMS_BY_MODEL) para mantener robustez del pipeline."),
  
        ...spacer(1),
        h3("Nota Técnica: Modelo de Ejecución de MMT (Many Model Training)"),
        para("Una preocupación común al ver imports de snowflake.ml.modeling.* dentro de la función de entrenamiento es si se está instanciando \"un modelo distribuido dentro de un nodo\". La clarificación es la siguiente:"),
        bulletRuns([run("ManyModelTraining es el sistema distribuido: ", { bold: true }), run("particiona el dataset de entrada (por ejemplo, por STATS_NTILE_GROUP) y programa una partición por worker en el runtime administrado por Snowflake.")]),
        bulletRuns([run("Dentro de cada worker, ", { bold: true }), run("la función de entrenamiento opera sobre un pandas DataFrame con únicamente los datos de esa partición. El entrenamiento ejecutado es completamente local — no hay entrenamiento distribuido anidado dentro del estimador.")]),
        bulletRuns([run("Los estimadores en snowflake.ml.modeling.* son wrappers compatibles con el runtime: ", { bold: true }), run("están diseñados para funcionar de forma confiable dentro del entorno de ejecución de Snowflake. Utilizar librerías nativas directamente puede romper la serialización en el runtime.")]),
        bulletRuns([run("Cada modelo entrenado es independiente: ", { bold: true }), run("cada STATS_NTILE_GROUP produce un artefacto de modelo separado que se registra individualmente en Model Registry.")]),
        ...spacer(1),
        infoBox("Resumen: Modelo de Ejecución MMT", [
          "ManyModelTraining distribuye el trabajo entre particiones, y cada partición entrena un modelo local estándar sobre su subconjunto de datos.",
          "No hay distribución anidada ni entrenamiento distribuido dentro de los estimadores individuales."
        ]),
  
        ...spacer(1),
        h2("6.3 Notebook: 05_create_partitioned_model"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook crea un modelo particionado que agrupa los modelos entrenados por segmento en un único modelo lógico. Este modelo actúa como un wrapper que selecciona automáticamente el modelo correspondiente dependiendo del valor de la columna de segmentación."),
  
        h3("Código"),
        para("Python, Snowpark, Snowflake ML Registry"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo particionado (UNIBOX_CUSTBPR_WEEKLY_FORECAST)"],
          ["Parámetro", "STATS_NTILE_GROUP_COL", "Columna utilizada para seleccionar el modelo (STATS_NTILE_GROUP)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__TRAIN", "Dataset utilizado para identificar los grupos de entrenamiento"],
          ["Model Registry", "unibox_custbpr_weekly_forecast__group_stat_*", "16 modelos previamente entrenados (con alias PRODUCTION)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Model Registry", "UNIBOX_CUSTBPR_WEEKLY_FORECAST", "Modelo particionado que agrupa todos los modelos por segmento (con alias PRODUCTION en DEV)"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consulta el dataset FEAT_CUSTBPR_WEEKLY__TRAIN para identificar los 16 segmentos existentes."),
        numbered("Para cada grupo se carga el modelo correspondiente desde el Model Registry utilizando el alias PRODUCTION."),
        numbered("Se construye una estructura que mapea cada grupo con su modelo correspondiente."),
        numbered("Se define una clase CustomModel (PartitionedModel) que encapsula los 16 submodelos e implementa el método predict() que enruta automáticamente cada fila al submodelo correcto."),
        numbered("El modelo particionado se registra en Snowflake Model Registry con nombre UNIBOX_CUSTBPR_WEEKLY_FORECAST, versión v_{VERSION_DATE}, alias PRODUCTION (en DEV) y task TABULAR_REGRESSION."),
        numbered("Este modelo unificado permite inferencia simplificada: un único llamado a MODEL()!PREDICT maneja automáticamente el enrutamiento interno."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 7. BASELINE GENERATION
        // ══════════════════════════════════════════════════
        h1("7. Baseline Generation (Training Data)"),
        para("Esta etapa genera los baselines de referencia utilizando el conjunto de holdout del entrenamiento. Los baselines se utilizarán posteriormente para detectar drift en producción."),
        ...spacer(1),
  
        h2("7.1 Notebook: 06a_setup_baselines"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook inicializa la infraestructura de observabilidad y ejecuta inferencia batch sobre el conjunto de holdout para generar predicciones de referencia (baseline)."),
        para("El notebook crea las tablas necesarias para almacenar histogramas y métricas de baseline, carga el modelo PRODUCTION desde Model Registry, ejecuta inferencia particionada sobre los datos de holdout y almacena las predicciones que servirán como referencia para los procesos de monitoreo de drift."),
  
        h3("Código"),
        para("Python, Snowpark, SQL, Snowflake ML Model Registry"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo registrado en Snowflake Model Registry"],
          ["Parámetro", "PARTITION_COL", "Columna utilizada para particionar el modelo (STATS_NTILE_GROUP)"],
          ["Parámetro", "TARGET_COL", "Variable objetivo utilizada durante entrenamiento (UNI_BOX_WEEK)"],
          ["Parámetro", "PREDICTION_COL", "Columna donde se almacenará la predicción (PREDICTED_UNI_BOX_WEEK)"],
          ["Parámetro", "TIME_COL", "Columna temporal utilizada para batching (week)"],
          ["Parámetro", "BASELINE_ALIAS", "Alias del modelo en registry (PRODUCTION)"],
          ["Parámetro", "N_BINS", "Número de bins para histogramas de drift (default: 20)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__HOLDOUT", "Dataset de holdout para generar baseline"],
          ["Tabla", "TRAIN_CUST_CATEGORY_LOOKUP", "Tabla de referencia de categorías de clientes"],
          ["Servicio Snowflake", "Snowflake Model Registry", "Repositorio donde se encuentra el modelo PRODUCTION"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PREDICTIONS_BL", "Tabla que almacena las predicciones baseline con metadatos completos"],
          ["Tabla", "OBS_PREDICTIONS_BL_VW", "Vista transient que join predictions con categorías de clientes"],
          ["Tabla", "OBS_DATA_HIST_BL", "Tabla para almacenar histogramas de features baseline (creada, poblada en 06b)"],
          ["Tabla", "OBS_PRED_HIST_BL", "Tabla para almacenar histogramas de predicciones baseline (creada, poblada en 06c)"],
          ["Tabla", "OBS_PERFORMANCE_BL", "Tabla para almacenar métricas de desempeño baseline (creada, poblada en 06d)"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se crean las tablas de baseline si no existen: OBS_DATA_HIST_BL, OBS_PRED_HIST_BL, OBS_PERFORMANCE_BL, OBS_PREDICTIONS_BL."),
        numbered("Se crea la lookup table TRAIN_CUST_CATEGORY_LOOKUP para mapear categorías de clientes."),
        numbered("Se crea la vista FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW combinando holdout data con categorías."),
        numbered("Se conecta al Model Registry y se obtiene la versión asociada al alias PRODUCTION."),
        numbered("Se verifica si ya existen predicciones baseline para esta versión del modelo."),
        numbered("Si no existen predicciones previas: se itera sobre cada valor único de TIME_COL (week) y se ejecuta MODEL()!PREDICT con particionamiento por STATS_NTILE_GROUP. Las predicciones se insertan en OBS_PREDICTIONS_BL con RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, PREDICTION, BKCC, CALMONTH, LDTS."),
        numbered("Se crea la tabla transient OBS_PREDICTIONS_BL_VW mediante join con categorías de clientes."),
  
        ...spacer(1),
        h2("7.2 Notebook: 06b_data_drift_baseline"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook genera el baseline estadístico de las variables de entrada utilizadas por el modelo, el cual será utilizado para detectar data drift durante la operación del modelo en producción."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "PARTITION_COL", "Columna utilizada para segmentar los datos durante el análisis"],
          ["Parámetro", "TARGET_COL", "Variable objetivo utilizada en el modelo"],
          ["Parámetro", "TIME_COL", "Columna temporal utilizada para identificar periodos"],
          ["Parámetro", "N_BINS", "Número de bins utilizados para generar histogramas de distribución"],
          ["Vista", "TRAIN_DATASET_HOLDOUT_VW", "Vista con el dataset de referencia para generar el baseline de features"],
          ["Vista", "DA_PREDICTIONS_BASELINE_VW", "Vista con las predicciones generadas por el modelo"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_DATA_HIST_BL", "Tabla que contiene los histogramas base para detectar desviaciones en la distribución de las features"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consulta el dataset de referencia desde TRAIN_DATASET_HOLDOUT_VW."),
        numbered("Se identifican las variables utilizadas por el modelo que deben ser monitoreadas para detectar data drift."),
        numbered("Se excluyen columnas que no deben participar en el análisis de drift."),
        numbered("Para cada feature se calculan histogramas de distribución utilizando un número definido de bins."),
        numbered("Los histogramas se calculan por segmentos definidos por STATS_NTILE_GROUP y CUST_CATEGORY."),
        numbered("Los resultados se almacenan en la tabla OBS_DATA_HIST_BL."),
  
        ...spacer(1),
        h2("7.3 Notebook: 06c_prediction_drift_baseline"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook genera el baseline de la distribución de las predicciones del modelo utilizando las predicciones almacenadas por el script 06a, el cual será utilizado para detectar prediction drift durante la operación del modelo en producción."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "PREDICTION_COL", "Columna que contiene las predicciones (PREDICTED_UNI_BOX_WEEK)"],
          ["Parámetro", "PARTITION_COL", "Columna utilizada para segmentar los datos (STATS_NTILE_GROUP)"],
          ["Parámetro", "TIME_COL", "Columna temporal utilizada en el análisis (week)"],
          ["Parámetro", "N_BINS", "Número de bins utilizados para construir histogramas (default: 20)"],
          ["Tabla", "OBS_PREDICTIONS_BL_VW", "Vista que contiene las predicciones baseline generadas por 06a"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PRED_HIST_BL", "Tabla que contiene los histogramas base de la distribución de las predicciones"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consulta la vista OBS_PREDICTIONS_BL_VW con las predicciones de referencia."),
        numbered("Se identifican todas las combinaciones únicas de (MODEL_NAME, MODEL_VERSION) presentes en las predicciones baseline."),
        numbered("Para cada combinación se agrupan las predicciones por segmentos (STATS_NTILE_GROUP, CUST_CATEGORY) y se generan histogramas mediante WIDTH_BUCKET para crear bins equiespaciados."),
        numbered("Se calcula el conteo de predicciones por bin y se normaliza para obtener frecuencias relativas."),
        numbered("Cada histograma se almacena en OBS_PRED_HIST_BL con RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_MAP (histograma en JSON), CALMONTH y LDTS."),
  
        ...spacer(1),
        h2("7.4 Notebook: 06d_performance_drift_baseline"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook genera el baseline de métricas de desempeño del modelo utilizando las predicciones baseline y los valores reales del conjunto de holdout, el cual será utilizado para detectar performance drift durante la operación del modelo en producción."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "TARGET_COL", "Variable objetivo utilizada para evaluar el modelo (UNI_BOX_WEEK)"],
          ["Parámetro", "PREDICTION_COL", "Columna que contiene las predicciones (PREDICTED_UNI_BOX_WEEK)"],
          ["Parámetro", "PARTITION_COL", "Columna utilizada para segmentar el análisis (STATS_NTILE_GROUP)"],
          ["Parámetro", "TIME_COL", "Columna temporal utilizada para evaluación (week)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW", "Vista con los valores reales del conjunto de holdout"],
          ["Tabla", "OBS_PREDICTIONS_BL_VW", "Vista con las predicciones baseline generadas por 06a"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PERFORMANCE_BL", "Tabla que almacena las métricas base de desempeño del modelo"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se consultan los valores reales (actuals) desde FEAT_CUSTBPR_WEEKLY__HOLDOUT_VW."),
        numbered("Se consultan las predicciones baseline desde OBS_PREDICTIONS_BL_VW."),
        numbered("Se realiza un join entre actuals y predictions utilizando las claves: customer_id, brand_pres_ret, prod_key, week."),
        numbered("Para cada combinación de (MODEL_NAME, MODEL_VERSION, STATS_NTILE_GROUP, CUST_CATEGORY) se calculan métricas de desempeño:"),
        bulletRuns([run("WAPE ", { bold: true }), run("(Weighted Absolute Percentage Error): suma de errores absolutos dividida por suma de valores reales")], 1),
        bulletRuns([run("RMSE ", { bold: true }), run("(Root Mean Squared Error): raíz cuadrada del error cuadrático medio")], 1),
        bulletRuns([run("MAE ", { bold: true }), run("(Mean Absolute Error): promedio de errores absolutos")], 1),
        bulletRuns([run("F1_BINARY: ", { bold: true }), run("métrica binaria basada en threshold (target > 0)")], 1),
        numbered("Las métricas se almacenan en OBS_PERFORMANCE_BL con RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE, BKCC, CALMONTH, LDTS."),
        numbered("Las métricas baseline servirán como referencia para detectar degradación del modelo en producción."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 8. ENVIRONMENT PROMOTION
        // ══════════════════════════════════════════════════
        h1("8. Environment Promotion"),
        para("Esta etapa sincroniza artefactos entre ambientes (DEV → QA/PROD), copiando baselines y modelos registrados para preparar el ambiente objetivo."),
        ...spacer(1),
  
        h2("8.1 Notebook: 07a_copy_baselines"),
  
        h3("Objetivo del Notebook"),
        para("Copiar las tablas de baseline (histogramas y métricas de referencia) desde el ambiente de desarrollo hacia el ambiente objetivo (QA o PROD). Este script sincroniza únicamente los registros faltantes, evitando duplicados."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "SRC_DATABASE", "Base de datos fuente (desarrollo)"],
          ["Parámetro", "TGT_DATABASE", "Base de datos destino (QA/PROD)"],
          ["Parámetro", "SRC_STORAGE_SCHEMA", "Schema fuente donde residen los baselines"],
          ["Parámetro", "TGT_STORAGE_SCHEMA", "Schema destino donde se copiarán los baselines"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo para filtrar registros a copiar"],
          ["Tabla", "OBS_DATA_HIST_BL (SRC)", "Tabla fuente de histogramas de data drift baseline"],
          ["Tabla", "OBS_PRED_HIST_BL (SRC)", "Tabla fuente de histogramas de prediction drift baseline"],
          ["Tabla", "OBS_PERFORMANCE_BL (SRC)", "Tabla fuente de métricas de performance baseline"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_DATA_HIST_BL (TGT)", "Tabla destino de histogramas de data drift baseline"],
          ["Tabla", "OBS_PRED_HIST_BL (TGT)", "Tabla destino de histogramas de prediction drift baseline"],
          ["Tabla", "OBS_PERFORMANCE_BL (TGT)", "Tabla destino de métricas de performance baseline"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Para cada par de tablas (fuente, destino):"),
        bullet("Se crea la tabla destino si no existe (usando CREATE TABLE ... LIKE)", 1),
        bullet("Se identifican combinaciones (MODEL_NAME, MODEL_VERSION, AGGREGATED_COL) que existen en fuente pero no en destino", 1),
        bullet("Se copian únicamente los registros faltantes mediante INSERT INTO ... SELECT con filtro de exclusión", 1),
        bullet("Se reporta el número de registros copiados", 1),
        numbered("Las tablas sincronizadas son: OBS_DATA_HIST_BL, OBS_PRED_HIST_BL y OBS_PERFORMANCE_BL."),
        numbered("Este script NO copia los modelos del registry (eso lo hace 07b)."),
  
        ...spacer(1),
        h2("8.2 Notebook: 07b_copy_models"),
  
        h3("Objetivo del Notebook"),
        para("Copiar los modelos registrados desde el Model Registry de desarrollo hacia el Model Registry del ambiente objetivo (QA o PROD), aplicando tags de producción para control de versiones operativas."),
  
        h3("Código"),
        para("Python, Snowpark, Snowflake ML Registry"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "SRC_DATABASE", "Base de datos fuente (desarrollo)"],
          ["Parámetro", "TGT_DATABASE", "Base de datos destino (QA/PROD)"],
          ["Parámetro", "SRC_MODELS_SCHEMA", "Schema fuente donde reside el Model Registry de desarrollo"],
          ["Parámetro", "TGT_MODELS_SCHEMA", "Schema destino donde se copiará el modelo"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo particionado a copiar"],
          ["Parámetro", "USE_CASE", "Token del caso de uso para tags (ejemplo: CLIENTA_DEFAULT)"],
          ["Model Registry", "{MODEL_NAME} (SRC)", "Modelo particionado fuente con alias PRODUCTION"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Model Registry", "{MODEL_NAME} (TGT)", "Modelo particionado copiado en ambiente objetivo"],
          ["Model Tag", "PRODUCTION_{USE_CASE}", "Tag aplicado a la versión activa de producción"],
          ["Model Tag", "ROLLBACK_VERSION_{USE_CASE}", "Tag aplicado a la versión de rollback"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se conecta al Model Registry fuente y se obtiene el modelo mediante alias PRODUCTION."),
        numbered("Se verifica si el modelo ya existe en el registry destino. Si NO existe: se copia el modelo completo. Si SÍ existe: se verifica si la versión específica ya está presente."),
        numbered("Se aplican tags operativos al modelo en el ambiente destino:"),
        bullet("PRODUCTION_{USE_CASE}: identifica la versión activa de producción", 1),
        bullet("ROLLBACK_VERSION_{USE_CASE}: identifica la versión previa para rollback", 1),
        numbered("El modelo copiado mantiene todas sus propiedades: submodelos por segmento, lógica de particionamiento, métricas y metadatos."),
        numbered("Los tags permiten gestionar promociones y rollbacks de forma controlada sin depender de alias globales."),
        numbered("Este script NO copia las tablas de baseline (eso lo hace 07a)."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 9. INFERENCE
        // ══════════════════════════════════════════════════
        h1("9. Inference (Production Data)"),
        h2("9.1 Notebook: 08_partitioned_inference_batch"),
  
        h3("Objetivo del Notebook"),
        para("Ejecutar el proceso de inferencia batch sobre el dataset de inferencia productivo utilizando el modelo particionado identificado mediante tags de producción. Este script detecta automáticamente qué combinaciones (versión del modelo, semana) faltan en la tabla de predicciones y las procesa de forma incremental."),
  
        h3("Código"),
        para("Python, Snowpark, SQL, Snowflake ML Model Registry"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "MODEL_NAME", "Nombre del modelo particionado registrado (UNIBOX_CUSTBPR_WEEKLY_FORECAST)"],
          ["Parámetro", "USE_CASE", "Token del caso de uso para identificar tag de producción (ej: CLIENTA_DEFAULT)"],
          ["Parámetro", "PRODUCTION_TAG", "Tag que identifica la versión activa (PRODUCTION_{USE_CASE})"],
          ["Parámetro", "MIN_INFERENCE_TIME", "Filtro opcional de semanas mínimas a procesar (None = todas)"],
          ["Parámetro", "INFERENCE_SAMPLE_FRACTION", "Fracción de muestreo opcional (None = dataset completo)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__INF", "Dataset de inferencia limpio (creado en script 01)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__INF_VW", "Vista con categorías de clientes para inferencia"],
          ["Tabla", "INFERENCE_CUST_CATEGORY_LOOKUP", "Lookup de categorías de clientes para inferencia"],
          ["Tabla", "GROUND_TRUTH_DATASET_STRUCTURED", "Dataset con valores reales para evaluación posterior"],
          ["Tabla", "ACTUALS_TABLE_VW", "Vista que expone actuals para join con predicciones"],
          ["Modelo", "Partitioned Model (PRODUCTION tag)", "Modelo registrado con tag de producción activo"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PREDICTIONS", "Tabla que contiene las predicciones generadas en producción"],
          ["Tabla", "OBS_PREDICTIONS_VW", "Vista transient que join predictions con categorías"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se crean tablas y vistas auxiliares si no existen: INFERENCE_CUST_CATEGORY_LOOKUP, FEAT_CUSTBPR_WEEKLY__INF_VW, ACTUALS_TABLE_VW."),
        numbered("Se crea la tabla OBS_PREDICTIONS con esquema completo si no existe."),
        numbered("Se conecta al Model Registry y se obtiene la versión del modelo mediante el tag PRODUCTION_{USE_CASE}."),
        numbered("Se identifica qué combinaciones (MODEL_VERSION, WEEK) faltan en OBS_PREDICTIONS."),
        numbered("Para cada WEEK faltante:"),
        bullet("Se carga el batch de datos correspondiente", 1),
        bullet("Se aplica muestreo opcional si INFERENCE_SAMPLE_FRACTION está configurado", 1),
        bullet("Se ejecuta MODEL()!PREDICT con particionamiento por STATS_NTILE_GROUP", 1),
        bullet("Las predicciones se insertan en OBS_PREDICTIONS con RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, PREDICTION, BKCC, CALMONTH, LDTS", 1),
        numbered("Se crea/actualiza la tabla transient OBS_PREDICTIONS_VW mediante join con categorías."),
        numbered("El proceso es incremental: solo procesa semanas faltantes, permitiendo ejecuciones repetidas sin duplicados."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 10. ML OBSERVABILITY
        // ══════════════════════════════════════════════════
        h1("10. ML Observability (Production Data)"),
        para("Esta etapa monitorea el comportamiento del modelo en producción, detectando drift en datos, predicciones y desempeño mediante comparación con baselines."),
        ...spacer(1),
  
        h2("10.1 Notebook: 09a_setup_observability"),
  
        h3("Objetivo del Notebook"),
        para("Inicializar la infraestructura de observabilidad para producción, creando las tablas de landing necesarias para almacenar histogramas de drift, métricas de drift con thresholds y alertas, y métricas de desempeño."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "Parámetros Constantes", "Consultar la sección 3.5 Parámetros Constantes"],
          ["Parámetro", "STORAGE_SCHEMA", "Esquema que contendrá las tablas de monitoreo productivas"],
          ["Parámetro", "N_BINS", "Número de bins para histogramas (default: 20)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_DATA_HIST", "Tabla para histogramas de features en producción"],
          ["Tabla", "OBS_DATA_DRIFT", "Tabla para métricas de data drift con thresholds y alertas"],
          ["Tabla", "OBS_PRED_HIST", "Tabla para histogramas de predicciones en producción"],
          ["Tabla", "OBS_PRED_DRIFT", "Tabla para métricas de prediction drift con thresholds y alertas"],
          ["Tabla", "OBS_PERFORMANCE", "Tabla para métricas de desempeño con thresholds y alertas"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se crean las tablas de landing para histogramas con esquema estandarizado: OBS_DATA_HIST y OBS_PRED_HIST. Esquema: RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_MAP (OBJECT con histograma), CALMONTH, LDTS."),
        numbered("Se crean las tablas de landing para métricas de drift: OBS_DATA_DRIFT y OBS_PRED_DRIFT. Esquema: RECORD_ID, MODEL_NAME, MODEL_VERSION, ENTITY_MAP, AGGREGATED_COL, AGGREGATED_VALUE, METRIC_COL, METRIC_VALUE, THRESHOLD_WARNING, THRESHOLD_CRITICAL, ALERT_LEVEL, BKCC, CALMONTH, LDTS."),
        numbered("Se crea la tabla OBS_PERFORMANCE con el mismo esquema extendido con thresholds y alert_level."),
        numbered("Estas tablas serán pobladas por los scripts 09b, 09c y 09d."),
  
        ...spacer(1),
        h2("10.2 Notebook: 09b_data_drift"),
  
        h3("Objetivo del Notebook"),
        para("Detectar cambios en la distribución de las variables de entrada del modelo comparando los datos de inferencia recientes con los histogramas baseline mediante divergencia de Kullback-Leibler (KL)."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "N_BINS", "Número de bins para histogramas (debe coincidir con baseline)"],
          ["Parámetro", "THRESHOLD_WARNING", "Umbral de KL divergence para alerta WARNING (default: 0.1)"],
          ["Parámetro", "THRESHOLD_CRITICAL", "Umbral de KL divergence para alerta CRITICAL (default: 0.3)"],
          ["Tabla", "FEAT_CUSTBPR_WEEKLY__INF_VW", "Vista con features de inferencia recientes"],
          ["Tabla", "OBS_PREDICTIONS_VW", "Predicciones recientes con metadatos (para identificar versión del modelo)"],
          ["Tabla", "OBS_DATA_HIST_BL", "Histogramas baseline de features (generados en 06b)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_DATA_HIST", "Histogramas de features calculados sobre datos de inferencia recientes"],
          ["Tabla", "OBS_DATA_DRIFT", "Métricas de data drift (KL divergence) con niveles de alerta"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_DATA_HIST."),
        numbered("Para cada combinación faltante:"),
        bullet("Se consultan las features de inferencia desde FEAT_CUSTBPR_WEEKLY__INF_VW", 1),
        bullet("Se identifican las columnas de features (excluyendo metadata)", 1),
        bullet("Para cada feature y cada segmento (STATS_NTILE_GROUP, CUST_CATEGORY): se calculan histogramas de distribución usando N_BINS bins y se normalizan las frecuencias para obtener distribuciones de probabilidad", 1),
        bullet("Se almacenan en OBS_DATA_HIST", 1),
        numbered("Se comparan histogramas actuales vs baseline mediante KL divergence: sum(p * log(p / q)) donde p=actual, q=baseline."),
        numbered("Se asigna nivel de alerta según thresholds:"),
        bullet("OK: KL < THRESHOLD_WARNING", 1),
        bullet("WARNING: THRESHOLD_WARNING <= KL < THRESHOLD_CRITICAL", 1),
        bullet("CRITICAL: KL >= THRESHOLD_CRITICAL", 1),
        numbered("Se almacenan métricas en OBS_DATA_DRIFT."),
  
        ...spacer(1),
        h2("10.3 Notebook: 09c_prediction_drift"),
  
        h3("Objetivo del Notebook"),
        para("Monitorear cambios en la distribución de las predicciones generadas por el modelo en producción comparando contra histogramas baseline mediante divergencia KL."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "N_BINS", "Número de bins para histogramas (debe coincidir con baseline)"],
          ["Parámetro", "THRESHOLD_WARNING", "Umbral de KL divergence para alerta WARNING (default: 0.1)"],
          ["Parámetro", "THRESHOLD_CRITICAL", "Umbral de KL divergence para alerta CRITICAL (default: 0.3)"],
          ["Tabla", "OBS_PREDICTIONS_VW", "Vista con predicciones recientes de producción"],
          ["Tabla", "OBS_PRED_HIST_BL", "Histogramas baseline de predicciones (generados en 06c)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PRED_HIST", "Histogramas de predicciones calculados sobre inferencia reciente"],
          ["Tabla", "OBS_PRED_DRIFT", "Métricas de prediction drift (KL divergence) con niveles de alerta"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_PRED_HIST."),
        numbered("Para cada combinación faltante: se consultan las predicciones desde OBS_PREDICTIONS_VW, se calculan histogramas de distribución de PREDICTION usando N_BINS bins por segmento, se normalizan las frecuencias y se almacenan en OBS_PRED_HIST."),
        numbered("Se comparan histogramas actuales vs baseline mediante KL divergence y se asigna nivel de alerta (OK / WARNING / CRITICAL)."),
        numbered("Se almacenan métricas en OBS_PRED_DRIFT."),
        numbered("Cambios significativos en la distribución de predicciones pueden indicar:"),
        bullet("Modificaciones en el comportamiento del modelo", 1),
        bullet("Cambios en los patrones de datos de entrada", 1),
        bullet("Drift conceptual en el dominio del problema", 1),
  
        ...spacer(1),
        h2("10.4 Notebook: 09d_performance_drift"),
  
        h3("Objetivo del Notebook"),
        para("Evaluar la degradación del desempeño del modelo en producción comparando métricas actuales contra baseline, detectando deterioro en la capacidad predictiva mediante comparación con thresholds configurados."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "PERF_JOIN_KEYS", "Claves para join actuals-predictions (CUSTOMER_ID, BRAND_PRES_RET, WEEK)"],
          ["Parámetro", "PERF_METRIC_NAMES", "Métricas a calcular (wape, rmse, mae, f1_binary)"],
          ["Parámetro", "THRESHOLD_WARNING", "% de degradación para WARNING (default: 10%)"],
          ["Parámetro", "THRESHOLD_CRITICAL", "% de degradación para CRITICAL (default: 25%)"],
          ["Tabla", "OBS_PREDICTIONS_VW", "Vista con predicciones recientes de producción"],
          ["Tabla", "ACTUALS_TABLE_VW", "Vista con valores reales disponibles para evaluación"],
          ["Tabla", "OBS_PERFORMANCE_BL", "Métricas de desempeño baseline (generadas en 06d)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Tabla", "OBS_PERFORMANCE", "Métricas de desempeño calculadas sobre producción con niveles de alerta"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark."),
        numbered("Se identifican combinaciones (MODEL_NAME, MODEL_VERSION) presentes en OBS_PREDICTIONS_VW pero faltantes en OBS_PERFORMANCE."),
        numbered("Para cada combinación faltante: se hace join entre OBS_PREDICTIONS_VW y ACTUALS_TABLE_VW usando PERF_JOIN_KEYS. Para cada segmento se calculan métricas:"),
        bulletRuns([run("WAPE: ", { bold: true }), run("suma de errores absolutos / suma de valores reales")], 1),
        bulletRuns([run("RMSE: ", { bold: true }), run("raíz cuadrada del error cuadrático medio")], 1),
        bulletRuns([run("MAE: ", { bold: true }), run("promedio de errores absolutos")], 1),
        bulletRuns([run("F1_BINARY: ", { bold: true }), run("métrica binaria (threshold: target > 0)")], 1),
        numbered("Se comparan métricas actuales vs baseline: se calcula % de degradación = (actual - baseline) / baseline * 100."),
        numbered("Para métricas de error (WAPE, RMSE, MAE): degradación = aumento. Para métricas de accuracy (F1): degradación = disminución."),
        numbered("Se asigna nivel de alerta según thresholds (OK / WARNING / CRITICAL) y se almacenan en OBS_PERFORMANCE."),
        numbered("El monitoreo continuo permite identificar cuándo el modelo requiere reentrenamiento por cambios en datos (concept drift), cambios en el comportamiento del sistema o deterioro natural del modelo."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 11. ALERTING & NOTIFICATIONS
        // ══════════════════════════════════════════════════
        h1("11. Alerting & Notifications"),
        h2("11.1 Notebook: 10_alertas"),
  
        h3("Objetivo del Notebook"),
        para("Este notebook consolida y reporta alertas generadas por la capa de observabilidad, consultando las tablas OBS_PERFORMANCE (performance drift), OBS_DATA_DRIFT (data drift) y OBS_PRED_DRIFT (prediction drift)."),
        para("El objetivo es identificar registros con ALERT_LEVEL en estado WARNING o CRITICAL (configurable) para un MODEL_NAME dado y construir un reporte unificado, listo para ser enviado por correo (HTML) y/o publicado a una cola/event bus (JSON) usando integraciones de notificación de Snowflake."),
  
        h3("Código"),
        para("Python, Snowpark, SQL"),
  
        h3("Entradas"),
        ioTable([
          ["Parámetro", "DATABASE", "Base de datos utilizada (BD_AA_DEV)"],
          ["Parámetro", "FEATURES_SCHEMA", "Esquema donde residen las tablas de observabilidad (SC_FEATURES_BMX)"],
          ["Parámetro", "MODEL_NAME", "Modelo a monitorear (ej: UNIBOX_CUSTBPR_WEEKLY_FORECAST)"],
          ["Parámetro", "REPORT_ALERT_THRESHOLD", "Nivel mínimo a reportar (por defecto CRITICAL)"],
          ["Parámetro", "LDTS_AFTER", "Filtro temporal para considerar solo métricas recientes (LDTS > LDTS_AFTER)"],
          ["Parámetro", "PERFORMANCE_METRICS", "Métricas a incluir desde performance (ej: rmse, wape)"],
          ["Parámetro", "DATA_DRIFT_FEATURE_METRICS", "Métricas a incluir desde feature drift (ej: jensen-shannon)"],
          ["Parámetro", "DATA_DRIFT_SEGMENT_METRICS", "Métricas a incluir desde population drift (ej: population_stability_index)"],
          ["Parámetro", "PRED_DRIFT_METRICS", "Métricas a incluir desde prediction drift (ej: jensen-shannon)"],
          ["Parámetro", "AGGREGATED_COLS", "Dimensiones a incluir en el reporte (ej: stats_ntile_group)"],
          ["Tabla", "OBS_PERFORMANCE", "Métricas de performance con thresholds y ALERT_LEVEL"],
          ["Tabla", "OBS_DATA_DRIFT", "Métricas de data drift con thresholds y ALERT_LEVEL"],
          ["Tabla", "OBS_PRED_DRIFT", "Métricas de prediction drift con thresholds y ALERT_LEVEL"],
          ["Configuración", "NOTIFICATION_INTEGRATION", "Nombre de la integración de email (si se habilita envío)"],
          ["Configuración", "QUEUE_NOTIFICATION_INTEGRATION", "Nombre de la integración para cola/event bus (si se habilita)"],
          ["Parámetro", "EMAIL_RECIPIENTS", "Lista de destinatarios del reporte (si se habilita envío)"],
        ]),
  
        ...spacer(1),
        h3("Salidas"),
        ioTable([
          ["Reporte (DataFrame)", "Consolidated alerts", "Resultado consolidado (union) de alertas provenientes de las 3 tablas"],
          ["Reporte (HTML)", "email_html", "Cuerpo HTML del reporte para envío por email"],
          ["Payload (JSON)", "queue_payload", "Mensaje JSON para publicar a cola/event bus"],
          ["Notificación (opcional)", "SYSTEM$SEND_SNOWFLAKE_NOTIFICATION", "Envío por email (TEXT_HTML) y/o publicación (APPLICATION_JSON) si la integración existe"],
        ]),
  
        ...spacer(1),
        h3("Proceso Técnico"),
        resetNum(), numbered("Se inicializa la sesión Snowpark y se selecciona DATABASE y FEATURES_SCHEMA."),
        numbered("Se define el set de tablas de observabilidad y las métricas relevantes por fuente."),
        numbered("Se consulta cada tabla filtrando por MODEL_NAME, ALERT_LEVEL >= REPORT_ALERT_THRESHOLD, METRIC_COL (solo métricas configuradas por fuente), AGGREGATED_COL y LDTS > LDTS_AFTER (si aplica)."),
        numbered("Se normaliza el esquema entre fuentes y se agrega la columna SOURCE_TABLE para identificar el origen (performance / feature drift / population drift / prediction drift)."),
        numbered("Se construye un reporte consolidado (unión de todas las fuentes) ordenado por severidad."),
        numbered("Se genera el HTML del reporte con resumen por (MODEL_VERSION, fuente) y secciones de detalle con anclas por fuente y versión."),
        numbered("Se construye un payload JSON con el detalle de alertas para integración con sistemas externos."),
        numbered("El notebook incluye (comentado) el envío/publicación vía SYSTEM$SEND_SNOWFLAKE_NOTIFICATION, que requiere que las Notification Integrations estén configuradas en la cuenta."),
  
      ]
    }]
  });
  
  Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("/mnt/user-data/outputs/KT_Snowflake_MLOps.docx", buffer);
    console.log("✅ Documento KT generado exitosamente");
  }).catch(err => {
    console.error("❌ Error:", err);
    process.exit(1);
  });