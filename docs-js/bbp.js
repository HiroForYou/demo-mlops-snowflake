const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    HeadingLevel, AlignmentType, LevelFormat, BorderStyle, WidthType,
    ShadingType, VerticalAlign, PageBreak, ImageRun, Header, Footer,
    NumberFormat, TabStopType, TabStopPosition, PositionalTab,
    PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader
  } = require('docx');
  const fs = require('fs');
  const path = require('path');
  
  // ─── helpers ───────────────────────────────────────────────────────────────
  const BLUE = "1F4E79";
  const LIGHT_BLUE = "2E75B6";
  const MEDIUM_BLUE = "D6E4F0";
  const LIGHT_GRAY = "F5F5F5";
  const DARK_GRAY = "595959";
  const WHITE = "FFFFFF";
  
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
  
  let _numGroupCounter = 0;
  let _numCurrentRef = null;
  function numberedGroup() {
    // Call this before a new numbered list to start fresh numbering
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
  
  function badge(text, color = LIGHT_BLUE) {
    return new Paragraph({
      children: [
        new TextRun({ text: `  ${text}  `, font: "Arial", size: 18, bold: true, color: WHITE, shading: { type: ShadingType.CLEAR, fill: color } })
      ],
      spacing: { before: 60, after: 60 }
    });
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
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360], rows, borders: { top: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, bottom: border1, left: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, right: border1 } });
  }
  
  function pendingBox(text) {
    return infoBox("⚠️  PENDIENTE DE DEFINICIÓN", [text], "FFF3CD");
  }
  
  function sectionTag(label, color) {
    // color: "2E75B6" for POC, "1F7A4D" for PROD, "7030A0" for FUTURO
    return new Paragraph({
      children: [
        new TextRun({ text: `  ▶  ${label}  `, font: "Arial", size: 16, bold: true, color: WHITE, shading: { type: ShadingType.CLEAR, fill: color } })
      ],
      spacing: { before: 40, after: 60 }
    });
  }
  
  function twoColTable(col1Items, col2Items, header1, header2) {
    const headerRow = new TableRow({
      children: [
        new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: header1, font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
        new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: header2, font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
      ]
    });
    const maxRows = Math.max(col1Items.length, col2Items.length);
    const dataRows = [...Array(maxRows)].map((_, i) => new TableRow({
      children: [
        new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: col1Items[i] || "", font: "Arial", size: 20 })] })] }),
        new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: col2Items[i] || "", font: "Arial", size: 20 })] })] }),
      ]
    }));
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 4680], rows: [headerRow, ...dataRows] });
  }
  
  function threeColTable(rows3, headers) {
    const w = [2500, 3430, 3430];
    const headerRow = new TableRow({
      children: headers.map((h, i) => new TableCell({
        borders, shading: { fill: BLUE, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        width: { size: w[i], type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })]
      }))
    });
    const dataRows = rows3.map((row, ri) => new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        shading: { fill: ri % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        width: { size: w[ci], type: WidthType.DXA },
        children: (Array.isArray(cell) ? cell : [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 20 })] })])
      }))
    }));
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: w, rows: [headerRow, ...dataRows] });
  }
  
  function codeBlock(lines) {
    return new Table({
      width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
      rows: [new TableRow({ children: [new TableCell({
        borders: noBorders,
        shading: { fill: "2B2B2B", type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 200 },
        width: { size: 9360, type: WidthType.DXA },
        children: lines.map(l => new Paragraph({ children: [new TextRun({ text: l, font: "Courier New", size: 18, color: "F8F8F2" })] }))
      })] })]
    });
  }
  
  // ─── load logo image ────────────────────────────────────────────────────────
  const logoPath = "/home/claude/bbp_unpacked/word/media/image1.png";
  const logoData = fs.readFileSync(logoPath);
  
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
        {
          reference: "numbers_1",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_2",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_3",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_4",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_5",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_6",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_7",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_8",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_9",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_10",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_11",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_12",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_13",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_14",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_15",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_16",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_17",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_18",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_19",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_20",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_21",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_22",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_23",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_24",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_25",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
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
              new TextRun({ text: "BBP — Migración del Modelo de Pronóstico de Ventas: Databricks → Snowflake", font: "Arial", size: 16, color: DARK_GRAY }),
              new TextRun({ text: "   |   Versión 3.0   |   Confidencial", font: "Arial", size: 16, color: DARK_GRAY, italics: true })
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
          children: [new ImageRun({ data: logoData, transformation: { width: 180, height: 73 }, type: "png" })],
          spacing: { before: 400, after: 600 }
        }),
        new Paragraph({
          children: [new TextRun({ text: "BUSINESS BLUEPRINT", font: "Arial", size: 48, bold: true, color: BLUE, allCaps: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 120 }
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
          children: [new TextRun({ text: "Versión 3.0  ·  Enero 2025  ·  Confidencial", font: "Arial", size: 20, color: DARK_GRAY, italics: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 600 }
        }),
  
        // Tabla de aprobación
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 2340, 2340, 2340],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, columnSpan: 4, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "BUSINESS BLUEPRINT — APROBACIÓN", font: "Arial", size: 20, bold: true, color: WHITE, allCaps: true })] })] }),
            ]}),
            new TableRow({ children: ["Documento", "Nombre", "Fecha de aprobación", "Firma"].map(h => new TableCell({ borders, shading: { fill: MEDIUM_BLUE, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 18, bold: true })] })] })) }),
            ...[1, 2, 3].map(() => new TableRow({ children: ["", "", "", ""].map(() => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: " ", font: "Arial", size: 18 })] })] })) }))
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // LEYENDA DE ESTRUCTURA
        // ══════════════════════════════════════════════════
        h1("Guía de Lectura del Documento"),
        para("Este documento está organizado en tres bloques estratégicos claramente diferenciados. Cada sección está etiquetada con un indicador de alcance para facilitar la comprensión del estado actual frente a la evolución futura:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [1800, 7560],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: LIGHT_BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ POC ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Implementado y validado en la Prueba de Concepto. Componentes reales, ejecutables y demostrados.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "1F7A4D", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ PROD ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Mejoras y componentes a formalizar en la siguiente fase hacia ambiente productivo.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "7030A0", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ FUTURO ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Roadmap estratégico: automatización avanzada, escalabilidad y observabilidad ampliada.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "C55A11", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ PENDIENTE ]", font: "Arial", size: 18, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Componente o decisión aún no definida formalmente. Requiere validación con el cliente.", font: "Arial", size: 20 })] })] }),
            ]}),
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 1. INTRODUCCIÓN
        // ══════════════════════════════════════════════════
        h1("1. Introducción"),
        para("Arca Continental manifestó interés en evaluar de extremo a extremo las capacidades de Snowflake para desplegar modelos de ML mediante MLOps (Model Registry, Feature Store, ML Observability) y abordar la migración de su modelo de Pronóstico de Ventas Semanales, actualmente en ejecución en Databricks, hacia Snowflake."),
        para("Este documento describe la arquitectura técnica propuesta, el alcance de la Prueba de Concepto (POC) realizada y la evolución planificada hacia un ambiente productivo robusto, gobernado y escalable."),
  
        ...spacer(1),
        h1("2. Objetivos"),
        para("El objetivo principal es plantear y validar la migración del modelo de ML de Pronóstico de Ventas Semanales (uni_box_week), el cual fue analizado mediante el Snowpark Migration Accelerator (SMA), hacia Snowflake, aprovechando la oportunidad para evaluar el ciclo de vida completo de MLOps con las mejores prácticas de la plataforma."),
        para("Los objetivos específicos incluyen:"),
        bullet("Validar la factibilidad técnica del ciclo completo MLOps en Snowflake (entrenamiento, registro, inferencia, monitoreo)."),
        bullet("Reproducir funcionalmente la arquitectura MLOps actualmente implementada en Databricks."),
        bullet("Simplificar la operación mediante capacidades nativas de Snowflake ML, reduciendo la complejidad y la dependencia de componentes externos."),
        bullet("Establecer las bases técnicas para la evolución hacia un ambiente productivo automatizado, gobernado y auditable."),
  
        ...spacer(1),
        h1("3. Alcance"),
        h2("3.1 Estándar Arquitectónico Adoptado"),
        para("La arquitectura propuesta homologa el uso de tablas físicas (tablas fijas) como estándar de almacenamiento en lugar de Dynamic Tables o Snowflake Tasks. Esta decisión responde a las restricciones de permisos actuales del entorno de desarrollo, y se mantendrá como base hasta que se defina formalmente la estrategia de orquestación productiva."),
        infoBox("Decisiones de Estandarización", [
          "• Tablas físicas (fijas) como estándar de almacenamiento para features y resultados.",
          "• Cálculo de lags y ventanas temporales mediante SQL, integrados en dichas tablas.",
          "• Python (Snowpark) y SQL como únicos lenguajes en los notebooks.",
          "• Sin uso de Snowflake Tasks ni tablas dinámicas en la POC por restricciones de permisos.",
          "• Orquestación externa o manual hasta definición formal de la estrategia productiva."
        ]),
        ...spacer(1),
        para("El uso exclusivo de Python y SQL responde a los siguientes criterios técnicos:"),
        bullet("Compatibilidad nativa con las APIs de Snowflake ML (snowflake.ml.modeling, Feature Store, Registry)."),
        bullet("Estandarización del stack tecnológico, facilitando el mantenimiento y la transferencia de conocimiento."),
        bullet("Gobernanza y auditabilidad: los notebooks en Python + SQL son inspeccionables, versionables y reproducibles dentro del ecosistema Snowflake."),
        bullet("Eliminación de dependencias externas (frameworks adicionales, entornos virtuales complejos, librerías de orquestación de terceros)."),
  
        ...spacer(1),
        h2("3.2 Lo que Incluye la POC"),
        sectionTag("POC", LIGHT_BLUE),
        bullet("Ejecución del entrenamiento de modelos sobre datasets previamente preparados y disponibles en Snowflake."),
        bullet("Implementación del Feature Store como tablas físicas con features calculadas mediante SQL (lags, rolling aggregations)."),
        bullet("Búsqueda de hiperparámetros (HPO) por grupo utilizando Bayesian Search (snowflake.ml.modeling.tune)."),
        bullet("Entrenamiento de 16 sub-modelos en paralelo mediante Many Model Training (MMT)."),
        bullet("Registro y versionado de modelos en el Snowflake Model Registry."),
        bullet("Inferencia batch particionada mediante sintaxis SQL nativa."),
        bullet("Configuración inicial de ML Observability para monitoreo de desempeño y drift."),
  
        ...spacer(1),
        h2("3.3 Lo que NO Incluye la POC"),
        bullet("Ingesta, limpieza o transformación de datos desde fuentes origen (los datasets son provistos ya curados por Arca Continental)."),
        bullet("Diseño o implementación de pipelines de datos bajo el modelo Medallón."),
        bullet("Definición o ajuste de métricas de negocio, KPIs o umbrales finales de aceptación."),
        bullet("Orquestación completamente automatizada end-to-end (Snowflake Tasks, ML Jobs)."),
        bullet("Parametrización dinámica mediante tablas de configuración centralizadas."),
        bullet("CI/CD automatizado con validación en cada etapa."),
        bullet("Real-time inference y endpoints de serving (SPCS)."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 4. DATASETS
        // ══════════════════════════════════════════════════
        h1("4. Datasets"),
        para("Los datasets utilizados en la POC fueron provistos por Arca Continental de forma curada y lista para su uso como fuente de entrenamiento e inferencia:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2400, 2400, 4560],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Dataset", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Volumen", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Descripción", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Training", font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "~96 millones de filas (2.9 GB)", font: "Arial", size: 20 })] })] }),
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Ventas semanales por cliente. Información del año 2025 dividida por semanas. Segmentado en 16 grupos (STATS_NTILE_GROUP).", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Inference", font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "~24.7 millones de filas (807.2 MB)", font: "Arial", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Misma granularidad que el dataset de entrenamiento. Utilizado para validar predicciones y evaluar el comportamiento productivo del modelo.", font: "Arial", size: 20 })] })] }),
            ]}),
          ]
        }),
        ...spacer(1),
        h3("Gestión del Test Set y Predicciones Productivas"),
        sectionTag("POC", LIGHT_BLUE),
        para("Una consideración crítica de diseño es la diferenciación entre el conjunto de prueba (test set) utilizado durante el entrenamiento y las predicciones productivas semanales. Estas dos operaciones sirven propósitos distintos y deben mantenerse separadas arquitectónicamente:"),
        bullet("Test Set: Subconjunto del dataset de entrenamiento reservado para evaluación de métricas. El split 80/20 se realiza de forma estratificada por grupo (STATS_NTILE_GROUP) para mantener la distribución representativa en cada segmento. Se verifica la homogeneidad mediante KS test o comparación de medias entre train y test por grupo."),
        bullet("Predicciones productivas: Se ejecutan únicamente sobre el dataset de inferencia correspondiente a la semana en curso, sin mezcla con el test set de entrenamiento."),
        ...spacer(1),
        pendingBox("Temporalidad y automatización semanal: La estrategia de cómo se parametrizará la semana de inferencia (rolling window, fecha fija, trigger automático) está pendiente de definición. Se evaluará si Snowflake Tasks permite conversión automática de fechas a partir del número de semana como parámetro de ejecución."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 5. ARQUITECTURA ACTUAL (DATABRICKS)
        // ══════════════════════════════════════════════════
        h1("5. Arquitectura Actual en Databricks"),
        para("El siguiente diagrama representa la arquitectura MLOps en Databricks, compartida por Arca Continental como referencia para el análisis y definición de la arquitectura objetivo en Snowflake."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image2.png"), transformation: { width: 620, height: 330 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        para("La arquitectura actual implementa un enfoque de entrenamiento de múltiples modelos especializados. Los puntos clave son:"),
        ...spacer(1),
        resetNum(),
        numbered("Segmentación del dataset de entrenamiento en 16 grupos basados en patrones recientes de venta (últimas 4 semanas) y volumen histórico por cuartiles (0-25%, 25-50%, 50-75%, 75-100%)."),
        numbered("Entrenamiento distribuido manual: 16 sub-modelos entrenados individualmente con AutoML de Databricks (60 min timeout por modelo), resultando en ~16 horas secuenciales de cómputo."),
        numbered("Ensemble Orchestration Custom: librería \"Pandas Ensemble\" personalizada que unifica los 16 sub-modelos en un objeto lógico único con ruteo automático por grupo en inferencia."),
        numbered("Gestión con MLflow: versionado por timestamp, aliases por fecha de ejecución (ej. 20250910), tags con execution_date y cutoff_date."),
        numbered("Monitoreo manual con alertas no proactivas: Data Drift (Jensen-Shannon >0.2 Warning, >0.4 Critical), Prediction Drift por distribución histórica vs actual, y Performance Drift con WAPE + F1 Score retroactivo."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 6. ARQUITECTURA PROPUESTA EN SNOWFLAKE
        // ══════════════════════════════════════════════════
        h1("6. Arquitectura Propuesta en Snowflake"),
        h2("6.1 Visión General de la POC"),
        sectionTag("POC", LIGHT_BLUE),
        para("La arquitectura propuesta para la POC en Snowflake tiene como objetivo reproducir el flujo MLOps de Databricks manteniendo la equivalencia funcional del modelo de Pronóstico de Ventas, pero simplificando su operación mediante capacidades nativas de Snowflake ML."),
        para("Procesos como el entrenamiento de múltiples sub-modelos, el ruteo de inferencias, el versionado de modelos y el monitoreo del desempeño se resuelven mediante HPO manual + MMT, ML Experiments y ML Registry, en lugar de AutoML propietario y componentes custom."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image3.png"), transformation: { width: 620, height: 290 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
  
        ...spacer(1),
        h2("6.2 Comparativa de Capacidades MLOps"),
        para("La siguiente tabla resume la equivalencia funcional entre Databricks y Snowflake para cada componente del ciclo MLOps:"),
        ...spacer(1),
        threeColTable([
          ["Entrenamiento", "AutoML Databricks + loops manuales de segmentación", "HPO Bayesian Search + Many Model Training (MMT)"],
          ["Inferencia", "Pandas Ensemble custom (PyFunc) con ruteo manual", "Partitioned Model Inference con ruteo automático nativo"],
          ["Gestión de Modelos", "MLflow externo con aliases y tags", "Snowflake Model Registry integrado (versiones, aliases, tags)"],
          ["Orquestación", "Notebooks manuales + loops secuenciales", "POC: manual / PROD: Snowflake Tasks o ML Jobs"],
          ["Observabilidad", "Notebooks custom de monitoreo desacoplados", "ML Observability nativo con monitores configurables"],
          ["Feature Management", "Tablas Delta en Databricks", "Tablas físicas + SQL (lags calculados, sin FeatureView por permisos)"],
          ["Seguridad", "Permisos a nivel de catálogo Databricks", "RBAC nativo Snowflake (roles ML_DEV, ML_OPS, ML_CONSUMER)"],
          ["Deep Learning", "No utilizado", "No aplica: datos tabulares estructurados. Gradient boosting supera a redes neuronales en este dominio con menor overhead operativo."],
        ], ["Componente", "Databricks (actual)", "Snowflake (propuesto)"]),
  
        ...spacer(1),
        h2("6.3 Propuesta Conceptual AutoML en Snowflake"),
        para("Snowflake no dispone actualmente de una solución AutoML nativa equivalente a la de Databricks. Sin embargo, es posible construir un enfoque equivalente combinando HPO (Hyperparameter Optimization) y MMT (Many Model Training). Ambas estrategias se orquestan de forma controlada con el objetivo de equilibrar los tiempos de ejecución y evitar desbalances injustificados entre fases del pipeline."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image9.png"), transformation: { width: 560, height: 305 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        para("El enfoque opera en dos etapas diferenciadas:"),
        bullet("Etapa 1 — HPO: explora el espacio de hiperparámetros y selecciona el algoritmo óptimo por segmento. Actúa como una forma de orquestación manual controlada. Se ejecuta puntualmente, no en cada ciclo de reentrenamiento."),
        bullet("Etapa 2 — MMT: utiliza los hiperparámetros validados por HPO para entrenar los 16 modelos finales de forma paralela (~7 minutos en cluster de 5 nodos). La arquitectura simplifica componentes intermedios, delegando trazabilidad al Model Registry y ML Experiments."),
        para("Esta combinación reproduce el comportamiento de un AutoML completo (exploración de modelos + entrenamiento optimizado) sin depender de soluciones propietarias externas, manteniéndose íntegramente dentro del ecosistema Snowflake. La descripción detallada de la propuesta conceptual y el diagrama de arquitectura se encuentran en la sección 8.3."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 7. FEATURE STORE
        // ══════════════════════════════════════════════════
        h1("7. Feature Store"),
        h2("7.1 Implementación en la POC — Tablas Físicas con Cálculo Manual"),
        sectionTag("POC", LIGHT_BLUE),
        para("En la POC, el Feature Store se implementa mediante tablas físicas estáticas en el schema SC_FEATURES_BMX. Esta decisión responde a las restricciones de permisos del entorno de desarrollo (sin Tasks ni Dynamic Tables) y al objetivo de la POC: validar el ciclo MLOps completo con el menor overhead operativo posible."),
        para("El flujo de construcción de features se ejecuta manualmente en cada ciclo:"),
        resetNum(),
        numbered("Las tablas de ventas, clientes y productos (capa Gold) se toman como fuente única de datos."),
        numbered("Se ejecutan transformaciones SQL directamente sobre Snowflake para calcular las features de lag y ventanas temporales necesarias para el modelo."),
        numbered("Los resultados se materializan en la tabla UNI_BOX_FEATURES, lista para ser consumida por el proceso de entrenamiento (HPO + MMT) y por inferencia."),
        numbered("Se mantiene una tabla FEATURE_VERSIONS que registra el timestamp del snapshot activo y su identificador, garantizando trazabilidad entre la versión de features y la versión del modelo entrenado."),
        ...spacer(1),
        para("Las features disponibles en la POC incluyen ventanas temporales de ventas pasadas (SUM_PAST_4_WEEKS, AVG_PAST_12_WEEKS, MAX_PAST_24_WEEKS, entre otras), métricas por punto de venta (NUM_COOLERS, NUM_DOORS) y variables categóricas de segmento (PHARM_SUPER_CONV, WINES_LIQUOR, GROCERIES, SPEC_FOODS)."),
        ...spacer(1),
        infoBox("Cálculo de Lags mediante Window Functions SQL", [
          "Los lags y ventanas temporales se calculan con funciones de ventana estándar de SQL. Ejemplo:",
          "  SUM(ventas) OVER (PARTITION BY ENTITY_ID ORDER BY SEMANA ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)",
          "Este enfoque es reproducible, auditable y no requiere dependencias adicionales. El cálculo se ejecuta manualmente antes de cada ciclo de entrenamiento o inferencia.",
        ]),
  
        ...spacer(1),
  
        h2("7.2 Evolución del Feature Store"),
  
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
  
  let _numGroupCounter = 0;
  let _numCurrentRef = null;
  function numberedGroup() {
    // Call this before a new numbered list to start fresh numbering
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
  
  function badge(text, color = LIGHT_BLUE) {
    return new Paragraph({
      children: [
        new TextRun({ text: `  ${text}  `, font: "Arial", size: 18, bold: true, color: WHITE, shading: { type: ShadingType.CLEAR, fill: color } })
      ],
      spacing: { before: 60, after: 60 }
    });
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
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360], rows, borders: { top: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, bottom: border1, left: { style: BorderStyle.SINGLE, size: 4, color: LIGHT_BLUE }, right: border1 } });
  }
  
  function pendingBox(text) {
    return infoBox("⚠️  PENDIENTE DE DEFINICIÓN", [text], "FFF3CD");
  }
  
  function sectionTag(label, color) {
    // color: "2E75B6" for POC, "1F7A4D" for PROD, "7030A0" for FUTURO
    return new Paragraph({
      children: [
        new TextRun({ text: `  ▶  ${label}  `, font: "Arial", size: 16, bold: true, color: WHITE, shading: { type: ShadingType.CLEAR, fill: color } })
      ],
      spacing: { before: 40, after: 60 }
    });
  }
  
  function twoColTable(col1Items, col2Items, header1, header2) {
    const headerRow = new TableRow({
      children: [
        new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: header1, font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
        new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: header2, font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
      ]
    });
    const maxRows = Math.max(col1Items.length, col2Items.length);
    const dataRows = [...Array(maxRows)].map((_, i) => new TableRow({
      children: [
        new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: col1Items[i] || "", font: "Arial", size: 20 })] })] }),
        new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4680, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: col2Items[i] || "", font: "Arial", size: 20 })] })] }),
      ]
    }));
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 4680], rows: [headerRow, ...dataRows] });
  }
  
  function threeColTable(rows3, headers) {
    const w = [2500, 3430, 3430];
    const headerRow = new TableRow({
      children: headers.map((h, i) => new TableCell({
        borders, shading: { fill: BLUE, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        width: { size: w[i], type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })]
      }))
    });
    const dataRows = rows3.map((row, ri) => new TableRow({
      children: row.map((cell, ci) => new TableCell({
        borders,
        shading: { fill: ri % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 120, right: 120 },
        width: { size: w[ci], type: WidthType.DXA },
        children: (Array.isArray(cell) ? cell : [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 20 })] })])
      }))
    }));
    return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: w, rows: [headerRow, ...dataRows] });
  }
  
  function codeBlock(lines) {
    return new Table({
      width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
      rows: [new TableRow({ children: [new TableCell({
        borders: noBorders,
        shading: { fill: "2B2B2B", type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 200 },
        width: { size: 9360, type: WidthType.DXA },
        children: lines.map(l => new Paragraph({ children: [new TextRun({ text: l, font: "Courier New", size: 18, color: "F8F8F2" })] }))
      })] })]
    });
  }
  
  // ─── load logo image ────────────────────────────────────────────────────────
  const logoPath = "/home/claude/bbp_unpacked/word/media/image1.png";
  const logoData = fs.readFileSync(logoPath);
  
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
        {
          reference: "numbers_1",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_2",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_3",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_4",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_5",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_6",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_7",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_8",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_9",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_10",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_11",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_12",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_13",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_14",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_15",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_16",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_17",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_18",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_19",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_20",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_21",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_22",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_23",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_24",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
        {
          reference: "numbers_25",
          levels: [
            { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
            { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
          ]
        },
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
              new TextRun({ text: "BBP — Migración del Modelo de Pronóstico de Ventas: Databricks → Snowflake", font: "Arial", size: 16, color: DARK_GRAY }),
              new TextRun({ text: "   |   Versión 3.0   |   Confidencial", font: "Arial", size: 16, color: DARK_GRAY, italics: true })
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
          children: [new ImageRun({ data: logoData, transformation: { width: 180, height: 73 }, type: "png" })],
          spacing: { before: 400, after: 600 }
        }),
        new Paragraph({
          children: [new TextRun({ text: "BUSINESS BLUEPRINT", font: "Arial", size: 48, bold: true, color: BLUE, allCaps: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 120 }
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
          children: [new TextRun({ text: "Versión 3.0  ·  Enero 2025  ·  Confidencial", font: "Arial", size: 20, color: DARK_GRAY, italics: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 600 }
        }),
  
        // Tabla de aprobación
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 2340, 2340, 2340],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, columnSpan: 4, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "BUSINESS BLUEPRINT — APROBACIÓN", font: "Arial", size: 20, bold: true, color: WHITE, allCaps: true })] })] }),
            ]}),
            new TableRow({ children: ["Documento", "Nombre", "Fecha de aprobación", "Firma"].map(h => new TableCell({ borders, shading: { fill: MEDIUM_BLUE, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 18, bold: true })] })] })) }),
            ...[1, 2, 3].map(() => new TableRow({ children: ["", "", "", ""].map(() => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2340, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: " ", font: "Arial", size: 18 })] })] })) }))
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // LEYENDA DE ESTRUCTURA
        // ══════════════════════════════════════════════════
        h1("Guía de Lectura del Documento"),
        para("Este documento está organizado en tres bloques estratégicos claramente diferenciados. Cada sección está etiquetada con un indicador de alcance para facilitar la comprensión del estado actual frente a la evolución futura:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [1800, 7560],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: LIGHT_BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ POC ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Implementado y validado en la Prueba de Concepto. Componentes reales, ejecutables y demostrados.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "1F7A4D", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ PROD ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Mejoras y componentes a formalizar en la siguiente fase hacia ambiente productivo.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "7030A0", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ FUTURO ]", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Roadmap estratégico: automatización avanzada, escalabilidad y observabilidad ampliada.", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: "C55A11", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 1800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[ PENDIENTE ]", font: "Arial", size: 18, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 7560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Componente o decisión aún no definida formalmente. Requiere validación con el cliente.", font: "Arial", size: 20 })] })] }),
            ]}),
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 1. INTRODUCCIÓN
        // ══════════════════════════════════════════════════
        h1("1. Introducción"),
        para("Arca Continental manifestó interés en evaluar de extremo a extremo las capacidades de Snowflake para desplegar modelos de ML mediante MLOps (Model Registry, Feature Store, ML Observability) y abordar la migración de su modelo de Pronóstico de Ventas Semanales, actualmente en ejecución en Databricks, hacia Snowflake."),
        para("Este documento describe la arquitectura técnica propuesta, el alcance de la Prueba de Concepto (POC) realizada y la evolución planificada hacia un ambiente productivo robusto, gobernado y escalable."),
  
        ...spacer(1),
        h1("2. Objetivos"),
        para("El objetivo principal es plantear y validar la migración del modelo de ML de Pronóstico de Ventas Semanales (uni_box_week), el cual fue analizado mediante el Snowpark Migration Accelerator (SMA), hacia Snowflake, aprovechando la oportunidad para evaluar el ciclo de vida completo de MLOps con las mejores prácticas de la plataforma."),
        para("Los objetivos específicos incluyen:"),
        bullet("Validar la factibilidad técnica del ciclo completo MLOps en Snowflake (entrenamiento, registro, inferencia, monitoreo)."),
        bullet("Reproducir funcionalmente la arquitectura MLOps actualmente implementada en Databricks."),
        bullet("Simplificar la operación mediante capacidades nativas de Snowflake ML, reduciendo la complejidad y la dependencia de componentes externos."),
        bullet("Establecer las bases técnicas para la evolución hacia un ambiente productivo automatizado, gobernado y auditable."),
  
        ...spacer(1),
        h1("3. Alcance"),
        h2("3.1 Estándar Arquitectónico Adoptado"),
        para("La arquitectura propuesta homologa el uso de tablas físicas (tablas fijas) como estándar de almacenamiento en lugar de Dynamic Tables o Snowflake Tasks. Esta decisión responde a las restricciones de permisos actuales del entorno de desarrollo, y se mantendrá como base hasta que se defina formalmente la estrategia de orquestación productiva."),
        infoBox("Decisiones de Estandarización", [
          "• Tablas físicas (fijas) como estándar de almacenamiento para features y resultados.",
          "• Cálculo de lags y ventanas temporales mediante SQL, integrados en dichas tablas.",
          "• Python (Snowpark) y SQL como únicos lenguajes en los notebooks.",
          "• Sin uso de Snowflake Tasks ni tablas dinámicas en la POC por restricciones de permisos.",
          "• Orquestación externa o manual hasta definición formal de la estrategia productiva."
        ]),
        ...spacer(1),
        para("El uso exclusivo de Python y SQL responde a los siguientes criterios técnicos:"),
        bullet("Compatibilidad nativa con las APIs de Snowflake ML (snowflake.ml.modeling, Feature Store, Registry)."),
        bullet("Estandarización del stack tecnológico, facilitando el mantenimiento y la transferencia de conocimiento."),
        bullet("Gobernanza y auditabilidad: los notebooks en Python + SQL son inspeccionables, versionables y reproducibles dentro del ecosistema Snowflake."),
        bullet("Eliminación de dependencias externas (frameworks adicionales, entornos virtuales complejos, librerías de orquestación de terceros)."),
  
        ...spacer(1),
        h2("3.2 Lo que Incluye la POC"),
        sectionTag("POC", LIGHT_BLUE),
        bullet("Ejecución del entrenamiento de modelos sobre datasets previamente preparados y disponibles en Snowflake."),
        bullet("Implementación del Feature Store como tablas físicas con features calculadas mediante SQL (lags, rolling aggregations)."),
        bullet("Búsqueda de hiperparámetros (HPO) por grupo utilizando Bayesian Search (snowflake.ml.modeling.tune)."),
        bullet("Entrenamiento de 16 sub-modelos en paralelo mediante Many Model Training (MMT)."),
        bullet("Registro y versionado de modelos en el Snowflake Model Registry."),
        bullet("Inferencia batch particionada mediante sintaxis SQL nativa."),
        bullet("Configuración inicial de ML Observability para monitoreo de desempeño y drift."),
  
        ...spacer(1),
        h2("3.3 Lo que NO Incluye la POC"),
        bullet("Ingesta, limpieza o transformación de datos desde fuentes origen (los datasets son provistos ya curados por Arca Continental)."),
        bullet("Diseño o implementación de pipelines de datos bajo el modelo Medallón."),
        bullet("Definición o ajuste de métricas de negocio, KPIs o umbrales finales de aceptación."),
        bullet("Orquestación completamente automatizada end-to-end (Snowflake Tasks, ML Jobs)."),
        bullet("Parametrización dinámica mediante tablas de configuración centralizadas."),
        bullet("CI/CD automatizado con validación en cada etapa."),
        bullet("Real-time inference y endpoints de serving (SPCS)."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 4. DATASETS
        // ══════════════════════════════════════════════════
        h1("4. Datasets"),
        para("Los datasets utilizados en la POC fueron provistos por Arca Continental de forma curada y lista para su uso como fuente de entrenamiento e inferencia:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2400, 2400, 4560],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Dataset", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Volumen", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Descripción", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Training", font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "~96 millones de filas (2.9 GB)", font: "Arial", size: 20 })] })] }),
              new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Ventas semanales por cliente. Información del año 2025 dividida por semanas. Segmentado en 16 grupos (STATS_NTILE_GROUP).", font: "Arial", size: 20 })] })] }),
            ]}),
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Inference", font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "~24.7 millones de filas (807.2 MB)", font: "Arial", size: 20 })] })] }),
              new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 4560, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Misma granularidad que el dataset de entrenamiento. Utilizado para validar predicciones y evaluar el comportamiento productivo del modelo.", font: "Arial", size: 20 })] })] }),
            ]}),
          ]
        }),
        ...spacer(1),
        h3("Gestión del Test Set y Predicciones Productivas"),
        sectionTag("POC", LIGHT_BLUE),
        para("Una consideración crítica de diseño es la diferenciación entre el conjunto de prueba (test set) utilizado durante el entrenamiento y las predicciones productivas semanales. Estas dos operaciones sirven propósitos distintos y deben mantenerse separadas arquitectónicamente:"),
        bullet("Test Set: Subconjunto del dataset de entrenamiento reservado para evaluación de métricas. El split 80/20 se realiza de forma estratificada por grupo (STATS_NTILE_GROUP) para mantener la distribución representativa en cada segmento. Se verifica la homogeneidad mediante KS test o comparación de medias entre train y test por grupo."),
        bullet("Predicciones productivas: Se ejecutan únicamente sobre el dataset de inferencia correspondiente a la semana en curso, sin mezcla con el test set de entrenamiento."),
        ...spacer(1),
        pendingBox("Temporalidad y automatización semanal: La estrategia de cómo se parametrizará la semana de inferencia (rolling window, fecha fija, trigger automático) está pendiente de definición. Se evaluará si Snowflake Tasks permite conversión automática de fechas a partir del número de semana como parámetro de ejecución."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 5. ARQUITECTURA ACTUAL (DATABRICKS)
        // ══════════════════════════════════════════════════
        h1("5. Arquitectura Actual en Databricks"),
        para("El siguiente diagrama representa la arquitectura MLOps en Databricks, compartida por Arca Continental como referencia para el análisis y definición de la arquitectura objetivo en Snowflake."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image2.png"), transformation: { width: 620, height: 330 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        para("La arquitectura actual implementa un enfoque de entrenamiento de múltiples modelos especializados. Los puntos clave son:"),
        ...spacer(1),
        resetNum(),
        numbered("Segmentación del dataset de entrenamiento en 16 grupos basados en patrones recientes de venta (últimas 4 semanas) y volumen histórico por cuartiles (0-25%, 25-50%, 50-75%, 75-100%)."),
        numbered("Entrenamiento distribuido manual: 16 sub-modelos entrenados individualmente con AutoML de Databricks (60 min timeout por modelo), resultando en ~16 horas secuenciales de cómputo."),
        numbered("Ensemble Orchestration Custom: librería \"Pandas Ensemble\" personalizada que unifica los 16 sub-modelos en un objeto lógico único con ruteo automático por grupo en inferencia."),
        numbered("Gestión con MLflow: versionado por timestamp, aliases por fecha de ejecución (ej. 20250910), tags con execution_date y cutoff_date."),
        numbered("Monitoreo manual con alertas no proactivas: Data Drift (Jensen-Shannon >0.2 Warning, >0.4 Critical), Prediction Drift por distribución histórica vs actual, y Performance Drift con WAPE + F1 Score retroactivo."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 6. ARQUITECTURA PROPUESTA EN SNOWFLAKE
        // ══════════════════════════════════════════════════
        h1("6. Arquitectura Propuesta en Snowflake"),
        h2("6.1 Visión General de la POC"),
        sectionTag("POC", LIGHT_BLUE),
        para("La arquitectura propuesta para la POC en Snowflake tiene como objetivo reproducir el flujo MLOps de Databricks manteniendo la equivalencia funcional del modelo de Pronóstico de Ventas, pero simplificando su operación mediante capacidades nativas de Snowflake ML."),
        para("Procesos como el entrenamiento de múltiples sub-modelos, el ruteo de inferencias, el versionado de modelos y el monitoreo del desempeño se resuelven mediante HPO manual + MMT, ML Experiments y ML Registry, en lugar de AutoML propietario y componentes custom."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image3.png"), transformation: { width: 620, height: 290 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
  
        ...spacer(1),
        h2("6.2 Comparativa de Capacidades MLOps"),
        para("La siguiente tabla resume la equivalencia funcional entre Databricks y Snowflake para cada componente del ciclo MLOps:"),
        ...spacer(1),
        threeColTable([
          ["Entrenamiento", "AutoML Databricks + loops manuales de segmentación", "HPO Bayesian Search + Many Model Training (MMT)"],
          ["Inferencia", "Pandas Ensemble custom (PyFunc) con ruteo manual", "Partitioned Model Inference con ruteo automático nativo"],
          ["Gestión de Modelos", "MLflow externo con aliases y tags", "Snowflake Model Registry integrado (versiones, aliases, tags)"],
          ["Orquestación", "Notebooks manuales + loops secuenciales", "POC: manual / PROD: Snowflake Tasks o ML Jobs"],
          ["Observabilidad", "Notebooks custom de monitoreo desacoplados", "ML Observability nativo con monitores configurables"],
          ["Feature Management", "Tablas Delta en Databricks", "Tablas físicas + SQL (lags calculados, sin FeatureView por permisos)"],
          ["Seguridad", "Permisos a nivel de catálogo Databricks", "RBAC nativo Snowflake (roles ML_DEV, ML_OPS, ML_CONSUMER)"],
          ["Deep Learning", "No utilizado", "No aplica: datos tabulares estructurados. Gradient boosting supera a redes neuronales en este dominio con menor overhead operativo."],
        ], ["Componente", "Databricks (actual)", "Snowflake (propuesto)"]),
  
        ...spacer(1),
        h2("6.3 Propuesta Conceptual AutoML en Snowflake"),
        para("Snowflake no dispone actualmente de una solución AutoML nativa equivalente a la de Databricks. Sin embargo, es posible construir un enfoque equivalente combinando HPO (Hyperparameter Optimization) y MMT (Many Model Training). Ambas estrategias se orquestan de forma controlada con el objetivo de equilibrar los tiempos de ejecución y evitar desbalances injustificados entre fases del pipeline."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image9.png"), transformation: { width: 560, height: 305 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        para("El enfoque opera en dos etapas diferenciadas:"),
        bullet("Etapa 1 — HPO: explora el espacio de hiperparámetros y selecciona el algoritmo óptimo por segmento. Actúa como una forma de orquestación manual controlada. Se ejecuta puntualmente, no en cada ciclo de reentrenamiento."),
        bullet("Etapa 2 — MMT: utiliza los hiperparámetros validados por HPO para entrenar los 16 modelos finales de forma paralela (~7 minutos en cluster de 5 nodos). La arquitectura simplifica componentes intermedios, delegando trazabilidad al Model Registry y ML Experiments."),
        para("Esta combinación reproduce el comportamiento de un AutoML completo (exploración de modelos + entrenamiento optimizado) sin depender de soluciones propietarias externas, manteniéndose íntegramente dentro del ecosistema Snowflake. La descripción detallada de la propuesta conceptual y el diagrama de arquitectura se encuentran en la sección 8.3."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 7. FEATURE STORE
        // ══════════════════════════════════════════════════
        h1("7. Feature Store"),
        h2("7.1 Implementación en la POC — Arquitectura Batch Incremental"),
        sectionTag("POC", LIGHT_BLUE),
        para("El Feature Store de la POC implementa una arquitectura batch incremental con trazabilidad por lote (batch). Esta arquitectura fue diseñada a partir de los requerimientos de Arca Continental: trazabilidad completa por batch procesado, capacidad de auditoría y re-ejecución de lotes específicos, y compatibilidad futura con patrones de upsert/merge."),
        para("Se adopta la Opción B (Histórico + Latest), que combina una tabla histórica append-only con una tabla de estado actual optimizada para inferencia. La Opción A (solo histórico con PARTITION BY para obtener el último estado) es funcionalmente equivalente y puede utilizarse como alternativa simplificada si se prefiere reducir la cantidad de tablas gestionadas."),
        ...spacer(1),
  
        h3("Componentes del Feature Store"),
        ...spacer(1),
  
        // Component table
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2600, 2380, 4380],
          rows: [
            new TableRow({ children:
              ["Tabla", "Schema", "Descripción"].map(h =>
                new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })
              )
            }),
            ...[
              ["TRAIN_TRANSACTIONS", "SC_FEATURES_BMX", "Tabla de transacciones fuente con columnas técnicas de control de batch: BATCH_ID, PROCESS_STATUS (PENDING / IN_PROGRESS / PROCESSED / FAILED), PROCESSED_AT. Define el 'corte' de cada ejecución."],
              ["FEATURE_BATCH_RUNS", "SC_FEATURES_BMX", "Tabla de control de batch: BATCH_ID, RUN_TS_START, RUN_TS_END, STATUS, conteos de entrada/salida y notas. Permite auditoría completa del proceso."],
              ["FEATURES_UNI_BOX_BATCH", "SC_FEATURES_BMX", "Staging por batch: contiene únicamente las features calculadas en la ejecución actual, incluyendo ENTITY_ID, FEATURE_TS, BATCH_ID y columnas de features. Se sobreescribe en cada ejecución."],
              ["FEATURES_UNI_BOX_HIST", "SC_FEATURES_BMX", "Feature Store histórico (append-only). Acumula todas las ejecuciones con su BATCH_ID. Fuente principal para entrenamiento: permite seleccionar features de un batch específico o un rango temporal."],
              ["FEATURES_UNI_BOX_LATEST", "SC_FEATURES_BMX", "Feature Store de estado actual (MERGE por ENTITY_ID). Contiene la versión más reciente de cada entidad. Fuente optimizada para inferencia: no requiere PARTITION BY para obtener el último registro."],
            ].map(([tbl, schema, desc], i) =>
              new TableRow({ children: [tbl, schema, desc].map((t, ci) =>
                new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
                  margins: { top: 60, bottom: 60, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 19, bold: ci === 0 })] })] })
              )})
            ),
          ]
        }),
        ...spacer(1),
  
        h3("Flujo del Pipeline Batch"),
        para("Cada ejecución del Feature Store sigue este flujo controlado:"),
        resetNum(),
        numbered("Generación de BATCH_ID e inicio del registro en FEATURE_BATCH_RUNS (STATUS = IN_PROGRESS, RUN_TS_START)."),
        numbered("Selección de transacciones con PROCESS_STATUS = PENDING desde TRAIN_TRANSACTIONS."),
        numbered("Marcado de transacciones seleccionadas como IN_PROGRESS con el BATCH_ID actual. Esto 'congela' el corte: cualquier nueva transacción que llegue en paralelo quedará en el siguiente batch."),
        numbered("Cálculo de features para el batch actual mediante SQL (lags, rolling aggregations, ventanas temporales). Resultado materializado en FEATURES_UNI_BOX_BATCH."),
        numbered("INSERT en FEATURES_UNI_BOX_HIST (append). El historial queda inmutable y trazable por BATCH_ID."),
        numbered("MERGE en FEATURES_UNI_BOX_LATEST por ENTITY_ID: actualiza entidades existentes e inserta nuevas. Mantiene siempre el estado más reciente listo para inferencia."),
        numbered("Marcado de transacciones como PROCESSED en TRAIN_TRANSACTIONS."),
        numbered("Cierre del batch en FEATURE_BATCH_RUNS: STATUS = SUCCESS (o FAILED si hubo error), RUN_TS_END y conteos de registros procesados."),
        ...spacer(1),
  
        infoBox("Flujo de datos resumido", [
          "TRAIN_TRANSACTIONS (PENDING)",
          "→ Batch selector (asigna BATCH_ID, marca IN_PROGRESS)",
          "→ Feature Engineering SQL (lags, rolling windows, agregaciones)",
          "→ FEATURES_UNI_BOX_BATCH (staging del batch actual)",
          "→ FEATURES_UNI_BOX_HIST (append histórico con BATCH_ID)",
          "→ FEATURES_UNI_BOX_LATEST (merge estado actual)",
          "→ TRAIN_TRANSACTIONS (PROCESSED) + FEATURE_BATCH_RUNS (SUCCESS/FAILED)",
        ]),
        ...spacer(1),
  
        h3("Uso por Componente del Pipeline"),
        bullet("Entrenamiento (HPO + MMT): consume FEATURES_UNI_BOX_HIST filtrando por BATCH_ID o rango temporal. Permite re-entrenar con exactamente las mismas features de un batch anterior, garantizando reproducibilidad completa."),
        bullet("Inferencia batch: consume FEATURES_UNI_BOX_LATEST directamente sin necesidad de filtros adicionales. Al estar pre-calculado el estado actual, la consulta es eficiente y no requiere lógica de deduplicación."),
        bullet("Auditoría: ante cualquier incidencia, es posible consultar FEATURE_BATCH_RUNS para identificar el batch problemático, revisar FEATURES_UNI_BOX_HIST para ese BATCH_ID, y comparar con ejecuciones anteriores o posteriores."),
        ...spacer(1),
  
        infoBox("Cálculo de Features con Window Functions SQL", [
          "Los lags y ventanas temporales se calculan con funciones de ventana estándar de SQL. Ejemplo:",
          "  SUM(ventas) OVER (PARTITION BY ENTITY_ID ORDER BY FEATURE_TS ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)",
          "Este cálculo se ejecuta sobre el subconjunto del batch actual (FEATURES_UNI_BOX_BATCH) antes de persistir en HIST y LATEST. El resultado es reproducible, auditable y compatible con el engine de Snowflake sin dependencias adicionales."
        ]),
  
        ...spacer(1),
        h2("7.2 Evolución del Feature Store — Arquitectura Batch Incremental"),
        sectionTag("PROD", "1F7A4D"),
        para("En producción cercana, el Feature Store evoluciona hacia una arquitectura batch incremental con trazabilidad completa por lote. Esta arquitectura fue diseñada a partir de los requerimientos de Arca Continental: trazabilidad por batch procesado, capacidad de auditoría y re-ejecución de lotes específicos, y compatibilidad con el esquema de cortes que ya manejan actualmente."),
        ...spacer(1),
        h3("Componentes de la Arquitectura"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2600, 2200, 4560],
          rows: [
            new TableRow({ children:
              ["Tabla", "Schema", "Descripción"].map(h =>
                new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })
              )
            }),
            ...[
              ["TRAIN_TRANSACTIONS", "SC_FEATURES_BMX", "Tabla fuente con columnas técnicas de control: BATCH_ID, PROCESS_STATUS (PENDING / IN_PROGRESS / PROCESSED / FAILED), PROCESSED_AT. Define el corte de cada ejecución."],
              ["FEATURE_BATCH_RUNS", "SC_FEATURES_BMX", "Control de auditoría por lote: BATCH_ID, RUN_TS_START, RUN_TS_END, STATUS, conteos de registros procesados y notas de la ejecución."],
              ["FEATURES_UNI_BOX_BATCH", "SC_FEATURES_BMX", "Staging del batch actual: features calculadas únicamente en esta ejecución (ENTITY_ID, FEATURE_TS, BATCH_ID + columnas de features). Se sobreescribe en cada corrida."],
              ["FEATURES_UNI_BOX_HIST", "SC_FEATURES_BMX", "Histórico append-only: acumula todas las ejecuciones con su BATCH_ID. Fuente para entrenamiento: permite reproducir el estado de features de cualquier batch pasado."],
              ["FEATURES_UNI_BOX_LATEST", "SC_FEATURES_BMX", "Estado actual DROP/RECREATE: contiene las entidades activas del último batch ejecutado. Fuente optimizada para inferencia. No mantiene entidades que no sean transaccionalmente nuevas en el lote."],
            ].map(([tbl, schema, desc], i) =>
              new TableRow({ children: [tbl, schema, desc].map((t, ci) =>
                new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
                  margins: { top: 60, bottom: 60, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 19, bold: ci === 0 })] })] })
              )})
            ),
          ]
        }),
        ...spacer(1),
        infoBox("Decisión clave: DROP/RECREATE en FEATURES_UNI_BOX_LATEST (no MERGE)", [
          "La tabla LATEST no se actualiza mediante MERGE/UPSERT por ENTITY_ID. En cada ejecución de batch, la tabla se elimina y se recrea con el contenido del batch actual.",
          "Esto significa que LATEST refleja exactamente las entidades transaccionalmente activas en el último lote, sin arrastrar entidades inactivas de ejecuciones anteriores.",
          "El histórico completo siempre está disponible en FEATURES_UNI_BOX_HIST (append-only), donde cada registro mantiene su BATCH_ID para trazabilidad.",
          "Para añadir nuevas features: se aplica ALTER TABLE en HIST y LATEST para agregar columnas. Las features no se eliminan a nivel estructural, solo se excluyen del cómputo si ya no son necesarias, preservando así la trazabilidad del histórico.",
        ], MEDIUM_BLUE),
        ...spacer(1),
        h3("Flujo del Pipeline Batch"),
        resetNum(),
        numbered("Generación de BATCH_ID e inicio del registro en FEATURE_BATCH_RUNS (STATUS = IN_PROGRESS, RUN_TS_START)."),
        numbered("Selección de transacciones con PROCESS_STATUS = PENDING desde TRAIN_TRANSACTIONS."),
        numbered("Marcado IN_PROGRESS con el BATCH_ID actual: congela el corte. Nuevas transacciones que lleguen en paralelo quedan en PENDING para el siguiente batch."),
        numbered("Cálculo de features para el batch actual (lags, rolling aggregations, ventanas temporales) mediante SQL sobre el subconjunto marcado. Resultado en FEATURES_UNI_BOX_BATCH."),
        numbered("INSERT en FEATURES_UNI_BOX_HIST (append). El histórico es inmutable y siempre trazable por BATCH_ID."),
        numbered("DROP y RECREATE de FEATURES_UNI_BOX_LATEST con el contenido del batch actual. La tabla refleja solo las entidades activas del lote, sin entidades inactivas previas."),
        numbered("Marcado de transacciones como PROCESSED en TRAIN_TRANSACTIONS."),
        numbered("Cierre del batch en FEATURE_BATCH_RUNS: STATUS = SUCCESS o FAILED, RUN_TS_END y conteos finales."),
        ...spacer(1),
        h3("Uso por Componente"),
        bullet("Entrenamiento (HPO + MMT): consume FEATURES_UNI_BOX_HIST filtrando por BATCH_ID o rango temporal. Permite reproducir exactamente el estado de features de cualquier ejecución pasada."),
        bullet("Inferencia batch: consume FEATURES_UNI_BOX_LATEST directamente. Al ser DROP/RECREATE, el contenido es siempre el del último batch activo sin necesidad de filtros adicionales."),
        ...spacer(1),
        h3("Orquestación del Feature Store"),
        para("Este pipeline puede orquestarse de tres formas según los permisos disponibles en el entorno de Arca Continental:"),
        bullet("Opción A — Snowflake Tasks (si se habilitan permisos): pipeline nativo con dependencias y logging integrado. Actualmente no disponible en la POC por restricciones de permisos."),
        bullet("Opción B — Snowflake ML Jobs: alternativa orientada a cargas ML, con mejor integración con el Model Registry. Requiere permisos similares a Tasks pero puede habilitarse de forma independiente."),
        bullet("Opción C — Orquestador externo (Dagster u otro): el pipeline completo se implementa en notebooks orquestados desde Dagster, invocando las operaciones SQL en Snowflake vía conector. Esta opción no requiere permisos adicionales en Snowflake."),
        ...spacer(1),
        pendingBox("Estrategia de orquestación del Feature Store: pendiente de confirmación con Arca Continental. Se debe definir qué opción es viable en el entorno productivo y con qué periodicidad se ejecutará el pipeline (semanal o por trigger del proceso upstream)."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 8. HPO Y MMT
        // ══════════════════════════════════════════════════
        h1("8. Entrenamiento: HPO y Many Model Training"),
        h2("8.1 HPO — Hyperparameter Optimization"),
        sectionTag("POC", LIGHT_BLUE),
        h3("Qué es y qué resuelve"),
        para("HPO (Hyperparameter Optimization) es el proceso de búsqueda sistemática de la combinación óptima de hiperparámetros para cada algoritmo y segmento. En lugar de definir manualmente estos valores (como se hace en Databricks), HPO automatiza la exploración y selecciona la configuración que minimiza el error de validación."),
        para("En la POC, HPO también puede considerarse una forma de orquestación manual, ya que es el componente que controla qué algoritmo y qué configuración se asigna a cada uno de los 16 grupos, registrando los resultados de forma trazable en ML Experiments."),
        ...spacer(1),
        h3("Implementación y decisiones técnicas"),
        para("El proceso de HPO sigue el siguiente flujo para cada uno de los 16 grupos de segmentación:"),
        resetNum(),
        numbered("Se carga el dataset del grupo desde la tabla de features (UNI_BOX_FEATURES) o directamente desde la tabla de entrenamiento limpia como fallback."),
        numbered("Se aplica un muestreo del 20% del dataset del grupo para la fase de búsqueda. Aunque en primera instancia pueda parecer una decisión arbitraria, tiene una justificación técnica sólida: con ~96M de filas (2.9 GB) distribuidos en 16 grupos, entrenar 15 trials con el 100% de los datos implicaría tiempos de exploración de muchas horas por grupo. El 20% mantiene suficiente representatividad estadística para discriminar configuraciones buenas de malas, que es el objetivo del HPO, sin incurrir en el costo de entrenamiento completo. El entrenamiento final con MMT sí utiliza el 100% de los datos."),
        numbered("Se realiza un split 80/20 dentro del subconjunto muestreado para train/validation, estratificando por la distribución del target."),
        numbered("Se ejecuta Bayesian Search (BayesOpt) con 15 trials y hasta 4 trials concurrentes por grupo. El proceso está estructurado en un bucle secuencial a nivel lógico (un grupo tras otro), pero internamente cada trial de HPO se ejecuta en paralelo sobre el cluster de 5 nodos, al igual que MMT. Esto reduce significativamente el tiempo real de exploración respecto a una ejecución puramente secuencial."),
        numbered("Los hiperparámetros optimizados incluyen: número de estimadores, profundidad máxima, tasa de aprendizaje, parámetros de regularización (reg_alpha, reg_lambda), subsampling y colsample_bytree."),
        numbered("Los resultados (mejores parámetros, RMSE, MAE, WAPE por grupo) se registran en ML Experiments para trazabilidad, con fallback a tabla HYPERPARAMETER_RESULTS si ML Experiments no está disponible."),
        ...spacer(1),
        infoBox("¿Por qué Bayesian Search y no Random Search o Grid Search?", [
          "Grid Search evalúa todas las combinaciones posibles del espacio de hiperparámetros: con 9 parámetros y rangos continuos, el número de combinaciones es computacionalmente prohibitivo.",
          "Random Search muestrea combinaciones al azar de forma ciega, sin aprender de las evaluaciones anteriores. Es mejor que Grid, pero ineficiente en espacios de alta dimensión.",
          "Bayesian Search (BayesOpt) construye un modelo probabilístico del espacio de hiperparámetros a partir de los trials previos, y dirige los siguientes trials hacia las regiones con mayor probabilidad de mejorar la métrica objetivo. Esto permite encontrar configuraciones óptimas con un número mucho menor de evaluaciones, lo que se traduce directamente en menor tiempo de cómputo y menor costo de warehouse.",
          "Con 15 trials por grupo sobre un cluster de 5 nodos M, el HPO completo (16 grupos) tarda aproximadamente 1 hora. Este número de trials representa un equilibrio probado entre calidad de la solución encontrada y tiempo total de ejecución.",
        ]),
        ...spacer(1),
        para("El mapeo de algoritmo por grupo es el siguiente (resultado del análisis de la POC):"),
        ...spacer(1),
        codeBlock([
          'GROUP_MODEL = {',
          '  "group_stat_0_1": "LGBMRegressor",',
          '  "group_stat_0_2": "LGBMRegressor",',
          '  "group_stat_0_3": "LGBMRegressor",',
          '  "group_stat_0_4": "LGBMRegressor",',
          '  "group_stat_1_1": "LGBMRegressor",',
          '  "group_stat_1_2": "LGBMRegressor",',
          '  "group_stat_1_3": "XGBRegressor",   # grupos de mayor volumen',
          '  "group_stat_1_4": "XGBRegressor",',
          '  "group_stat_2_1": "LGBMRegressor",',
          '  "group_stat_2_2": "LGBMRegressor",',
          '  "group_stat_2_3": "XGBRegressor",',
          '  "group_stat_2_4": "XGBRegressor",',
          '  "group_stat_3_1": "LGBMRegressor",',
          '  "group_stat_3_2": "LGBMRegressor",',
          '  "group_stat_3_3": "LGBMRegressor",',
          '  "group_stat_3_4": "XGBRegressor",',
          '}',
        ]),
        ...spacer(1),
        h3("Frecuencia de ejecución"),
        bullet("En la POC: HPO se ejecuta manualmente una vez antes del primer entrenamiento completo."),
        bullet("En producción: HPO se ejecutará de forma puntual (no en cada ciclo semanal) y solo cuando sea necesario re-explorar el espacio de hiperparámetros (ej. cambio significativo en la distribución de datos, degradación de performance detectada por los monitores)."),
        bullet("Los ciclos recurrentes de reentrenamiento utilizarán directamente los parámetros validados, concentrando el paralelismo en MMT."),
        ...spacer(1),
        pendingBox("Frecuencia de reentrenamiento: La periodicidad exacta de los ciclos de reentrenamiento (semanal, mensual, bajo demanda por drift) está pendiente de definición conjunta con Arca Continental."),
  
        ...spacer(1),
        h2("8.2 MMT — Many Model Training"),
        sectionTag("POC", LIGHT_BLUE),
        h3("Qué es y qué resuelve"),
        para("Many Model Training (MMT) es una funcionalidad nativa de Snowflake que permite entrenar múltiples modelos de forma paralela, donde cada modelo se especializa en un subconjunto específico de los datos definido por una columna de partición. En el contexto de ARCA, esta columna es STATS_NTILE_GROUP, que define los 16 segmentos del modelo."),
        para("Mientras que HPO optimiza la configuración para cada grupo, MMT entrena los 16 modelos finales de forma simultánea usando esos parámetros validados. Esto simplifica radicalmente la orquestación: en Databricks se requieren loops manuales, diccionarios de tracking (experiment_submodel_map, group_submodel_map) y gestión manual de artefactos; MMT automatiza todo con una sola llamada especificando la columna de partición."),
        para("En la arquitectura actual de la POC se están removiendo y simplificando ciertos componentes de orquestación que existían en versiones anteriores del diseño. Esta simplificación reduce el overhead operativo y centraliza la lógica dentro del framework nativo de Snowflake, con la justificación técnica de que la trazabilidad se delega íntegramente al Model Registry y a ML Experiments."),
        ...spacer(1),
        h3("Implementación y tiempos de ejecución"),
        para("El proceso de MMT sigue el siguiente flujo:"),
        resetNum(),
        numbered("Se cargan los hiperparámetros óptimos por grupo desde ML Experiments o desde la tabla de resultados de HPO."),
        numbered("Se carga el dataset completo (100% de los datos, ~96M filas / 2.9 GB) desde la tabla de features."),
        numbered("Se define la función de entrenamiento por segmento, que recibe los hiperparámetros específicos del grupo e instancia el modelo correspondiente (XGBRegressor o LGBMRegressor según el mapeo por grupo)."),
        numbered("Se ejecuta ManyModelTraining con particionamiento por STATS_NTILE_GROUP. Snowflake distribuye automáticamente los 16 grupos entre los nodos del cluster, ejecutándolos en paralelo sin intervención manual."),
        numbered("El entrenamiento completo de los 16 sub-modelos sobre el 100% del dataset (~96M filas / 2.9 GB) tarda aproximadamente 7 minutos en el cluster de 5 nodos M. Este tiempo contrasta con las ~16 horas secuenciales del proceso en Databricks, ilustrando el beneficio de la paralelización nativa."),
        numbered("Cada modelo se registra en el Snowflake Model Registry con su versión, métricas (RMSE, MAE, WAPE), metadatos de feature version y timestamp de entrenamiento."),
        numbered("Se configura el alias PRODUCTION para la versión aprobada de cada modelo."),
        ...spacer(1),
        codeBlock([
          '# Fragmento ilustrativo del flujo MMT',
          'from snowflake.ml.modeling.distributors.many_model import ManyModelTraining',
          '',
          'mmt = ManyModelTraining(',
          '    session=session,',
          '    partition_column="STATS_NTILE_GROUP",',
          '    train_func=train_func_with_hpo_params,  # función que carga params por grupo',
          ')',
          'mmt.run(train_df)  # Snowflake paraleliza automáticamente los 16 grupos',
        ]),
        ...spacer(1),
        h3("Escalabilidad de Modelos"),
        para("La estrategia de escalamiento adoptada es la segmentación por columna de partición (STATS_NTILE_GROUP). Este enfoque fue seleccionado sobre alternativas como partición lógica por región o distribución computacional genérica, porque:"),
        bullet("Preserva la lógica de negocio existente en Databricks (16 grupos basados en patrones de venta y volumen)."),
        bullet("Alinea el escalamiento con el patrón de MMT de Snowflake, que está optimizado para la paralelización por partición."),
        bullet("Permite asignar algoritmos distintos por segmento (XGBoost para grupos de alto volumen, LightGBM para grupos de baja densidad) sin modificar la arquitectura base."),
        bullet("El código escala automáticamente sin modificaciones: al aumentar el número de grupos o el tamaño del warehouse, MMT distribuye la carga entre nodos disponibles."),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 3120, 3120],
          rows: [
            new TableRow({ children: ["Aspecto", "POC", "Producción"].map(h => new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })) }),
            new TableRow({ children: ["Warehouse", "X-Small / Small con escalado automático", "Large / X-Large con multi-cluster habilitado"].map(t => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }),
            new TableRow({ children: ["Nodos de cómputo", "Pool de 5 nodos (escalable hasta 2 mínimo)", "Multi-nodo con elasticidad dinámica"].map(t => new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }),
            new TableRow({ children: ["HPO frecuencia", "Una vez (puntual)", "Bajo demanda o ante drift significativo"].map(t => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }),
            new TableRow({ children: ["MMT frecuencia", "Manual, una vez por ciclo de experimentación", "Por definir: semanal, mensual o por trigger de drift"].map(t => new TableCell({ borders, shading: { fill: LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }),
            new TableRow({ children: ["GPU", "No requerido", "Disponible para modelos de mayor complejidad"].map(t => new TableCell({ borders, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3120, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }),
          ]
        }),
        ...spacer(1),
        ...spacer(1),
        h2("8.3 Propuesta Conceptual: AutoML en Snowflake"),
        sectionTag("FUTURO", "7030A0"),
        infoBox("Contexto: ¿por qué los modelos están fijos en el código hoy?", [
          "En la POC actual, el algoritmo asignado a cada grupo (XGBoost o LightGBM) está definido explícitamente en el código de entrenamiento. Esto significa que para cambiar el modelo de un segmento hay que modificar el código manualmente.",
          "Arca Continental señaló la necesidad de automatizar esta selección, de modo que el sistema pueda evaluar múltiples algoritmos por segmento y elegir el óptimo sin intervención humana.",
          "La propuesta conceptual que se describe a continuación resuelve exactamente esta necesidad, elevando el HPO actual a un AutoML completo dentro del ecosistema Snowflake.",
        ], MEDIUM_BLUE),
        ...spacer(1),
        para("Snowflake no dispone actualmente de una solución AutoML nativa equivalente a la de Databricks. No obstante, es posible construir un enfoque AutoML utilizando las capacidades existentes de Hyperparameter Optimization (HPO) y Many Model Training (MMT)."),
        para("En la POC actual, el proceso de HPO optimiza hiperparámetros sobre un conjunto fijo de algoritmos definidos en el código. La evolución propuesta consiste en ampliar esta lógica para que la exploración no solo ocurra dentro del espacio de hiperparámetros, sino también a nivel de selección de modelo."),
        ...spacer(1),
        para("Conceptualmente, esto se implementaría mediante:"),
        bullet("Un diccionario o tabla de configuración que defina múltiples algoritmos candidatos (por ejemplo, XGBoost, LightGBM, modelos lineales, etc.)."),
        bullet("Para cada algoritmo, un espacio de búsqueda de hiperparámetros estructurado."),
        bullet("Una capa de orquestación que itere sobre combinaciones de (modelo, espacio de búsqueda, segmento)."),
        bullet("Registro unificado de métricas en Model Registry para comparar desempeño entre algoritmos y configuraciones."),
        ...spacer(1),
        para("En este esquema, cada ejecución evaluaría múltiples combinaciones modelo-configuración, distribuidas sobre el compute pool mediante HPO. Posteriormente, se seleccionaría automáticamente el mejor modelo por segmento según la métrica objetivo definida."),
        para("Este enfoque no forma parte del alcance actual de la POC, pero puede implementarse como una capa adicional de orquestación sobre HPO y MMT, manteniéndose completamente dentro del ecosistema Snowflake. De esta manera, se simula un comportamiento AutoML sin depender de una solución externa propietaria."),
        ...spacer(1),
        para("A continuación, la representación gráfica de la solución propuesta:"),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/automl_diagram.png"), transformation: { width: 490, height: 324 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 120, after: 180 }
        }),
        ...spacer(1),
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 9. MODEL REGISTRY
        // ══════════════════════════════════════════════════
        h1("9. Model Registry"),
        sectionTag("POC", LIGHT_BLUE),
        h2("9.1 Función y Estructura"),
        para("El Snowflake Model Registry es un repositorio integrado que centraliza el gobierno del ciclo de vida de los modelos directamente dentro de Snowflake, reemplazando MLflow como componente externo. Permite gestionar múltiples versiones, comparar performance entre ellas y controlar qué versión está activa en cada ambiente."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image10.png"), transformation: { width: 620, height: 284 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        h2("9.2 Artefactos Almacenados por Versión"),
        para("Cada versión registrada en el Model Registry almacena los siguientes artefactos y metadatos:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [3000, 6360],
          rows: [
            new TableRow({ children: [
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Artefacto / Metadato", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
              new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: 6360, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Descripción", font: "Arial", size: 20, bold: true, color: WHITE })] })] }),
            ]}),
            ...[
              ["Modelo serializado", "Objeto del modelo entrenado (XGBRegressor / LGBMRegressor) serializado en formato compatible con Snowflake ML."],
              ["Métricas de performance", "RMSE, MAE, WAPE y MAPE por grupo (16 valores de cada métrica) registradas en el momento del entrenamiento."],
              ["Hiperparámetros", "Configuración óptima encontrada por HPO para cada grupo (n_estimators, max_depth, learning_rate, etc.)."],
              ["Feature version ID", "Identificador de la versión de features utilizada para el entrenamiento, garantizando reproducibilidad."],
              ["Feature snapshot timestamp", "Timestamp del snapshot de features usado, para point-in-time consistency."],
              ["Tags de versión activa", "Tags asignados al modelo para identificar qué versión está activa por proyecto o grupo de consumo (ej. act_general, act_grupo1). Controlados vía RBAC por ML_OPS_ROLE."],
              ["Autor y timestamp", "Usuario que registró el modelo y fecha/hora de registro para trazabilidad y auditoría."],
              ["Dependencias", "Versiones de librerías utilizadas (XGBoost, LightGBM, Snowflake ML) para reproducibilidad del entorno."],
              ["Baseline de drift", "Datos de referencia calculados a partir del set de entrenamiento, usados por ML Observability para comparación."],
            ].map(([art, desc], i) => new TableRow({ children: [
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: art, font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 6360, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: desc, font: "Arial", size: 20 })] })] }),
            ]}))
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 10. INFERENCIA PARTICIONADA
        // ══════════════════════════════════════════════════
        h1("10. Inferencia Batch Particionada"),
        sectionTag("POC", LIGHT_BLUE),
        h2("10.1 Funcionamiento"),
        para("Partitioned Model Inference es el complemento natural de MMT. Una vez registrado el modelo particionado en el Registry, el motor de Snowflake resuelve automáticamente qué sub-modelo corresponde a cada registro en función del valor de la columna STATS_NTILE_GROUP, sin necesidad de lógica condicional adicional ni enrutador externo."),
        para("Este comportamiento es fundamentalmente diferente a una PyFunc de MLflow, donde todo el ruteo debe programarse explícitamente dentro del código. En Snowflake, el ruteo queda encapsulado en el objeto particionado y es gestionado nativamente por el motor del warehouse."),
        ...spacer(1),
        h2("10.2 Flujo de Ejecución"),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image7.jpg"), transformation: { width: 620, height: 138 }, type: "jpg" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        para("El proceso de inferencia batch sigue este flujo:"),
        resetNum(),
        numbered("Se carga el dataset de inferencia desde la tabla fuente (INFERENCE_DATASET_CLEANED)."),
        numbered("Se recupera la referencia al modelo particionado con alias PRODUCTION desde el Model Registry."),
        numbered("Se ejecuta la inferencia mediante sintaxis SQL nativa con la cláusula TABLE(MODEL(...)!PREDICT(...) OVER (PARTITION BY STATS_NTILE_GROUP))."),
        numbered("Los resultados se insertan en la tabla INFERENCE_PREDICTIONS con los campos: CUSTOMER_ID, STATS_NTILE_GROUP, WEEK, BRAND_PRES_RET, PROD_KEY, PREDICTED_UNI_BOX_WEEK, MODEL_VERSION y PREDICTION_TIMESTAMP."),
        numbered("Opcionalmente, si el volumen lo requiere, la inferencia puede ejecutarse en batches (BATCH_SIZE configurable) usando Snowpark order_by + limit/offset."),
        ...spacer(1),
        codeBlock([
          '-- Sintaxis SQL nativa de inferencia particionada',
          'INSERT INTO INFERENCE_PREDICTIONS (...)',
          'SELECT p.CUSTOMER_ID, p.WEEK, p.predicted_uni_box_week, ...',
          'FROM INFERENCE_INPUT t,',
          'TABLE(',
          '  MODEL(BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED, PRODUCTION)',
          '  !PREDICT(t.CUSTOMER_ID, t.STATS_NTILE_GROUP, t.WEEK, ...)',
          '  OVER (PARTITION BY t.STATS_NTILE_GROUP)',
          ') p',
        ]),
        ...spacer(1),
        infoBox("Deployment en producción", ["La estrategia de deployment está definida: Tags para gestión de versiones activas por proyecto, y Export/Import de artefactos para la promoción entre ambientes DEV → QA → PROD. Ver sección 12 para el detalle completo."]),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 11. ML OBSERVABILITY
        // ══════════════════════════════════════════════════
        h1("11. ML Observability — Monitoreo de Datos y Modelos"),
        h2("11.1 Estrategia de Monitoreo"),
        sectionTag("POC", LIGHT_BLUE),
        para("ML Observability es la solución de monitoreo continuo que detecta automáticamente degradación de performance o cambios significativos en los datos de entrada. La estrategia se basa en tres pilares:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2200, 3580, 3580],
          rows: [
            new TableRow({ children: ["Tipo de Drift", "Qué detecta", "Métricas utilizadas"].map(h => new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: h === "Tipo de Drift" ? 2200 : 3580, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })) }),
            ...[
              ["Data Drift", "Cambios en la distribución de features entre entrenamiento y producción.", "Jensen-Shannon Divergence, Kolmogorov-Smirnov, PSI, Wasserstein Distance (numéricas); Chi-square, Cramér's V (categóricas)."],
              ["Performance Drift", "Degradación de las métricas del modelo comparadas contra el baseline del test set.", "RMSE, MAE comparados contra intervalos de confianza del test set histórico."],
              ["Prediction Drift", "Cambios en la distribución de predicciones del modelo.", "Comparación de distribuciones de outputs actuales vs históricos mediante métodos estadísticos."],
            ].map(([t, d, m], i) => new TableRow({ children: [t, d, m].map((txt, ci) => new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: ci === 0 ? 2200 : 3580, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: txt, font: "Arial", size: 20, bold: ci === 0 })] })] })) }))
          ]
        }),
        ...spacer(1),
        para("La estrategia de monitoreo se mantendrá activa tanto en la POC como en producción, evolucionando en automatización y granularidad. El baseline para drift detection se genera una única vez post-entrenamiento en DEV y se promueve junto con el modelo como parte del paquete de artefactos, garantizando consistencia entre ambientes."),
        ...spacer(1),
        h2("11.2 Visualización por Segmento y Agregada"),
        para("El módulo de Performance Drift garantiza visibilidad en dos niveles:"),
        bullet("Por grupo/segmento: métricas de RMSE, MAE, WAPE y drift calculadas individualmente para cada uno de los 16 grupos (STATS_NTILE_GROUP), permitiendo identificar qué segmento específico está degradándose."),
        bullet("Agregada general: vista consolidada de las métricas promediadas o ponderadas sobre todos los grupos, expuesta en dashboards Snowsight para una visión ejecutiva del comportamiento del modelo."),
        ...spacer(1),
        h2("11.3 Configuración del Monitoreo"),
        sectionTag("POC", LIGHT_BLUE),
        para("En la POC, los Model Monitors se configuran programáticamente mediante SQL/Python para garantizar consistencia y versionado. El flujo es:"),
        resetNum(),
        numbered("Se prepara el baseline UNA SOLA VEZ post-entrenamiento con datos de training del ambiente DEV."),
        numbered("Se crean Model Monitors por modelo, especificando la tabla de inference logs, las columnas de features/predicciones, la tabla baseline y la ventana de agregación."),
        numbered("Los monitors se refrescan automáticamente según la frecuencia configurada (horaria, diaria), generando tablas de drift, performance y prediction stats."),
        numbered("Los resultados se exponen en dashboards Snowsight con alertas configurables por umbral."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image8.png"), transformation: { width: 620, height: 394 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        pendingBox("Revisión de la estrategia de monitoreo con Snowflake: Se recomienda alinear formalmente con Snowflake la estrategia definitiva de monitoreo de datos, especificar si la configuración actual de monitores se mantendrá en producción o evolucionará, y definir los umbrales de alertamiento (Warning/Critical) en función de los KPIs de negocio de Arca Continental."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 12. DEPLOYMENT Y AMBIENTES
        // ══════════════════════════════════════════════════
        h1("12. Estrategia de Deployment y Promoción de Modelos"),
        sectionTag("POC", LIGHT_BLUE),
        para("Esta sección describe la estrategia completa de deployment: cómo se separan los ambientes, cómo se gestionan múltiples versiones activas en producción mediante Tags, y cómo se promueve un modelo entre ambientes usando Export/Import de artefactos."),
  
        h2("12.1 Separación por Ambientes"),
        para("La estrategia define tres ambientes separados por base de datos en Snowflake, con roles de acceso diferenciados para cada etapa del ciclo de vida del modelo:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2000, 2500, 4860],
          rows: [
            new TableRow({ children: ["Ambiente", "Base de Datos", "Propósito"].map(h => new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: h === "Ambiente" ? 2000 : h === "Base de Datos" ? 2500 : 4860, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })) }),
            ...[
              ["DEV", "ML_DEV_DB", "Desarrollo, experimentación, entrenamiento y validación inicial. Acceso a ML_DEV_ROLE."],
              ["QA / Staging", "ML_QA_DB", "Validación funcional y de calidad antes de producción. Acceso a ML_OPS_ROLE."],
              ["PROD", "ML_PRD_DB", "Ambiente productivo. Solo ML_OPS_ROLE puede realizar promociones. Los consumidores operan con ML_CONSUMER_ROLE."],
            ].map(([amb, db, prop], i) => new TableRow({ children: [amb, db, prop].map((t, ci) => new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: ci === 0 ? 2000 : ci === 1 ? 2500 : 4860, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 20 })] })] })) }))
          ]
        }),
        ...spacer(1),
  
        h2("12.2 Gestión de Versiones Activas mediante Tags"),
        para("Arca Continental opera con un escenario donde un mismo modelo puede tener múltiples versiones activas simultáneamente en producción, asociadas a distintos proyectos o grupos de consumo. La solución adoptada es el uso de Tags en el Snowflake Model Registry."),
        para("Un Tag es una etiqueta con valor variable asignada directamente al objeto modelo en el Registry, manipulable mediante SQL y con permisos controlables vía RBAC. Cada proyecto o grupo de consumo define su propio Tag, permitiendo que distintos consumidores apunten a versiones diferentes del mismo modelo sin conflictos y sin modificar el código de inferencia:"),
        ...spacer(1),
        codeBlock([
          '-- Asignar versión activa por proyecto',
          "ALTER MODEL UNI_BOX_REGRESSION_PARTITIONED SET TAG act_general  = 'v20260103';",
          "ALTER MODEL UNI_BOX_REGRESSION_PARTITIONED SET TAG act_grupo1   = 'v20260103';",
          "ALTER MODEL UNI_BOX_REGRESSION_PARTITIONED SET TAG act_grupo2   = 'v20251205';",
          '',
          '-- Consultar versión activa para un proyecto (en el notebook de inferencia)',
          "SET version = (SELECT SYSTEM$GET_TAG('act_grupo1',",
          "               'UNI_BOX_REGRESSION_PARTITIONED', 'MODULE'));",
        ]),
        ...spacer(1),
        para("Ventajas del enfoque con Tags: control de acceso granular por RBAC (solo ML_OPS_ROLE modifica tags de producción), trazabilidad completa en QUERY_HISTORY de Snowflake, y sin necesidad de cambios de código en los procesos consumidores al actualizar una versión. El límite de 50 tags por modelo deja margen suficiente para los 16 grupos actuales con capacidad de crecimiento."),
        ...spacer(1),
  
        h2("12.3 Flujo de Promoción DEV → QA → PROD"),
        para("El modelo NO se reentrena en producción. Se entrena UNA VEZ en DEV, se valida en QA y se promueve el artefacto binario validado a PROD mediante Export/Import. Esto garantiza que el objeto que opera en producción es exactamente el mismo que fue aprobado en QA, sin riesgo de variaciones por reentrenamiento."),
        ...spacer(1),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image14.png"), transformation: { width: 620, height: 130 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        ...spacer(1),
        resetNum(),
        numbered("Entrenamiento en DEV: HPO + MMT se ejecutan sobre los datos de entrenamiento. Los modelos se registran en ML_DEV_DB.SC_MODELS con versión, métricas y metadatos completos."),
        numbered("Validación en DEV: El equipo ML revisa métricas por grupo (RMSE, MAE, WAPE) como gate de calidad. Manual en la POC; automatizable con umbrales en producción cercana."),
        numbered("Export DEV → Stage: ML_OPS_ROLE exporta el artefacto del modelo validado al stage de DEV."),
        numbered("Copia al stage de QA: el contenido del stage se transfiere al stage receptor. Si ambos ambientes usan stages externos (S3/Azure/GCS), la copia se realiza directamente entre buckets."),
        numbered("Import en QA: el modelo se importa al Model Registry de QA con los mismos metadatos y métricas. Se ejecutan pruebas funcionales de inferencia sobre datos de QA."),
        numbered("Aprobación de QA: verificación de consistencia de predicciones. Gate de calidad antes de promover a PROD."),
        numbered("Export QA → Stage PROD + Import en PROD: mismo mecanismo. El artefacto binario importado es idéntico al aprobado en QA."),
        numbered("Actualización de Tag en PROD: ML_OPS_ROLE asigna el Tag del proyecto a la nueva versión importada. Los procesos de inferencia leen el Tag para resolver qué versión invocar, sin cambios de código."),
        ...spacer(1),
        codeBlock([
          '-- Export desde DEV (o QA)',
          'EXPORT MODEL BD_AA_DEV.SC_MODELS_BMX.UNI_BOX_REGRESSION_PARTITIONED',
          'TO @BD_AA_DEV.SC_MODELS_BMX.MODEL_STAGE/unibox_v20260103/;',
          '',
          '-- Copia entre stages (si external stage: GET/PUT o copia entre buckets)',
          '',
          '-- Import en PROD',
          'IMPORT MODEL UNI_BOX_REGRESSION_PARTITIONED',
          'FROM @BD_AA_PRD.SC_MODELS_BMX.MODEL_STAGE/unibox_v20260103/;',
          '',
          '-- Actualizar tag de versión activa',
          "ALTER MODEL UNI_BOX_REGRESSION_PARTITIONED SET TAG act_general = 'v20260103';",
        ]),
        ...spacer(1),
  
        h2("12.4 Control de Acceso RBAC"),
        para("El gobierno del ciclo de vida se implementa mediante cuatro roles diferenciados:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [2400, 6960],
          rows: [
            new TableRow({ children: ["Rol", "Permisos"].map(h => new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: h === "Rol" ? 2400 : 6960, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })) }),
            ...[
              ["ML_DEV_ROLE", "Lectura/escritura en schemas DEV. Registro de modelos en Model Registry de DEV. Sin acceso a QA ni PROD."],
              ["ML_OPS_ROLE", "Permisos completos para Export/Import de modelos y actualización de Tags en todos los ambientes. OWNERSHIP en schemas de producción. Único rol autorizado para modificar versiones activas en PROD."],
              ["ML_CONSUMER_ROLE", "Solo lectura y ejecución de modelos en producción (SELECT/USAGE sobre funciones de inferencia). Sin permisos de modificación de modelos ni Tags."],
              ["ML_MONITOR_ROLE", "Lectura de logs de inferencia, métricas y tablas de monitoreo. Sin permisos de modificación sobre modelos o schemas de entrenamiento."],
            ].map(([rol, perm], i) => new TableRow({ children: [
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 2400, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: rol, font: "Arial", size: 20, bold: true, color: LIGHT_BLUE })] })] }),
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 6960, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: perm, font: "Arial", size: 20 })] })] }),
            ]}))
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 13. COMPARATIVA POC VS PRODUCCIÓN
        // ══════════════════════════════════════════════════
        h1("13. Comparativa: POC vs. Producción"),
        para("La siguiente tabla diferencia tres horizontes: la POC ya implementada, la producción cercana (mejoras inmediatas sin grandes cambios de arquitectura) y la producción lejana (evolución completa del sistema a mediano/largo plazo):"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2000, 2453, 2453, 2454],
          rows: [
            // Header row
            new TableRow({ children: 
              ["Aspecto", "POC (actual)", "Prod. Cercano", "Prod. Lejano"].map((h, ci) =>
                new TableCell({
                  borders, shading: { fill: ci === 0 ? BLUE : ci === 1 ? "2E5E8E" : ci === 2 ? "1F7A4D" : "7030A0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  width: { size: ci === 0 ? 2000 : ci === 1 ? 2453 : ci === 2 ? 2453 : 2454, type: WidthType.DXA },
                  children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })]
                })
              )
            }),
            // Data rows: [aspect, poc, prod_cercano, prod_lejano]
            ...[
              [
                "Datos",
                "Históricos estáticos curados por Arca Continental",
                "Mismos datos con proceso de validación y limpieza documentado",
                "Actualización periódica automatizada desde fuentes origen"
              ],
              [
                "Inferencia",
                "Batch manual sobre INFERENCE_DATASET_CLEANED",
                "Batch periódico (semanal) con trigger definido",
                "Batch + Real-time (SPCS con endpoints HTTPS)"
              ],
              [
                "Feature Store",
                "Tablas físicas estáticas, sin actualización automática",
                "Refresh manual o vía orquestador externo; sin Tasks por permisos",
                "Tasks / ML Jobs nativos o FeatureViews con governance integrado"
              ],
              [
                "HPO",
                "Manual, puntual, una sola ejecución",
                "Bajo demanda ante degradación de métricas detectada",
                "Trigger automático por drift; selección de modelo extendida (AutoML)"
              ],
              [
                "MMT — Reentrenamiento",
                "Manual, una vez por ciclo de experimentación",
                "Periódico (frecuencia por definir con Arca Continental)",
                "Automático con aprobación basada en umbrales de métricas"
              ],
              [
                "Orquestación",
                "Ejecución manual de notebooks paso a paso",
                "Scripts parametrizados con ejecución controlada por ML_OPS",
                "Snowflake Tasks / ML Jobs con DAG de dependencias y logging"
              ],
              [
                "CI/CD",
                "Manual — promoción controlada por ML_OPS_ROLE",
                "Semi-automatizado con checklist de validación por ambiente",
                "Pipeline CI/CD automatizado con gates de calidad por etapa"
              ],
              [
                "Escalamiento",
                "Warehouse X-Small/Small, pool de 5 nodos M",
                "Warehouse Medium/Large; ajuste manual según carga",
                "Warehouse Large/X-Large con multi-cluster y auto-scaling dinámico"
              ],
              [
                "Versionado multi-proyecto",
                "Tags por versión activa en el Model Registry",
                "Tags por proyecto (pendiente validación con Arca)",
                "Gobierno centralizado de versiones activas por proyecto/región"
              ],
              [
                "Monitoreo",
                "Configuración inicial de ML Observability",
                "Monitores activos con umbrales Warning/Critical definidos",
                "Dashboards avanzados, alertas proactivas y auto-reentrenamiento"
              ],
              [
                "Selección de modelos",
                "Algoritmo fijo por segmento (definido en código)",
                "Selección validada manualmente por el equipo ML",
                "AutoML: selección automática del mejor modelo por segmento"
              ],
            ].map(([asp, poc, near, far], i) =>
              new TableRow({ children: [asp, poc, near, far].map((t, ci) =>
                new TableCell({
                  borders,
                  shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR },
                  margins: { top: 60, bottom: 60, left: 120, right: 120 },
                  width: { size: ci === 0 ? 2000 : ci === 1 ? 2453 : ci === 2 ? 2453 : 2454, type: WidthType.DXA },
                  children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 19, bold: ci === 0 })] })]
                })
              )})
            ),
          ]
        }),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 14. ORQUESTACIÓN
        // ══════════════════════════════════════════════════
        h1("14. Orquestación del Pipeline"),
        h2("14.1 POC — Orquestación Manual"),
        sectionTag("POC", LIGHT_BLUE),
        para("En la POC, los notebooks se ejecutan manualmente en secuencia. Esta decisión permite mantener trazabilidad y control completo durante la fase de validación, sin introducir complejidad operativa innecesaria."),
        para("La secuencia de ejecución es:"),
        resetNum(),
        numbered("Paso 1 — Validación y limpieza de datos: verificación de estructura, calidad y compatibilidad entre los datasets de entrenamiento e inferencia."),
        numbered("Paso 2 — Construcción del Feature Store: cálculo de features mediante SQL (lags, rolling aggregations) y materialización en tablas físicas."),
        numbered("Paso 3 — HPO Bayesian Search: búsqueda de hiperparámetros óptimos por grupo (~1h en cluster de 5 nodos M, 15 trials por grupo)."),
        numbered("Paso 4 — MMT: entrenamiento paralelo de 16 sub-modelos con hiperparámetros validados (~7 minutos en cluster de 5 nodos M)."),
        numbered("Paso 5 — Creación del modelo particionado: construcción del wrapper unificado y registro en Model Registry con alias PRODUCTION."),
        numbered("Paso 6 — Inferencia batch: ejecución de predicciones con sintaxis SQL nativa sobre el dataset de inferencia."),
        ...spacer(1),
        h2("14.2 Producción — Orquestación Automatizada"),
        sectionTag("FUTURO", "7030A0"),
        para("En producción, el pipeline se automatizará mediante Snowflake Tasks o ML Jobs, eliminando la dependencia de ejecución manual. El flujo típico será:"),
        new Paragraph({
          children: [new ImageRun({ data: fs.readFileSync("/home/claude/bbp_unpacked/word/media/image13.png"), transformation: { width: 620, height: 56 }, type: "png" })],
          alignment: AlignmentType.CENTER, spacing: { before: 100, after: 100 }
        }),
        resetNum(),
        numbered("Task/Job diario o semanal: Validación y limpieza de datos entrantes."),
        numbered("Task/Job dependiente: Actualización del Feature Store con las nuevas semanas disponibles."),
        numbered("Task/Job condicional: Re-entrenamiento (MMT) si se detecta drift significativo o se alcanza el umbral de reentrenamiento periódico."),
        numbered("Task/Job programado: Inferencia batch semanal sobre el dataset de la semana en curso."),
        numbered("Task/Job de monitoreo: Generación de estadísticas de drift y performance, y disparo de alertas si se superan umbrales."),
        ...spacer(1),
        pendingBox("La estrategia de automatización de la temporalidad semanal (parámetro de semana para inferencia, conversión automática de fechas) está pendiente de definición. Se evaluará si Snowflake Tasks puede manejar este parámetro nativamente."),
  
        pageBreak(),
  
        // ══════════════════════════════════════════════════
        // 15. CONSIDERACIONES
        // ══════════════════════════════════════════════════
        h1("15. Consideraciones y Responsabilidades"),
        h2("15.1 Responsabilidades de Arca Continental"),
        resetNum(),
        numbered("Proveer accesos a Snowflake con los permisos necesarios para trabajar en el ambiente de desarrollo."),
        numbered("Participar en la transferencia de conocimiento sobre la funcionalidad actual del modelo en Databricks."),
        numbered("Entregar el dataset de entrenamiento y el de inferencia de forma curada y lista para usarse como fuente de la POC."),
        numbered("Acompañar en la fase de certificación de datos para la aprobación de la migración de los modelos creados."),
        numbered("Definir y validar los umbrales de aceptación de métricas (RMSE, WAPE) que determinarán la aprobación del modelo en cada ambiente."),
        numbered("Confirmar la frecuencia de reentrenamiento y validar el esquema de Tags por proyecto definido en la sección 12."),
        ...spacer(1),
        h2("15.2 Responsabilidades de Seidor Analytics"),
        resetNum(),
        numbered("Desarrollar la migración técnica del modelo de Pronóstico de Ventas Semanales desde Databricks hacia Snowflake."),
        numbered("Implementar las mejores prácticas con las herramientas nativas de Snowflake para maximizar el aprovechamiento de la plataforma."),
        numbered("Entregar los modelos entrenados y las predicciones versionadas y certificadas para su aprobación."),
        numbered("Entregar documentación de desarrollo, certificación de datos y playbook de componentes desarrollados."),
        numbered("Definir y documentar formalmente la estrategia de deployment en producción una vez alineada con Arca Continental."),
        ...spacer(1),
        h2("15.3 Puntos Pendientes de Definición"),
        sectionTag("PENDIENTE", "C55A11"),
        para("Los siguientes puntos requieren validación y aprobación formal por ambas partes antes de la transición a producción:"),
        ...spacer(1),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [3600, 5760],
          rows: [
            new TableRow({ children: ["Punto Pendiente", "Detalle"].map(h => new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, width: { size: h === "Punto Pendiente" ? 3600 : 5760, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: h, font: "Arial", size: 20, bold: true, color: WHITE })] })] })) }),
            ...[
              ["Estrategia de deployment PROD", "Aliases únicamente vs. Tags por grupo de proyecto vs. combinación. Impacta el flujo de promoción y los permisos RBAC."],
              ["Frecuencia de reentrenamiento", "Semanal, mensual o bajo demanda por drift. Impacta costos y operativa del equipo."],
              ["Temporalidad semanal de inferencia", "Cómo se parametriza la semana en curso para la ejecución de inferencia. Evaluación de automatización con Snowflake Tasks."],
              ["Revisión estrategia de monitoreo con Snowflake", "Alinear formalmente la estrategia de ML Observability y definir si la configuración actual se mantiene o evoluciona en producción."],
              ["Umbrales de alertamiento", "Valores de Warning/Critical para Data Drift, Performance Drift y Prediction Drift, basados en KPIs de negocio de Arca."],
            ].map(([pt, det], i) => new TableRow({ children: [
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 3600, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: pt, font: "Arial", size: 20, bold: true })] })] }),
              new TableCell({ borders, shading: { fill: i % 2 === 0 ? WHITE : LIGHT_GRAY, type: ShadingType.CLEAR }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, width: { size: 5760, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: det, font: "Arial", size: 20 })] })] }),
            ]}))
          ]
        }),
      ]
    }]
  });
  
  Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("/home/claude/BBP_ARCA_Snowflake_V3_Mejorado.docx", buffer);
    console.log("✅ Document created successfully");
  }).catch(err => {
    console.error("❌ Error:", err);
    process.exit(1);
  });