# MorphoStat

**An end-to-end morphological post-processing pipeline for cell-imaging assays.**

Point MorphoStat at a folder of FilamentSensor / ImageJ / CellProfiler CSV
exports and it will, in one command, ingest and aggregate the files, clean and
normalize the measurements across replicates, run the statistically appropriate
differential tests between your control and treatment groups, and emit
publication-ready **editable vector figures** (SVG + PDF) plus full statistics
tables and a written report.

It exists to remove the manual, error-prone step between *image quantification*
(which the imaging tools already do well) and the *statistical / visual
endpoint* — the part that is repeated by hand in almost every cell-morphology
paper and is a known bottleneck for reproducibility and throughput.

---

## Why

Tools such as **FilamentSensor 2.0** (Hauke et al., 2023, *PLOS ONE*) and
ImageJ/FIJI-based pipelines such as **Lusca** (Šimunić et al., 2024, *Sci Rep*)
extract rich cytoskeletal and morphological measurements, but they stop at a
table of numbers. The downstream work — aggregating replicates, choosing the
right test, correcting for multiple comparisons, and drawing consistent figures
— is done manually and differently each time. MorphoStat standardizes that
endpoint.

---

## Install

```bash
git clone <this-repo>
cd MorphoStat
pip install -e .
```

Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn.

## One-click usage

```bash
morphostat run \
    --input  path/to/csv_folder \
    --group-col  treatment \
    --control    DMSO \
    --batch-col  Plate \
    --source     cellprofiler \
    --outdir     results
```

Outputs land in `results/`:

```
results/
├── REPORT.md                         # human-readable summary
├── run_manifest.json                 # machine-readable run record
├── figures/
│   ├── violin_panel.{svg,pdf,png}    # feature distributions by group
│   ├── pca.{svg,pdf,png}             # morphological-signature clustering
│   ├── correlation_heatmap.{svg,pdf,png}
│   └── volcano_<group>.{svg,pdf,png} # effect size vs FDR
└── tables/
    ├── pairwise_vs_control.csv       # per-feature stats, every group vs control
    ├── omnibus_across_groups.csv     # ANOVA / Kruskal across all groups
    ├── normalized_features.csv       # cleaned + normalized matrix
    └── cleaning_report.txt           # ingestion / QC audit trail
```

Or from Python:

```python
from morphostat import run_pipeline
res = run_pipeline("csv_folder", "results",
                   group_col="treatment", control="DMSO",
                   batch_col="Plate", source="cellprofiler")
```

## Inputs

MorphoStat is schema-tolerant. It auto-detects numeric morphology features and
drops obvious non-features (pixel coordinates, object/image indices, identifiers).

- **`--source cellprofiler`** — per-object or per-well CellProfiler exports
  (`*_AreaShape_*`, `*_Intensity_*`, `*_Texture_*`, …).
- **`--source filamentsensor`** — FilamentSensor per-filament tables. With
  `--fs-sample-col <col>` the per-filament rows are collapsed to per-image
  features: mean/total filament length, mean width, circular-mean orientation,
  and the **nematic order parameter** `S = ⟨cos 2(θ − ⟨θ⟩)⟩`, plus filament
  count. Column matching is tolerant (length/width/angle/order are found by
  pattern); override with a custom map if your export differs.
- **`--source generic`** — any tidy table with a group column and numeric
  feature columns (e.g. ImageJ "Measure" Area / Circ. / AR / Solidity exports).

Each input file may be one replicate / plate / condition; files are concatenated
on column union, so missing columns in some files are handled gracefully.

## Methods

**Cleaning.** Coerce features to numeric; drop columns above a missing-value
threshold or with near-zero variance; impute remaining gaps (median by default).

**Normalization (`--normalization`).**
- `robust_z_to_control` *(default)* — per batch/plate, each feature is z-scored
  to the control group's **median and MAD** (`x' = (x − med_ctrl)/MAD_ctrl`).
  This is the standard negative-control plate normalization used in image-based
  morphological profiling; it removes plate-to-plate batch effects. Batches with
  no control fall back to pooled-control statistics.
- `standard` — global mean/SD z-score. `none` — pass-through.

**Statistics.** For every feature, each group is compared to control:
Shapiro–Wilk tests normality and Levene tests equal variance, then the test is
selected automatically — Student's *t* (normal, equal var), Welch's *t* (normal,
unequal var), or Mann–Whitney *U* (non-normal). Effect sizes are reported as
Hedges' *g* (parametric) or Cliff's δ (non-parametric). An omnibus one-way
**ANOVA** or **Kruskal–Wallis** (with η² / ε²) is run across all groups.
All p-values are corrected with **Benjamini–Hochberg FDR**.

**Figures.** Violin+box+strip panels, PCA of the full signature with 95%
confidence ellipses, a clustered feature-correlation heatmap, and an
effect-size-vs-FDR volcano. Text is preserved as editable text in the SVG
(`svg.fonttype='none'`) so figures open fully editable in Illustrator/Inkscape.

---

## Validation on real data (BBBC021, MCF-7)

MorphoStat was validated on the **BBBC021** image-based screen (Ljosa et al.,
2013, *J Biomol Screen*; Broad Bioimage Benchmark Collection) — real **MCF-7
breast-cancer cells** treated with a compendium of mechanistically distinct
compounds, stained for DNA / Actin (phalloidin) / β-Tubulin, segmented and
quantified per cell in CellProfiler and averaged per well. We compared the DMSO
negative control against the cytoskeleton-relevant mechanism classes.

The pipeline recovered the **known pharmacology** with no manual tuning:

| Mechanism class | Recovered morphological signature (vs DMSO, FDR<0.05) |
|---|---|
| **Actin disruptors** (cytochalasin/latrunculin) | smaller cells (Area ↓), reduced boundary smoothness (Solidity ↓, FormFactor ↓), redistributed brighter F-actin (mean Actin intensity ↑) — the classic rounding/arborization phenotype |
| **Microtubule destabilizers** (nocodazole, vinblastine) | larger, more elongated, irregular cells with **increased** F-actin (Actin intensity ↑, FDR≈1e-21) — the RhoA-driven stress-fiber induction these drugs are known to cause |
| **Microtubule stabilizers** (taxol/epothilone) | strong tubulin-intensity and texture shifts; distinct from destabilizers |
| **Aurora-kinase inhibitors** | large multinucleated cells (Area ↑) — the expected cytokinesis-failure phenotype |

PCA of the full feature set separated each mechanism class from the tight DMSO
control cluster, and the correlation heatmap recovered coherent Actin-, Tubulin-
, and shape-feature blocks.

Reproduce the demo dataset:

```bash
git clone --depth 1 https://github.com/cytomining/cytominergallery.git
python examples/prepare_bbbc021.py cytominergallery/inst/extdata demo_input
morphostat run --input demo_input --group-col moa --control DMSO \
               --batch-col Plate --source cellprofiler --outdir results
```

Two worked result sets ship with this repository:

- `results_cytoskeleton/` — focused DMSO vs the four cytoskeleton-relevant
  mechanism classes (the cleanest figures for a figure panel).
- `results_full_screen/` — the complete 13-mechanism screen (632 wells,
  473 features, 4,619 significant feature×group hits) demonstrating scalability.

## Tests

Engineering smoke tests exercise every code path (ingestion, the FilamentSensor
per-filament adapter, cleaning, replicate normalization, auto-selected
statistics, the vector figure writers, and the full pipeline) on tiny synthetic
fixtures:

```bash
pip install -e .
pytest -q          # 16 tests
```

These tests verify the plumbing only; the *scientific* validation is the real
BBBC021 analysis above. The synthetic fixtures are never presented as results.

---

## Limitations & notes

- BBBC021 here is *per-well aggregated* real data; for single-cell exports the
  same pipeline applies with `--source cellprofiler` and the object table.
- Cliff's δ saturates at ±1 for fully separated distributions (common for potent
  compounds); interpret alongside the omnibus effect sizes.
- The FilamentSensor adapter's column matching is tolerant but version-dependent;
  verify the mapping on your first export.

## References

- Hauke L, Primeßnig A, Eltzner B, Radwitz J, Huckemann SF, Rehfeldt F. (2023).
  FilamentSensor 2.0: an open-source modular toolbox for 2D/3D cytoskeletal
  filament tracking. *PLOS ONE* 18:e0279336.
- Šimunić I, Jagečić D, Isaković J, Dobrivojević Radmilović M, Mitrečić D. (2024).
  Lusca: FIJI (ImageJ) based tool for automated morphological analysis. *Sci Rep* 14.
- Ljosa V, Sokolnicki KL, Carpenter AE. (2012/2013). Annotated high-throughput
  microscopy image sets for validation / BBBC021. *Nat Methods* / *J Biomol Screen*.

## License

MIT.
