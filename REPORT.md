# MorphoStat analysis report

- **Input:** `demo_input_cytoskeleton`  (source preset: `cellprofiler`)
- **Samples analysed:** 450
- **Features retained:** 473
- **Grouping:** `moa`  |  **control:** `DMSO`  |  **batch/replicate key:** `Plate`
- **Significance:** Benjamini-Hochberg FDR < 0.05

## Group sizes
- DMSO: 330
- Microtubule destabilizers: 42
- Aurora kinase inhibitors: 36
- Microtubule stabilizers: 27
- Actin disruptors: 15

## Differential morphology (each group vs control)
Total significant feature-by-group hits: **1640** of 1892 tests.

| treatment group | significant features | top feature | top |effect| |
|---|---|---|---|
| Microtubule destabilizers | 426 | Nuclei_Texture_AngularSecondMoment_CorrActin_3_0 | 1.00 (cliffs_delta) |
| Microtubule stabilizers | 422 | Cells_Intensity_MaxIntensityEdge_CorrTub | 1.00 (cliffs_delta) |
| Aurora kinase inhibitors | 406 | Nuclei_Texture_AngularSecondMoment_CorrActin_3_0 | 1.00 (cliffs_delta) |
| Actin disruptors | 386 | Cells_Intensity_MedianIntensity_CorrTub | 1.00 (cliffs_delta) |

## Highlighted feature panel (auto-selected by largest mean effect)
- Cells_AreaShape_Area
- Cells_AreaShape_FormFactor
- Cells_AreaShape_Eccentricity
- Cells_AreaShape_Solidity
- Cells_Intensity_MeanIntensity_CorrActin
- Cells_Intensity_MeanIntensity_CorrTub

## Test selection summary
- mann_whitney_u: 1892 comparisons

## Files
- `figures/violin_panel.svg|pdf` - feature distributions by group
- `figures/pca.svg|pdf` - morphological signature clustering
- `figures/correlation_heatmap.svg|pdf` - feature correlation structure
- `figures/volcano_*.svg|pdf` - effect size vs FDR for strongest responder
- `tables/pairwise_vs_control.csv` - full per-feature statistics
- `tables/omnibus_across_groups.csv` - ANOVA/Kruskal across all groups
- `tables/normalized_features.csv` - cleaned, normalized matrix
- `tables/cleaning_report.txt` - ingestion/cleaning audit trail

---

```
MorphoStat cleaning report
==========================
rows: 450 -> 450
features: 473 -> 473
values imputed (median): 0

dropped (>NaN threshold)  [0]: 
dropped (near-zero var)   [0]: 
dropped (non-feature)     [0]: 
```