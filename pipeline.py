"""
morphostat.pipeline
===================
One-call orchestration: folder of CSVs in -> figures, stats tables, and a report out.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from . import io as mio
from . import stats as mstats
from . import viz as mviz


def _auto_feature_panel(pairwise: pd.DataFrame, feature_cols, k: int = 6) -> list[str]:
    """Pick an interpretable panel: the features with the largest mean |effect|
    across treatment groups (these are the most discriminating morphology readouts)."""
    if pairwise is None or not len(pairwise):
        return list(feature_cols)[:k]
    rank = (pairwise.dropna(subset=["effect"])
            .assign(abseff=lambda d: d["effect"].abs())
            .groupby("feature")["abseff"].mean()
            .sort_values(ascending=False))
    return rank.index[:k].tolist()


def run_pipeline(
    input_path: str,
    outdir: str,
    group_col: str,
    control: str,
    *,
    source: str = "generic",            # {"generic","cellprofiler","filamentsensor"}
    batch_col: str | None = None,
    meta_cols: Sequence[str] = (),
    explicit_features: Sequence[str] | None = None,
    normalization: str = "robust_z_to_control",
    impute: str = "median",
    nan_frac_thresh: float = 0.5,
    alpha: float = 0.05,
    panel_features: Sequence[str] | None = None,
    fs_sample_col: str | None = None,
    pattern: str = "*.csv",
    recursive: bool = False,
    max_heatmap_features: int = 40,
) -> dict:
    """Run the full MorphoStat pipeline and write all outputs under ``outdir``."""
    t0 = datetime.now()
    fig_dir = os.path.join(outdir, "figures")
    tbl_dir = os.path.join(outdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # 1. INGEST -----------------------------------------------------------
    raw = mio.read_folder(input_path, pattern=pattern, recursive=recursive)
    if source == "filamentsensor" and fs_sample_col:
        # collapse per-filament rows to per-sample features, then re-attach metadata
        agg = mio.aggregate_filamentsensor(raw, sample_col=fs_sample_col)
        keep_meta = [c for c in ([group_col, batch_col, *meta_cols]) if c and c in raw.columns]
        if keep_meta:
            meta = raw[[fs_sample_col, *keep_meta]].drop_duplicates(fs_sample_col)
            raw = agg.merge(meta, on=fs_sample_col, how="left")
        else:
            raw = agg

    all_meta = [c for c in ([group_col, batch_col, *meta_cols, fs_sample_col]) if c]
    feature_cols = mio.detect_feature_columns(raw, meta_cols=all_meta,
                                              explicit_features=explicit_features)
    if not feature_cols:
        raise ValueError("No numeric morphology features detected. "
                         "Pass explicit_features or check meta_cols.")

    # 2. CLEAN ------------------------------------------------------------
    clean_df, feature_cols, report = mio.clean(
        raw, feature_cols, nan_frac_thresh=nan_frac_thresh, impute=impute)
    with open(os.path.join(tbl_dir, "cleaning_report.txt"), "w") as fh:
        fh.write(report.as_text())

    if group_col not in clean_df.columns:
        raise ValueError(f"group_col {group_col!r} not present in data.")
    if control not in set(clean_df[group_col].astype(str)):
        raise ValueError(f"control group {control!r} not found in column {group_col!r}.")
    clean_df[group_col] = clean_df[group_col].astype(str)
    control = str(control)

    # 3. NORMALIZE --------------------------------------------------------
    norm_df = mio.normalize(clean_df, feature_cols, method=normalization,
                            batch_col=batch_col, group_col=group_col, control=control)
    keep_cols = [c for c in [group_col, batch_col, fs_sample_col, *meta_cols] if c and c in norm_df.columns]
    norm_df[list(dict.fromkeys(keep_cols + list(feature_cols)))].to_csv(
        os.path.join(tbl_dir, "normalized_features.csv"), index=False)

    # 4. STATS ------------------------------------------------------------
    pairwise = mstats.pairwise_vs_control(norm_df, feature_cols, group_col, control, alpha=alpha)
    omni = mstats.omnibus(norm_df, feature_cols, group_col, alpha=alpha)
    pairwise.to_csv(os.path.join(tbl_dir, "pairwise_vs_control.csv"), index=False)
    omni.to_csv(os.path.join(tbl_dir, "omnibus_across_groups.csv"), index=False)

    # 5. VISUALIZE --------------------------------------------------------
    order = [control] + [g for g in norm_df[group_col].unique() if g != control]
    if panel_features is None:
        panel_features = _auto_feature_panel(pairwise, feature_cols, k=6)

    figures = {}
    figures["violin_panel"] = mviz.violin_panel(
        norm_df, panel_features, group_col, fig_dir, control=control, order=order,
        title="Morphological features by treatment group (robust z to control)")
    figures["pca"], _ = mviz.pca_plot(
        norm_df, feature_cols, group_col, fig_dir, control=control, order=order,
        title="Morphological signature (PCA of all features)")
    figures["correlation_heatmap"] = mviz.correlation_heatmap(
        norm_df, feature_cols, fig_dir, max_features=max_heatmap_features)
    treat_groups = [g for g in order if g != control]
    if treat_groups:
        # volcano for the treatment with the strongest overall response
        strongest = (pairwise.dropna(subset=["effect"])
                     .assign(a=lambda d: d["effect"].abs())
                     .groupby("group")["a"].mean().sort_values(ascending=False))
        target = strongest.index[0] if len(strongest) else treat_groups[0]
        figures["volcano"] = mviz.effect_volcano(pairwise, target, fig_dir, alpha=alpha)

    # 6. REPORT + MANIFEST ------------------------------------------------
    report_md = _build_report(input_path, source, group_col, control, batch_col,
                              feature_cols, norm_df, pairwise, omni, report, alpha,
                              panel_features)
    with open(os.path.join(outdir, "REPORT.md"), "w") as fh:
        fh.write(report_md)

    manifest = dict(
        timestamp=t0.isoformat(), input=input_path, source=source,
        group_col=group_col, control=control, batch_col=batch_col,
        normalization=normalization, alpha=alpha,
        n_samples=int(len(norm_df)), n_features=int(len(feature_cols)),
        groups={k: int(v) for k, v in norm_df[group_col].value_counts().items()},
        n_significant_pairwise=int(pairwise["significant"].sum()) if "significant" in pairwise else 0,
        figures={k: v for k, v in figures.items()},
        runtime_sec=(datetime.now() - t0).total_seconds(),
    )
    with open(os.path.join(outdir, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    return dict(manifest=manifest, pairwise=pairwise, omnibus=omni,
                normalized=norm_df, feature_cols=feature_cols,
                cleaning_report=report, figures=figures)


def _build_report(input_path, source, group_col, control, batch_col, feature_cols,
                  norm_df, pairwise, omni, clean_report, alpha, panel) -> str:
    n_sig = int(pairwise["significant"].sum()) if "significant" in pairwise else 0
    lines = []
    lines.append("# MorphoStat analysis report\n")
    lines.append(f"- **Input:** `{input_path}`  (source preset: `{source}`)")
    lines.append(f"- **Samples analysed:** {len(norm_df)}")
    lines.append(f"- **Features retained:** {len(feature_cols)}")
    lines.append(f"- **Grouping:** `{group_col}`  |  **control:** `{control}`"
                 + (f"  |  **batch/replicate key:** `{batch_col}`" if batch_col else ""))
    lines.append(f"- **Significance:** Benjamini-Hochberg FDR < {alpha}\n")

    lines.append("## Group sizes")
    vc = norm_df[group_col].value_counts()
    for g, n in vc.items():
        lines.append(f"- {g}: {n}")
    lines.append("")

    lines.append("## Differential morphology (each group vs control)")
    lines.append(f"Total significant feature-by-group hits: **{n_sig}** "
                 f"of {len(pairwise)} tests.\n")
    if "group" in pairwise.columns:
        per_group = (pairwise.groupby("group")["significant"].sum()
                     .sort_values(ascending=False))
        lines.append("| treatment group | significant features | top feature | top |effect| |")
        lines.append("|---|---|---|---|")
        for g, n in per_group.items():
            sub = pairwise[(pairwise["group"] == g)].dropna(subset=["effect"])
            if len(sub):
                top = sub.reindex(sub["effect"].abs().sort_values(ascending=False).index).iloc[0]
                lines.append(f"| {g} | {int(n)} | {top['feature']} | {top['effect']:.2f} ({top['effect_type']}) |")
            else:
                lines.append(f"| {g} | {int(n)} | - | - |")
    lines.append("")

    lines.append("## Highlighted feature panel (auto-selected by largest mean effect)")
    for f in panel:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## Test selection summary")
    if "test" in pairwise.columns:
        for t, n in pairwise["test"].value_counts().items():
            lines.append(f"- {t}: {n} comparisons")
    lines.append("")

    lines.append("## Files")
    lines.append("- `figures/violin_panel.svg|pdf` - feature distributions by group")
    lines.append("- `figures/pca.svg|pdf` - morphological signature clustering")
    lines.append("- `figures/correlation_heatmap.svg|pdf` - feature correlation structure")
    lines.append("- `figures/volcano_*.svg|pdf` - effect size vs FDR for strongest responder")
    lines.append("- `tables/pairwise_vs_control.csv` - full per-feature statistics")
    lines.append("- `tables/omnibus_across_groups.csv` - ANOVA/Kruskal across all groups")
    lines.append("- `tables/normalized_features.csv` - cleaned, normalized matrix")
    lines.append("- `tables/cleaning_report.txt` - ingestion/cleaning audit trail")
    lines.append("\n---\n")
    lines.append("```")
    lines.append(clean_report.as_text())
    lines.append("```")
    return "\n".join(lines)
