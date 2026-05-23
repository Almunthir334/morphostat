"""
morphostat.viz
==============
Publication-ready, *editable* vector figures (SVG + PDF).

All text is kept as real text in the SVG (``svg.fonttype='none'``) so figures
remain fully editable in Illustrator / Inkscape / Affinity. Every helper saves
both a .svg and a .pdf and returns the matplotlib Figure.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats as _sps
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def set_style():
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "svg.fonttype": "none",          # keep text editable in vector output
        "pdf.fonttype": 42,              # TrueType, editable in PDF
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _save(fig, outdir: str, name: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for ext in ("svg", "pdf", "png"):   # svg+pdf are the editable deliverables; png is a preview
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        paths[ext] = p
    return paths


def _palette(groups: Sequence[str], control: str | None = None) -> dict:
    groups = list(dict.fromkeys(groups))
    base = sns.color_palette("colorblind", n_colors=max(len(groups), 3))
    pal = {g: base[i % len(base)] for i, g in enumerate(groups)}
    if control in pal:
        pal[control] = (0.45, 0.45, 0.45)  # grey control
    return pal


# ---------------------------------------------------------------------------
def violin_panel(
    df: pd.DataFrame, features: Sequence[str], group_col: str,
    outdir: str, name: str = "violin_panel", control: str | None = None,
    order: Sequence[str] | None = None, ncols: int = 3, title: str | None = None,
) -> dict[str, str]:
    """Grid of violin + box + jittered points for selected features by group."""
    set_style()
    features = [f for f in features if f in df.columns]
    n = len(features); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    if order is None:
        order = list(df[group_col].dropna().unique())
        if control in order:
            order = [control] + [g for g in order if g != control]
    pal = _palette(order, control)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False)
    for i, feat in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        sns.violinplot(data=df, x=group_col, y=feat, order=order, hue=group_col,
                       palette=pal, inner=None, cut=0, linewidth=0.8, legend=False, ax=ax)
        for coll in ax.collections:
            coll.set_alpha(0.55)
        sns.boxplot(data=df, x=group_col, y=feat, order=order, width=0.18,
                    showcaps=True, boxprops={"facecolor": "white", "zorder": 3},
                    whiskerprops={"linewidth": 0.8}, fliersize=0, linewidth=0.8, ax=ax)
        sns.stripplot(data=df, x=group_col, y=feat, order=order, color="black",
                      size=1.6, alpha=0.35, jitter=0.18, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(feat, fontsize=8)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title, y=1.005, fontsize=11)
    fig.tight_layout()
    paths = _save(fig, outdir, name)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
def pca_plot(
    df: pd.DataFrame, feature_cols: Sequence[str], group_col: str,
    outdir: str, name: str = "pca", control: str | None = None,
    order: Sequence[str] | None = None, draw_ellipses: bool = True,
    title: str | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    """PCA scatter of the full morphological signature, coloured by group."""
    set_style()
    X = df[list(feature_cols)].to_numpy(float)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_ * 100
    scores = df[[group_col]].copy()
    scores["PC1"], scores["PC2"] = pcs[:, 0], pcs[:, 1]

    if order is None:
        order = list(df[group_col].dropna().unique())
        if control in order:
            order = [control] + [g for g in order if g != control]
    pal = _palette(order, control)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for g in order:
        sub = scores[scores[group_col] == g]
        ax.scatter(sub["PC1"], sub["PC2"], s=22, alpha=0.8, label=str(g),
                   color=pal[g], edgecolor="white", linewidth=0.3)
        if draw_ellipses and len(sub) >= 3:
            _confidence_ellipse(sub["PC1"].to_numpy(), sub["PC2"].to_numpy(), ax,
                                n_std=2.0, edgecolor=pal[g], facecolor="none",
                                linewidth=1.0, alpha=0.7)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.axhline(0, color="grey", lw=0.5, ls=":"); ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.legend(title=group_col, fontsize=7, title_fontsize=8, frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    paths = _save(fig, outdir, name)
    plt.close(fig)
    return paths, scores


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if x.size < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    e = Ellipse((np.mean(x), np.mean(y)), width=w, height=h, angle=theta, **kwargs)
    ax.add_patch(e)


# ---------------------------------------------------------------------------
def correlation_heatmap(
    df: pd.DataFrame, feature_cols: Sequence[str], outdir: str,
    name: str = "correlation_heatmap", method: str = "pearson",
    cluster: bool = True, max_features: int = 40, title: str | None = None,
) -> dict[str, str]:
    """Clustered feature-feature correlation heatmap."""
    set_style()
    feats = list(feature_cols)
    if len(feats) > max_features:  # keep the most variable features for readability
        var = df[feats].var().sort_values(ascending=False)
        feats = var.index[:max_features].tolist()
    corr = df[feats].corr(method=method)
    if cluster and len(feats) > 2:
        d = (1 - corr.abs()).to_numpy(dtype=float).copy()
        d = (d + d.T) / 2.0           # enforce exact symmetry
        np.fill_diagonal(d, 0.0)
        Z = linkage(squareform(d, checks=False), method="average")
        idx = leaves_list(Z)
        corr = corr.iloc[idx, idx]
    sz = max(6, 0.22 * len(feats))
    fig, ax = plt.subplots(figsize=(sz, sz * 0.92))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.5, "label": f"{method} r"},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.tick_params(labelsize=5)
    ax.set_title(title or f"Cytoskeletal feature correlation ({method})")
    fig.tight_layout()
    paths = _save(fig, outdir, name)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
def effect_volcano(
    pairwise: pd.DataFrame, group: str, outdir: str,
    name: str | None = None, alpha: float = 0.05, top_n_labels: int = 12,
) -> dict[str, str]:
    """Volcano-style summary: effect size vs -log10(FDR) for one treatment group."""
    set_style()
    sub = pairwise[pairwise["group"] == group].copy()
    sub = sub.dropna(subset=["effect", "fdr_bh"])
    sub["nlog10fdr"] = -np.log10(sub["fdr_bh"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sig = sub["fdr_bh"] < alpha
    ax.scatter(sub.loc[~sig, "effect"], sub.loc[~sig, "nlog10fdr"], s=14,
               color="lightgrey", alpha=0.7, label="ns")
    ax.scatter(sub.loc[sig & (sub["effect"] > 0), "effect"], sub.loc[sig & (sub["effect"] > 0), "nlog10fdr"],
               s=18, color="#c0392b", alpha=0.8, label=f"up (FDR<{alpha})")
    ax.scatter(sub.loc[sig & (sub["effect"] < 0), "effect"], sub.loc[sig & (sub["effect"] < 0), "nlog10fdr"],
               s=18, color="#2471a3", alpha=0.8, label=f"down (FDR<{alpha})")
    ax.axhline(-np.log10(alpha), color="grey", ls="--", lw=0.7)
    ax.axvline(0, color="grey", ls=":", lw=0.6)
    et = sub["effect_type"].dropna().iloc[0] if sub["effect_type"].notna().any() else "effect"
    ax.set_xlabel(f"effect size ({et}),  {group} vs control")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Morphological response: {group}")

    # ---- decluttered labelling: stack top hits per side with leader lines ----
    ymax = sub["nlog10fdr"].max()
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.5, ymax * 1.18)
    sig_df = sub.loc[sig].dropna(subset=["effect"]).drop_duplicates("feature")
    sig_df = sig_df.reindex(sig_df["effect"].abs().sort_values(ascending=False).index)
    for side, x_anchor, ha in [(1, 1.30, "right"), (-1, -1.30, "left")]:
        side_hits = sig_df[np.sign(sig_df["effect"]) == side].head(max(1, top_n_labels // 2))
        n = len(side_hits)
        for k, (_, r) in enumerate(side_hits.iterrows()):
            y_lab = ymax * (1.08 - 0.11 * k)
            ax.annotate(_short(r["feature"], 30),
                        xy=(r["effect"], r["nlog10fdr"]),
                        xytext=(x_anchor, y_lab), ha=ha, va="center", fontsize=5.5,
                        color="#444444",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#bbbbbb",
                                        shrinkA=0, shrinkB=2))
    ax.legend(fontsize=7, frameon=False, loc="lower center")
    fig.tight_layout()
    paths = _save(fig, outdir, name or f"volcano_{_safe(group)}")
    plt.close(fig)
    return paths


def _short(s: str, n: int = 26) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))
