"""
morphostat.stats
================
Automated differential morphology statistics.

Workflow per feature:
  1. Shapiro-Wilk normality test on each group.
  2. Levene test for equality of variances.
  3. Test selection:
       2 groups  : both-normal & equal-var -> Student t ; both-normal & unequal-var -> Welch t ;
                    otherwise -> Mann-Whitney U (rank-sum).
       >2 groups : all-normal & homoscedastic -> one-way ANOVA (+Tukey HSD post-hoc) ;
                    otherwise -> Kruskal-Wallis (+Dunn post-hoc).
  4. Effect sizes: Hedges' g (parametric) / Cliff's delta (non-parametric);
     eta^2 / epsilon^2 for the omnibus.
  5. Benjamini-Hochberg FDR across all reported comparisons.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _shapiro_p(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    if n > 5000:  # Shapiro is unreliable / errors for very large n -> subsample
        rng = np.random.default_rng(0)
        x = rng.choice(x, 5000, replace=False)
    if np.allclose(x, x[0]):
        return 0.0
    try:
        return float(stats.shapiro(x).pvalue)
    except Exception:
        return np.nan


def hedges_g(treat: np.ndarray, control: np.ndarray) -> float:
    """Standardized mean difference (treatment - control), small-sample corrected.
    Positive => treatment mean is higher than control."""
    t = treat[np.isfinite(treat)]; c = control[np.isfinite(control)]
    n1, n2 = len(t), len(c)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp2 = ((n1 - 1) * np.var(t, ddof=1) + (n2 - 1) * np.var(c, ddof=1)) / (n1 + n2 - 2)
    sp = np.sqrt(sp2)
    if sp == 0:
        return 0.0
    d = (np.mean(t) - np.mean(c)) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)  # bias correction
    return float(d * J)


def cliffs_delta(treat: np.ndarray, control: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]; >0 => treatment tends to exceed control."""
    t = treat[np.isfinite(treat)]; c = control[np.isfinite(control)]
    if len(t) == 0 or len(c) == 0:
        return np.nan
    # rank-based O(n log n) computation
    allv = np.concatenate([t, c])
    ranks = stats.rankdata(allv)
    rt = ranks[: len(t)].sum()
    delta = (2 * rt - len(t) * (len(t) + 1)) / (len(t) * len(c)) - 1
    return float(delta)


# ---------------------------------------------------------------------------
# pairwise: each group vs control
# ---------------------------------------------------------------------------
def compare_two(control_vals: np.ndarray, treat_vals: np.ndarray, alpha: float = ALPHA) -> dict:
    c = np.asarray(control_vals, float); t = np.asarray(treat_vals, float)
    c = c[np.isfinite(c)]; t = t[np.isfinite(t)]
    res = dict(n_control=len(c), n_treat=len(t),
               shapiro_p_control=_shapiro_p(c), shapiro_p_treat=_shapiro_p(t),
               levene_p=np.nan, test=None, statistic=np.nan, p_value=np.nan,
               effect=np.nan, effect_type=None, direction=None)
    if len(c) < 3 or len(t) < 3:
        res["test"] = "insufficient_n"
        return res
    try:
        res["levene_p"] = float(stats.levene(c, t, center="median").pvalue)
    except Exception:
        res["levene_p"] = np.nan

    normal = (res["shapiro_p_control"] > alpha) and (res["shapiro_p_treat"] > alpha)
    equal_var = (res["levene_p"] > alpha) if np.isfinite(res["levene_p"]) else False

    if normal:
        st = stats.ttest_ind(t, c, equal_var=equal_var)
        res["test"] = "students_t" if equal_var else "welchs_t"
        res["statistic"], res["p_value"] = float(st.statistic), float(st.pvalue)
        res["effect"], res["effect_type"] = hedges_g(t, c), "hedges_g"
    else:
        st = stats.mannwhitneyu(t, c, alternative="two-sided")
        res["test"] = "mann_whitney_u"
        res["statistic"], res["p_value"] = float(st.statistic), float(st.pvalue)
        res["effect"], res["effect_type"] = cliffs_delta(t, c), "cliffs_delta"

    res["direction"] = "up" if (np.median(t) >= np.median(c)) else "down"
    return res


def pairwise_vs_control(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_col: str,
    control: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Compare every non-control group with the control group, for every feature.
    BH-FDR is applied across the full set of (feature x comparison) tests."""
    groups = [g for g in df[group_col].dropna().unique() if g != control]
    rows = []
    ctrl_df = df[df[group_col] == control]
    for feat in feature_cols:
        cvals = ctrl_df[feat].to_numpy(float)
        for g in groups:
            tvals = df.loc[df[group_col] == g, feat].to_numpy(float)
            r = compare_two(cvals, tvals, alpha=alpha)
            r.update(feature=feat, group=g, control=control)
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        valid = out["p_value"].notna()
        out["fdr_bh"] = np.nan
        if valid.any():
            out.loc[valid, "fdr_bh"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
        out["significant"] = out["fdr_bh"] < alpha
    cols = ["feature", "group", "control", "test", "statistic", "p_value", "fdr_bh",
            "significant", "effect", "effect_type", "direction",
            "n_control", "n_treat", "shapiro_p_control", "shapiro_p_treat", "levene_p"]
    return out[[c for c in cols if c in out.columns]]


# ---------------------------------------------------------------------------
# omnibus across all groups
# ---------------------------------------------------------------------------
def _eta_squared_anova(groups_vals: list[np.ndarray]) -> float:
    grand = np.concatenate(groups_vals)
    gm = grand.mean()
    ss_between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups_vals)
    ss_total = ((grand - gm) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def omnibus(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_col: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    levels = [l for l in df[group_col].dropna().unique()]
    rows = []
    for feat in feature_cols:
        gv = [df.loc[df[group_col] == l, feat].to_numpy(float) for l in levels]
        gv = [g[np.isfinite(g)] for g in gv]
        gv = [g for g in gv if len(g) >= 3]
        if len(gv) < 2:
            continue
        all_normal = all(_shapiro_p(g) > alpha for g in gv)
        try:
            levene_p = float(stats.levene(*gv, center="median").pvalue)
        except Exception:
            levene_p = np.nan
        homoscedastic = (levene_p > alpha) if np.isfinite(levene_p) else False
        if all_normal and homoscedastic:
            st = stats.f_oneway(*gv)
            test, eff, eff_type = "anova", _eta_squared_anova(gv), "eta_squared"
        else:
            st = stats.kruskal(*gv)
            n = sum(len(g) for g in gv); k = len(gv)
            eps2 = (st.statistic - k + 1) / (n - k) if (n - k) > 0 else np.nan
            test, eff, eff_type = "kruskal_wallis", float(eps2), "epsilon_squared"
        rows.append(dict(feature=feat, test=test, statistic=float(st.statistic),
                         p_value=float(st.pvalue), effect=eff, effect_type=eff_type,
                         n_groups=len(gv), levene_p=levene_p))
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
        out["significant"] = out["fdr_bh"] < alpha
    return out


def tukey_posthoc(df: pd.DataFrame, feature: str, group_col: str) -> pd.DataFrame:
    """Tukey HSD post-hoc for one feature across all groups (parametric)."""
    sub = df[[feature, group_col]].dropna()
    res = pairwise_tukeyhsd(sub[feature].to_numpy(float), sub[group_col].astype(str).to_numpy())
    tbl = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    tbl.insert(0, "feature", feature)
    return tbl
