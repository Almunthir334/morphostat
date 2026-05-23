"""
MorphoStat smoke tests
=======================
These are *engineering* smoke tests, not a scientific validation. They use tiny,
deliberately synthetic toy tables whose only purpose is to exercise every code
path (ingestion, the FilamentSensor per-filament adapter, cleaning, replicate
normalization, the auto-selected statistics, the figure writers, and the full
CLI-equivalent pipeline) and confirm the package is wired together correctly.

The *biological* validation of MorphoStat is performed separately on the real
BBBC021 MCF-7 dataset (see README.md and examples/prepare_bbbc021.py); none of
the numbers below should be read as biological results.

Run with:  pytest -q
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import morphostat as ms
from morphostat import io as mio
from morphostat import stats as mstats
from morphostat import viz as mviz


# ---------------------------------------------------------------------------
# Toy-data fixtures (SYNTHETIC, format-only — not biological data)
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(0)


def _toy_generic_frame(n_per_group: int = 30) -> pd.DataFrame:
    """A small generic morphology table with a clear control/treatment shift.

    Two groups ('ctrl', 'drug') over two batches; 'drug' has a shifted 'area'
    and 'solidity' so the differential stats have something real to find. Also
    includes a non-feature column ('object_number') and a constant column to
    make sure cleaning/feature-detection drop them.
    """
    rows = []
    for batch in ("plateA", "plateB"):
        for grp, shift in (("ctrl", 0.0), ("drug", 2.5)):
            for i in range(n_per_group):
                rows.append(dict(
                    group=grp,
                    batch=batch,
                    object_number=i,                      # identifier -> dropped
                    area=RNG.normal(100 + shift * 8, 10),
                    solidity=RNG.normal(0.9 - shift * 0.05, 0.03),
                    eccentricity=RNG.normal(0.5, 0.05),
                    constant_col=1.0,                     # zero variance -> dropped
                ))
    return pd.DataFrame(rows)


def _toy_filamentsensor_frame(n_filaments: int = 200) -> pd.DataFrame:
    """A SYNTHETIC *per-filament* table in FilamentSensor-like wide format.

    One row per filament; columns named like a FilamentSensor export so the
    tolerant column-pattern matcher resolves Length/Width/Angle. Two images per
    group so the per-sample aggregation has something to collapse. This only
    proves the adapter math and column matching run — it is NOT real filament
    data.
    """
    rows = []
    for grp, ang_center, len_mu in (("ctrl", 90.0, 40.0), ("treated", 10.0, 60.0)):
        for img in (1, 2):
            sample = f"{grp}_img{img}"
            for _ in range(n_filaments // 2):
                rows.append(dict(
                    image_id=sample,
                    group=grp,
                    Length=max(1.0, RNG.normal(len_mu, 8)),
                    Width=max(0.5, RNG.normal(3.0, 0.5)),
                    Angle=(RNG.normal(ang_center, 12)) % 180.0,  # degrees, 0–180
                ))
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_csv_folder(tmp_path):
    """Write the toy generic frame out as two per-batch CSVs (multi-file ingest)."""
    df = _toy_generic_frame()
    folder = tmp_path / "csvs"
    folder.mkdir()
    for batch, g in df.groupby("batch"):
        g.to_csv(folder / f"{batch}.csv", index=False)
    return str(folder)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------
def test_package_exposes_public_api():
    assert hasattr(ms, "run_pipeline")
    assert hasattr(ms, "io") and hasattr(ms, "stats") and hasattr(ms, "viz")
    assert isinstance(ms.__version__, str)


# ---------------------------------------------------------------------------
# Ingestion + cleaning + normalization
# ---------------------------------------------------------------------------
def test_read_folder_concatenates_multiple_csvs(toy_csv_folder):
    df = mio.read_folder(toy_csv_folder)
    assert len(df) == 120                       # 30 * 2 groups * 2 batches
    assert "__source_file" in df.columns
    assert df["__source_file"].nunique() == 2   # two per-batch files


def test_read_folder_raises_on_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        mio.read_folder(str(empty))


def test_detect_feature_columns_drops_meta_and_identifiers(toy_csv_folder):
    df = mio.read_folder(toy_csv_folder)
    feats = mio.detect_feature_columns(df, meta_cols=["group", "batch"])
    assert set(feats) >= {"area", "solidity", "eccentricity"}
    assert "object_number" not in feats         # identifier pattern
    assert "group" not in feats and "batch" not in feats


def test_clean_drops_zero_variance_and_reports(toy_csv_folder):
    df = mio.read_folder(toy_csv_folder)
    feats = mio.detect_feature_columns(df, meta_cols=["group", "batch"])
    feats = list(feats) + ["constant_col"]      # force a zero-variance column in
    clean_df, kept, rep = mio.clean(df, feats, impute="median")
    assert "constant_col" in rep.dropped_low_variance
    assert "constant_col" not in kept
    assert rep.n_features_out == len(kept)


def test_clean_imputes_missing_values():
    df = _toy_generic_frame(20)
    df.loc[0:4, "area"] = np.nan                # inject 5 NaNs
    clean_df, kept, rep = mio.clean(df, ["area", "solidity"], impute="median")
    assert rep.n_values_imputed == 5
    assert clean_df["area"].isna().sum() == 0


def test_normalize_robust_z_centers_control_near_zero():
    df = _toy_generic_frame(40)
    feats = ["area", "solidity", "eccentricity"]
    norm = mio.normalize(df, feats, method="robust_z_to_control",
                         batch_col="batch", group_col="group", control="ctrl")
    # control median per batch should map to ~0 after robust z to control
    ctrl_med = norm.loc[norm["group"] == "ctrl", "area"].median()
    assert abs(ctrl_med) < 0.5


# ---------------------------------------------------------------------------
# FilamentSensor adapter (SYNTHETIC format-only smoke test)
# ---------------------------------------------------------------------------
def test_filamentsensor_adapter_collapses_per_filament_to_per_sample():
    """Format-only: confirms the per-filament -> per-sample aggregation runs and
    that the nematic order parameter S lands in its valid [-1, 1] range. Uses
    synthetic data; not a biological result."""
    fil = _toy_filamentsensor_frame()
    agg = mio.aggregate_filamentsensor(fil, sample_col="image_id")

    # one row per image/sample
    assert len(agg) == fil["image_id"].nunique() == 4
    # tolerant patterns should have resolved Length/Width/Angle -> canonical cols
    assert "filament_length_mean" in agg.columns
    assert "filament_width_mean" in agg.columns
    assert "order_parameter_S" in agg.columns
    assert "filament_count" in agg.columns
    # order parameter is bounded
    assert agg["order_parameter_S"].between(-1.0, 1.0).all()
    # counts add back up to the input
    assert agg["filament_count"].sum() == len(fil)


def test_filamentsensor_order_parameter_high_for_aligned_filaments():
    """If all filaments share one angle, S should be ~1 (perfectly aligned).
    Pure math check on the adapter, synthetic input."""
    n = 100
    fil = pd.DataFrame(dict(
        image_id=["s1"] * n,
        Angle=np.full(n, 45.0),     # all identical
        Length=np.full(n, 10.0),
    ))
    agg = mio.aggregate_filamentsensor(fil, sample_col="image_id")
    assert agg["order_parameter_S"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Statistics: auto test selection + effect sizes + FDR
# ---------------------------------------------------------------------------
def test_compare_two_picks_parametric_for_normal_data():
    # near-perfect normal samples via the inverse normal CDF on a uniform grid;
    # this reliably passes Shapiro-Wilk, so the selector should pick a t-test.
    from scipy.stats import norm
    grid = (np.arange(200) + 0.5) / 200
    c = norm.ppf(grid)
    t = norm.ppf(grid) + 1.0          # location shift, equal variance/shape
    res = mstats.compare_two(c, t)
    assert res["test"] in {"students_t", "welchs_t"}
    assert res["effect_type"] == "hedges_g"
    assert res["p_value"] < 0.05
    assert res["direction"] == "up"


def test_compare_two_falls_back_to_nonparametric_for_skewed_data():
    c = RNG.exponential(1.0, 200)               # strongly non-normal
    t = RNG.exponential(2.0, 200)
    res = mstats.compare_two(c, t)
    assert res["test"] == "mann_whitney_u"
    assert res["effect_type"] == "cliffs_delta"


def test_compare_two_handles_insufficient_n():
    res = mstats.compare_two([1.0, 2.0], [3.0])
    assert res["test"] == "insufficient_n"


def test_pairwise_vs_control_runs_and_applies_fdr():
    df = _toy_generic_frame(40)
    res = mstats.pairwise_vs_control(df, ["area", "solidity", "eccentricity"],
                                     group_col="group", control="ctrl")
    assert {"feature", "group", "fdr_bh", "significant", "effect"} <= set(res.columns)
    assert (res["fdr_bh"].dropna() >= 0).all() and (res["fdr_bh"].dropna() <= 1).all()
    # the engineered 'area' shift should come back significant
    area_hit = res[(res["feature"] == "area") & (res["group"] == "drug")]
    assert bool(area_hit["significant"].iloc[0])


def test_omnibus_across_groups():
    df = _toy_generic_frame(40)
    res = mstats.omnibus(df, ["area", "solidity"], group_col="group")
    assert {"feature", "test", "p_value", "fdr_bh"} <= set(res.columns)
    assert res["test"].isin({"anova", "kruskal_wallis"}).all()


# ---------------------------------------------------------------------------
# Visualization: editable vector output
# ---------------------------------------------------------------------------
def test_viz_writes_editable_vector_files(tmp_path):
    df = _toy_generic_frame(40)
    out = str(tmp_path / "figs")
    os.makedirs(out, exist_ok=True)

    paths = mviz.violin_panel(df, ["area", "solidity"], group_col="group",
                              control="ctrl", outdir=out, name="violin")
    assert os.path.exists(paths["svg"]) and os.path.exists(paths["pdf"])

    # editable text: matplotlib writes glyphs as <text>, not paths, when
    # svg.fonttype='none' (set in viz.set_style)
    with open(paths["svg"], encoding="utf-8") as fh:
        svg = fh.read()
    assert "<text" in svg

    mviz.pca_plot(df, ["area", "solidity", "eccentricity"], group_col="group",
                  control="ctrl", outdir=out, name="pca")
    mviz.correlation_heatmap(df, ["area", "solidity", "eccentricity"],
                             outdir=out, name="heat")
    assert os.path.exists(os.path.join(out, "pca.svg"))
    assert os.path.exists(os.path.join(out, "heat.svg"))


# ---------------------------------------------------------------------------
# Full pipeline (CLI-equivalent) end-to-end on toy data
# ---------------------------------------------------------------------------
def test_run_pipeline_end_to_end(toy_csv_folder, tmp_path):
    outdir = str(tmp_path / "results")
    summary = ms.run_pipeline(
        input_path=toy_csv_folder,
        outdir=outdir,
        group_col="group",
        control="ctrl",
        source="generic",
        batch_col="batch",
        meta_cols=["object_number"],
        normalization="robust_z_to_control",
        impute="median",
    )
    # expected artifacts on disk
    assert os.path.exists(os.path.join(outdir, "REPORT.md"))
    assert os.path.exists(os.path.join(outdir, "run_manifest.json"))
    assert os.path.exists(os.path.join(outdir, "tables", "pairwise_vs_control.csv"))
    assert os.path.exists(os.path.join(outdir, "figures", "pca.svg"))

    # manifest is valid JSON and records the run
    with open(os.path.join(outdir, "run_manifest.json")) as fh:
        manifest = json.load(fh)
    assert manifest["control"] == "ctrl"
    assert manifest["n_significant_pairwise"] >= 1   # engineered area/solidity shifts
    assert isinstance(summary, dict)
