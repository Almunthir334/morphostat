"""
morphostat.io
=============
Data ingestion, cleaning, and replicate-aware normalization for
morphological measurement tables (FilamentSensor, ImageJ/CellProfiler, or generic).

The ingestion layer is deliberately tolerant: column headers differ between
FilamentSensor versions, ImageJ "Measure" exports, and CellProfiler. Rather than
hard-coding a single schema, we detect numeric morphology features automatically
and let the caller override via explicit column lists.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column patterns that are *not* morphology features even though they're numeric.
# (object/image indices, pixel coordinates, identifiers, batch keys).
# ---------------------------------------------------------------------------
_NON_FEATURE_PATTERNS = [
    r"center_?x", r"center_?y", r"location_center", r"_x$", r"_y$",
    r"object_?number", r"image_?number", r"^objectnumber$", r"^imagenumber$",
    r"parent", r"children", r"^id$", r"^index$", r"unnamed",
]
_NON_FEATURE_RE = re.compile("|".join(_NON_FEATURE_PATTERNS), re.IGNORECASE)

# FilamentSensor 2.0 / ImageJ FilamentSensor export columns vary by version and
# by whether you export per-filament or per-image tables. These tolerant patterns
# map common headers onto canonical feature names. Override `fs_column_map` if your
# export differs.
FILAMENTSENSOR_PATTERNS = {
    "filament_length": [r"\blength\b", r"len(_|$)", r"filament.?length"],
    "filament_width":  [r"\bwidth\b", r"thick"],
    "filament_angle":  [r"\bangle\b", r"orient", r"theta"],
    "order_parameter": [r"order.?param", r"\border\b", r"\bs_?value\b", r"^s$"],
    "curvature":       [r"curv"],
    "filament_count":  [r"count", r"n_?filament", r"number.?of.?filament"],
}


@dataclass
class CleaningReport:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_features_in: int = 0
    n_features_out: int = 0
    dropped_high_nan: list = field(default_factory=list)
    dropped_low_variance: list = field(default_factory=list)
    dropped_non_feature: list = field(default_factory=list)
    n_values_imputed: int = 0
    impute_strategy: str = "median"

    def as_text(self) -> str:
        lines = [
            "MorphoStat cleaning report",
            "==========================",
            f"rows: {self.n_rows_in} -> {self.n_rows_out}",
            f"features: {self.n_features_in} -> {self.n_features_out}",
            f"values imputed ({self.impute_strategy}): {self.n_values_imputed}",
            "",
            f"dropped (>NaN threshold)  [{len(self.dropped_high_nan)}]: "
            + ", ".join(self.dropped_high_nan[:30]) + (" ..." if len(self.dropped_high_nan) > 30 else ""),
            f"dropped (near-zero var)   [{len(self.dropped_low_variance)}]: "
            + ", ".join(self.dropped_low_variance[:30]) + (" ..." if len(self.dropped_low_variance) > 30 else ""),
            f"dropped (non-feature)     [{len(self.dropped_non_feature)}]: "
            + ", ".join(self.dropped_non_feature[:30]) + (" ..." if len(self.dropped_non_feature) > 30 else ""),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def read_folder(
    input_path: str,
    pattern: str = "*.csv",
    recursive: bool = False,
    sep: str | None = None,
    add_source_col: bool = True,
) -> pd.DataFrame:
    """Read and concatenate every CSV in a folder (or a single CSV file).

    Multiple files are aligned by column name (outer join of columns), so
    per-plate / per-replicate exports concatenate cleanly even if some columns
    are missing from a given file. A ``__source_file`` column records origin.
    """
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        search = os.path.join(input_path, "**", pattern) if recursive else os.path.join(input_path, pattern)
        files = sorted(glob.glob(search, recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {input_path!r}")

    frames = []
    sources = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=sep, engine="python") if sep is None else pd.read_csv(f, sep=sep)
        except Exception as exc:  # pragma: no cover - surfaced to user
            warnings.warn(f"Skipping {f}: {exc}")
            continue
        frames.append(df)
        sources.append((os.path.basename(f), len(df)))
    if not frames:
        raise ValueError("No readable CSV files were found.")
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    if add_source_col:
        # build the source column separately and concat along axis=1 (a single
        # block join) rather than inserting into a wide frame, which avoids
        # pandas' fragmentation warning on 500+ column morphology tables.
        src = pd.Series(
            np.repeat([s[0] for s in sources], [s[1] for s in sources]),
            index=combined.index, name="__source_file",
        )
        combined = pd.concat([combined, src], axis=1)
    return combined


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name, re.IGNORECASE) for p in patterns)


def aggregate_filamentsensor(
    df: pd.DataFrame,
    sample_col: str,
    fs_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collapse a *per-filament* FilamentSensor table to *per-sample* features.

    Each input row is one filament; output rows are one image/sample
    (``sample_col``). Angles are summarised with the circular mean and the
    nematic order parameter S = <cos 2(theta - <theta>)>; lengths/widths with
    the mean; and a filament count is added.
    """
    # resolve canonical -> actual column
    canon = {}
    if fs_column_map:
        canon.update(fs_column_map)
    else:
        for canonical, pats in FILAMENTSENSOR_PATTERNS.items():
            for c in df.columns:
                if _match_any(str(c), pats):
                    canon[canonical] = c
                    break

    out_rows = []
    for sample, g in df.groupby(sample_col):
        row = {sample_col: sample, "filament_count": len(g)}
        if "filament_length" in canon:
            row["filament_length_mean"] = g[canon["filament_length"]].mean()
            row["filament_length_total"] = g[canon["filament_length"]].sum()
        if "filament_width" in canon:
            row["filament_width_mean"] = g[canon["filament_width"]].mean()
        if "filament_angle" in canon:
            ang = np.deg2rad(g[canon["filament_angle"]].to_numpy(dtype=float))
            # nematic (line) statistics: use 2*theta
            mean_ang = 0.5 * np.angle(np.mean(np.exp(2j * ang)))
            S = np.mean(np.cos(2 * (ang - mean_ang)))
            row["mean_orientation_deg"] = np.rad2deg(mean_ang)
            row["order_parameter_S"] = float(S)
        if "order_parameter" in canon:  # if FS already exported S per filament/image
            row["order_parameter_S"] = g[canon["order_parameter"]].mean()
        if "curvature" in canon:
            row["curvature_mean"] = g[canon["curvature"]].mean()
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def detect_feature_columns(
    df: pd.DataFrame,
    meta_cols: Sequence[str] = (),
    explicit_features: Sequence[str] | None = None,
) -> list[str]:
    """Return the list of numeric morphology-feature columns.

    Drops anything in ``meta_cols``, anything matching the non-feature patterns
    (coordinates, indices, identifiers), and any non-numeric column.
    """
    if explicit_features is not None:
        return [c for c in explicit_features if c in df.columns]

    meta = set(meta_cols) | {"__source_file"}
    feats = []
    for c in df.columns:
        if c in meta:
            continue
        if _NON_FEATURE_RE.search(str(c)):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.9:  # mostly-numeric text column
                feats.append(c)
    return feats


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    nan_frac_thresh: float = 0.5,
    variance_thresh: float = 1e-12,
    impute: str = "median",  # {"median","mean","drop","none"}
) -> tuple[pd.DataFrame, list[str], CleaningReport]:
    """Clean a feature matrix: coerce to numeric, drop unusable columns,
    handle missing values. Returns (clean_df, kept_feature_cols, report)."""
    rep = CleaningReport(n_rows_in=len(df), n_features_in=len(feature_cols),
                         impute_strategy=impute)
    work = df.copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    kept = list(feature_cols)

    # drop high-NaN columns
    nan_frac = work[kept].isna().mean()
    rep.dropped_high_nan = nan_frac.index[nan_frac > nan_frac_thresh].tolist()
    kept = [c for c in kept if c not in rep.dropped_high_nan]

    # drop near-zero-variance columns
    var = work[kept].var(numeric_only=True)
    rep.dropped_low_variance = var.index[(var.fillna(0) <= variance_thresh)].tolist()
    kept = [c for c in kept if c not in rep.dropped_low_variance]

    # missing-value handling
    if impute == "drop":
        before = len(work)
        work = work.dropna(subset=kept)
        rep.n_rows_out = len(work)
        rep.n_values_imputed = 0
        _ = before
    elif impute in ("median", "mean"):
        n_missing = int(work[kept].isna().sum().sum())
        fill = work[kept].median() if impute == "median" else work[kept].mean()
        work[kept] = work[kept].fillna(fill)
        rep.n_values_imputed = n_missing
        rep.n_rows_out = len(work)
    else:  # none
        rep.n_rows_out = len(work)

    rep.n_features_out = len(kept)
    return work.reset_index(drop=True), kept, rep


# ---------------------------------------------------------------------------
# Normalization across replicates / batches
# ---------------------------------------------------------------------------
def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def normalize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    method: str = "robust_z_to_control",
    batch_col: str | None = None,
    group_col: str | None = None,
    control: str | None = None,
) -> pd.DataFrame:
    """Normalize features to remove plate/replicate batch effects.

    Methods
    -------
    robust_z_to_control : per batch, z-score each feature using the *control*
        group's median and MAD  -> x' = (x - median_ctrl) / MAD_ctrl.
        This is the standard negative-control plate normalization used in
        image-based morphological profiling. Batches lacking control wells fall
        back to the pooled-control statistics.
    standard : global mean/SD z-score (ignores batch).
    none : pass-through.
    """
    out = df.copy()
    feats = list(feature_cols)
    if method == "none":
        return out
    if method == "standard":
        mu = out[feats].mean()
        sd = out[feats].std(ddof=0).replace(0, np.nan)
        out[feats] = (out[feats] - mu) / sd
        out[feats] = out[feats].fillna(0.0)
        return out
    if method != "robust_z_to_control":
        raise ValueError(f"Unknown normalization method: {method}")

    if group_col is None or control is None:
        raise ValueError("robust_z_to_control requires group_col and control.")

    # pooled control statistics (fallback)
    pooled_ctrl = out[out[group_col] == control]
    pooled_med = pooled_ctrl[feats].median()
    pooled_mad = pooled_ctrl[feats].apply(lambda s: _mad(s.to_numpy(dtype=float)))

    def _transform(block: pd.DataFrame) -> pd.DataFrame:
        ctrl = block[block[group_col] == control]
        if len(ctrl) >= 2:
            med = ctrl[feats].median()
            mad = ctrl[feats].apply(lambda s: _mad(s.to_numpy(dtype=float)))
        else:
            med, mad = pooled_med, pooled_mad
        mad = mad.replace(0, np.nan).fillna(pooled_mad).replace(0, np.nan).fillna(1.0)
        block = block.copy()
        block[feats] = (block[feats] - med) / mad
        return block

    if batch_col is not None and batch_col in out.columns:
        out = out.groupby(batch_col, group_keys=False)[out.columns.tolist()].apply(_transform)
    else:
        out = _transform(out)
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.reset_index(drop=True)
