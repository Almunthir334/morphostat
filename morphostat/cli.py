"""
morphostat.cli
==============
"One-click" command line interface.

    morphostat run --input <folder> --group-col moa --control DMSO \
                   --batch-col Plate --source cellprofiler --outdir results
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="morphostat",
        description="Automated morphological post-processing for FilamentSensor / "
                    "ImageJ / CellProfiler outputs: clean -> normalize -> stats -> figures.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full pipeline on a folder of CSVs")
    r.add_argument("--input", required=True, help="folder of CSVs (or a single CSV)")
    r.add_argument("--outdir", default="morphostat_results", help="output directory")
    r.add_argument("--group-col", required=True, help="column holding the treatment/condition label")
    r.add_argument("--control", required=True, help="value in --group-col denoting the control group")
    r.add_argument("--source", default="generic",
                   choices=["generic", "cellprofiler", "filamentsensor"],
                   help="input preset")
    r.add_argument("--batch-col", default=None, help="replicate/plate column for batch normalization")
    r.add_argument("--meta-cols", nargs="*", default=[], help="extra non-feature columns to exclude")
    r.add_argument("--features", nargs="*", default=None,
                   help="explicit feature columns (default: auto-detect)")
    r.add_argument("--panel", nargs="*", default=None,
                   help="feature columns to show in the violin panel (default: auto)")
    r.add_argument("--normalization", default="robust_z_to_control",
                   choices=["robust_z_to_control", "standard", "none"])
    r.add_argument("--impute", default="median", choices=["median", "mean", "drop", "none"])
    r.add_argument("--nan-frac-thresh", type=float, default=0.5)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--fs-sample-col", default=None,
                   help="(filamentsensor) per-filament -> per-sample grouping column")
    r.add_argument("--pattern", default="*.csv")
    r.add_argument("--recursive", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        res = run_pipeline(
            input_path=args.input, outdir=args.outdir,
            group_col=args.group_col, control=args.control, source=args.source,
            batch_col=args.batch_col, meta_cols=args.meta_cols,
            explicit_features=args.features, panel_features=args.panel,
            normalization=args.normalization, impute=args.impute,
            nan_frac_thresh=args.nan_frac_thresh, alpha=args.alpha,
            fs_sample_col=args.fs_sample_col, pattern=args.pattern,
            recursive=args.recursive,
        )
        m = res["manifest"]
        print(f"\n[OK] {m['n_samples']} samples x {m['n_features']} features "
              f"-> {m['n_significant_pairwise']} significant hits (FDR<{m['alpha']})")
        print(f"     outputs written to: {args.outdir}/")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
