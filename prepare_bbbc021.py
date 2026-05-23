"""
examples/prepare_bbbc021.py
===========================
Assemble a REAL demo dataset for MorphoStat from the BBBC021 MCF-7 screen
(Ljosa et al. 2013, J Biomol Screen), as bundled by the Broad Institute
`cytominergallery` R package.

This is genuine experimental data: MCF-7 breast-cancer cells treated with a
compendium of mechanistically distinct compounds, stained for DNA / Actin
(phalloidin) / beta-Tubulin, segmented and quantified per cell with CellProfiler,
then averaged per well. We join the per-well feature table to the image
metadata (compound, concentration) and the mechanism-of-action key, then write
ONE CSV PER PLATE into a folder so the pipeline's multi-file ingestion is
exercised exactly as it would be on real per-replicate FilamentSensor exports.

Usage:
    python examples/prepare_bbbc021.py <path_to_cytominergallery/inst/extdata> <out_folder>
"""
import os
import sys
import pandas as pd


def main(extdata_dir: str, out_dir: str):
    feat = pd.read_csv(os.path.join(extdata_dir, "ljosa_jbiomolscreen_2013_per_well_mean.csv.gz"))
    img = pd.read_csv(os.path.join(extdata_dir, "BBBC021_v1_image.csv.gz"))
    moa = pd.read_csv(os.path.join(extdata_dir, "BBBC021_v1_moa.csv"))

    key = (img[["Image_Metadata_Plate_DAPI", "Image_Metadata_Well_DAPI",
                "Image_Metadata_Compound", "Image_Metadata_Concentration"]]
           .drop_duplicates())
    key.columns = ["Plate", "Well", "compound", "concentration"]

    df = (feat.rename(columns={"Image_Metadata_Plate": "Plate",
                               "Image_Metadata_Well": "Well"})
              .merge(key, on=["Plate", "Well"], how="left")
              .merge(moa, on=["compound", "concentration"], how="left"))
    # DMSO is the negative control
    df.loc[df["compound"].str.upper() == "DMSO", "moa"] = "DMSO"
    df = df[df["moa"].notna()].copy()

    # move metadata columns to the front for readability
    meta = ["Plate", "Well", "compound", "concentration", "moa"]
    df = df[meta + [c for c in df.columns if c not in meta]]

    os.makedirs(out_dir, exist_ok=True)
    for plate, g in df.groupby("Plate"):
        g.to_csv(os.path.join(out_dir, f"{plate}.csv"), index=False)
    print(f"Wrote {df['Plate'].nunique()} per-plate CSVs ({len(df)} wells, "
          f"{df.shape[1]} columns) to {out_dir}")
    print("MOA groups:\n", df["moa"].value_counts().to_string())


if __name__ == "__main__":
    extdata = sys.argv[1] if len(sys.argv) > 1 else "cytominergallery/inst/extdata"
    out = sys.argv[2] if len(sys.argv) > 2 else "demo_input_bbbc021"
    main(extdata, out)
