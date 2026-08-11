"""
Cross-species validation of human scBACs and the published Buckley mouse clocks.

This script separates the cross-species/model-sensitivity analyses from the
APOE4/artemisinin intervention workflow.

Final-analysis scope
--------------------
1. Young-versus-aged mouse atlas:
   a. fixed adult human scBACs -> human scBAC-equivalent cellular age;
   b. fixed published Buckley mouse clocks -> mouse biological-age estimate.
2. Converse response-only sensitivity analysis:
   apply the available Buckley mouse clocks to the independent GSE254569 adult
   human validation cohort and report cell- and donor-level correlations.

Manuscript-result reproduction uses supplied precomputed prediction metadata
when available. Regeneration from raw h5ad files is optional and uses only fixed
pretrained/released clocks; no clock is trained here.

The released Buckley coefficient files used in this study did not contain
intercepts. Therefore mouse-clock predictions are not recalibrated and should
not be interpreted as calibrated human ages in the converse analysis.
"""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import spearmanr, pearsonr, ranksums

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ensemble_brain_age_pred as eba_pred
from mouse_clock_utils import predict_mouse_clock_celltypes


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(os.environ.get("SCBACS_PROJECT_ROOT", ".")).resolve()
DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results" / "mouse_cross_species_validation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Published Buckley clock coefficient tables.
MOUSE_CLOCK_DIR = MODEL_ROOT / "mouse_brain_clock"
BUCKLEY_MOUSE_CLOCK = MOUSE_CLOCK_DIR / "scMouseBrainAgeClock.csv"
BUCKLEY_MOUSE_CLOCK_HUMAN_GENEID = (
    MOUSE_CLOCK_DIR / "scMouseBrainAgeClock_humanGeneID.csv"
)

# Young/aged mouse atlas used for Extended Data Fig. 10a-b.
YOUNG_OLD_MOUSE_H5AD = DATA_DIR / "BrainAgingSpatialAtlas.h5ad"
YOUNG_OLD_ORTHOLOGUE = DATA_DIR / "BrainAgingSpatialAtlas_feature.csv"

# Historical precomputed human-scBAC table from the working analysis.
YOUNG_OLD_HUMAN_SCBAC_METADATA = DATA_DIR / "meta_young&old_mice.csv"
# Public precomputed Buckley prediction table; create it once with regeneration.
YOUNG_OLD_MOUSE_CLOCK_METADATA = DATA_DIR / "meta_young_old_mice_Buckley_clock.csv"

REGENERATE_YOUNG_OLD_HUMAN_SCBAC = False
REGENERATE_YOUNG_OLD_MOUSE_CLOCK = False

# Converse mouse-on-human response-only analysis (external validation dataset 2).
GSE254569_H5AD = DATA_DIR / "GSE254569_adata_RNA.h5ad"
GSE254569_MOUSE_CLOCK_METADATA = DATA_DIR / "GSE254569_mouse_model_age.csv"
REGENERATE_GSE254569_MOUSE_CLOCK = False

MOUSE_CLOCK_CELLTYPES = ["Ast", "Oli", "Mic"]
HUMAN_SCBAC_CELLTYPES = ["Exc", "Inh", "OPC", "Mic", "Ast", "Oli"]

YOUNG_STAGE = "4-week-old stage"
AGED_STAGE = "20-month-old stage and over"


# =============================================================================
# 2. SMALL HELPERS
# =============================================================================
def harmonize_mouse_atlas_celltypes(adata):
    """Map source atlas labels to the scBAC/public-analysis lineage labels."""
    out = adata.copy()
    if "cell_type" not in out.obs.columns and "celltype" in out.obs.columns:
        return out
    mapping = {
        "neuron": "Exc",
        "inhibitory interneuron": "Inh",
        "oligodendrocyte": "Oli",
        "astrocyte": "Ast",
        "microglial cell": "Mic",
        "oligodendrocyte precursor cell": "OPC",
        "endothelial cell": "End",
        "pericyte": "Per",
        "macrophage": "CAMs",
    }
    out.obs["celltype"] = out.obs["cell_type"].astype(str).replace(mapping)
    return out


def map_mouse_to_human_from_table(adata, mapping_file):
    """Map mouse features to human symbols using the released orthologue table."""
    mapping = pd.read_csv(mapping_file, index_col=0)
    required = {"mouseGene", "humanGene"}
    missing = required.difference(mapping.columns)
    if missing:
        raise KeyError(
            "Orthologue table is missing columns: {}".format(sorted(missing))
        )
    out = adata.copy()
    if "feature_name" in out.var.columns:
        out.var_names = out.var["feature_name"].astype(str).values
    mapping = mapping.dropna(subset=["mouseGene", "humanGene"]).copy()
    mapping["mouseGene"] = mapping["mouseGene"].astype(str)
    mapping["humanGene"] = mapping["humanGene"].astype(str)
    mapping = mapping.drop_duplicates("mouseGene", keep="first").set_index("mouseGene")
    common = out.var_names.astype(str).intersection(mapping.index)
    out = out[:, common].copy()
    human = mapping.loc[common, "humanGene"].to_numpy(str)
    keep = ~pd.Index(human).duplicated(keep="first")
    out = out[:, keep].copy()
    out.var_names = pd.Index(human[keep])
    return out


def two_group_cell_tests(df, value_col, group_col, group_a, group_b):
    """Two-sided Wilcoxon rank-sum tests within each cell type."""
    rows = []
    for ct, g in df.groupby("celltype", observed=True):
        x = pd.to_numeric(
            g.loc[g[group_col].astype(str) == str(group_a), value_col],
            errors="coerce",
        ).dropna()
        y = pd.to_numeric(
            g.loc[g[group_col].astype(str) == str(group_b), value_col],
            errors="coerce",
        ).dropna()
        if len(x) == 0 or len(y) == 0:
            continue
        z, p = ranksums(x, y)
        rows.append(
            {
                "celltype": ct,
                "group_a": group_a,
                "group_b": group_b,
                "median_a": x.median(),
                "median_b": y.median(),
                "median_difference": x.median() - y.median(),
                "Wilcoxon_z": z,
                "P_value": p,
                "N_cells_a": len(x),
                "N_cells_b": len(y),
            }
        )
    return pd.DataFrame(rows)


def correlation_summary(df):
    """Cell- and donor-level Pearson/Spearman correlations for the converse test."""
    rows = []
    for ct, g in df.groupby("celltype", observed=True):
        cell = g[["Pred_age", "Age_at_death"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        donor = (
            g.groupby("donor_id", observed=True)
            .agg(Pred_age=("Pred_age", "median"), Age_at_death=("Age_at_death", "median"))
            .reset_index()
        )
        donor[["Pred_age", "Age_at_death"]] = donor[
            ["Pred_age", "Age_at_death"]
        ].apply(pd.to_numeric, errors="coerce")
        donor = donor.dropna()
        for level, d in [("cell", cell), ("donor_median", donor)]:
            if len(d) < 3:
                continue
            pearson, pearson_p = pearsonr(d["Age_at_death"], d["Pred_age"])
            spearman, spearman_p = spearmanr(d["Age_at_death"], d["Pred_age"])
            rows.append(
                {
                    "celltype": ct,
                    "level": level,
                    "N": len(d),
                    "Pearson_r": pearson,
                    "Pearson_P": pearson_p,
                    "Spearman_rho": spearman,
                    "Spearman_P": spearman_p,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 3. YOUNG-VERSUS-AGED MOUSE: FIXED HUMAN scBAC
# =============================================================================
human_mouse = pd.DataFrame()
if REGENERATE_YOUNG_OLD_HUMAN_SCBAC:
    if not YOUNG_OLD_MOUSE_H5AD.exists():
        raise FileNotFoundError(YOUNG_OLD_MOUSE_H5AD)
    if not YOUNG_OLD_ORTHOLOGUE.exists():
        raise FileNotFoundError(YOUNG_OLD_ORTHOLOGUE)

    adata = harmonize_mouse_atlas_celltypes(sc.read_h5ad(YOUNG_OLD_MOUSE_H5AD))
    mapped = map_mouse_to_human_from_table(adata, YOUNG_OLD_ORTHOLOGUE)

    predictor = eba_pred.UnifiedAgePredictor(
        DEVICE,
        max_workers=4,
        stages=["Adult"],
        model_types=["transf", "elasticnet", "clm"],
        model_root=str(MODEL_ROOT),
        strict_five_folds=True,
    )
    human_mouse = eba_pred.predict_all_celltypes(
        adata_obj=mapped,
        celltypes=HUMAN_SCBAC_CELLTYPES,
        predictor=predictor,
        norm=False,
        chunk_size=50000,
        include_benchmarking=False,
    )
    human_mouse["human_scBAC_equivalent_age"] = human_mouse["Ensemble_Adult"]
    for col in ["development_stage", "celltype"]:
        if col not in human_mouse.columns and col in mapped.obs.columns:
            human_mouse[col] = mapped.obs.loc[human_mouse.index, col].values
    human_mouse.to_csv(
        RESULT_DIR / "meta_young_old_mice_human_scBAC_regenerated.csv"
    )
elif YOUNG_OLD_HUMAN_SCBAC_METADATA.exists():
    human_mouse = pd.read_csv(YOUNG_OLD_HUMAN_SCBAC_METADATA, index_col=0)
    if "human_scBAC_equivalent_age" not in human_mouse.columns:
        if "Ensemble_Adult" in human_mouse.columns:
            human_mouse["human_scBAC_equivalent_age"] = pd.to_numeric(
                human_mouse["Ensemble_Adult"], errors="coerce"
            )

if not human_mouse.empty:
    tests = two_group_cell_tests(
        human_mouse,
        value_col="human_scBAC_equivalent_age",
        group_col="development_stage",
        group_a=AGED_STAGE,
        group_b=YOUNG_STAGE,
    )
    tests.to_csv(
        RESULT_DIR / "young_vs_aged_human_scBAC_equivalent_age_tests.csv",
        index=False,
    )


# =============================================================================
# 4. YOUNG-VERSUS-AGED MOUSE: FIXED BUCKLEY MOUSE CLOCK
# =============================================================================
buckley_mouse = pd.DataFrame()
if REGENERATE_YOUNG_OLD_MOUSE_CLOCK:
    if not YOUNG_OLD_MOUSE_H5AD.exists():
        raise FileNotFoundError(YOUNG_OLD_MOUSE_H5AD)
    adata = harmonize_mouse_atlas_celltypes(sc.read_h5ad(YOUNG_OLD_MOUSE_H5AD))
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).values
    buckley_mouse = predict_mouse_clock_celltypes(
        adata,
        param_file=BUCKLEY_MOUSE_CLOCK,
        cell_types=MOUSE_CLOCK_CELLTYPES,
        celltype_col="celltype",
        norm=False,
        extra_obs_cols=["development_stage"],
    )
    buckley_mouse.to_csv(
        RESULT_DIR / "meta_young_old_mice_Buckley_clock_regenerated.csv"
    )
elif YOUNG_OLD_MOUSE_CLOCK_METADATA.exists():
    buckley_mouse = pd.read_csv(
        YOUNG_OLD_MOUSE_CLOCK_METADATA,
        index_col=0,
    )

if not buckley_mouse.empty:
    tests = two_group_cell_tests(
        buckley_mouse,
        value_col="Pred_age",
        group_col="development_stage",
        group_a=AGED_STAGE,
        group_b=YOUNG_STAGE,
    )
    tests.to_csv(
        RESULT_DIR / "young_vs_aged_Buckley_mouse_clock_tests.csv",
        index=False,
    )


# =============================================================================
# 5. CONVERSE TEST: BUCKLEY MOUSE CLOCK ON HUMAN GSE254569
# =============================================================================
human_transfer = pd.DataFrame()
if REGENERATE_GSE254569_MOUSE_CLOCK:
    if not GSE254569_H5AD.exists():
        raise FileNotFoundError(GSE254569_H5AD)

    gse = sc.read_h5ad(GSE254569_H5AD)
    if "counts" in gse.layers:
        gse.X = gse.layers["counts"].copy()

    if "major_celltypes" not in gse.obs.columns:
        raise KeyError("GSE254569 requires obs['major_celltypes']")
    gse.obs["celltype"] = gse.obs["major_celltypes"].replace(
        {
            "Oligodendrocyte": "Oli",
            "In_Neurons": "aNSC_NPC",
            "Exc_Neurons": "aNSC_NPC",
            "Astrocytes": "Ast",
            "OPC": "OPC",
            "Microglia": "Mic",
            "Endothelial": "End",
        }
    )
    gse.obs["status"] = gse.obs["Disease_Status"].map({0: "CT"}).fillna("Disease")
    gse.obs["Age_at_death"] = pd.to_numeric(gse.obs["Age"], errors="coerce")
    gse.obs["donor_id"] = gse.obs["Donor"].astype(str)
    controls = gse[gse.obs["status"] == "CT"].copy()

    available = [
        ct for ct in ["Oli", "aNSC_NPC", "Ast", "Mic"]
        if ct in pd.read_csv(BUCKLEY_MOUSE_CLOCK_HUMAN_GENEID, nrows=1).columns
    ]
    human_transfer = predict_mouse_clock_celltypes(
        controls,
        param_file=BUCKLEY_MOUSE_CLOCK_HUMAN_GENEID,
        cell_types=available,
        celltype_col="celltype",
        norm=True,
        extra_obs_cols=["Age_at_death", "donor_id"],
    )
    human_transfer.to_csv(
        RESULT_DIR / "GSE254569_Buckley_mouse_clock_regenerated.csv"
    )
elif GSE254569_MOUSE_CLOCK_METADATA.exists():
    human_transfer = pd.read_csv(
        GSE254569_MOUSE_CLOCK_METADATA,
        index_col=0,
    )

if not human_transfer.empty:
    correlation_summary(human_transfer).to_csv(
        RESULT_DIR / "GSE254569_Buckley_mouse_clock_correlations.csv",
        index=False,
    )


print("Mouse cross-species validation complete.")
