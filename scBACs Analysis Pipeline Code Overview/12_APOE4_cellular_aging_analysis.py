"""
Sex-dependent APOE4 effects on scBAC-predicted cellular age.

Final-analysis scope:
- ROSMAP PFC discovery and SEAAD+GSE263468+GSE237718+GSE157827 replication
- cell-level and donor-median APOE4 x sex x AD interaction models
- male- and female-stratified APOE4 x AD models
- adjusted AD-vs-non-AD differences within four sex/APOE4 strata
- the same framework across Inh subtypes when subtype metadata are available
- reproducible APOE4-dependent cellular-age genes and the subset jointly aligned
  with AD-associated expression changes

Exploratory compensatory gene intersections and alternative regression models from the
working script are intentionally omitted.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.stats import spearmanr

from downstream_analysis_utils import (
    MAIN_CELLTYPES, add_scbac_adult, make_donor_uid, donor_median_summary,
    encode_apoe4, run_apoe_interaction_models, run_status_within_sex_apoe_strata,
    normalize_log1p, bh_fdr,
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path('./')
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results_" / "APOE4_cellular_aging"
FIGURE_DIR = PROJECT_ROOT / "figures_" / "APOE4_cellular_aging"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_METADATA = DATA_DIR / "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
SEAAD_METADATA = DATA_DIR / "meta_SEAAD_PFC_replication.csv"
GSE263468_METADATA = DATA_DIR / "meta_GSE263468_PFC_AD_replication.csv"
APOE_TEMPORAL_METADATA = DATA_DIR / "meta_APOE4_Temporal_cortex_replication.csv"  # expected to contain GSE237718/GSE157827

ROSMAP_INH_METADATA = DATA_DIR / "meta_ROSMAC_PFC_Inh_Subclass_discovery.csv"
SEAAD_INH_METADATA = DATA_DIR / "meta_SEAAD_PFC_Inh_Subclass_replication.csv"
GSE263468_INH_METADATA = DATA_DIR / "meta_GSE263468_Inh_Subclass_replication.csv"
APOE_INH_METADATA = DATA_DIR / "meta_APOE4_Temporal_cortex_Inh_Subclass_replication.csv"
SUBTYPE_COL = "sub_celltype_celltypist_raw"

APOE_GENOTYPE_CANDIDATES = ["APOE Genotype", "APOE_genotype", "APOE", "apoe_genotype"]
RUN_GENE_ANALYSIS = False
ROSMAP_EXPRESSION = DATA_DIR / "sce_test.h5ad"
REPLICATION_EXPRESSION = DATA_DIR / "sce_APOE4_aging_replication.h5ad"

# =============================================================================
# 2. HARMONIZATION
# =============================================================================
def find_genotype_col(df):
    if "apoe4" in df.columns:
        return None
    for col in APOE_GENOTYPE_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError("No APOE genotype/carrier column found.")


def prepare_apoe(df, resource):
    d = df.copy()
    d = d[d["celltype"].isin(MAIN_CELLTYPES)]
    d = d[d["status"].isin(["CT", "MCI", "AD"])]
    d = add_scbac_adult(d)
    d = make_donor_uid(d)
    if "apoe4" not in d.columns:
        genotype = find_genotype_col(d)
        d["apoe4"] = encode_apoe4(d[genotype])
    else:
        d["apoe4"] = pd.to_numeric(d["apoe4"], errors="coerce")
    d["resource"] = resource
    return d.dropna(subset=["apoe4", "Sex", "Age_at_death", "status"])

meta = pd.read_csv(MAIN_METADATA, index_col=0)
dis = meta[(meta["dataset"].astype(str) == "ROSMAP.MIT") & (meta["sub_tissue"].astype(str) == "Prefrontal cortex")].copy()
dis = prepare_apoe(dis, "Discovery")

rep_parts = []
for path in [SEAAD_METADATA, GSE263468_METADATA, APOE_TEMPORAL_METADATA]:
    if path.exists():
        rep_parts.append(pd.read_csv(path, index_col=0))

        
rep = prepare_apoe(pd.concat(rep_parts, axis=0), "Replication") if rep_parts else pd.DataFrame()
rep = rep[rep['sub_tissue'].isin(['Prefrontal cortex','Temporal cortex'])].copy()
# =============================================================================
# 3. CELL-LEVEL APOE4 x SEX x AD MODELS
# =============================================================================
for name, cells in [("discovery", dis), ("replication", rep)]:
    if cells.empty:
        continue
    for sex_stratum in [None, "Female", "Male"]:
        res = run_apoe_interaction_models(
            cells, outcome_col="scBAC_adult", sex_stratum=sex_stratum,
            min_n=10,
        )
        suffix = "combined" if sex_stratum is None else sex_stratum.lower()
        res.to_csv(RESULT_DIR / f"APOE4_celllevel_{name}_{suffix}.csv", index=False)

# =============================================================================
# 4. DONOR-LEVEL COMBINED AND STRATIFIED APOE4/AD/SEX ANALYSES
# =============================================================================
def add_analysis_donor_id(df):
    d = df.copy()
    dataset = d["dataset"].astype(str) if "dataset" in d.columns else pd.Series("dataset", index=d.index)
    donor = d["donor_id"].astype(str) if "donor_id" in d.columns else d["donor_uid"].astype(str)
    tissue = d["sub_tissue"].astype(str) if "sub_tissue" in d.columns else pd.Series("Cortex", index=d.index)
    d["donor_uid_analysis"] = dataset + "::" + donor + "::" + tissue
    return d

def encode_sex_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.astype(str).str.lower().map({"female": 0.0, "male": 1.0, "f": 0.0, "m": 1.0})

def encode_status_binary(series):
    return series.astype(str).map({"CT": 0.0, "MCI": 0.0, "AD": 1.0, "non-AD": 0.0, "Non-AD": 0.0})

def add_fdr_by_stratum(df, strata):
    out = df.copy()
    if out.empty:
        return out
    out["FDR"] = out.groupby(strata, observed=True)["P_value"].transform(bh_fdr)
    return out

def run_ad_effect_within_sex_apoe(donor_df, outcome_col="scBAC_adult", min_donors=5):
    work = donor_df.copy()
    work["Sex_numeric"] = encode_sex_numeric(work["Sex"])
    work["APOE4_numeric"] = pd.to_numeric(work["apoe4"], errors="coerce")
    work["AD_status"] = encode_status_binary(work["status"])
    work["PMI_numeric"] = pd.to_numeric(work["PMI"], errors="coerce")
    rows = []
    for sex_code, sex_name in [(0.0, "Female"), (1.0, "Male")]:
        for apoe_code, apoe_name in [(0.0, "Non-carrier"), (1.0, "Carrier")]:
            strata = work[(work["Sex_numeric"] == sex_code) & (work["APOE4_numeric"] == apoe_code)].copy()
            for ct, g in strata.groupby("celltype", observed=True):
                g = g.dropna(subset=[outcome_col, "AD_status", "Age_at_death", "PMI_numeric"]).copy()
                if len(g) < min_donors or g["AD_status"].nunique() < 2:
                    continue
                g["Age_c"] = pd.to_numeric(g["Age_at_death"], errors="coerce") - pd.to_numeric(g["Age_at_death"], errors="coerce").mean()
                g["Age_c2"] = g["Age_c"] ** 2
                g = g.dropna(subset=["Age_c", "Age_c2"])
                if len(g) < min_donors:
                    continue
                fit = smf.ols(f"{outcome_col} ~ AD_status + Age_c + Age_c2 + PMI_numeric", data=g).fit()
                ci = fit.conf_int().loc["AD_status"]
                rows.append({"CellType": ct, "Sex": sex_name, "Sex_numeric": sex_code, "APOE4": apoe_name, "apoe4": apoe_code, "Effect": "AD_vs_nonAD", "Effect_Size": fit.params["AD_status"], "Std_Error": fit.bse["AD_status"], "CI95_Low": ci.iloc[0], "CI95_High": ci.iloc[1], "P_value": fit.pvalues["AD_status"], "N_Donors": len(g), "N_AD": int((g["AD_status"] == 1).sum()), "N_nonAD": int((g["AD_status"] == 0).sum())})
    return add_fdr_by_stratum(pd.DataFrame(rows), ["Sex", "APOE4"])

def run_apoe_effect_within_sex_disease(donor_df, outcome_col="scBAC_adult", min_donors=5):
    work = donor_df.copy()
    work["Sex_numeric"] = encode_sex_numeric(work["Sex"])
    work["APOE4_numeric"] = pd.to_numeric(work["apoe4"], errors="coerce")
    work["AD_status"] = encode_status_binary(work["status"])
    work["PMI_numeric"] = pd.to_numeric(work["PMI"], errors="coerce")
    rows = []
    for sex_code, sex_name in [(0.0, "Female"), (1.0, "Male")]:
        for disease_code, disease_name in [(0.0, "non-AD"), (1.0, "AD")]:
            strata = work[(work["Sex_numeric"] == sex_code) & (work["AD_status"] == disease_code)].copy()
            for ct, g in strata.groupby("celltype", observed=True):
                g = g.dropna(subset=[outcome_col, "APOE4_numeric", "Age_at_death", "PMI_numeric"]).copy()
                if len(g) < min_donors or g["APOE4_numeric"].nunique() < 2:
                    continue
                g["Age_c"] = pd.to_numeric(g["Age_at_death"], errors="coerce") - pd.to_numeric(g["Age_at_death"], errors="coerce").mean()
                g["Age_c2"] = g["Age_c"] ** 2
                g = g.dropna(subset=["Age_c", "Age_c2"])
                if len(g) < min_donors:
                    continue
                fit = smf.ols(f"{outcome_col} ~ APOE4_numeric + Age_c + Age_c2 + PMI_numeric", data=g).fit()
                ci = fit.conf_int().loc["APOE4_numeric"]
                rows.append({"CellType": ct, "Sex": sex_name, "Sex_numeric": sex_code, "Disease": disease_name, "AD_status": disease_code, "Effect": "APOE4_carrier_vs_noncarrier", "Effect_Size": fit.params["APOE4_numeric"], "Std_Error": fit.bse["APOE4_numeric"], "CI95_Low": ci.iloc[0], "CI95_High": ci.iloc[1], "P_value": fit.pvalues["APOE4_numeric"], "N_Donors": len(g), "N_Carrier": int((g["APOE4_numeric"] == 1).sum()), "N_NonCarrier": int((g["APOE4_numeric"] == 0).sum())})
    return add_fdr_by_stratum(pd.DataFrame(rows), ["Sex", "Disease"])

def donor_group_summary(donor_df, outcome_col="scBAC_adult"):
    d = donor_df.copy()
    d["Sex_label"] = encode_sex_numeric(d["Sex"]).map({0.0: "Female", 1.0: "Male"})
    d["APOE4_label"] = pd.to_numeric(d["apoe4"], errors="coerce").map({0.0: "Non-carrier", 1.0: "Carrier"})
    d["Disease_label"] = encode_status_binary(d["status"]).map({0.0: "non-AD", 1.0: "AD"})
    return d.groupby(["celltype", "Sex_label", "APOE4_label", "Disease_label"], observed=True)[outcome_col].agg(["count", "mean", "median", "std", "sem"]).reset_index()

def plot_donor_interaction(raw_df, stats_df, output_file, outcome_col="scBAC_adult"):
    d = raw_df.copy()
    d["Sex_numeric"] = encode_sex_numeric(d["Sex"])
    d["APOE4_numeric"] = pd.to_numeric(d["apoe4"], errors="coerce")
    d["AD_status"] = encode_status_binary(d["status"])
    celltypes = [ct for ct in ["Ast", "Exc", "Inh", "Oli", "OPC", "Mic"] if ct in d["celltype"].astype(str).unique()]
    fig, axes = plt.subplots(len(celltypes), 2, figsize=(9, max(4, 3.2 * len(celltypes))), squeeze=False)
    colors = {0.0: "#2166ac", 1.0: "#b2182b"}
    markers = {0.0: "o", 1.0: "s"}
    linestyles = {0.0: "-", 1.0: "--"}
    for row, ct in enumerate(celltypes):
        row_data = d[d["celltype"].astype(str) == ct].copy()
        row_summary = row_data.groupby(["Sex_numeric", "APOE4_numeric", "AD_status"], observed=True)[outcome_col].agg(["mean", "sem"]).reset_index()
        y_values = np.r_[row_summary["mean"] - row_summary["sem"], row_summary["mean"] + row_summary["sem"]]
        y_values = y_values[np.isfinite(y_values)]
        if len(y_values):
            ymin, ymax = y_values.min(), y_values.max()
            margin = max((ymax - ymin) * 0.35, 1.0)
            ylim = (ymin - 0.15 * margin, ymax + margin)
        else:
            ylim = None
        for col, (sex_code, sex_name) in enumerate([(1.0, "Male"), (0.0, "Female")]):
            ax = axes[row, col]
            sub = row_data[row_data["Sex_numeric"] == sex_code].copy()
            if sub.empty:
                ax.axis("off")
                continue
            for apoe_code in [0.0, 1.0]:
                g = sub[sub["APOE4_numeric"] == apoe_code]
                summary = g.groupby("AD_status", observed=True)[outcome_col].agg(["mean", "sem"])
                xs, ys, errs = [], [], []
                for status_code in [0.0, 1.0]:
                    if status_code in summary.index:
                        xs.append(status_code); ys.append(summary.loc[status_code, "mean"]); errs.append(summary.loc[status_code, "sem"])
                if xs:
                    ax.errorbar(xs, ys, yerr=errs, marker=markers[apoe_code], linestyle=linestyles[apoe_code], color=colors[apoe_code], linewidth=1.4, markersize=5, capsize=3)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xticks([0, 1], ["non-AD", "AD"])
            ax.set_title(f"{ct} | {sex_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(f"{outcome_col} (years)" if col == 0 else "")
            legend = []
            for apoe_code, apoe_label in [(0.0, "APOE4-"), (1.0, "APOE4+")]:
                n_nonad = len(sub[(sub["AD_status"] == 0) & (sub["APOE4_numeric"] == apoe_code)])
                n_ad = len(sub[(sub["AD_status"] == 1) & (sub["APOE4_numeric"] == apoe_code)])
                stat = stats_df[(stats_df["CellType"] == ct) & (stats_df["Sex_numeric"] == sex_code) & (stats_df["apoe4"] == apoe_code)]
                if not stat.empty:
                    r = stat.iloc[0]
                    ptxt = f"{r['P_value']:.2e}" if r["P_value"] < 0.001 else f"{r['P_value']:.3f}"
                    label = f"{apoe_label}: β={r['Effect_Size']:+.2f}y, P={ptxt}, AD/non-AD={n_ad}/{n_nonad}"
                else:
                    label = f"{apoe_label}: AD/non-AD={n_ad}/{n_nonad}"
                legend.append(Line2D([0], [0], color=colors[apoe_code], marker=markers[apoe_code], linestyle=linestyles[apoe_code], label=label, markersize=5))
            ax.legend(handles=legend, frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.3)
            sns.despine(ax=ax)
    plt.subplots_adjust(hspace=0.45, wspace=0.2)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_stratified_forest(stats_df, stratum_col, output_file, title_prefix):
    if stats_df.empty:
        return
    cell_order = [ct for ct in ["Ast", "Exc", "Inh", "Oli", "OPC", "Mic"] if ct in stats_df["CellType"].astype(str).unique()]
    strata = [(sex, stratum) for sex in ["Female", "Male"] for stratum in stats_df.loc[stats_df["Sex"] == sex, stratum_col].dropna().unique()]
    if not strata:
        return
    fig, axes = plt.subplots(len(strata), 1, figsize=(7.5, max(3.0, 2.6 * len(strata))), squeeze=False)
    for idx, (sex, stratum) in enumerate(strata):
        ax = axes[idx, 0]
        g = stats_df[(stats_df["Sex"] == sex) & (stats_df[stratum_col] == stratum)].copy()
        g["CellType"] = pd.Categorical(g["CellType"], categories=cell_order, ordered=True)
        g = g.sort_values("CellType")
        x = np.arange(len(g))
        lower = g["Effect_Size"].to_numpy(float) - g["CI95_Low"].to_numpy(float)
        upper = g["CI95_High"].to_numpy(float) - g["Effect_Size"].to_numpy(float)
        ax.errorbar(x, g["Effect_Size"], yerr=np.vstack([lower, upper]), fmt="o", capsize=3, linewidth=1.2)
        ax.axhline(0, linewidth=0.8, linestyle="--", color="0.4")
        ax.set_xticks(x, g["CellType"].astype(str), rotation=0)
        ax.set_ylabel("Effect size (years)")
        ax.set_title(f"{title_prefix}: {sex}, {stratum}", fontsize=10, fontweight="bold")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_donor_counts(summary_df, output_file):
    d = summary_df.copy()
    d["Stratum"] = d["Sex_label"].astype(str) + " | " + d["APOE4_label"].astype(str) + " | " + d["Disease_label"].astype(str)
    pivot = d.pivot_table(index="celltype", columns="Stratum", values="count", fill_value=0, observed=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.85 * pivot.shape[1]), 4.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Number of donors"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Cell type")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)

dis_for_donor = add_analysis_donor_id(dis)
rep_for_donor = add_analysis_donor_id(rep) if not rep.empty else pd.DataFrame()
metadata_cols = ["status", "Sex", "Age_at_death", "PMI", "dataset", "sub_tissue", "apoe4", "resource"]
dis_donor = donor_median_summary(dis_for_donor, ["scBAC_adult"], donor_col="donor_uid_analysis", metadata_cols=metadata_cols)
rep_donor = donor_median_summary(rep_for_donor, ["scBAC_adult"], donor_col="donor_uid_analysis", metadata_cols=metadata_cols) if not rep_for_donor.empty else pd.DataFrame()
dis_donor["analysis_resource"] = "Discovery"
if not rep_donor.empty:
    rep_donor["analysis_resource"] = "Replication"

    
combined_donor = pd.concat([dis_donor, rep_donor], axis=0, ignore_index=True) if not rep_donor.empty else dis_donor.copy()
combined_donor.to_csv(RESULT_DIR / "APOE4_donorlevel_discovery_replication_combined.csv", index=False)

# Run the original three-way framework on each cohort and on discovery+replication.
for name, donor in [("discovery", dis_donor), ("replication", rep_donor), ("combined_dis_rep", combined_donor)]:
    if donor.empty:
        continue
    for sex_stratum in [None, "Female", "Male"]:
        res = run_apoe_interaction_models(donor, outcome_col="scBAC_adult", sex_stratum=sex_stratum, min_n=5)
        suffix = "combined" if sex_stratum is None else sex_stratum.lower()
        res.to_csv(RESULT_DIR / f"APOE4_donorlevel_{name}_{suffix}.csv", index=False)

# Disease effect within the four Sex x APOE4 strata.
ad_strata = run_ad_effect_within_sex_apoe(combined_donor, outcome_col="scBAC_adult", min_donors=5)
ad_strata.to_csv(RESULT_DIR / "AD_effect_within_sex_APOE4_strata_combined_dis_rep.csv", index=False)

# APOE4 effect within Sex x disease-state strata.
apoe_strata = run_apoe_effect_within_sex_disease(combined_donor, outcome_col="scBAC_adult", min_donors=5)
apoe_strata.to_csv(RESULT_DIR / "APOE4_effect_within_sex_disease_strata_combined_dis_rep.csv", index=False)

# Donor counts and descriptive summaries for all 2 x 2 x 2 strata.
group_summary = donor_group_summary(combined_donor, outcome_col="scBAC_adult")
group_summary.to_csv(RESULT_DIR / "donor_summary_by_sex_APOE4_disease_celltype.csv", index=False)

# Visualization: interaction trajectories and two complementary forest plots.
plot_donor_interaction(combined_donor, ad_strata, FIGURE_DIR / "donorlevel_sex_APOE4_AD_interaction_combined_dis_rep.pdf", outcome_col="scBAC_adult")
plot_stratified_forest(ad_strata, "APOE4", FIGURE_DIR / "donorlevel_AD_effect_within_sex_APOE4_strata.pdf", "AD vs non-AD")
plot_stratified_forest(apoe_strata, "Disease", FIGURE_DIR / "donorlevel_APOE4_effect_within_sex_disease_strata.pdf", "APOE4 carrier vs non-carrier")
plot_donor_counts(group_summary, FIGURE_DIR / "donorlevel_counts_by_sex_APOE4_disease.pdf")

# =============================================================================
# 5. INH SUBTYPE APOE4 ANALYSIS
# =============================================================================
def load_subtype(paths, resource):
    parts = []
    for path in paths:
        if Path(path).exists():
            parts.append(pd.read_csv(path, index_col=0))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, axis=0)
    d = add_scbac_adult(d); d = make_donor_uid(d)
    if "apoe4" not in d.columns:
        try:
            col = find_genotype_col(d); d["apoe4"] = encode_apoe4(d[col])
        except KeyError:
            return pd.DataFrame()
    if SUBTYPE_COL not in d.columns:
        for c in ["sub_celltype2", "sub_celltype_celltypist", "sub_celltype_celltypist_coarse"]:
            if c in d.columns:
                d[SUBTYPE_COL] = d[c]; break
    d["resource"] = resource
    return d

sub_dis = load_subtype([ROSMAP_INH_METADATA], "Discovery")
sub_rep = load_subtype([SEAAD_INH_METADATA, GSE263468_INH_METADATA, APOE_INH_METADATA], "Replication")

for name, cells in [("discovery", sub_dis), ("replication", sub_rep)]:
    if cells.empty or SUBTYPE_COL not in cells.columns:
        continue
    tmp = cells.rename(columns={SUBTYPE_COL: "analysis_subtype"}).copy()
    donor = donor_median_summary(
        tmp, ["scBAC_adult"], celltype_col="analysis_subtype",
        metadata_cols=["status", "Sex", "Age_at_death", "PMI", "dataset", "apoe4", "resource"]
    )
    for sex_stratum in [None, "Female", "Male"]:
        res = run_apoe_interaction_models(
            donor, outcome_col="scBAC_adult", celltype_col="analysis_subtype",
            sex_stratum=sex_stratum, min_n=5,
        )
        suffix = "combined" if sex_stratum is None else sex_stratum.lower()
        res.rename(columns={"CellType": "Inh_Subtype"}).to_csv(
            RESULT_DIR / f"APOE4_Inh_subtype_{name}_{suffix}.csv", index=False
        )

# =============================================================================
# 6. REPRODUCIBLE APOE4-DEPENDENT CELLULAR-AGE GENES
# =============================================================================
def attach_metadata(adata, metadata):
    ids = adata.obs_names.intersection(metadata.index.astype(str))
    a = adata[ids].copy()
    m = metadata.loc[ids]
    for col in m.columns:
        a.obs[col] = m[col].values
    return a


def donor_gene_age_analysis(adata, sex_name, resource_name):
    """DE APOE4 carrier/non-carrier + donor pseudobulk Spearman with donor-median age."""
    sex_code = 1 if sex_name == "Male" else 0
    sex_series = adata.obs["Sex"].astype(str).str.lower().map({"male": 1, "female": 0})
    a = adata[sex_series == sex_code].copy()
    a = a[a.obs["status"].isin(["CT", "MCI", "AD"])].copy()
    a = normalize_log1p(a)
    results = []
    for ct in MAIN_CELLTYPES:
        sub = a[a.obs["celltype"] == ct].copy()
        if sub.n_obs == 0 or sub.obs["apoe4"].nunique() < 2:
            continue
        sub.obs["APOE4_group"] = sub.obs["apoe4"].map({0.0: "Non-carrier", 1.0: "Carrier"})
        sc.tl.rank_genes_groups(sub, "APOE4_group", groups=["Carrier"], reference="Non-carrier", method="wilcoxon", pts=True)
        deg = sc.get.rank_genes_groups_df(sub, group="Carrier")
        deg = deg.rename(columns={"names": "gene", "logfoldchanges": "APOE4_log2FC", "pvals_adj": "APOE4_FDR", "pvals": "APOE4_P"})

        donor_rows = []
        expression_rows = []
        for donor, idx in sub.obs.groupby("donor_uid", observed=True).indices.items():
            idx = np.asarray(idx, dtype=int)
            block = sub.X[idx]
            mean_expr = np.asarray(block.mean(axis=0)).reshape(-1) if sparse.issparse(block) else np.asarray(block).mean(axis=0)
            expression_rows.append(mean_expr)
            donor_rows.append({"donor_uid": donor, "cell_age": sub.obs.iloc[idx]["scBAC_adult"].median()})
        expr = np.vstack(expression_rows)
        ages = np.array([x["cell_age"] for x in donor_rows], dtype=float)
        corr_rows = []
        for j, gene in enumerate(sub.var_names.astype(str)):
            x = expr[:, j]
            valid = np.isfinite(x) & np.isfinite(ages)
            if valid.sum() < 5 or np.unique(x[valid]).size < 2:
                continue
            rho, p = spearmanr(x[valid], ages[valid])
            corr_rows.append({"gene": gene, "Age_rho": rho, "Age_P": p})
        corr = pd.DataFrame(corr_rows)
        if not corr.empty:
            corr["Age_FDR"] = bh_fdr(corr["Age_P"])
        merged = deg.merge(corr, on="gene", how="inner")
        merged["celltype"] = ct; merged["Sex"] = sex_name; merged["resource"] = resource_name
        results.append(merged)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def ad_deg_by_sex_celltype(adata, sex_name, resource_name):
    """AD-vs-non-AD DE within sex and cell type for the joint APOE4/AD/aging signature."""
    sex_code = 1 if sex_name == "Male" else 0
    sex_series = adata.obs["Sex"].astype(str).str.lower().map({"male": 1, "female": 0})
    a = adata[sex_series == sex_code].copy()
    a = a[a.obs["status"].isin(["CT", "MCI", "AD"])].copy()
    a.obs["AD_group"] = np.where(a.obs["status"].astype(str) == "AD", "AD", "non-AD")
    a = normalize_log1p(a)
    rows = []
    for ct in MAIN_CELLTYPES:
        sub = a[a.obs["celltype"] == ct].copy()
        if sub.n_obs == 0 or sub.obs["AD_group"].nunique() < 2:
            continue
        sc.tl.rank_genes_groups(sub, "AD_group", groups=["AD"], reference="non-AD", method="wilcoxon", pts=True)
        deg = sc.get.rank_genes_groups_df(sub, group="AD").rename(
            columns={"names": "gene", "logfoldchanges": "AD_log2FC", "pvals": "AD_P", "pvals_adj": "AD_FDR"}
        )
        deg["celltype"] = ct; deg["Sex"] = sex_name; deg["resource"] = resource_name
        rows.append(deg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def integrate_discovery_replication(dis_res, rep_res):
    keys = ["gene", "celltype", "Sex"]
    m = dis_res.merge(rep_res, on=keys, suffixes=("_dis", "_rep"))
    m["APOE4_direction_concordant"] = np.sign(m["APOE4_log2FC_dis"]) == np.sign(m["APOE4_log2FC_rep"])
    m["Age_direction_concordant"] = np.sign(m["Age_rho_dis"]) == np.sign(m["Age_rho_rep"])
    m["reproducible"] = (
        (m["APOE4_FDR_dis"] < 0.05) & (m["APOE4_P_rep"] < 0.05) &
        (m["Age_FDR_dis"] < 0.05) & (m["Age_P_rep"] < 0.05) &
        m["APOE4_direction_concordant"] & m["Age_direction_concordant"]
    )
    m["aging_consistent"] = (
        ((m["APOE4_log2FC_dis"] > 0) & (m["Age_rho_dis"] > 0)) |
        ((m["APOE4_log2FC_dis"] < 0) & (m["Age_rho_dis"] < 0))
    )
    return m

if RUN_GENE_ANALYSIS and ROSMAP_EXPRESSION.exists() and not rep.empty:
    # Discovery expression is aligned to the discovery metadata above.
    ad_dis = attach_metadata(sc.read_h5ad(ROSMAP_EXPRESSION), dis)
    # Replication expression path should represent the combined APOE replication resource;
    # if stored as separate h5ad files, run this function per file and concatenate results.
    if REPLICATION_EXPRESSION.exists():
        ad_rep = attach_metadata(sc.read_h5ad(REPLICATION_EXPRESSION), rep)
        dis_gene = pd.concat([donor_gene_age_analysis(ad_dis, s, "Discovery") for s in ["Female", "Male"]], ignore_index=True)
        rep_gene = pd.concat([donor_gene_age_analysis(ad_rep, s, "Replication") for s in ["Female", "Male"]], ignore_index=True)
        dis_gene.to_csv(RESULT_DIR / "APOE4_age_genes_discovery_all.csv", index=False)
        rep_gene.to_csv(RESULT_DIR / "APOE4_age_genes_replication_all.csv", index=False)
        integrated = integrate_discovery_replication(dis_gene, rep_gene)
        reproducible = integrated[integrated["reproducible"] & integrated["aging_consistent"]].copy()
        reproducible.to_csv(RESULT_DIR / "APOE4_age_genes_reproducible.csv", index=False)

        # Further intersection with concordant AD-vs-non-AD DE, as specified in the final Methods.
        ad_dis = pd.concat([ad_deg_by_sex_celltype(ad_dis, s, "Discovery") for s in ["Female", "Male"]], ignore_index=True)
        ad_rep = pd.concat([ad_deg_by_sex_celltype(ad_rep, s, "Replication") for s in ["Female", "Male"]], ignore_index=True)
        if not ad_dis.empty and not ad_rep.empty:
            ad_joint = ad_dis.merge(ad_rep, on=["gene", "celltype", "Sex"], suffixes=("_dis", "_rep"))
            ad_joint = ad_joint[
                (ad_joint["AD_FDR_dis"] < 0.05) & (ad_joint["AD_P_rep"] < 0.05) &
                (np.sign(ad_joint["AD_log2FC_dis"]) == np.sign(ad_joint["AD_log2FC_rep"]))
            ].copy()
            joint = reproducible.merge(
                ad_joint[["gene", "celltype", "Sex", "AD_log2FC_dis", "AD_log2FC_rep", "AD_FDR_dis", "AD_P_rep"]],
                on=["gene", "celltype", "Sex"], how="inner"
            )
            # Retain AD changes concordant with the APOE4/age direction.
            joint = joint[np.sign(joint["AD_log2FC_dis"]) == np.sign(joint["APOE4_log2FC_dis"])].copy()
            joint.to_csv(RESULT_DIR / "APOE4_AD_cellular_age_genes_reproducible.csv", index=False)
            go_dir = RESULT_DIR / "GO_input_gene_lists"
            go_dir.mkdir(exist_ok=True)
            for (sex, ct), d in joint.groupby(["Sex", "celltype"], observed=True):
                for direction, mask in [("up_age_positive", d["APOE4_log2FC_dis"] > 0), ("down_age_negative", d["APOE4_log2FC_dis"] < 0)]:
                    pd.Series(sorted(d.loc[mask, "gene"].astype(str).unique()), name="gene").to_csv(
                        go_dir / f"{sex}_{ct}_{direction}.csv", index=False
                    )

print("APOE4 cellular-age analysis complete.")
