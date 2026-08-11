"""
Shared utilities for the downstream scBAC analyses used in the manuscript.

This module contains only methods retained in the final analysis:
- OLS regression for NDD/APOE4 cellular-age and AASO analyses
- donor-median summaries and AD-odds logistic regression
- control-derived cell-level RAA and AASO reconstruction
- donor-level pseudobulk expression and gene association tests
- Scanpy Wilcoxon differential expression with a minimum-expression filter

Historical mixed-model, GAM, Granger-causality and exploratory sensitivity helpers
are intentionally omitted because they are not part of the final Methods.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import sparse, stats
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

MAIN_CELLTYPES = ["Exc", "Inh", "Ast", "Oli", "OPC", "Mic"]

SEX_MAP = {
    "Female": 0, "female": 0, "F": 0, "f": 0,
    "Male": 1, "male": 1, "M": 1, "m": 1,
    0: 0, 1: 1,
}

APOE4_NONCARRIER = {
    "E2/E2", "E2/E3", "E3/E2", "E3/E3",
    "2/2", "2/3", "3/2", "3/3",
    "ε2/ε2", "ε2/ε3", "ε3/ε2", "ε3/ε3",
}
APOE4_CARRIER = {
    "E2/E4", "E4/E2", "E3/E4", "E4/E3", "E4/E4",
    "2/4", "4/2", "3/4", "4/3", "4/4",
    "ε2/ε4", "ε4/ε2", "ε3/ε4", "ε4/ε3", "ε4/ε4",
}


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR while preserving missing values."""
    p = pd.to_numeric(pd.Series(p_values), errors="coerce")
    out = np.full(len(p), np.nan, dtype=float)
    valid = p.notna().values
    if valid.sum():
        out[valid] = multipletests(p.loc[valid].values, method="fdr_bh")[1]
    return out


def encode_sex(series):
    """Encode Female=0 and Male=1 without relying on categorical ordering."""
    return series.map(SEX_MAP)


def encode_apoe4(series):
    """Encode APOE4 carrier status from common genotype notations."""
    def _one(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip().replace(" ", "")
        value = value.replace("APOE", "").replace("e", "E")
        if value in APOE4_CARRIER:
            return 1.0
        if value in APOE4_NONCARRIER:
            return 0.0
        return np.nan
    return series.map(_one)


def normalize_status_nonad_ad(series):
    """Final APOE4 analysis coding: AD=1; CT/MCI=0."""
    return series.astype(str).map({"AD": 1.0, "CT": 0.0, "MCI": 0.0})


def add_scbac_adult(df, output_col="scBAC_adult"):
    """Resolve/build the adult scBAC column from old or rewritten pipeline names."""
    df = df.copy()
    candidates = [output_col, "Ensemble_Adult", "scBAC_age"]
    for col in candidates:
        if col in df.columns:
            df[output_col] = pd.to_numeric(df[col], errors="coerce")
            return df

    component_sets = [
        ["elasticnet_Adult", "clm_Adult", "transf_Adult"],
        ["elasticnet_adult", "clm_adult", "transformer_adult"],
    ]
    for cols in component_sets:
        if all(c in df.columns for c in cols):
            df[output_col] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            return df
    raise KeyError("Could not find or construct adult scBAC predictions.")


def make_donor_uid(df, donor_col="donor_id", dataset_col="dataset", output_col="donor_uid"):
    """Create a study-safe donor identifier."""
    out = df.copy()
    if dataset_col in out.columns:
        out[output_col] = out[dataset_col].astype(str) + "::" + out[donor_col].astype(str)
    else:
        out[output_col] = out[donor_col].astype(str)
    return out


def center_age_within_groups(df, group_cols, age_col="Age_at_death"):
    """Center chronological age separately within each requested analysis stratum."""
    out = df.copy()
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
    out["Age_c"] = out[age_col] - out.groupby(group_cols, observed=True)[age_col].transform("mean")
    out["Age_c2"] = out["Age_c"] ** 2
    return out


def run_ndd_cellular_age_ols(
    df,
    disease_label,
    outcome_col="scBAC_adult",
    tissue_col="sub_tissue",
    celltype_col="celltype",
    status_col="status",
    sex_col="Sex",
    age_col="Age_at_death",
    min_cells=10,
    include_sex_by_disease=False,
):
    """Final cell-level NDD model, fitted separately by tissue x cell type."""
    work = df[df[status_col].isin(["CT", disease_label])].copy()
    work["Disease"] = work[status_col].map({"CT": 0.0, disease_label: 1.0})
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work = center_age_within_groups(work, [tissue_col, celltype_col], age_col=age_col)

    formula = (
        f"{outcome_col} ~ Disease + Sex_numeric + Age_c + Age_c2 + "
        "Sex_numeric:Age_c + Sex_numeric:Age_c2"
    )
    terms = ["Disease"]
    if include_sex_by_disease:
        formula += " + Sex_numeric:Disease"
        terms.append("Sex_numeric:Disease")

    rows = []
    for (tissue, ct), g in work.groupby([tissue_col, celltype_col], observed=True):
        g = g.dropna(subset=[outcome_col, "Disease", "Sex_numeric", "Age_c", "Age_c2"])
        if len(g) < min_cells or g["Disease"].nunique() < 2:
            continue
        fit = smf.ols(formula, data=g).fit()
        for term in terms:
            if term not in fit.params:
                continue
            ci = fit.conf_int().loc[term]
            rows.append({
                "Disease": disease_label,
                "Sub_Tissue": tissue,
                "CellType": ct,
                "Term": term,
                "Effect_Size_Years": fit.params[term],
                "Std_Error": fit.bse[term],
                "CI95_Low": ci.iloc[0],
                "CI95_High": ci.iloc[1],
                "P_value": fit.pvalues[term],
                "N_Cells": len(g),
                "N_Donors": g.get("donor_uid", g.get("donor_id", pd.Series(index=g.index))).nunique(),
                "Mean_Age_Reference": g[age_col].mean(),
                "R_squared": fit.rsquared,
            })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["FDR"] = res.groupby("Term", observed=True)["P_value"].transform(bh_fdr)
    return res


def donor_median_summary(
    df,
    outcome_cols,
    donor_col="donor_uid",
    celltype_col="celltype",
    metadata_cols=None,
):
    """Summarize cell-level ages as donor x cell-type medians."""
    metadata_cols = metadata_cols or []
    agg = {c: "median" for c in outcome_cols}
    agg.update({c: "first" for c in metadata_cols if c in df.columns})
    out = df.groupby([donor_col, celltype_col], observed=True).agg(agg).reset_index()
    return out


def run_donor_ad_age_ols(
    donor_df,
    outcome_col="scBAC_adult",
    status_col="status",
    celltype_col="celltype",
    sex_col="Sex",
    age_col="Age_at_death",
    min_donors=10,
):
    """Donor-median AD-vs-CT OLS using the final NDD covariate structure."""
    work = donor_df[donor_df[status_col].isin(["CT", "AD"])].copy()
    work["Disease"] = work[status_col].map({"CT": 0.0, "AD": 1.0})
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work = center_age_within_groups(work, [celltype_col], age_col=age_col)
    formula = (
        f"{outcome_col} ~ Disease + Sex_numeric + Age_c + Age_c2 + PMI+"
        "Sex_numeric:Age_c + Sex_numeric:Age_c2"
    )
    rows = []
    for ct, g in work.groupby(celltype_col, observed=True):
        g = g.dropna(subset=[outcome_col, "Disease", "Sex_numeric", "Age_c", "Age_c2"])
        if len(g) < min_donors or g["Disease"].nunique() < 2:
            continue
        fit = smf.ols(formula, data=g).fit()
        ci = fit.conf_int().loc["Disease"]
        rows.append({
            "CellType": ct,
            "Effect_Size_Years": fit.params["Disease"],
            "Std_Error": fit.bse["Disease"],
            "CI95_Low": ci.iloc[0],
            "CI95_High": ci.iloc[1],
            "P_value": fit.pvalues["Disease"],
            "N_Donors": len(g),
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["FDR"] = bh_fdr(res["P_value"])
    return res


def run_ad_odds_logistic(
    donor_df,
    outcome_col="scBAC_adult",
    status_col="status",
    celltype_col="celltype",
    sex_col="Sex",
    age_col="Age_at_death",
    pmi_col="PMI",
    min_donors=10,
):
    """AD odds per 1-SD increase in donor-median cellular age."""
    work = donor_df[donor_df[status_col].isin(["CT", "AD"])].copy()
    work["AD"] = work[status_col].map({"CT": 0.0, "AD": 1.0})
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work = center_age_within_groups(work, [celltype_col], age_col=age_col)
    rows = []
    for ct, g in work.groupby(celltype_col, observed=True):
        required = [outcome_col, "AD", "Sex_numeric", "Age_c", "Age_c2", pmi_col]
        g = g.dropna(subset=required).copy()
        if len(g) < min_donors or g["AD"].nunique() < 2:
            continue
        sd = g[outcome_col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        g["CellAge_z"] = (g[outcome_col] - g[outcome_col].mean()) / sd
        formula = (
            f"AD ~ CellAge_z + Age_c + Age_c2 + Sex_numeric + "
            f"Sex_numeric:Age_c + Sex_numeric:Age_c2 + {pmi_col}"
        )
        fit = smf.logit(formula, data=g).fit(disp=False)
        beta = fit.params["CellAge_z"]
        ci = fit.conf_int().loc["CellAge_z"]
        rows.append({
            "CellType": ct,
            "OR_per_1SD_CellAge": np.exp(beta),
            "CI95_Low": np.exp(ci.iloc[0]),
            "CI95_High": np.exp(ci.iloc[1]),
            "Beta": beta,
            "P_value": fit.pvalues["CellAge_z"],
            "N_Donors": len(g),
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["FDR"] = bh_fdr(res["P_value"])
    return res


def run_pathology_associations(
    donor_df,
    phenotypes,
    outcome_col="scBAC_adult",
    celltype_col="celltype",
    sex_col="Sex",
    age_col="Age_at_death",
    pmi_col="PMI",
    min_donors=8,
):
    """Donor-level cellular-age association with AD pathology/cognition."""
    work = donor_df.copy()
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work[age_col] = pd.to_numeric(work[age_col], errors="coerce")
    work["Age2"] = work[age_col] ** 2
    rows = []
    for phenotype in phenotypes:
        if phenotype not in work.columns:
            continue
        for ct, g in work.groupby(celltype_col, observed=True):
            req = [outcome_col, phenotype, "Sex_numeric", age_col, "Age2", pmi_col]
            g = g.dropna(subset=req).copy()
            if len(g) < min_donors:
                continue
            formula = (
                f"{outcome_col} ~ Q('{phenotype}') + Sex_numeric + {age_col} + Age2 + {pmi_col} + "
                f"Sex_numeric:{age_col} + Sex_numeric:Age2"
            )
            fit = smf.ols(formula, data=g).fit()
            term = "Q('{}')".format(phenotype)
            if term not in fit.params:
                continue
            ci = fit.conf_int().loc[term]
            rows.append({
                "Phenotype": phenotype,
                "CellType": ct,
                "Effect_Size": fit.params[term],
                "Std_Error": fit.bse[term],
                "CI95_Low": ci.iloc[0],
                "CI95_High": ci.iloc[1],
                "P_value": fit.pvalues[term],
                "N_Donors": len(g),
            })
    res = pd.DataFrame(rows)
    if not res.empty:
        # Final manuscript states correction across phenotype-cell-type tests.
        res["FDR"] = bh_fdr(res["P_value"])
    return res


def fit_raa_reference(
    controls,
    predicted_age_col="scBAC_adult",
    donor_col="donor_uid",
    celltype_col="celltype",
    age_col="Age_at_death",
    sex_col="Sex",
):
    """Fit CellAge ~ chronological age + sex + donor cell count in controls."""
    work = controls.copy()
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work["donor_cell_count"] = work.groupby(donor_col, observed=True)[donor_col].transform("size")
    models = {}
    for ct, g in work.groupby(celltype_col, observed=True):
        g = g.dropna(subset=[predicted_age_col, age_col, "Sex_numeric", "donor_cell_count"])
        if len(g) < 10:
            continue
        X = sm.add_constant(g[[age_col, "Sex_numeric", "donor_cell_count"]], has_constant="add")
        fit = sm.OLS(g[predicted_age_col].astype(float), X.astype(float)).fit()
        models[str(ct)] = {
            "features": list(X.columns),
            "coef": fit.params.to_dict(),
            "n_cells": int(len(g)),
        }
    return models


def apply_raa_reference(
    df,
    models,
    predicted_age_col="scBAC_adult",
    donor_col="donor_uid",
    celltype_col="celltype",
    age_col="Age_at_death",
    sex_col="Sex",
    output_col="RAA",
):
    """Apply fixed control-reference coefficients to cells without refitting."""
    out = df.copy()
    out["Sex_numeric"] = encode_sex(out[sex_col])
    out["donor_cell_count"] = out.groupby(donor_col, observed=True)[donor_col].transform("size")
    out[output_col] = np.nan
    for ct, g in out.groupby(celltype_col, observed=True):
        info = models.get(str(ct))
        if info is None:
            continue
        features = info["features"]
        valid = g[[predicted_age_col, age_col, "Sex_numeric", "donor_cell_count"]].notna().all(axis=1)
        gg = g.loc[valid].copy()
        if gg.empty:
            continue
        X = pd.DataFrame(index=gg.index)
        for f in features:
            if f == "const":
                X[f] = 1.0
            else:
                X[f] = pd.to_numeric(gg[f], errors="coerce")
        expected = np.zeros(len(gg), dtype=float)
        for f in features:
            expected += float(info["coef"][f]) * X[f].to_numpy(dtype=float)
        out.loc[gg.index, output_col] = pd.to_numeric(gg[predicted_age_col], errors="coerce") - expected
    return out.drop(columns=["Sex_numeric"], errors="ignore")


def save_raa_reference(models, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"models": models, "method": "OLS"}, fh)


def load_raa_reference(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["models"] if isinstance(obj, dict) and "models" in obj else obj


def compute_raa_thresholds(controls_with_raa, raa_col="RAA", celltype_col="celltype", quantile=0.75):
    rows = []
    for ct, g in controls_with_raa.groupby(celltype_col, observed=True):
        x = pd.to_numeric(g[raa_col], errors="coerce").dropna()
        if x.empty:
            continue
        rows.append({"celltype": ct, "quantile": quantile, "threshold": x.quantile(quantile), "n_cells": len(x)})
    return pd.DataFrame(rows)


def estimate_aaso_one_donor(
    df,
    threshold,
    predicted_age_col="scBAC_adult",
    raa_col="RAA",
    min_cells=8,
    n_grid=100,
    window_half_width=2.0,
):
    """Reconstruct one donor-specific RAA trajectory and return its first from-below crossing."""
    d = df[[predicted_age_col, raa_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(predicted_age_col)
    if len(d) < min_cells:
        return np.nan
    q5, q95 = np.percentile(d[raa_col], [5, 95])
    span = q95 - q5
    lo, hi = q5 - 1.5 * span, q95 + 1.5 * span
    d = d[d[raa_col].between(lo, hi)]
    if len(d) < 6 or d[predicted_age_col].nunique() < 2:
        return np.nan

    x_min, x_max = d[predicted_age_col].min(), d[predicted_age_col].max()
    grid = np.linspace(x_min, x_max, n_grid)
    smooth = np.full(n_grid, np.nan)
    x = d[predicted_age_col].to_numpy(float)
    y = d[raa_col].to_numpy(float)
    for i, point in enumerate(grid):
        mask = np.abs(x - point) <= window_half_width
        if mask.any():
            smooth[i] = np.nanmean(y[mask])
    good = np.isfinite(smooth)
    if good.sum() < 4:
        return np.nan
    smooth = interp1d(grid[good], smooth[good], kind="linear", bounds_error=False, fill_value="extrapolate")(grid)

    above = smooth >= threshold
    crossings = np.where((~above[:-1]) & above[1:])[0]
    if len(crossings) == 0:
        return np.nan
    i = int(crossings[0])
    x0, x1 = grid[i], grid[i + 1]
    y0, y1 = smooth[i], smooth[i + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def compute_aaso(
    cells,
    thresholds,
    donor_col="donor_uid",
    celltype_col="celltype",
    predicted_age_col="scBAC_adult",
    raa_col="RAA",
    min_cells=8,
    metadata_cols=None,
):
    """Compute AASO for each donor x cell type using fixed cell-type thresholds."""
    metadata_cols = metadata_cols or []
    threshold_map = thresholds.set_index("celltype")["threshold"].to_dict()
    rows = []
    for (donor, ct), g in cells.groupby([donor_col, celltype_col], observed=True):
        if ct not in threshold_map:
            continue
        onset = estimate_aaso_one_donor(
            g, threshold_map[ct], predicted_age_col=predicted_age_col,
            raa_col=raa_col, min_cells=min_cells,
        )
        row = {donor_col: donor, celltype_col: ct, "AASO": onset, "N_Cells": len(g)}
        for col in metadata_cols:
            if col in g.columns:
                row[col] = g[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_aaso_across_celltypes(aaso_df, celltype_col="celltype", min_donors=3):
    """One-way ANOVA plus pairwise Welch t-tests with BH correction."""
    groups = {
        ct: g["AASO"].dropna().to_numpy(float)
        for ct, g in aaso_df.groupby(celltype_col, observed=True)
        if g["AASO"].notna().sum() >= min_donors
    }
    anova = {"F": np.nan, "P_value": np.nan, "N_CellTypes": len(groups)}
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups.values())
        anova.update({"F": f, "P_value": p})
    rows = []
    keys = list(groups)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            stat, p = stats.ttest_ind(groups[a], groups[b], equal_var=False, nan_policy="omit")
            rows.append({"CellType_1": a, "CellType_2": b, "Welch_t": stat, "P_value": p})
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["FDR"] = bh_fdr(pairwise["P_value"])
    return pd.DataFrame([anova]), pairwise


def run_aaso_disease_ols(
    aaso_df,
    disease_label="AD",
    status_col="status",
    celltype_col="celltype",
    sex_col="Sex",
    age_col="Age_at_death",
    pmi_col="PMI",
    min_donors=8,
):
    """Disease effect on donor AASO using the donor-level covariate framework."""
    work = aaso_df[aaso_df[status_col].isin(["CT", disease_label])].copy()
    work["Disease"] = work[status_col].map({"CT": 0.0, disease_label: 1.0})
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work[age_col] = pd.to_numeric(work[age_col], errors="coerce")
    work["Age2"] = work[age_col] ** 2
    rows = []
    for ct, g in work.groupby(celltype_col, observed=True):
        req = ["AASO", "Disease", "Sex_numeric", age_col, "Age2"]
        if pmi_col in g.columns:
            req.append(pmi_col)
        g = g.dropna(subset=req).copy()
        if len(g) < min_donors or g["Disease"].nunique() < 2:
            continue
        formula = f"AASO ~ Disease + Sex_numeric + {age_col} + Age2 + Sex_numeric:{age_col} + Sex_numeric:Age2"
        if pmi_col in g.columns:
            formula += f" + {pmi_col}"
        fit = smf.ols(formula, data=g).fit()
        ci = fit.conf_int().loc["Disease"]
        rows.append({
            "Disease": disease_label, "CellType": ct,
            "Effect_Size_Years": fit.params["Disease"], "Std_Error": fit.bse["Disease"],
            "CI95_Low": ci.iloc[0], "CI95_High": ci.iloc[1],
            "P_value": fit.pvalues["Disease"], "N_Donors": len(g),
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["FDR"] = bh_fdr(res["P_value"])
    return res


def run_apoe_interaction_models(
    df,
    outcome_col,
    celltype_col="celltype",
    age_col="Age_at_death",
    sex_col="Sex",
    status_col="status",
    apoe_col="apoe4",
    pmi_col="PMI",
    min_n=10,
    sex_stratum=None,
):
    """Final APOE4 x sex x AD interaction model, with local age centering by cell type."""
    work = df.copy()
    work["Sex_numeric"] = encode_sex(work[sex_col])
    if apoe_col not in work.columns:
        raise KeyError(apoe_col)
    work["APOE4"] = pd.to_numeric(work[apoe_col], errors="coerce")
    work["Status"] = normalize_status_nonad_ad(work[status_col])
    if sex_stratum is not None:
        code = 1 if str(sex_stratum).lower().startswith("m") else 0
        work = work[work["Sex_numeric"] == code].copy()
    work = center_age_within_groups(work, [celltype_col], age_col=age_col)

    if sex_stratum is None:
        formula = (
            f"{outcome_col} ~ APOE4 + Status + Sex_numeric + Sex_numeric:APOE4 + "
            "Sex_numeric:Status + APOE4:Status + Sex_numeric:APOE4:Status + "
            f"Age_c + Age_c2 + {pmi_col} + Sex_numeric:Age_c + Sex_numeric:Age_c2"
        )
        terms = ["APOE4", "APOE4:Status", "Sex_numeric:APOE4:Status"]
    else:
        formula = f"{outcome_col} ~ APOE4 + Status + APOE4:Status + Age_c + Age_c2 + {pmi_col}"
        terms = ["APOE4", "APOE4:Status"]

    rows = []
    for ct, g in work.groupby(celltype_col, observed=True):
        req = [outcome_col, "APOE4", "Status", "Age_c", "Age_c2", pmi_col]
        if sex_stratum is None:
            req.append("Sex_numeric")
        g = g.dropna(subset=req).copy()
        if len(g) < min_n or g["APOE4"].nunique() < 2:
            continue
        try:
            fit = smf.ols(formula, data=g).fit()
        except Exception:
            continue
        for term in terms:
            if term not in fit.params:
                continue
            ci = fit.conf_int().loc[term]
            rows.append({
                "CellType": ct, "Sex_Stratum": sex_stratum or "Combined", "Term": term,
                "Effect_Size": fit.params[term], "Std_Error": fit.bse[term],
                "CI95_Low": ci.iloc[0], "CI95_High": ci.iloc[1],
                "P_value": fit.pvalues[term], "N": len(g),
            })
    res = pd.DataFrame(rows)
    if not res.empty:
        # Correct across cell types separately for each term and sex stratum.
        res["FDR"] = res.groupby(["Sex_Stratum", "Term"], observed=True)["P_value"].transform(bh_fdr)
    return res


def run_status_within_sex_apoe_strata(
    donor_df,
    outcome_col="scBAC_adult",
    celltype_col="celltype",
    status_col="status",
    sex_col="Sex",
    apoe_col="apoe4",
    age_col="Age_at_death",
    pmi_col="PMI",
    min_donors=5,
):
    """Adjusted AD-vs-non-AD cellular-age difference within four sex/APOE4 strata."""
    work = donor_df.copy()
    work["Sex_numeric"] = encode_sex(work[sex_col])
    work["APOE4"] = pd.to_numeric(work[apoe_col], errors="coerce")
    work["Status"] = normalize_status_nonad_ad(work[status_col])
    rows = []
    for sex_code, sex_name in [(0, "Female"), (1, "Male")]:
        for apoe_code, apoe_name in [(0, "Non-carrier"), (1, "Carrier")]:
            subset = work[(work["Sex_numeric"] == sex_code) & (work["APOE4"] == apoe_code)].copy()
            subset = center_age_within_groups(subset, [celltype_col], age_col=age_col)
            for ct, g in subset.groupby(celltype_col, observed=True):
                g = g.dropna(subset=[outcome_col, "Status", "Age_c", "Age_c2", pmi_col])
                if len(g) < min_donors or g["Status"].nunique() < 2:
                    continue
                fit = smf.ols(f"{outcome_col} ~ Status + Age_c + Age_c2 + {pmi_col}", data=g).fit()
                ci = fit.conf_int().loc["Status"]
                rows.append({
                    "Sex": sex_name, "APOE4": apoe_name, "CellType": ct,
                    "AD_Effect_Years": fit.params["Status"], "Std_Error": fit.bse["Status"],
                    "CI95_Low": ci.iloc[0], "CI95_High": ci.iloc[1],
                    "P_value": fit.pvalues["Status"], "N_Donors": len(g),
                })
    return pd.DataFrame(rows)


def normalize_log1p(adata, target_sum=1e4):
    out = adata.copy()
    out.X = out.X.astype(np.float32)
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out


def attach_metadata_by_index(adata, metadata):
    """Align an AnnData object and external metadata by cell identifier."""
    ids = adata.obs_names.intersection(metadata.index)
    out = adata[ids].copy()
    meta = metadata.loc[ids]
    for col in meta.columns:
        out.obs[col] = meta[col].values
    return out


def donor_pseudobulk_expression(
    adata,
    donor_col="donor_uid",
    celltype_col="celltype",
    min_cells=5,
    target_sum=1e4,
):
    """Mean log-normalized expression per donor x cell type."""
    a = normalize_log1p(adata, target_sum=target_sum)
    rows = []
    meta_rows = []
    for (donor, ct), idx in a.obs.groupby([donor_col, celltype_col], observed=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) < min_cells:
            continue
        block = a.X[idx]
        mean = np.asarray(block.mean(axis=0)).reshape(-1) if sparse.issparse(block) else np.asarray(block).mean(axis=0)
        rows.append(mean)
        meta_rows.append({donor_col: donor, celltype_col: ct, "n_cells": len(idx)})
    expr = pd.DataFrame(rows, columns=a.var_names.astype(str))
    meta = pd.DataFrame(meta_rows)
    return expr, meta


def gene_spearman_with_donor_metric(expr_df, meta_df, metric_series, donor_col="donor_uid"):
    """Gene-wise Spearman correlation with a donor-level metric."""
    metric = pd.Series(metric_series)
    if metric.index.name != donor_col:
        metric.index = metric.index.astype(str)
    donors = meta_df[donor_col].astype(str)
    y = donors.map(metric).to_numpy(float)
    rows = []
    for gene in expr_df.columns:
        x = pd.to_numeric(expr_df[gene], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
            continue
        rho, p = spearmanr(x[valid], y[valid])
        rows.append({"gene": gene, "spearman_rho": rho, "P_value": p, "N_Donors": int(valid.sum())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = bh_fdr(out["P_value"])
    return out


def gene_ols_with_aaso(
    expr_df,
    meta_df,
    donor_covariates,
    donor_col="donor_uid",
    age_col="Age_at_death",
    sex_col="Sex",
    pmi_col=None,
):
    """Gene-AASO association: AASO ~ gene + age + sex (+ PMI when requested)."""
    base = meta_df.copy()
    base[donor_col] = base[donor_col].astype(str)
    cov = donor_covariates.copy()
    cov[donor_col] = cov[donor_col].astype(str)
    base = base.merge(cov, on=donor_col, how="left", suffixes=("", "_cov"))
    base["Sex_numeric"] = encode_sex(base[sex_col]) if sex_col in base.columns else np.nan
    rows = []
    for gene in expr_df.columns:
        d = base.copy()
        d["GeneExpression"] = pd.to_numeric(expr_df[gene], errors="coerce").values
        req = ["AASO", "GeneExpression", age_col]
        formula = f"AASO ~ GeneExpression + {age_col}"
        if sex_col in d.columns:
            req.append("Sex_numeric")
            formula += " + Sex_numeric"
        if pmi_col is not None and pmi_col in d.columns:
            req.append(pmi_col)
            formula += f" + {pmi_col}"
        d = d.dropna(subset=req)
        if len(d) < 5 or d["GeneExpression"].nunique() < 2:
            continue
        fit = smf.ols(formula, data=d).fit()
        rows.append({"gene": gene, "Beta_Gene": fit.params["GeneExpression"], "P_value": fit.pvalues["GeneExpression"], "N_Donors": len(d)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = bh_fdr(out["P_value"])
    return out


def scanpy_wilcoxon_deg(
    adata,
    groupby,
    case_label,
    reference_label,
    celltype_col="celltype",
    celltypes=None,
    min_fraction=0.10,
    target_sum=1e4,
):
    """Cell-type-stratified Scanpy Wilcoxon DEG with >=10% case/reference expression."""
    celltypes = celltypes or MAIN_CELLTYPES
    rows = []
    for ct in celltypes:
        sub = adata[(adata.obs[celltype_col].astype(str) == str(ct)) & adata.obs[groupby].isin([case_label, reference_label])].copy()
        if sub.n_obs == 0 or sub.obs[groupby].nunique() < 2:
            continue
        sub = normalize_log1p(sub, target_sum=target_sum)
        sc.tl.rank_genes_groups(sub, groupby=groupby, groups=[case_label], reference=reference_label, method="wilcoxon", pts=True)
        res = sc.get.rank_genes_groups_df(sub, group=case_label)
        pts = sub.uns["rank_genes_groups"].get("pts")
        if pts is not None and case_label in pts.columns:
            frac_case = pts[case_label]
            pts_rest = sub.uns["rank_genes_groups"].get("pts_rest")
            if pts_rest is not None and case_label in pts_rest.columns:
                frac_ref = pts_rest[case_label]
                keep = frac_case.index[(frac_case >= min_fraction) | (frac_ref >= min_fraction)]
            else:
                keep = frac_case.index[frac_case >= min_fraction]
            res = res[res["names"].isin(keep)]
        res["celltype"] = ct
        res["case"] = case_label
        res["reference"] = reference_label
        rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
