"""Convenience end-to-end RAA/AASO analysis workflow."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .raa import apply_raa, fit_raa_reference
from .aaso import calculate_aaso
from .utils import bh_fdr
from .visualization import plot_aaso_by_celltype, plot_raa_curves


def compare_aaso_celltypes(aaso_df):
    rows = []
    celltypes = sorted(aaso_df["celltype"].dropna().astype(str).unique())
    for i, a in enumerate(celltypes):
        x = pd.to_numeric(aaso_df.loc[aaso_df["celltype"].astype(str) == a, "AASO"], errors="coerce").dropna()
        for b in celltypes[i + 1:]:
            y = pd.to_numeric(aaso_df.loc[aaso_df["celltype"].astype(str) == b, "AASO"], errors="coerce").dropna()
            if len(x) < 2 or len(y) < 2:
                continue
            stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
            rows.append({"CellType_1": a, "CellType_2": b, "Mean_1": x.mean(), "Mean_2": y.mean(), "Difference": x.mean() - y.mean(), "T_stat": stat, "P_value": p, "N_1": len(x), "N_2": len(y)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = bh_fdr(out["P_value"])
    return out


def analyze_aging(
    data,
    output_prefix,
    *,
    donor_col="donor_id",
    celltype_col="celltype",
    predicted_age_col="scBAC_age",
    chronological_age_col="Age_at_death",
    sex_col="Sex",
    raa_reference=None,
    raa_clock_name="Ensemble_Adult",
    control_data=None,
    control_group_col=None,
    control_groups=None,
    quantile="q75",
    metadata_cols=None,
    status_col=None,
    disease_name=None,
    control_name=None,
):
    """Calculate RAA and AASO and write standard tables/figures."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if raa_reference is None and control_data is not None:
        ref_dir = prefix.parent / (prefix.name + "_raa_reference")
        raa_df, _, thresholds = fit_raa_reference(
            control_data, ref_dir, donor_col=donor_col, celltype_col=celltype_col,
            predicted_age_col=predicted_age_col, chronological_age_col=chronological_age_col,
            sex_col=sex_col, group_col=control_group_col, control_groups=control_groups,
        )
        # Apply the fitted reference to the full analysis data, not only controls.
        raa_df, thresholds = apply_raa(
            df, reference=ref_dir, donor_col=donor_col, celltype_col=celltype_col,
            predicted_age_col=predicted_age_col, chronological_age_col=chronological_age_col,
            sex_col=sex_col,
        )
    else:
        raa_df, thresholds = apply_raa(
            df, reference=raa_reference, clock_name=raa_clock_name,
            donor_col=donor_col, celltype_col=celltype_col,
            predicted_age_col=predicted_age_col, chronological_age_col=chronological_age_col,
            sex_col=sex_col,
        )
    if thresholds is None:
        raise FileNotFoundError("AASO requires a thresholds JSON in the RAA reference bundle.")
    raa_path = prefix.parent / (prefix.name + "_RAA.csv")
    raa_df.to_csv(raa_path, index=False)
    aaso_metadata = list(metadata_cols or [])
    if status_col is not None and status_col not in aaso_metadata:
        aaso_metadata.append(status_col)
    aaso, curves = calculate_aaso(
        raa_df, thresholds, donor_col=donor_col, celltype_col=celltype_col,
        predicted_age_col=predicted_age_col, raa_col="RAA", quantile=quantile,
        metadata_cols=aaso_metadata,
    )
    aaso_path = prefix.parent / (prefix.name + "_AASO.csv")
    aaso.to_csv(aaso_path, index=False)
    summary = aaso.groupby("celltype", observed=True)["AASO"].agg(["count", "mean", "median", "std", "sem"]).reset_index() if not aaso.empty else pd.DataFrame()
    summary.to_csv(prefix.parent / (prefix.name + "_AASO_summary.csv"), index=False)
    pairwise = compare_aaso_celltypes(aaso) if not aaso.empty else pd.DataFrame()
    pairwise.to_csv(prefix.parent / (prefix.name + "_AASO_pairwise.csv"), index=False)
    disease_effect = pd.DataFrame()
    if status_col is not None and disease_name is not None and status_col in aaso.columns:
        controls = [control_name] if control_name is not None else [x for x in aaso[status_col].dropna().astype(str).unique() if x != str(disease_name)]
        rows = []
        for ct, group in aaso.groupby("celltype", observed=True):
            x = pd.to_numeric(group.loc[group[status_col].astype(str) == str(disease_name), "AASO"], errors="coerce").dropna()
            y = pd.to_numeric(group.loc[group[status_col].astype(str).isin([str(v) for v in controls]), "AASO"], errors="coerce").dropna()
            if len(x) >= 2 and len(y) >= 2:
                stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                rows.append({"CellType": ct, "Disease": disease_name, "Controls": ",".join(map(str, controls)), "Disease_Mean_AASO": x.mean(), "Control_Mean_AASO": y.mean(), "Difference_Years": x.mean()-y.mean(), "T_stat": stat, "P_value": p, "N_Disease": len(x), "N_Control": len(y)})
        disease_effect = pd.DataFrame(rows)
        if not disease_effect.empty:
            disease_effect["FDR"] = bh_fdr(disease_effect["P_value"])
        disease_effect.to_csv(prefix.parent / (prefix.name + "_AASO_disease_effect.csv"), index=False)
    plot_aaso_by_celltype(aaso, prefix.parent / (prefix.name + "_AASO_boxplot.pdf"))
    plot_raa_curves(curves, prefix.parent / (prefix.name + "_RAA_curve_examples.pdf"))
    return {"raa": raa_df, "aaso": aaso, "summary": summary, "pairwise": pairwise, "disease_effect": disease_effect, "curves": curves}
