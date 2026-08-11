"""Age at aging acceleration onset (AASO) estimation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def smooth_donor_raa_curve(donor_cell_data, predicted_age_col="scBAC_age", raa_col="RAA", n_points=100, window_years=2.0):
    if len(donor_cell_data) < 8:
        return None, None
    d = donor_cell_data[[predicted_age_col, raa_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(predicted_age_col)
    if len(d) < 8:
        return None, None
    age_pred = d[predicted_age_col].to_numpy(float)
    raa = d[raa_col].to_numpy(float)
    q5, q95 = np.percentile(raa, [5, 95])
    spread = q95 - q5
    keep = (raa >= q5 - 1.5 * spread) & (raa <= q95 + 1.5 * spread)
    age_pred = age_pred[keep]
    raa = raa[keep]
    if len(age_pred) < 6:
        return None, None
    grid = np.linspace(age_pred.min(), age_pred.max(), int(n_points))
    distances = np.abs(grid[:, None] - age_pred)
    smooth = np.full(len(grid), np.nan, dtype=float)
    for i in range(len(grid)):
        mask = distances[i] <= float(window_years)
        if np.any(mask):
            smooth[i] = np.mean(raa[mask])
    valid = np.isfinite(smooth)
    if valid.sum() < 4:
        return None, None
    interpolator = interp1d(grid[valid], smooth[valid], kind="linear", bounds_error=False, fill_value="extrapolate")
    return grid, interpolator(grid)


def first_threshold_crossing(age_grid, raa_grid, target):
    indices = np.where(np.asarray(raa_grid) > float(target))[0]
    if len(indices) == 0:
        return None, "not_crossed"
    first = int(indices[0])
    if first == 0:
        return None, "left_censored"
    x1, x2 = float(age_grid[first - 1]), float(age_grid[first])
    y1, y2 = float(raa_grid[first - 1]), float(raa_grid[first])
    if y2 == y1:
        return x1, "observed"
    crossing = x1 + (float(target) - y1) * (x2 - x1) / (y2 - y1)
    return float(crossing), "observed"


def calculate_aaso(
    data,
    thresholds,
    *,
    donor_col="donor_id",
    celltype_col="celltype",
    predicted_age_col="scBAC_age",
    raa_col="RAA",
    quantile="q75",
    min_cells=10,
    n_points=100,
    window_years=2.0,
    metadata_cols=None,
    include_censored=False,
):
    """Calculate donor × cell-type AASO using the manuscript first-crossing algorithm."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    required = [donor_col, celltype_col, predicted_age_col, raa_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("AASO input is missing required columns: {}".format(missing))
    rows = []
    curves = {}
    metadata_cols = list(metadata_cols or [])
    for (donor, ct), group in df.groupby([donor_col, celltype_col], observed=True):
        ct = str(ct)
        if len(group) < int(min_cells) or ct not in thresholds or quantile not in thresholds[ct]:
            continue
        target = float(thresholds[ct][quantile])
        x, y = smooth_donor_raa_curve(group, predicted_age_col, raa_col, n_points=n_points, window_years=window_years)
        if x is None:
            continue
        onset, censoring = first_threshold_crossing(x, y, target)
        curves[(str(donor), ct)] = {"predicted_age": x, "raa": y, "threshold": target, "aaso": onset, "censoring": censoring}
        if onset is None and not include_censored:
            continue
        row = {
            "donor_id": donor,
            "celltype": ct,
            "AASO": onset,
            "threshold_age": onset,
            "threshold_reference": target,
            "threshold_quantile": quantile,
            "n_cells": int(len(group)),
            "censoring": censoring,
        }
        for col in metadata_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows), curves
