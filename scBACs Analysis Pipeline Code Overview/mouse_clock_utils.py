"""
Utilities for applying the fixed published Buckley et al. mouse brain clocks.

The released coefficient tables are treated as fixed pretrained clocks. They are
never refitted, recalibrated, or tuned in this repository.

Important
---------
The released Buckley coefficient tables used in the study did not contain model
intercepts. Therefore predictions are calculated exactly as in the original
working analysis:

    predicted_age = expression @ released_coefficients

No intercept is estimated or added. Consequently, transferred predictions can
occasionally be below zero, especially in cross-species applications. Such
values should be interpreted only for relative comparisons, not as calibrated
chronological ages.

Normalization follows the original working code when ``norm=True``:
Scanpy library-size normalization to 10,000 counts followed by log1p.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def _read_clock_coefficients(param_file, cell_type):
    """Read non-zero coefficients for one published mouse-clock lineage."""
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(param_file)
    params = pd.read_csv(param_file)
    required = {"Gene", cell_type}
    missing = required.difference(params.columns)
    if missing:
        raise KeyError(
            "Mouse-clock coefficient file is missing columns: {}".format(
                sorted(missing)
            )
        )
    coef = params.loc[
        pd.to_numeric(params[cell_type], errors="coerce").fillna(0) != 0,
        ["Gene", cell_type],
    ].copy()
    coef["Gene"] = coef["Gene"].astype(str)
    coef[cell_type] = pd.to_numeric(coef[cell_type], errors="coerce")
    coef = coef.dropna(subset=[cell_type]).drop_duplicates("Gene", keep="first")
    return coef.set_index("Gene")[cell_type]


def predict_age_singlecell(
    adata,
    param_file,
    cell_type="Ast",
    layer=None,
    use_raw=False,
    norm=False,
):
    """Apply one fixed Buckley mouse clock to single-cell expression.

    This preserves the original no-intercept calculation. Missing clock genes
    contribute zero because only genes present in both the dataset and released
    coefficient table enter the dot product.
    """
    coef = _read_clock_coefficients(param_file, cell_type)

    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None")
        genes = pd.Index(adata.raw.var_names.astype(str))
        X = adata.raw.X
    else:
        work = adata.copy()
        if norm:
            if layer is not None:
                if layer not in work.layers:
                    raise KeyError("Layer '{}' not found".format(layer))
                work.X = work.layers[layer].copy()
            work.X = work.X.astype("float32")
            sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
            X = work.X
            genes = pd.Index(work.var_names.astype(str))
        else:
            genes = pd.Index(work.var_names.astype(str))
            if layer is not None:
                if layer not in work.layers:
                    raise KeyError("Layer '{}' not found".format(layer))
                X = work.layers[layer]
            else:
                X = work.X

    common_genes = coef.index.intersection(genes)
    if len(common_genes) == 0:
        return np.full(adata.n_obs, np.nan, dtype=float)

    positions = genes.get_indexer(common_genes)
    X_use = X[:, positions]
    beta = coef.loc[common_genes].to_numpy(dtype=float)

    if sparse.issparse(X_use):
        pred = X_use.dot(beta)
        pred = np.asarray(pred).reshape(-1)
    else:
        pred = np.asarray(X_use, dtype=float) @ beta
    return np.asarray(pred, dtype=float).reshape(-1)


def predict_mouse_clock_celltypes(
    adata,
    param_file,
    cell_types=("Ast", "Oli", "Mic"),
    celltype_col="celltype",
    norm=False,
    layer=None,
    use_raw=False,
    extra_obs_cols=None,
):
    """Apply fixed lineage-specific mouse clocks and return one prediction table."""
    if celltype_col not in adata.obs.columns:
        raise KeyError("adata.obs['{}'] is required".format(celltype_col))
    extra_obs_cols = list(extra_obs_cols or [])
    parts = []
    for ct in cell_types:
        sub = adata[adata.obs[celltype_col].astype(str) == str(ct)].copy()
        if sub.n_obs == 0:
            continue
        pred = predict_age_singlecell(
            sub,
            param_file=param_file,
            cell_type=ct,
            layer=layer,
            use_raw=use_raw,
            norm=norm,
        )
        out = pd.DataFrame(index=sub.obs_names.copy())
        out["Pred_age"] = pred
        out["celltype"] = ct
        for col in extra_obs_cols:
            if col in sub.obs.columns:
                out[col] = sub.obs[col].values
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0)
