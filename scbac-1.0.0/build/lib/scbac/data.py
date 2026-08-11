"""Input/output helpers for AnnData and tabular scBAC workflows."""
from pathlib import Path
import pandas as pd
import scanpy as sc


def read_anndata(path, count_layer=None):
    """Read an h5ad file and optionally use one layer as the expression matrix."""
    path = Path(path)
    if path.suffix.lower() != ".h5ad":
        raise ValueError("Age-clock training/prediction expects an .h5ad AnnData file.")
    adata = sc.read_h5ad(path)
    if count_layer is not None:
        if count_layer not in adata.layers:
            raise KeyError("AnnData layer {!r} was not found.".format(count_layer))
        adata = adata.copy()
        adata.X = adata.layers[count_layer].copy()
    return adata


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Tabular input must be .csv, .tsv, or .txt")


def write_table(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=index)
    else:
        df.to_csv(path, index=index)
    return path


def attach_predictions(adata, predictions, prefix=""):
    """Return a copy of AnnData with prediction columns copied to ``obs``."""
    out = adata.copy()
    common = out.obs_names.intersection(predictions.index)
    for column in predictions.columns:
        if column == "celltype":
            continue
        out.obs.loc[common, prefix + column] = predictions.loc[common, column].values
    return out
