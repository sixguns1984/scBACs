"""Publication-style visualizations for RAA and AASO workflows."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_aaso_by_celltype(aaso_df, output, celltype_col="celltype", aaso_col="AASO"):
    d = aaso_df.dropna(subset=[celltype_col, aaso_col]).copy()
    if d.empty:
        return None
    order = d.groupby(celltype_col, observed=True)[aaso_col].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(order)), 4.5))
    sns.boxplot(data=d, x=celltype_col, y=aaso_col, order=order, showfliers=False, ax=ax)
    sns.stripplot(data=d, x=celltype_col, y=aaso_col, order=order, color="black", alpha=0.35, size=2.5, ax=ax)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Age at aging acceleration onset (AASO)")
    sns.despine(ax=ax)
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def plot_raa_curves(curves, output, max_panels=6):
    keys = list(curves)[: int(max_panels)]
    if not keys:
        return None
    n = len(keys)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        info = curves[key]
        ax.plot(info["predicted_age"], info["raa"], linewidth=1.2)
        ax.axhline(info["threshold"], linestyle="--", linewidth=0.9)
        if info.get("aaso") is not None:
            ax.axvline(info["aaso"], linestyle=":", linewidth=0.9)
        ax.set_title("{} | {}".format(*key), fontsize=8)
        ax.set_xlabel("Predicted cellular age")
        ax.set_ylabel("RAA")
        sns.despine(ax=ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output
