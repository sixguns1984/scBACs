"""
Whole-life Pseudo-bulk Gene Expression Trajectory Analysis
===========================================================

Purpose:
1. Identify chronological age-associated genes in the training dataset
2. Replicate these associations in the independent multi-study validation dataset
3. Retain reproducible gene-cell-type pairs
4. Reconstruct donor-level lifespan expression trajectories
5. Cluster standardized trajectories using K-means
6. Evaluate K = 2-15 using the silhouette score and Davies-Bouldin index
7. Identify three data-supported resolutions: maximum silhouette, minimum Davies-Bouldin, and the visually identified silhouette-curve elbow
8. Retain K = 5 as the primary solution and K = 8 as a higher-resolution sensitivity analysis; K = 2 is retained only as a coarse diagnostic

Main analysis criteria:
- Training: FDR < 0.05
- Independent validation: nominal P < 0.05
- Same direction of Spearman correlation in both datasets

Trajectory analysis:
- donor-level pseudobulk expression
- cubic smoothing spline
- 200 equally spaced age points
- Z-score each fitted trajectory
- K-means clustering

Python version:
- Python 3.9.20

Usage:
- Edit paths and settings in Section 2
- Run the script section by section
- Command-line execution is also supported; examples are provided at the end
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from scipy.interpolate import UnivariateSpline
from scipy.stats import spearmanr, zscore

from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from statsmodels.stats.multitest import multipletests


# ============================================================================
# 2. PATHS AND ANALYSIS SETTINGS
# ============================================================================

PROJECT_ROOT = Path('./')

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "Expression_Trajectory"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

# Revised 13-study CT training resource
TRAIN_H5AD = os.path.join(
    DATA_DIR,
    "sce_train_CT.h5ad"
)
#sce_train_CT.h5ad: combined sce_train.h5ad and sce_train_added.h5ad
# Independent multi-study validation controls
VALIDATION_H5AD = os.path.join(
    DATA_DIR,
    "sce_test.h5ad"
)

DONOR_COL = "donor_id"
DATASET_COL = "dataset"
CELLTYPE_COL = "celltype"
AGE_COL = "Age_at_death"
STATUS_COL = "status"

CONTROL_LABEL = "CT"

CELLTYPES = [
    "Exc",
    "Inh",
    "Ast",
    "Oli",
    "OPC",
    "Mic",
    "Per",
    "End",
    "Fib",
    "CAM",
    "T_cell"
]

MIN_CELLS_PER_DONOR_CORRELATION = 5
MIN_DONORS_PER_CELLTYPE = 5

# The original trajectory code used at least 10 cells per donor/cell type
MIN_CELLS_PER_DONOR_TRAJECTORY = 10
MIN_DONORS_PER_TRAJECTORY = 15

N_AGE_POINTS = 200
SPLINE_SMOOTHING_SCALE = 0.5

N_JOBS = 8
RANDOM_SEED = 444

# Elbow identified by inspection of the silhouette-score curve in the original analysis.
# Keep this explicit rather than introducing an elbow algorithm that was not used.
SILHOUETTE_ELBOW_K = 8

DISCOVERY_FDR = 0.05
VALIDATION_P = 0.05

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sns.set_style("ticks")


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Lifespan gene-expression trajectory analysis for scBACs."
    )

    parser.add_argument(
        "--training-h5ad",
        required=True
    )

    parser.add_argument(
        "--validation-h5ad",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8
    )

    args = parser.parse_args()

    TRAIN_H5AD = args.training_h5ad
    VALIDATION_H5AD = args.validation_h5ad
    OUTPUT_DIR = os.path.abspath(
        args.output_dir
    )

    FIGURE_DIR = os.path.join(
        OUTPUT_DIR,
        "figures"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    N_JOBS = args.n_jobs


# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def add_analysis_donor_id(adata):
    """
    Create a study-specific donor ID when dataset information is available.
    """
    adata = adata.copy()
    if (
        DATASET_COL in adata.obs.columns
        and DONOR_COL in adata.obs.columns
    ):
        adata.obs[
            "donor_id_analysis"
        ] = (
            adata.obs[
                DATASET_COL
            ].astype(str)
            + "::"
            + adata.obs[
                DONOR_COL
            ].astype(str)
        )
    else:
        adata.obs[
            "donor_id_analysis"
        ] = (
            adata.obs[
                DONOR_COL
            ].astype(str)
        )
    return adata


def build_pseudobulk(
    adata,
    celltypes,
    genes_by_celltype=None,
    min_cells_per_donor=5
):
    """
    Build donor-level mean log-normalized expression for each cell type.

    Steps:
    1. select one cell type
    2. normalize each cell by library size
    3. log1p transform
    4. average cells from the same donor
    """
    pseudobulk_results = {}
    for celltype in celltypes:

        print(
            "\nBuilding pseudobulk:",
            celltype
        )
        sce_ct = adata[
            adata.obs[
                CELLTYPE_COL
            ] == celltype,
            :
        ].copy()
        if sce_ct.n_obs == 0:
            print(
                "No cells found. Skipping."
            )
            continue
        if genes_by_celltype is not None:
            genes = genes_by_celltype.get(
                celltype,
                []
            )
            genes = np.intersect1d(
                genes,
                sce_ct.var_names
            )
            if len(genes) == 0:
                print(
                    "No eligible genes. Skipping."
                )
                continue
            sce_ct = sce_ct[
                :,
                genes
            ].copy()
        sc.pp.normalize_per_cell(
            sce_ct
        )
        sc.pp.log1p(
            sce_ct
        )
        donor_expression = {}
        donor_metadata = []
        for donor in sce_ct.obs[
            "donor_id_analysis"
        ].unique():
            donor_mask = (
                sce_ct.obs[
                    "donor_id_analysis"
                ]
                == donor
            )
            donor_data = sce_ct[
                donor_mask,
                :
            ]
            if donor_data.n_obs < min_cells_per_donor:
                continue
            if sparse.issparse(
                donor_data.X
            ):
                mean_expression = np.asarray(
                    donor_data.X.mean(
                        axis=0
                    )
                ).ravel()
            else:
                mean_expression = np.asarray(
                    donor_data.X
                ).mean(
                    axis=0
                )
            donor_expression[
                donor
            ] = mean_expression
            donor_metadata.append(
                {
                    "donor_id_analysis":
                        donor,
                    "age":
                        float(
                            donor_data.obs[
                                AGE_COL
                            ].iloc[0]
                        ),
                    "celltype":
                        celltype,
                    "n_cells":
                        donor_data.n_obs
                }
            )
        if len(
            donor_expression
        ) == 0:

            print(
                "No eligible donors. Skipping."
            )
            continue
        expr_df = pd.DataFrame(
            donor_expression,
            index=sce_ct.var_names
        ).T
        meta_df = pd.DataFrame(
            donor_metadata
        )
        meta_df.index = (
            meta_df[
                "donor_id_analysis"
            ]
        )
        expr_df = expr_df.loc[
            meta_df.index,
            :
        ]
        pseudobulk_results[
            celltype
        ] = {
            "expr":
                expr_df,

            "meta":
                meta_df
        }
        print(
            "Donors:",
            expr_df.shape[0],
            "| Genes:",
            expr_df.shape[1]
        )
    return pseudobulk_results


def calculate_gene_age_correlation(
    pseudobulk_results,
    min_donors=5,
    n_jobs=8
):
    """
    Calculate Spearman correlation between donor-level expression and age.
    FDR correction is performed separately within each cell type.
    """
    all_results = []
    for celltype in pseudobulk_results:
        print(
            "\nAge correlation:",
            celltype
        )
        expr_df = pseudobulk_results[
            celltype
        ]["expr"]
        meta_df = pseudobulk_results[
            celltype
        ]["meta"]
        age = meta_df.loc[
            expr_df.index,
            "age"
        ].values
        if len(
            age
        ) < min_donors:
            print(
                "Too few donors. Skipping."
            )
            continue
        def correlate_gene(gene):
            expression = (
                expr_df[
                    gene
                ]
                .values
            )
            rho, p_value = spearmanr(
                age,
                expression
            )
            return (
                gene,
                rho,
                p_value
            )
        gene_results = Parallel(
            n_jobs=n_jobs
        )(
            delayed(
                correlate_gene
            )(
                gene
            )
            for gene in expr_df.columns
        )
        result_df = pd.DataFrame(
            gene_results,
            columns=[
                "gene",
                "spearman_rho",
                "p_value"
            ]
        )
        valid_p = result_df[
            "p_value"
        ].notna()
        result_df[
            "fdr"
        ] = np.nan
        if valid_p.sum() > 0:
            result_df.loc[
                valid_p,
                "fdr"
            ] = multipletests(
                result_df.loc[
                    valid_p,
                    "p_value"
                ],
                method="fdr_bh"
            )[1]
        result_df[
            "celltype"
        ] = celltype
        result_df[
            "n_donors"
        ] = len(
            age
        )
        all_results.append(
            result_df
        )
    return pd.concat(
        all_results,
        axis=0,
        ignore_index=True
    )


def fit_one_spline(
    gene,
    celltype,
    expression,
    age,
    age_grid
):
    """
    Fit one cubic smoothing spline and return the standardized trajectory.
    """
    valid = (
        np.isfinite(
            expression
        )
        & np.isfinite(
            age
        )
    )
    x = age[
        valid
    ]
    y = expression[
        valid
    ]
    if len(
        x
    ) < MIN_DONORS_PER_TRAJECTORY:
        return None
    if np.std(
        y
    ) < 1e-5:
        return None
    order = np.argsort(
        x
    )
    x = x[
        order
    ]
    y = y[
        order
    ]
    # Average donors with exactly the same age before spline fitting
    temp = pd.DataFrame(
        {
            "age":
                x,
            "expression":
                y
        }
    )
    temp = (
        temp.groupby(
            "age"
        )[
            "expression"
        ]
        .mean()
        .reset_index()
    )
    x = temp[
        "age"
    ].values
    y = temp[
        "expression"
    ].values
    if len(
        x
    ) < 5:
        return None
    smooth_factor = (
        len(
            x
        )
        * np.var(
            y
        )
        * SPLINE_SMOOTHING_SCALE
    )
    try:
        spline = UnivariateSpline(
            x,
            y,
            k=3,
            s=smooth_factor
        )
        fitted = spline(
            age_grid
        )
    except Exception as error:

        print(
            "Spline failed:",
            gene,
            celltype,
            error
        )
        return None
    if (
        np.any(
            ~np.isfinite(
                fitted
            )
        )
        or np.std(
            fitted
        ) < 1e-8
    ):
        return None
    fitted_z = zscore(
        fitted
    )
    rho, p_value = spearmanr(
        age_grid,
        fitted_z
    )
    return {
        "gene":
            gene,
        "celltype":
            celltype,
        "feature_id":
            "{}__{}".format(
                gene,
                celltype
            ),
        "trajectory":
            fitted_z,
        "trajectory_spearman_rho":
            rho,
        "trajectory_spearman_p":
            p_value
    }


def plot_cluster_trajectories(
    age_grid,
    trajectory_matrix,
    cluster_ids,
    cluster_name_map,
    output_pdf
):
    """
    Plot the median trajectory and interquartile range for each cluster.
    """
    n_clusters = len(
        np.unique(
            cluster_ids
        )
    )
    n_cols = 3
    n_rows = int(
        np.ceil(
            n_clusters / n_cols
        )
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            4 * n_cols,
            3.5 * n_rows
        ),
        sharex=True
    )
    axes = np.asarray(
        axes
    ).ravel()
    for cluster_id in range(
        n_clusters
    ):
        ax = axes[
            cluster_id
        ]
        cluster_matrix = trajectory_matrix[
            cluster_ids
            == cluster_id,
            :
        ]
        if cluster_matrix.shape[0] == 0:
            ax.set_visible(
                False
            )
            continue
        median = np.median(
            cluster_matrix,
            axis=0
        )
        q25 = np.percentile(
            cluster_matrix,
            25,
            axis=0
        )
        q75 = np.percentile(
            cluster_matrix,
            75,
            axis=0
        )
        rho, p_value = spearmanr(
            age_grid,
            median
        )
        ax.plot(
            age_grid,
            median,
            linewidth=2
        )
        ax.fill_between(
            age_grid,
            q25,
            q75,
            alpha=0.30
        )
        cluster_name = cluster_name_map.get(
            cluster_id,
            "Cluster {}".format(
                cluster_id
            )
        )
        ax.set_title(
            "{}\n(n={}, r={:.2f})".format(
                cluster_name,
                cluster_matrix.shape[0],
                rho
            ),
            fontsize=10
        )
        ax.set_xlabel(
            "Age (Years)"
        )
        ax.set_ylabel(
            "Expression Z-score"
        )
        sns.despine(
            ax=ax
        )
    for i in range(
        n_clusters,
        len(
            axes
        )
    ):
        axes[
            i
        ].set_visible(
            False
        )
    plt.tight_layout()
    plt.savefig(
        output_pdf,
        bbox_inches="tight",
        dpi=600
    )
    plt.show()


# ============================================================================
# 4. LOAD TRAINING AND INDEPENDENT VALIDATION DATA
# ============================================================================

print("\n" + "=" * 70)
print("Loading training and validation data")
print("=" * 70)

sce_train = sc.read_h5ad(
    TRAIN_H5AD
)

sce_validation = sc.read_h5ad(
    VALIDATION_H5AD
)

if STATUS_COL in sce_train.obs.columns:
    sce_train = sce_train[
        sce_train.obs[
            STATUS_COL
        ] == CONTROL_LABEL,
        :
    ].copy()

if STATUS_COL in sce_validation.obs.columns:
    sce_validation = sce_validation[
        sce_validation.obs[
            STATUS_COL
        ] == CONTROL_LABEL,
        :
    ].copy()

sce_train = add_analysis_donor_id(
    sce_train
)

sce_validation = add_analysis_donor_id(
    sce_validation
)

print(
    "Training shape:",
    sce_train.shape
)

print(
    "Validation shape:",
    sce_validation.shape
)


# ============================================================================
# 5. BUILD TRAINING PSEUDOBULK FOR AGE-CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("Training dataset: donor-level pseudobulk")
print("=" * 70)

training_pseudobulk = build_pseudobulk(
    sce_train,
    celltypes=CELLTYPES,
    genes_by_celltype=None,
    min_cells_per_donor=MIN_CELLS_PER_DONOR_CORRELATION
)


# ============================================================================
# 6. IDENTIFY CHRONOLOGICAL AGE-ASSOCIATED GENES IN TRAINING DATA
# ============================================================================

print("\n" + "=" * 70)
print("Training dataset: chronological-age gene correlations")
print("=" * 70)

training_age_correlations = calculate_gene_age_correlation(
    training_pseudobulk,
    min_donors=MIN_DONORS_PER_CELLTYPE,
    n_jobs=N_JOBS
)

training_age_correlations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "chronological_age_gene_correlations_training.csv"
    ),
    index=False
)

print(
    "Training significant gene-cell-type pairs:",
    (
        training_age_correlations[
            "fdr"
        ]
        < DISCOVERY_FDR
    ).sum()
)


# ============================================================================
# 7. BUILD VALIDATION PSEUDOBULK
# ============================================================================

print("\n" + "=" * 70)
print("Independent validation: donor-level pseudobulk")
print("=" * 70)

validation_pseudobulk = build_pseudobulk(
    sce_validation,
    celltypes=CELLTYPES,
    genes_by_celltype=None,
    min_cells_per_donor=MIN_CELLS_PER_DONOR_CORRELATION
)


# ============================================================================
# 8. REPLICATE AGE-ASSOCIATED GENES IN INDEPENDENT VALIDATION DATA
# ============================================================================

print("\n" + "=" * 70)
print("Independent validation: chronological-age gene correlations")
print("=" * 70)

validation_age_correlations = calculate_gene_age_correlation(
    validation_pseudobulk,
    min_donors=MIN_DONORS_PER_CELLTYPE,
    n_jobs=N_JOBS
)

validation_age_correlations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "chronological_age_gene_correlations_independent_validation.csv"
    ),
    index=False
)


# ============================================================================
# 9. IDENTIFY REPRODUCIBLE AGE-ASSOCIATED GENE-CELL-TYPE PAIRS
# ============================================================================

print("\n" + "=" * 70)
print("Selecting reproducible chronological age-associated genes")
print("=" * 70)

age_gene_pairs = pd.merge(
    training_age_correlations,
    validation_age_correlations[
        [
            "gene",
            "celltype",
            "spearman_rho",
            "p_value"
        ]
    ],
    on=[
        "gene",
        "celltype"
    ],
    how="inner",
    suffixes=(
        "_training",
        "_validation"
    )
)

age_gene_pairs[
    "same_direction"
] = (
    np.sign(
        age_gene_pairs[
            "spearman_rho_training"
        ]
    )
    ==
    np.sign(
        age_gene_pairs[
            "spearman_rho_validation"
        ]
    )
)

reproducible_age_genes = age_gene_pairs.loc[
    (
        age_gene_pairs[
            "fdr"
        ]
        < DISCOVERY_FDR
    )
    &
    (
        age_gene_pairs[
            "p_value_validation"
        ]
        < VALIDATION_P
    )
    &
    (
        age_gene_pairs[
            "same_direction"
        ]
    ),
    :
].copy()

reproducible_age_genes[
    "direction"
] = np.where(
    reproducible_age_genes[
        "spearman_rho_training"
    ]
    > 0,
    "Age-associated upregulated",
    "Age-associated downregulated"
)

reproducible_age_genes.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reproducible_chronological_age_associated_genes.csv"
    ),
    index=False
)

print(
    "Reproducible gene-cell-type pairs:",
    reproducible_age_genes.shape[0]
)

print(
    "Unique genes:",
    reproducible_age_genes[
        "gene"
    ].nunique()
)

print(
    "\nPairs by cell type:"
)

print(
    reproducible_age_genes[
        "celltype"
    ].value_counts()
)


# ============================================================================
# 10. REBUILD TRAINING PSEUDOBULK FOR LIFESPAN TRAJECTORIES
# ============================================================================

print("\n" + "=" * 70)
print("Building pseudobulk for lifespan trajectories")
print("=" * 70)

genes_by_celltype = {}

for celltype in CELLTYPES:
    genes_by_celltype[
        celltype
    ] = (
        reproducible_age_genes.loc[
            reproducible_age_genes[
                "celltype"
            ]
            == celltype,
            "gene"
        ]
        .unique()
        .tolist()
    )

trajectory_pseudobulk = build_pseudobulk(
    sce_train,
    celltypes=CELLTYPES,
    genes_by_celltype=genes_by_celltype,
    min_cells_per_donor=MIN_CELLS_PER_DONOR_TRAJECTORY
)


# ============================================================================
# 11. FIT CUBIC SMOOTHING SPLINES ACROSS THE LIFESPAN
# ============================================================================

print("\n" + "=" * 70)
print("Reconstructing lifespan expression trajectories")
print("=" * 70)

all_training_ages = []

for celltype in trajectory_pseudobulk:
    all_training_ages.extend(
        trajectory_pseudobulk[
            celltype
        ][
            "meta"
        ][
            "age"
        ].tolist()
    )

all_training_ages = np.asarray(
    all_training_ages
)

age_grid = np.linspace(
    np.min(
        all_training_ages
    ),
    np.max(
        all_training_ages
    ),
    N_AGE_POINTS
)

trajectory_results = []

for celltype in trajectory_pseudobulk:
    print(
        "\nFitting trajectories:",
        celltype
    )
    expr_df = trajectory_pseudobulk[
        celltype
    ][
        "expr"
    ]
    meta_df = trajectory_pseudobulk[
        celltype
    ][
        "meta"
    ]
    age = meta_df.loc[
        expr_df.index,
        "age"
    ].values
    celltype_results = Parallel(
        n_jobs=N_JOBS
    )(
        delayed(
            fit_one_spline
        )(
            gene,
            celltype,
            expr_df[
                gene
            ].values,
            age,
            age_grid
        )
        for gene in expr_df.columns
    )
    celltype_results = [
        result
        for result in celltype_results
        if result is not None
    ]
    trajectory_results.extend(
        celltype_results
    )
print(
    "Valid fitted trajectories:",
    len(
        trajectory_results
    )
)


# ============================================================================
# 12. BUILD AND SAVE STANDARDIZED TRAJECTORY MATRIX
# ============================================================================

trajectory_matrix = np.vstack(
    [
        result[
            "trajectory"
        ]
        for result in trajectory_results
    ]
)

trajectory_metadata = pd.DataFrame(
    [
        {
            "gene":
                result[
                    "gene"
                ],

            "celltype":
                result[
                    "celltype"
                ],

            "feature_id":
                result[
                    "feature_id"
                ],

            "trajectory_spearman_rho":
                result[
                    "trajectory_spearman_rho"
                ],

            "trajectory_spearman_p":
                result[
                    "trajectory_spearman_p"
                ]
        }
        for result in trajectory_results
    ]
)

trajectory_df = pd.DataFrame(
    trajectory_matrix.T,
    columns=trajectory_metadata[
        "feature_id"
    ]
)

trajectory_df.insert(
    0,
    "age",
    age_grid
)

trajectory_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lifespan_standardized_expression_trajectories.csv"
    ),
    index=False
)

trajectory_metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lifespan_trajectory_metadata.csv"
    ),
    index=False
)

print(
    "Trajectory matrix shape:",
    trajectory_matrix.shape
)


# ============================================================================
# 13. EVALUATE K = 2 TO K = 15 USING THE THREE PRESPECIFIED CRITERIA
# ============================================================================

print("\n" + "=" * 70)
print("Evaluating the number of trajectory clusters")
print("=" * 70)

cluster_metrics = []
max_k = min(
    15,
    trajectory_matrix.shape[0] - 1
)

for k in range(
    2,
    max_k + 1
):
    print(
        "Testing K =",
        k
    )
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=20
    )
    cluster_id = kmeans.fit_predict(
        trajectory_matrix
    )
    silhouette = silhouette_score(
        trajectory_matrix,
        cluster_id
    )
    davies_bouldin = davies_bouldin_score(
        trajectory_matrix,
        cluster_id
    )
    cluster_metrics.append(
        {
            "K": k,
            "Silhouette": silhouette,
            "Davies_Bouldin": davies_bouldin
        }
    )

cluster_metrics = pd.DataFrame(
    cluster_metrics
)

best_silhouette_row = cluster_metrics.loc[
    cluster_metrics["Silhouette"].idxmax()
]
best_db_row = cluster_metrics.loc[
    cluster_metrics["Davies_Bouldin"].idxmin()
]
silhouette_elbow_k = int(SILHOUETTE_ELBOW_K)
if silhouette_elbow_k not in cluster_metrics["K"].values:
    raise ValueError(
        "SILHOUETTE_ELBOW_K must fall within the evaluated K range."
    )

best_silhouette_k = int(
    best_silhouette_row["K"]
)
best_db_k = int(
    best_db_row["K"]
)
best_silhouette_score = float(
    best_silhouette_row["Silhouette"]
)
best_db_score = float(
    best_db_row["Davies_Bouldin"]
)

print(
    "\nHighest silhouette score:     K={} (score={:.3f})".format(
        best_silhouette_k,
        best_silhouette_score
    )
)
print(
    "Lowest Davies-Bouldin:       K={} (score={:.3f})".format(
        best_db_k,
        best_db_score
    )
)
print(
    "Silhouette elbow point:      K={}".format(
        silhouette_elbow_k
    )
)

cluster_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trajectory_clustering_metrics_K2_to_K15.csv"
    ),
    index=False
)

cluster_selection = pd.DataFrame(
    [
        {
            "Criterion": "Highest silhouette score",
            "Selected_K": best_silhouette_k,
            "Score": best_silhouette_score
        },
        {
            "Criterion": "Lowest Davies-Bouldin index",
            "Selected_K": best_db_k,
            "Score": best_db_score
        },
        {
            "Criterion": "Silhouette elbow point",
            "Selected_K": silhouette_elbow_k,
            "Score": np.nan
        }
    ]
)

cluster_selection.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trajectory_cluster_selection_summary.csv"
    ),
    index=False
)

print(
    "\nCluster-selection summary:"
)
print(
    cluster_selection
)


# ============================================================================
# 14. PLOT THE TWO METRIC CURVES AND THREE SELECTION CRITERIA
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8, 3.5)
)

axes[0].plot(
    cluster_metrics["K"],
    cluster_metrics["Silhouette"],
    marker="o"
)
axes[0].scatter(
    best_silhouette_k,
    best_silhouette_score,
    s=60,
    zorder=5
)
axes[0].scatter(
    silhouette_elbow_k,
    cluster_metrics.loc[
        cluster_metrics["K"] == silhouette_elbow_k,
        "Silhouette"
    ].iloc[0],
    s=60,
    zorder=5
)
axes[0].annotate(
    "maximum: K={}".format(best_silhouette_k),
    xy=(best_silhouette_k, best_silhouette_score),
    xytext=(5, 8),
    textcoords="offset points",
    fontsize=8
)
axes[0].annotate(
    "elbow: K={}".format(silhouette_elbow_k),
    xy=(
        silhouette_elbow_k,
        cluster_metrics.loc[
            cluster_metrics["K"] == silhouette_elbow_k,
            "Silhouette"
        ].iloc[0]
    ),
    xytext=(5, -14),
    textcoords="offset points",
    fontsize=8
)
axes[0].set_xlabel(
    "K"
)
axes[0].set_ylabel(
    "Silhouette score"
)

axes[1].plot(
    cluster_metrics["K"],
    cluster_metrics["Davies_Bouldin"],
    marker="o"
)
axes[1].scatter(
    best_db_k,
    best_db_score,
    s=60,
    zorder=5
)
axes[1].annotate(
    "minimum: K={}".format(best_db_k),
    xy=(best_db_k, best_db_score),
    xytext=(5, 8),
    textcoords="offset points",
    fontsize=8
)
axes[1].set_xlabel(
    "K"
)
axes[1].set_ylabel(
    "Davies-Bouldin index"
)

for ax in axes:
    sns.despine(
        ax=ax
    )

plt.tight_layout()
plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "trajectory_clustering_metrics_K2_to_K15.pdf"
    ),
    bbox_inches="tight",
    dpi=600
)
plt.show()


# ============================================================================
# 15. COARSE TWO-CLUSTER DIAGNOSTIC SOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("K = 2 coarse diagnostic trajectory clustering")
print("=" * 70)

kmeans2 = KMeans(
    n_clusters=2,
    random_state=RANDOM_SEED,
    n_init=20
)
cluster2 = kmeans2.fit_predict(
    trajectory_matrix
)
trajectory_metadata[
    "Cluster2_ID"
] = cluster2

cluster2_name_map = {
    0: "Coarse trajectory cluster 1",
    1: "Coarse trajectory cluster 2"
}

plot_cluster_trajectories(
    age_grid,
    trajectory_matrix,
    cluster2,
    cluster2_name_map,
    os.path.join(
        FIGURE_DIR,
        "lifespan_trajectory_clusters_K2_diagnostic.pdf"
    )
)

print(
    "K = 2 is retained only as a coarse diagnostic because it does not resolve "
    "the major temporal patterns required for downstream biological interpretation."
)


# ============================================================================
# 16. PRIMARY FIVE-CLUSTER SOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("K = 5 primary trajectory clustering")
print("=" * 70)

kmeans5 = KMeans(
    n_clusters=5,
    random_state=RANDOM_SEED,
    n_init=20
)
cluster5 = kmeans5.fit_predict(
    trajectory_matrix
)
trajectory_metadata[
    "Cluster5_ID"
] = cluster5

# Biological labels used in the final manuscript.
# Raw K-means cluster numbers are arbitrary, so inspect the median trajectories
# if the input data or feature set is changed.
cluster5_name_map = {
    0: "Progressive age-associated upregulation",
    1: "Biphasic lifespan remodeling",
    2: "Progressive age-associated decline",
    3: "Aging-associated late-life induction",
    4: "Mid-life compensation failure"
}

trajectory_metadata[
    "Cluster5"
] = trajectory_metadata[
    "Cluster5_ID"
].map(
    cluster5_name_map
)

print(
    trajectory_metadata[
        "Cluster5"
    ].value_counts()
)

plot_cluster_trajectories(
    age_grid,
    trajectory_matrix,
    cluster5,
    cluster5_name_map,
    os.path.join(
        FIGURE_DIR,
        "lifespan_trajectory_clusters_K5.pdf"
    )
)


# ============================================================================
# 17. EIGHT-CLUSTER HIGHER-RESOLUTION SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("K = 8 higher-resolution trajectory clustering")
print("=" * 70)

kmeans8 = KMeans(
    n_clusters=8,
    random_state=RANDOM_SEED,
    n_init=20
)
cluster8 = kmeans8.fit_predict(
    trajectory_matrix
)
trajectory_metadata[
    "Cluster8_ID"
] = cluster8

cluster8_name_map = {
    0: "Progressive age-associated upregulation",
    1: "Early maturation-linked decline",
    2: "Developmental shutdown",
    3: "Biphasic lifespan remodeling",
    4: "Mid-life compensation failure",
    5: "Homeostatic lifespan maintenance",
    6: "Post-maturational functional decline",
    7: "Aging-associated late-life induction"
}

trajectory_metadata[
    "Cluster8"
] = trajectory_metadata[
    "Cluster8_ID"
].map(
    cluster8_name_map
)

print(
    trajectory_metadata[
        "Cluster8"
    ].value_counts()
)

plot_cluster_trajectories(
    age_grid,
    trajectory_matrix,
    cluster8,
    cluster8_name_map,
    os.path.join(
        FIGURE_DIR,
        "lifespan_trajectory_clusters_K8.pdf"
    )
)


# ============================================================================
# 18. SAVE FINAL TRAJECTORY CLUSTER ASSIGNMENTS
# ============================================================================

trajectory_metadata.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lifespan_trajectory_cluster_assignments.csv"
    ),
    index=False
)

print(
    "\nSaved trajectory cluster assignments."
)
print(
    "Output directory:",
    OUTPUT_DIR
)


# ============================================================================
# 19. OPTIONAL: INSPECT ONE GENE TRAJECTORY
# ============================================================================

"""
Example:

gene_to_plot = "CDKN2A"
celltype_to_plot = "Inh"

feature_id = "{}__{}".format(
    gene_to_plot,
    celltype_to_plot
)

if feature_id in trajectory_df.columns:

    plt.figure(
        figsize=(5, 4)
    )

    plt.plot(
        trajectory_df["age"],
        trajectory_df[
            feature_id
        ],
        linewidth=2
    )

    plt.xlabel(
        "Age (Years)"
    )

    plt.ylabel(
        "Standardized expression trajectory"
    )

    plt.title(
        "{} - {}".format(
            gene_to_plot,
            celltype_to_plot
        )
    )

    sns.despine()
    plt.tight_layout()
    plt.show()
"""


# ============================================================================
# 20. COMMAND-LINE USAGE
# ============================================================================

"""
Example:

python 03_lifespan_expression_trajectory_analysis.py \
    --training-h5ad /path/to/sce_train_CT.h5ad \
    --validation-h5ad /path/to/sce_test.h5ad \
    --output-dir /path/to/results/Expression_Trajectory \
    --n-jobs 8
"""
