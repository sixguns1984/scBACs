"""
scBAC Model Evaluation from Precomputed Prediction Metadata
===========================================================

This script reproduces the scBAC model-evaluation analyses using the
PRECOMPUTED cell-level metadata tables supplied with the study.


Input metadata files
--------------------
The filenames are intentionally kept exactly the same as in the original code:

1. meta_human_cortex_scrna_atlas_CT_NDDs.csv
2. meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv

Expected prediction columns
---------------------------
elasticnet_Development, clm_Development, transf_Development
elasticnet_Adult,       clm_Adult,       transf_Adult
elasticnet_Full,        clm_Full,        transf_Full       (optional)
Benchmarking                                                (optional)

The script reconstructs:
Ensemble_Development, Ensemble_Adult, Ensemble_Full
elasticnet_adult_deve, clm_adult_deve, transf_adult_deve
Ensemble_adult_deve

Stage-specific routing:
Age <= 18 years -> Development model
Age > 18 years  -> Adult model

Performance:
- Spearman correlation
- mean absolute error (MAE)
- cell level
- donor level using the median predicted age per donor

Python version: 3.9.20
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


# ============================================================================
# 2. PATHS AND ANALYSIS SETTINGS
# ============================================================================
__file__='/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/code/revision1_final_code_and_data/dataset'
PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results_",
    "model_evaluation"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "figures_",
    "model_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

# Same filenames as the original evaluation code
ATLAS_META_FILE = os.path.join(
    DATA_DIR,
    "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
)

EXTERNAL_META_FILE = os.path.join(
    DATA_DIR,
    "meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv"
)

CELL_TYPES = [
    "Exc",
    "Inh",
    "Ast",
    "OPC",
    "Oli",
    "Mic"
]

OTHER_CELL_TYPES = [
    "End",
    "Fib",
    "Per",
    "CAM",
    "T_cell"
]

# Same analysis-group labels as the original metadata
VALIDATION_GROUPS = {
    "Independent multi-study":
        "Integrated_Original",

    "External1":
        "Alisa_et_al_full_life",

    "External2":
        "Frohlich_et_al_adult"
}

CONTROL_LABEL = "CT"
AGE_CUT = 18

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica"
]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sns.set_context(
    "paper",
    font_scale=1.2
)

sns.set_style(
    "ticks"
)


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate scBAC models from the precomputed prediction metadata."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing the two supplied precomputed metadata CSV files."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for evaluation tables and figures."
    )

    args = parser.parse_args()

    DATA_DIR = os.path.abspath(
        args.data_dir
    )

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

    ATLAS_META_FILE = os.path.join(
        DATA_DIR,
        "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
    )

    EXTERNAL_META_FILE = os.path.join(
        DATA_DIR,
        "meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv"
    )


# ============================================================================
# 3. PERFORMANCE FUNCTIONS
# ============================================================================

def make_donor_id(data):
    """
    Construct a study-specific donor ID.
    This prevents donor-ID collisions when several studies are integrated.
    """
    data = data.copy()
    if "dataset" in data.columns:
        data[
            "donor_id_eval"
        ] = (
            data[
                "dataset"
            ].astype(str)
            + "_"
            + data[
                "donor_id"
            ].astype(str)
        )
    else:
        data[
            "donor_id_eval"
        ] = (
            data[
                "donor_id"
            ].astype(str)
        )
    return data


def run_aging_evaluation_stats(
    full_obs,
    cell_types,
    model_cols,
    analysis_groups_map,
    level="Cell",
    min_samples=10,
    age_min=None,
    age_max=None,
    save_prefix="Final"
):
    """
    Calculate MAE and Spearman correlation for each cell type, dataset and model.
    age_min:
        if not None, retain Age_at_death > age_min
    age_max:
        if not None, retain Age_at_death <= age_max
    """
    available_models = [
        model
        for model in model_cols
        if model in full_obs.columns
    ]
    if len(
        available_models
    ) == 0:
        print(
            "None of the requested prediction columns were found."
        )
        return pd.DataFrame()
    working_df = full_obs.loc[
        full_obs[
            "status"
        ] == CONTROL_LABEL,
        :
    ].copy()
    for col in [
        "Age_at_death"
    ] + available_models:
        working_df[
            col
        ] = pd.to_numeric(
            working_df[
                col
            ],
            errors="coerce"
        )
    working_df = make_donor_id(
        working_df
    )
    if level == "Donor":
        group_cols = [
            "donor_id_eval",
            "celltype",
            "analysis_group",
            "Age_at_death"
        ]
        if "dataset" in working_df.columns:
            group_cols.append(
                "dataset"
            )
        working_df = (
            working_df.groupby(
                group_cols,
                observed=True
            )[
                available_models
            ]
            .median()
            .reset_index()
        )
        print(
            "Donor-level aggregation complete. N =",
            len(
                working_df
            )
        )
    else:
        print(
            "Cell-level analysis. n =",
            len(
                working_df
            )
        )
    evaluation_results = []
    for celltype in cell_types:
        celltype_data = working_df.loc[
            working_df[
                "celltype"
            ] == celltype,
            :
        ].copy()
        for dataset_label, analysis_group in analysis_groups_map.items():
            data = celltype_data.loc[
                celltype_data[
                    "analysis_group"
                ] == analysis_group,
                :
            ].copy()
            if age_min is not None:
                data = data.loc[
                    data[
                        "Age_at_death"
                    ] > age_min,
                    :
                ].copy()
            if age_max is not None:
                data = data.loc[
                    data[
                        "Age_at_death"
                    ] <= age_max,
                    :
                ].copy()
            if data.empty:
                continue
            for model in available_models:
                valid = data.dropna(
                    subset=[
                        "Age_at_death",
                        model
                    ]
                )
                if len(
                    valid
                ) < min_samples:
                    continue
                mae = mean_absolute_error(
                    valid[
                        "Age_at_death"
                    ],
                    valid[
                        model
                    ]
                )
                rho, p_value = spearmanr(
                    valid[
                        "Age_at_death"
                    ],
                    valid[
                        model
                    ]
                )
                evaluation_results.append(
                    {
                        "CellType":
                            celltype,
                        "Dataset":
                            dataset_label,
                        "Analysis_Group":
                            analysis_group,
                        "Model":
                            model,
                        "Level":
                            level,
                        "MAE":
                            mae,
                        "Spearman_R":
                            rho,
                        "Spearman_P":
                            p_value,
                        "N":
                            len(
                                valid
                            )
                    }
                )
    results = pd.DataFrame(
        evaluation_results
    )
    if results.empty:
        print(
            "No metrics calculated. Check filters and prediction columns."
        )
        return results
    output_file = "{}_{}_Evaluation_Metrics.csv".format(
        save_prefix,
        level
    )
    results.to_csv(
        output_file,
        index=False
    )
    print(
        "Saved:",
        output_file
    )
    return results


def plot_aging_evaluation_combined(
    df_metrics,
    level="Cell",
    available_models=None,
    save_prefix="Final"
):
    """
    Plot Spearman correlation and MAE across cell types and validation datasets.
    """
    if (
        df_metrics is None
        or df_metrics.empty
    ):
        return
    if available_models is None:
        available_models = (
            df_metrics[
                "Model"
            ]
            .drop_duplicates()
            .tolist()
        )
    model_rename_map = {
        "Benchmarking":
            "Benchmarking",
        "Ensemble_Adult":
            "Adult scBAC",
        "Ensemble_Full":
            "Unified full scBAC",
        "Ensemble_adult_deve":
            "Stage-specific scBAC",
        "elasticnet_adult_deve":
            "Elastic Net",
        "clm_adult_deve":
            "CLM",
        "transf_adult_deve":
            "Transformer"
    }
    plot_data = df_metrics.copy()
    plot_data[
        "Model_Display"
    ] = plot_data[
        "Model"
    ].map(
        lambda x:
        model_rename_map.get(
            x,
            x
        )
    )
    datasets = (
        plot_data[
            "Dataset"
        ]
        .drop_duplicates()
        .tolist()
    )
    fig, axes = plt.subplots(
        len(
            datasets
        ),
        2,
        figsize=(
            7.0,
            max(
                1.9
                * len(
                    datasets
                ),
                2.3
            )
        ),
        squeeze=False
    )
    celltype_order = [
        celltype
        for celltype in CELL_TYPES
        if celltype in plot_data[
            "CellType"
        ].unique()
    ]
    hue_order = [
        model_rename_map.get(
            model,
            model
        )
        for model in available_models
        if model in plot_data[
            "Model"
        ].unique()
    ]
    nature_colors = [
        "#1F77B4",
        "#AEC7E8",
        "#FF7F0E",
        "#F0E442",
        "#E64B35"
    ]
    for row_idx, dataset in enumerate(
        datasets
    ):
        subset = plot_data.loc[
            plot_data[
                "Dataset"
            ] == dataset,
            :
        ]
        ax_r = axes[
            row_idx,
            0
        ]
        sns.barplot(
            data=subset,
            x="CellType",
            y="Spearman_R",
            hue="Model_Display",
            order=celltype_order,
            hue_order=hue_order,
            palette=nature_colors[
                :len(
                    hue_order
                )
            ],
            edgecolor="black",
            linewidth=0.3,
            ax=ax_r
        )
        ax_r.set_ylim(
            0,
            1.0
        )
        ax_r.set_ylabel(
            "Spearman R"
        )
        ax_r.set_title(
            "{} ({} level)".format(
                dataset,
                level
            )
        )
        if row_idx == 0:
            ax_r.legend(
                frameon=False,
                fontsize=5
            )
        elif ax_r.get_legend() is not None:
            ax_r.get_legend().remove()
        sns.despine(
            ax=ax_r
        )
        ax_mae = axes[
            row_idx,
            1
        ]
        sns.barplot(
            data=subset,
            x="CellType",
            y="MAE",
            hue="Model_Display",
            order=celltype_order,
            hue_order=hue_order,
            palette=nature_colors[
                :len(
                    hue_order
                )
            ],
            edgecolor="black",
            linewidth=0.3,
            ax=ax_mae
        )
        ax_mae.set_ylim(
            0,
            25
        )
        ax_mae.set_ylabel(
            "MAE (years)"
        )
        ax_mae.set_title(
            "{} ({} level)".format(
                dataset,
                level
            )
        )
        if ax_mae.get_legend() is not None:
            ax_mae.get_legend().remove()
        sns.despine(
            ax=ax_mae
        )
        for ax in [
            ax_r,
            ax_mae
        ]:
            ax.set_xlabel(
                ""
            )
            ax.tick_params(
                axis="x",
                rotation=45
            )
    plt.tight_layout()
    output_file = "{}_{}_Combined_Evaluation.pdf".format(
        save_prefix,
        level
    )
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )
    print(
        "Saved:",
        output_file
    )
    plt.show()



def plot_celltype_age_comparison(
    df,
    celltypes,
    output_path,
    Age_col,
    cellage_col,
    title_suffix=""
):
    """
    Plot predicted cellular age versus chronological age.
    Cell-level density is shown with hexbin.
    Red points represent donor-median predicted cellular ages.
    """
    fixed_cols = 6
    total_width_inches = 220 / 25.4
    font_size = 5
    plt.style.use(
        "default"
    )
    data = make_donor_id(
        df
    )
    n_celltypes = len(
        celltypes
    )
    n_rows = (
        n_celltypes
        + fixed_cols
        - 1
    ) // fixed_cols
    single_plot_size = (
        total_width_inches
        / fixed_cols
    )
    fig, axes = plt.subplots(
        n_rows,
        fixed_cols,
        figsize=(
            total_width_inches,
            single_plot_size
            * n_rows
        ),
        squeeze=False
    )
    axes = axes.flatten()
    for idx in range(
        n_rows
        * fixed_cols
    ):
        if idx >= n_celltypes:
            fig.delaxes(
                axes[
                    idx
                ]
            )
            continue
        celltype = celltypes[
            idx
        ]
        celltype_data = data.loc[
            data[
                "celltype"
            ] == celltype,
            :
        ].copy()
        celltype_data = celltype_data.dropna(
            subset=[
                Age_col,
                cellage_col
            ]
        )
        ax = axes[
            idx
        ]
        if celltype_data.empty:
            ax.set_visible(
                False
            )
            continue
        n_cells = len(
            celltype_data
        )
        n_donors = celltype_data[
            "donor_id_eval"
        ].nunique()
        median_ages = (
            celltype_data.groupby(
                "donor_id_eval",
                observed=True
            )
            .agg(
                {
                    cellage_col:
                        "median",
                    Age_col:
                        "first"
                }
            )
            .reset_index()
        )
        hb = ax.hexbin(
            celltype_data[
                Age_col
            ],
            celltype_data[
                cellage_col
            ],
            gridsize=15,
            cmap="Blues",
            alpha=0.6,
            mincnt=1,
            linewidths=0.2,
            extent=[
                0,
                100,
                0,
                100
            ]
        )
        ax.scatter(
            median_ages[
                Age_col
            ],
            median_ages[
                cellage_col
            ],
            color="#E41A1C",
            s=15,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.4,
            zorder=5
        )
        if len(
            median_ages
        ) > 1:
            slope, intercept, _, _, _ = stats.linregress(
                median_ages[
                    Age_col
                ],
                median_ages[
                    cellage_col
                ]
            )
            x_reg = np.linspace(
                0,
                100,
                100
            )
            ax.plot(
                x_reg,
                slope
                * x_reg
                + intercept,
                color="#555555",
                linewidth=1
            )
        fig.colorbar(
            hb,
            ax=ax,
            shrink=0.4,
            aspect=15
        )
        rho_cell, _ = spearmanr(
            celltype_data[
                Age_col
            ],
            celltype_data[
                cellage_col
            ]
        )
        mae_cell = mean_absolute_error(
            celltype_data[
                Age_col
            ],
            celltype_data[
                cellage_col
            ]
        )
        rho_donor, _ = spearmanr(
            median_ages[
                Age_col
            ],
            median_ages[
                cellage_col
            ]
        )
        mae_donor = mean_absolute_error(
            median_ages[
                Age_col
            ],
            median_ages[
                cellage_col
            ]
        )
        stats_text = (
            "$\\rho_{{cell}}$ = {:.2f}\n"
            "MAE$_{{cell}}$ = {:.2f}\n"
            "$\\rho_{{donor}}$ = {:.2f}\n"
            "MAE$_{{donor}}$ = {:.2f}"
        ).format(
            rho_cell,
            mae_cell,
            rho_donor,
            mae_donor
        )
        ax.text(
            0.95,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=font_size,
            verticalalignment="bottom",
            horizontalalignment="right"
        )
        ax.set_title(
            "{}{}\n(N = {}, n = {})".format(
                celltype,
                title_suffix,
                n_donors,
                n_cells
            ),
            fontweight="bold"
        )
        ax.set_xlabel(
            "Chronological Age"
        )
        if idx % fixed_cols == 0:
            ax.set_ylabel(
                "Predicted Cell Age"
            )
        else:
            ax.set_ylabel(
                ""
            )
        ax.set_xlim(
            0,
            100
        )
        ax.set_ylim(
            0,
            100
        )
        ax.set_xticks(
            [
                0,
                25,
                50,
                75,
                100
            ]
        )
        ax.set_yticks(
            [
                0,
                25,
                50,
                75,
                100
            ]
        )
        ax.set_aspect(
            "equal"
        )
        ax.spines[
            "top"
        ].set_visible(
            False
        )
        ax.spines[
            "right"
        ].set_visible(
            False
        )
    plt.tight_layout(
        pad=0.4,
        h_pad=0.8,
        w_pad=0.8
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    print(
        "Saved:",
        output_path
    )
    plt.show()
    return fig



# ============================================================================
# 4. LOAD PRECOMPUTED PREDICTION METADATA
# ============================================================================

print("\n" + "=" * 70)
print("Loading precomputed scBAC prediction metadata")
print("=" * 70)

atlas_meta = pd.read_csv(
    ATLAS_META_FILE,
    index_col=0
)

external_meta = pd.read_csv(
    EXTERNAL_META_FILE,
    index_col=0
)

print(
    "Atlas metadata:",
    atlas_meta.shape
)

print(
    "External validation metadata:",
    external_meta.shape
)

full_obs = pd.concat(
    [
        atlas_meta,
        external_meta
    ],
    axis=0
)

print(
    "Combined metadata:",
    full_obs.shape
)

print(
    "\nAvailable analysis groups:"
)

print(
    full_obs[
        "analysis_group"
    ].value_counts()
)


# ============================================================================
# 5. RECONSTRUCT ENSEMBLE AND STAGE-SPECIFIC PREDICTIONS
# ============================================================================

print("\n" + "=" * 70)
print("Reconstructing ensemble predictions")
print("=" * 70)

adult_components = [
    "elasticnet_Adult",
    "clm_Adult",
    "transf_Adult"
]

if all(
    col in full_obs.columns
    for col in adult_components
):
    full_obs[
        "Ensemble_Adult"
    ] = full_obs[
        adult_components
    ].mean(
        axis=1
    )

development_components = [
    "elasticnet_Development",
    "clm_Development",
    "transf_Development"
]

if all(
    col in full_obs.columns
    for col in development_components
):
    full_obs[
        "Ensemble_Development"
    ] = full_obs[
        development_components
    ].mean(
        axis=1
    )

full_components = [
    "elasticnet_Full",
    "clm_Full",
    "transf_Full"
]

if all(
    col in full_obs.columns
    for col in full_components
):
    full_obs[
        "Ensemble_Full"
    ] = full_obs[
        full_components
    ].mean(
        axis=1
    )

for model in [
    "elasticnet",
    "clm",
    "transf"
]:
    adult_col = "{}_Adult".format(
        model
    )
    development_col = "{}_Development".format(
        model
    )
    output_col = "{}_adult_deve".format(
        model
    )
    if (
        adult_col in full_obs.columns
        and development_col in full_obs.columns
    ):
        full_obs[
            output_col
        ] = full_obs[
            adult_col
        ].values
        full_obs.loc[
            full_obs[
                "Age_at_death"
            ] <= AGE_CUT,
            output_col
        ] = full_obs.loc[
            full_obs[
                "Age_at_death"
            ] <= AGE_CUT,
            development_col
        ].values

if (
    "Ensemble_Adult" in full_obs.columns
    and "Ensemble_Development" in full_obs.columns
):
    full_obs[
        "Ensemble_adult_deve"
    ] = full_obs[
        "Ensemble_Adult"
    ].values
    full_obs.loc[
        full_obs[
            "Age_at_death"
        ] <= AGE_CUT,
        "Ensemble_adult_deve"
    ] = full_obs.loc[
        full_obs[
            "Age_at_death"
        ] <= AGE_CUT,
        "Ensemble_Development"
    ].values

print(
    "\nPrediction columns available:"
)

prediction_columns = [
    "Benchmarking",
    "elasticnet_Development",
    "clm_Development",
    "transf_Development",
    "Ensemble_Development",
    "elasticnet_Adult",
    "clm_Adult",
    "transf_Adult",
    "Ensemble_Adult",
    "elasticnet_Full",
    "clm_Full",
    "transf_Full",
    "Ensemble_Full",
    "elasticnet_adult_deve",
    "clm_adult_deve",
    "transf_adult_deve",
    "Ensemble_adult_deve"
]

print(
    [
        col
        for col in prediction_columns
        if col in full_obs.columns
    ]
)


# ============================================================================
# 6. CHECK AGE RANGE OF EACH DATASET
# ============================================================================

print("\n" + "=" * 70)
print("Age ranges of datasets represented in the metadata")
print("=" * 70)

for dataset in full_obs[
    "dataset"
].dropna().unique():
    dataset_age = pd.to_numeric(
        full_obs.loc[
            full_obs[
                "dataset"
            ] == dataset,
            "Age_at_death"
        ],
        errors="coerce"
    )
    print(
        "{}: {:.1f} - {:.1f}".format(
            dataset,
            dataset_age.min(),
            dataset_age.max()
        )
    )


# ============================================================================
# 7. STAGE-SPECIFIC COMPONENT MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("Stage-specific component model comparison")
print("=" * 70)

stage_specific_models = [
    "elasticnet_adult_deve",
    "clm_adult_deve",
    "transf_adult_deve",
    "Ensemble_adult_deve"
]

stage_specific_cell_metrics = run_aging_evaluation_stats(
    full_obs=full_obs,
    cell_types=CELL_TYPES,
    model_cols=stage_specific_models,
    analysis_groups_map=VALIDATION_GROUPS,
    level="Cell",
    min_samples=10,
    age_min=None,
    age_max=None,
    save_prefix=os.path.join(
        OUTPUT_DIR,
        "Stage_specific_component_comparison"
    )
)

plot_aging_evaluation_combined(
    stage_specific_cell_metrics,
    level="Cell",
    available_models=stage_specific_models,
    save_prefix=os.path.join(
        FIGURE_DIR,
        "Stage_specific_component_comparison"
    )
)

stage_specific_donor_metrics = run_aging_evaluation_stats(
    full_obs=full_obs,
    cell_types=CELL_TYPES,
    model_cols=stage_specific_models,
    analysis_groups_map=VALIDATION_GROUPS,
    level="Donor",
    min_samples=5,
    age_min=None,
    age_max=None,
    save_prefix=os.path.join(
        OUTPUT_DIR,
        "Stage_specific_component_comparison"
    )
)

plot_aging_evaluation_combined(
    stage_specific_donor_metrics,
    level="Donor",
    available_models=stage_specific_models,
    save_prefix=os.path.join(
        FIGURE_DIR,
        "Stage_specific_component_comparison"
    )
)


# ============================================================================
# 8. ADULT SCBAC / BENCHMARK / UNIFIED FULL-MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("Adult validation comparison")
print("=" * 70)

adult_comparison_models = [
    "Benchmarking",
    "Ensemble_Adult",
    "Ensemble_Full"
]

adult_comparison_models = [
    model
    for model in adult_comparison_models
    if model in full_obs.columns
]

if len(
    adult_comparison_models
) > 0:
    adult_cell_metrics = run_aging_evaluation_stats(
        full_obs=full_obs,
        cell_types=CELL_TYPES,
        model_cols=adult_comparison_models,
        analysis_groups_map=VALIDATION_GROUPS,
        level="Cell",
        min_samples=10,
        age_min=18,
        age_max=None,
        save_prefix=os.path.join(
            OUTPUT_DIR,
            "Adult_scBAC_benchmark_full_comparison"
        )
    )
    plot_aging_evaluation_combined(
        adult_cell_metrics,
        level="Cell",
        available_models=adult_comparison_models,
        save_prefix=os.path.join(
            FIGURE_DIR,
            "Adult_scBAC_benchmark_full_comparison"
        )
    )
    adult_donor_metrics = run_aging_evaluation_stats(
        full_obs=full_obs,
        cell_types=CELL_TYPES,
        model_cols=adult_comparison_models,
        analysis_groups_map=VALIDATION_GROUPS,
        level="Donor",
        min_samples=5,
        age_min=18,
        age_max=None,
        save_prefix=os.path.join(
            OUTPUT_DIR,
            "Adult_scBAC_benchmark_full_comparison"
        )
    )
    plot_aging_evaluation_combined(
        adult_donor_metrics,
        level="Donor",
        available_models=adult_comparison_models,
        save_prefix=os.path.join(
            FIGURE_DIR,
            "Adult_scBAC_benchmark_full_comparison"
        )
    )


# ============================================================================
# 9. DEVELOPMENTAL VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("Developmental validation")
print("=" * 70)

development_groups = {
    "External1":
        "Alisa_et_al_full_life"
}

development_models = [
    "Benchmarking",
    "Ensemble_Development",
    "Ensemble_adult_deve"
]

development_models = [
    model
    for model in development_models
    if model in full_obs.columns
]

if len(
    development_models
) > 0:
    development_cell_metrics = run_aging_evaluation_stats(
        full_obs=full_obs,
        cell_types=CELL_TYPES,
        model_cols=development_models,
        analysis_groups_map=development_groups,
        level="Cell",
        min_samples=10,
        age_min=None,
        age_max=18,
        save_prefix=os.path.join(
            OUTPUT_DIR,
            "Developmental_validation"
        )
    )
    development_donor_metrics = run_aging_evaluation_stats(
        full_obs=full_obs,
        cell_types=CELL_TYPES,
        model_cols=development_models,
        analysis_groups_map=development_groups,
        level="Donor",
        min_samples=3,
        age_min=None,
        age_max=18,
        save_prefix=os.path.join(
            OUTPUT_DIR,
            "Developmental_validation"
        )
    )


# ============================================================================
# 10. PREDICTED AGE VERSUS CHRONOLOGICAL AGE PLOTS
# ============================================================================

print("\n" + "=" * 70)
print("Predicted age versus chronological age plots")
print("=" * 70)

if "Ensemble_adult_deve" in full_obs.columns:
    for dataset_label, analysis_group in VALIDATION_GROUPS.items():
        df_plot = full_obs.loc[
            (
                full_obs[
                    "analysis_group"
                ] == analysis_group
            )
            &
            (
                full_obs[
                    "status"
                ] == CONTROL_LABEL
            ),
            :
        ].copy()
        if df_plot.empty:
            continue
        plot_celltype_age_comparison(
            df=df_plot,
            celltypes=CELL_TYPES,
            Age_col="Age_at_death",
            cellage_col="Ensemble_adult_deve",
            output_path=os.path.join(
                FIGURE_DIR,
                "scBAC_{}_validation_main_celltypes_adult_deve.pdf".format(
                    dataset_label
                )
            ),
            title_suffix=""
        )


# ============================================================================
# 11. OTHER CELL TYPES IN THE INDEPENDENT MULTI-STUDY VALIDATION SET
# ============================================================================

if "Ensemble_adult_deve" in full_obs.columns:

    df_internal = full_obs.loc[
        (
            full_obs[
                "analysis_group"
            ] == "Integrated_Original"
        )
        &
        (
            full_obs[
                "status"
            ] == CONTROL_LABEL
        ),
        :
    ].copy()

    available_other_celltypes = [
        celltype
        for celltype in OTHER_CELL_TYPES
        if celltype in df_internal[
            "celltype"
        ].unique()
    ]

    if len(
        available_other_celltypes
    ) > 0:

        plot_celltype_age_comparison(
            df=df_internal,
            celltypes=available_other_celltypes,
            Age_col="Age_at_death",
            cellage_col="Ensemble_adult_deve",
            output_path=os.path.join(
                FIGURE_DIR,
                "scBAC_Internal_validation_other_celltypes_adult_deve.pdf"
            ),
            title_suffix=""
        )


# ============================================================================
# 12. CORRELATION BETWEEN ELASTIC NET, CLM AND TRANSFORMER
# ============================================================================

print("\n" + "=" * 70)
print("Correlation between component model predictions")
print("=" * 70)

component_cols = [
    "clm_adult_deve",
    "elasticnet_adult_deve",
    "transf_adult_deve"
]

if all(
    col in full_obs.columns
    for col in component_cols
):

    component_data = full_obs.loc[
        (
            full_obs[
                "celltype"
            ].isin(
                CELL_TYPES
            )
        )
        &
        (
            full_obs[
                "status"
            ] == CONTROL_LABEL
        )
        &
        (
            full_obs[
                "analysis_group"
            ] == "Integrated_Original"
        ),
        component_cols
    ].copy()

    component_correlation = component_data.corr(
        method="spearman"
    )

    component_correlation.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clm_elasticnet_transformer_stage_specific_spearman_correlation.csv"
        )
    )

    plt.figure(
        figsize=(5.2, 4.5),
        dpi=300
    )

    sns.heatmap(
        component_correlation,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        square=True,
        linewidths=1,
        cbar_kws={
            "label":
                "Spearman correlation"
        }
    )

    plt.title(
        "Correlation between model predictions"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "clm_transf_elastic_adult_deve_corr_in_Integrated_Original.pdf"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================================
# 13. SAVE COMBINED METADATA USED FOR EVALUATION
# ============================================================================

full_obs.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "combined_precomputed_prediction_metadata_used_for_evaluation.csv.gz"
    ),
    compression="gzip"
)

print(
    "\nModel evaluation completed."
)


# ============================================================================
# 14. COMMAND-LINE USAGE
# ============================================================================

"""
The following two files must be present in --data-dir:

meta_human_cortex_scrna_atlas_CT_NDDs.csv
meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv


Example:

python 06_scBAC_model_evaluation.py \
    --data-dir /path/to/revision1_final_code_and_data/dataset \
    --output-dir /path/to/revision1_final_code_and_data/results/model_evaluation
"""
