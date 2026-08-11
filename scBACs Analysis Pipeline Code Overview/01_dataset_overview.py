"""
scBACs Dataset Overview and Data Distribution
=============================================

Purpose:
- Visualize donor age distributions in the training and validation datasets
- Compare the original and revised training datasets
- Summarize diagnosis, age, and sex distributions in the integrated human cortex atlas

Python version:
- Python 3.9.20

Usage:
- Edit the paths in Section 2 and run the script section by section
- Command-line execution is also supported; examples are provided at the end
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


# ============================================================================
# 2. PATHS AND BASIC SETTINGS
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures", "Data_distribution")

os.makedirs(FIGURE_DIR, exist_ok=True)

ATLAS_META_FILE = os.path.join(
    DATA_DIR,
    "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
)

ALS_FTLD_META_FILE = os.path.join(
    DATA_DIR,
    "meta_ALS_FTLD.csv"
)


EXTERNAL_META_FILE = os.path.join(
    DATA_DIR,
    "meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv"
)

NEW_TRAINING_DATASET = "GSE291605"
AGE_BIN_SIZE = 10

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Generate dataset overview figures for scBACs."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Project directory containing dataset/ and figures/."
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=10,
        help="Age-bin width in years."
    )
    args = parser.parse_args()
    PROJECT_ROOT = os.path.abspath(args.project_root)
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures", "Data_distribution")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    ATLAS_META_FILE = os.path.join(
        DATA_DIR,
        "meta_human_cortex_scrna_atlas_CT_NDDs.csv"
    )
    ALS_FTLD_META_FILE = os.path.join(
        DATA_DIR,
        "meta_ALS_FTLD.csv"
    )
    EXTERNAL_META_FILE = os.path.join(
        DATA_DIR,
        "meta_scBACs_external_validation_datasets_Alisa_et_al_Frohlich_et_al.csv"
    )
    AGE_BIN_SIZE = args.bin_size


# ============================================================================
# 3. FUNCTION FOR DONOR AGE DISTRIBUTION
# ============================================================================

def plot_age_distribution(
    data,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=10,
    figsize=(5.0, 3.2),
    color="#3C5488",
    edgecolor="#1A2E5A",
    ymax=None,
    output_pdf=None
):
    """
    Plot donor counts across chronological-age bins.
    If both dataset and donor_id are available, the two columns are used
    together to define unique donors.
    """
    df_plot = data.copy()
    if donor_col is not None and donor_col in df_plot.columns:
        if dataset_col in df_plot.columns:
            df_plot = df_plot.drop_duplicates(
                subset=[dataset_col, donor_col]
            )
        else:
            df_plot = df_plot.drop_duplicates(
                subset=[donor_col]
            )
    df_plot[age_col] = pd.to_numeric(
        df_plot[age_col],
        errors="coerce"
    )
    df_plot = df_plot.dropna(
        subset=[age_col]
    ).copy()
    max_age = df_plot[age_col].max()
    max_age_rounded = int(
        np.ceil(max_age / bin_size) * bin_size
    )
    bins = list(
        range(
            0,
            max_age_rounded + bin_size,
            bin_size
        )
    )
    labels = [
        "{}-{}".format(
            bins[i],
            bins[i + 1]
        )
        for i in range(len(bins) - 1)
    ]
    df_plot["Age_Group"] = pd.cut(
        df_plot[age_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=300
    )
    sns.countplot(
        data=df_plot,
        x="Age_Group",
        order=labels,
        color=color,
        edgecolor=edgecolor,
        linewidth=0.5,
        alpha=0.85,
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            padding=3,
            fontsize=8.5
        )
    ax.set_xlabel(
        "Age Range (Years)",
        fontsize=10
    )
    ax.set_ylabel(
        "Donor Count",
        fontsize=10
    )
    ax.tick_params(
        axis="x",
        labelsize=8.5,
        rotation=45
    )
    ax.tick_params(
        axis="y",
        labelsize=8.5
    )
    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    else:
        current_ymax = ax.get_ylim()[1]
        ax.set_ylim(
            top=current_ymax * 1.15
        )
    sns.despine(
        ax=ax,
        top=True,
        right=True
    )
    plt.tight_layout()
    if output_pdf is not None:
        plt.savefig(
            output_pdf,
            bbox_inches="tight",
            dpi=600,
            transparent=True
        )
        print(
            "Saved:",
            output_pdf
        )
    plt.show()
    return df_plot


# ============================================================================
# 4. LOAD MAIN ATLAS METADATA
# ============================================================================

print("\n" + "=" * 70)
print("Loading atlas metadata")
print("=" * 70)

meta = pd.read_csv(
    ATLAS_META_FILE,
    index_col=0
)

print(
    "Metadata shape:",
    meta.shape
)

print(
    "\nStudies in metadata:"
)

print(
    meta["dataset"].value_counts()
)


# ============================================================================
# 5. ORIGINAL AND REVISED TRAINING DATASETS
# ============================================================================

print("\n" + "=" * 70)
print("Training dataset")
print("=" * 70)

train = meta.loc[
    (meta["analysis_group"] == "Training")
    & (meta["status"] == "CT"),
    :
].copy()

train_old = train.loc[
    train["dataset"] != NEW_TRAINING_DATASET,
    :
].copy()

train_old_donor = train_old.drop_duplicates(
    subset=["dataset", "donor_id"]
).copy()

train_donor = train.drop_duplicates(
    subset=["dataset", "donor_id"]
).copy()

print(
    "Original training studies:",
    train_old["dataset"].nunique()
)

print(
    "Original training donors:",
    train_old_donor.shape[0]
)

print(
    "Revised training studies:",
    train["dataset"].nunique()
)

print(
    "Revised training donors:",
    train_donor.shape[0]
)


# ============================================================================
# 6. ORIGINAL TRAINING AGE DISTRIBUTION
# ============================================================================

train_old_age = plot_age_distribution(
    train_old,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=AGE_BIN_SIZE,
    figsize=(5.0, 3.2),
    color="#3C5488",
    edgecolor="#1A2E5A",
    output_pdf=os.path.join(
        FIGURE_DIR,
        "Training_Donor_Age_Distribution_original_12_studies.pdf"
    )
)


# ============================================================================
# 7. REVISED TRAINING AGE DISTRIBUTION
# ============================================================================

train_age = plot_age_distribution(
    train,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=AGE_BIN_SIZE,
    figsize=(5.0, 3.2),
    color="#3C5488",
    edgecolor="#1A2E5A",
    output_pdf=os.path.join(
        FIGURE_DIR,
        "Training_Donor_Age_Distribution_revised_13_studies.pdf"
    )
)


# ============================================================================
# 8. INDEPENDENT MULTI-STUDY VALIDATION DATASET
# ============================================================================

print("\n" + "=" * 70)
print("Independent multi-study validation dataset")
print("=" * 70)

if "Independent_multi_study_validation" in meta["analysis_group"].values:
    validation_group = "Independent_multi_study_validation"
else:
    validation_group = "Integrated_Original"

val_multi = meta.loc[
    (meta["analysis_group"] == validation_group)
    & (meta["status"] == "CT"),
    :
].copy()

val_multi_donor = val_multi.drop_duplicates(
    subset=["dataset", "donor_id"]
).copy()

print(
    "Validation studies:",
    val_multi["dataset"].nunique()
)

print(
    "Validation donors:",
    val_multi_donor.shape[0]
)

val_multi_age = plot_age_distribution(
    val_multi,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=AGE_BIN_SIZE,
    output_pdf=os.path.join(
        FIGURE_DIR,
        "Independent_multi_study_validation_Donor_Age_Distribution.pdf"
    )
)


# ============================================================================
# 9. LOAD EXTERNAL SINGLE-STUDY VALIDATION METADATA
# ============================================================================

print("\n" + "=" * 70)
print("External single-study validation datasets")
print("=" * 70)

if os.path.exists(EXTERNAL_META_FILE):
    external_meta_file = EXTERNAL_META_FILE
elif os.path.exists(EXTERNAL_META_FILE_LEGACY):
    external_meta_file = EXTERNAL_META_FILE_LEGACY
else:
    raise FileNotFoundError(
        "External validation metadata file was not found."
    )

meta_external = pd.read_csv(
    external_meta_file,
    index_col=0
)

print(
    "External validation metadata shape:",
    meta_external.shape
)


# ============================================================================
# 10. EXTERNAL VALIDATION DATASET 1: JEFFRIES ET AL.
# ============================================================================

if "Jeffries_et_al_full_life" in meta_external["analysis_group"].values:
    jeffries_group = "Jeffries_et_al_full_life"
else:
    jeffries_group = "Alisa_et_al_full_life"

val_jeffries = meta_external.loc[
    (meta_external["analysis_group"] == jeffries_group)
    & (meta_external["status"] == "CT"),
    :
].copy()

val_jeffries_donor = val_jeffries.drop_duplicates(
    subset=["dataset", "donor_id"]
).copy()

print(
    "Jeffries et al. donors:",
    val_jeffries_donor.shape[0]
)

val_jeffries_age = plot_age_distribution(
    val_jeffries,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=AGE_BIN_SIZE,
    output_pdf=os.path.join(
        FIGURE_DIR,
        "Alisa_et_al_lifespan_validation_Donor_Age_Distribution.pdf"
    )
)


# ============================================================================
# 11. EXTERNAL VALIDATION DATASET 2: FROHLICH ET AL.
# ============================================================================

val_frohlich = meta_external.loc[
    (meta_external["analysis_group"] == "Frohlich_et_al_adult")
    & (meta_external["status"] == "CT"),
    :
].copy()

val_frohlich_donor = val_frohlich.drop_duplicates(
    subset=["dataset", "donor_id"]
).copy()

print(
    "Frohlich et al. donors:",
    val_frohlich_donor.shape[0]
)

val_frohlich_age = plot_age_distribution(
    val_frohlich,
    age_col="Age_at_death",
    donor_col="donor_id",
    dataset_col="dataset",
    bin_size=AGE_BIN_SIZE,
    output_pdf=os.path.join(
        FIGURE_DIR,
        "Frohlich_et_al_adult_validation_Donor_Age_Distribution.pdf"
    )
)


# ============================================================================
# 12. INTEGRATED ATLAS DATA DISTRIBUTION
# ============================================================================

print("\n" + "=" * 70)
print("Integrated atlas data distribution")
print("=" * 70)

meta_als_ftld = pd.read_csv(
    ALS_FTLD_META_FILE,
    index_col=0
)

meta_atlas = pd.concat(
    [
        meta,
        meta_als_ftld
    ],
    axis=0
)

if "sub_tissue" in meta_atlas.columns:
    meta_atlas["sub_tissue"] = meta_atlas["sub_tissue"].replace(
        [
            "Midtemporal",
            "Primary-motor-cortex"
        ],
        [
            "Temporal cortex",
            "Primary motor cortex"
        ]
    )

meta_atlas = meta_atlas.loc[
    meta_atlas["status"] != "Dementia",
    :
].copy()

meta_atlas["donor_id2"] = (
    meta_atlas["dataset"].astype(str)
    + "-"
    + meta_atlas["donor_id"].astype(str)
)

meta_donor = meta_atlas.drop_duplicates(
    subset=["donor_id2"]
).copy()

print(
    "\nDonor counts by diagnosis:"
)

print(
    meta_donor["status"].value_counts()
)


# ============================================================================
# 13. DIAGNOSIS AND AGE DISTRIBUTION OF THE INTEGRATED ATLAS
# ============================================================================

palette = {
    "CT": "#4e79a7",
    "MCI": "#f28e2b",
    "AD": "#e15759",
    "PD": "#76b7b2",
    "FTD": "#59a14f",
    "ALS": "#edc948",
    "FTLD": "#b07aa1"
}


def plot_donut(ax, data, label_text):
    counts = data["status"].value_counts()
    colors = [
        palette.get(
            status,
            "#999999"
        )
        for status in counts.index
    ]
    wedges, _ = ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        wedgeprops={
            "width": 0.4,
            "edgecolor": "white"
        }
    )
    for i, wedge in enumerate(wedges):
        angle = (
            wedge.theta2
            - wedge.theta1
        ) / 2.0 + wedge.theta1
        x = np.cos(
            np.deg2rad(angle)
        ) * 0.75
        y = np.sin(
            np.deg2rad(angle)
        ) * 0.75
        ax.text(
            x,
            y,
            "{:,}".format(
                counts.iloc[i]
            ),
            ha="center",
            va="center",
            fontsize=12
        )
        label_x = np.cos(
            np.deg2rad(angle)
        ) * 1.15
        label_y = np.sin(
            np.deg2rad(angle)
        ) * 1.15
        ax.text(
            label_x,
            label_y,
            counts.index[i],
            ha="center",
            va="center",
            fontsize=12
        )
    ax.text(
        -1.4,
        1.2,
        label_text,
        fontsize=18,
        fontweight="bold"
    )


def plot_violin(ax, data):
    order = [
        status
        for status in [
            "CT",
            "MCI",
            "AD",
            "PD",
            "FTD",
            "ALS",
            "FTLD"
        ]
        if status in data["status"].values
    ]
    sns.violinplot(
        data=data,
        x="Age_at_death",
        y="status",
        order=order,
        palette=[
            palette.get(
                status,
                "#999999"
            )
            for status in order
        ],
        orient="h",
        inner="quartile",
        linewidth=1.2,
        cut=0,
        ax=ax
    )
    ax.set_xlabel(
        "Age at Death"
    )
    ax.set_ylabel(
        ""
    )
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )


fig = plt.figure(
    figsize=(14, 12)
)

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[
        1,
        1.2
    ],
    height_ratios=[
        1,
        1
    ],
    hspace=0.3
)

plot_donut(
    fig.add_subplot(
        gs[0, 0]
    ),
    meta_donor,
    "a"
)

plot_violin(
    fig.add_subplot(
        gs[0, 1]
    ),
    meta_donor
)

plot_donut(
    fig.add_subplot(
        gs[1, 0]
    ),
    meta_atlas,
    "e"
)

plot_violin(
    fig.add_subplot(
        gs[1, 1]
    ),
    meta_atlas
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "Integrated_atlas_diagnosis_age_distribution.pdf"
    ),
    bbox_inches="tight",
    dpi=600
)

plt.show()


# ============================================================================
# 14. SEX DISTRIBUTION OF THE INTEGRATED ATLAS
# ============================================================================

sex_palette = {
    "Female": "#4e79a7",
    "Male": "#ff9f1a"
}


def get_sex_percentage(data):
    percentage = (
        data.groupby(
            [
                "status",
                "Sex"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )
    for sex in [
        "Female",
        "Male"
    ]:
        if sex not in percentage.columns:
            percentage[sex] = 0
    percentage = (
        percentage[
            [
                "Female",
                "Male"
            ]
        ]
        .apply(
            lambda x:
            x / x.sum() * 100,
            axis=1
        )
    )
    return percentage


pct_donors = get_sex_percentage(
    meta_donor
)

pct_cells = get_sex_percentage(
    meta_atlas
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 12),
    sharex=True
)


def plot_stacked_bar(ax, data, label_text):
    data[
        [
            "Female",
            "Male"
        ]
    ].plot(
        kind="barh",
        stacked=True,
        color=[
            sex_palette["Female"],
            sex_palette["Male"]
        ],
        ax=ax,
        width=0.6,
        edgecolor="white",
        linewidth=0.5
    )
    ax.set_ylabel(
        ""
    )
    ax.set_xlabel(
        "Percentage (%)"
    )
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.15
        ),
        ncol=2,
        frameon=False
    )
    ax.text(
        -0.12,
        1.03,
        label_text,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold"
    )


plot_stacked_bar(
    axes[0],
    pct_donors,
    "d"
)

plot_stacked_bar(
    axes[1],
    pct_cells,
    "h"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "Integrated_atlas_sex_distribution.pdf"
    ),
    bbox_inches="tight",
    dpi=600
)

plt.show()


# ============================================================================
# 15. COMMAND-LINE USAGE
# ============================================================================

"""
Run with the default project structure:

python 01_dataset_overview.py \
    --project-root /path/to/revision1_final_code_and_data

Change the age-bin size:

python 01_dataset_overview.py \
    --project-root /path/to/revision1_final_code_and_data \
    --bin-size 10
"""
