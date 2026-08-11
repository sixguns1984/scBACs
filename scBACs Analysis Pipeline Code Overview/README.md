# Single-cell Brain Age Clocks (scBACs)

This repository contains the analysis code for the **single-cell Brain Age Clock (scBAC)** study. The pipeline constructs cell-type-specific transcriptomic aging clocks across the human lifespan, evaluates their performance across independent datasets, and applies them to neurodegenerative disease, APOE4-associated aging, aging-acceleration onset, neurogenesis, cross-species validation, and artemisinin intervention analyses.

The repository includes scripts for model training, fixed-model inference, benchmarking, downstream statistical analyses, and figure generation.

---

## Overview

The scBAC framework contains three complementary age-prediction models for each major CNS cell type:

- **Elastic Net**
- **Neural Cumulative Link Model (CLM)**
- **Transformer**

Predictions from the three models are averaged to generate the final ensemble scBAC estimate.

To reduce lifespan-related prediction bias, separate models are used for:

- **Development:** age `<= 18 years`
- **Adult:** age `> 18 years`
- **Full lifespan:** retained as a comparison model

For the primary stage-specific prediction, cells from donors aged `<= 18 years` are assigned the Development model and cells from donors aged `> 18 years` are assigned the Adult model.

The major CNS cell types analyzed throughout the study are:

`Exc`, `Inh`, `Ast`, `Oli`, `OPC`, and `Mic`.

---

## Repository structure

```text
.
├── 01_dataset_overview.py
├── 02_consensus_celltype_annotation.py
├── 03_lifespan_expression_trajectory_analysis.py
│
├── 04a_train_elasticnet_scBAC.py
├── 04b_train_clm_scBAC.py
├── 04c_train_transformer_scBAC.py
├── 04d_LOSO_scBAC_validation.py
├── 05_predict_scBAC_ensemble_fixed_pretrained.py
├── 06_scBAC_model_evaluation.py
├── 06b_mouse_cross_species_validation.py
│
├── 07_scBAC_molecular_feature_analysis.py
├── 08_scBAC_neurogenesis_validation.py
├── 09_inhibitory_neuron_subtype_annotation.py
│
├── 10_NDD_cellular_aging_analysis.py
├── 11_NDD_AASO_analysis.py
├── 12_APOE4_cellular_aging_analysis.py
├── 13_APOE4_AASO_analysis.py
├── 14_APOE4_CMap_artemisinin_analysis.py
│
├── elasticnet_clock.py
├── clm_clock.py
├── transformer_clock.py
├── ensemble_brain_age_pred.py
├── downstream_analysis_utils.py
└── mouse_clock_utils.py
```

The numbered scripts correspond to the main analysis workflow. The remaining Python files contain reusable model architectures, prediction utilities, statistical functions, and mouse-clock helpers.

---

## Analysis workflow

### 01. Dataset overview

**`01_dataset_overview.py`**

Summarizes the datasets, donor composition, age distributions, cell numbers, and study-level structure used for scBAC model development and downstream analyses.

---

### 02. Consensus cell-type annotation

**`02_consensus_celltype_annotation.py`**

Performs and/or summarizes consensus cell-type annotation across the integrated human cortical single-cell datasets.

---

### 03. Lifespan expression trajectories

**`03_lifespan_expression_trajectory_analysis.py`**

Identifies chronological-age-associated genes and characterizes cell-type-specific gene-expression trajectories across the human lifespan.

---

### 04. scBAC model development and cross-study validation

**`04a_train_elasticnet_scBAC.py`**  
Trains donor-grouped Elastic Net scBAC models.

**`04b_train_clm_scBAC.py`**  
Trains donor-grouped neural cumulative link models.

**`04c_train_transformer_scBAC.py`**  
Trains donor-grouped Transformer scBAC models.

**`04d_LOSO_scBAC_validation.py`**  
Performs leave-one-study-out (LOSO) validation across the studies used for model development.

These scripts are the model-training components of the repository. Training and validation use donor-level separation to prevent cells from the same donor from appearing in both training and validation folds.

---

### 05. Fixed pretrained scBAC prediction

**`05_predict_scBAC_ensemble_fixed_pretrained.py`**

Applies the released pretrained scBAC models to genuinely new single-cell datasets.

This script does **not** retrain, refit, recalibrate, or optimize the released models.

For each model family, predictions are averaged across the five pretrained folds, and the final ensemble prediction is the mean of Elastic Net, CLM, and Transformer predictions.

The reusable model loader is implemented in:

**`ensemble_brain_age_pred.py`**

---

### 06. Model evaluation

**`06_scBAC_model_evaluation.py`**

Evaluates scBAC prediction performance across independent validation datasets, including cell-level and donor-median analyses, correlation, mean absolute error (MAE), and comparison with the benchmarking brain-age model.

---

### 06b. Mouse cross-species validation

**`06b_mouse_cross_species_validation.py`**

Evaluates the relationship between human scBAC-equivalent transcriptomic aging and mouse brain aging using cross-species analyses and a published mouse brain-age clock.

Mouse-clock utilities are implemented in:

**`mouse_clock_utils.py`**

---

### 07. Molecular features of cellular aging

**`07_scBAC_molecular_feature_analysis.py`**

Identifies genes and biological processes associated with scBAC-predicted cellular aging, including model-associated molecular features and age-related expression programs.

---

### 08. Adult hippocampal neurogenesis validation

**`08_scBAC_neurogenesis_validation.py`**

Evaluates whether scBAC-derived aging states capture biological variation related to adult human hippocampal neurogenesis.

For this analysis, relative age acceleration (RAA) is recalculated within the hippocampal dataset using its own healthy adult controls. Neurogenic-cell abundance is defined using the original dataset's `Neurogenic` annotation.

---

### 09. Inhibitory-neuron subtype annotation

**`09_inhibitory_neuron_subtype_annotation.py`**

Annotates inhibitory-neuron populations and harmonizes interneuron subtypes for downstream Alzheimer’s disease analyses.

---

### 10. Neurodegenerative disease-associated cellular aging

**`10_NDD_cellular_aging_analysis.py`**

Tests cell-type-specific changes in predicted cellular biological age across neurodegenerative diseases, with a primary focus on Alzheimer’s disease and independent replication.

Analyses include donor-level cellular aging, disease associations, neuropathological measures, cognition, and inhibitory-neuron subtype vulnerability.

---

### 11. Neurodegenerative disease-associated aging-acceleration onset

**`11_NDD_AASO_analysis.py`**

Calculates **relative age acceleration (RAA)** and **age at aging acceleration onset (AASO)** using fixed adult-control RAA reference models and precomputed cell-type-specific thresholds.

AASO is defined as the predicted cellular age at which a donor-specific smoothed RAA trajectory first crosses the cell-type-specific accelerated-aging threshold from below.

The primary accelerated-aging threshold is the **75th percentile (Q75)** of adult-control RAA.

The script evaluates:

- cell-type-specific AASO
- AD discovery and replication cohorts
- additional neurodegenerative diseases
- associations with neuropathology and cognition
- inhibitory-neuron subtype AASO
- optional molecular features associated with earlier AASO

---

### 12. APOE4-associated cellular aging

**`12_APOE4_cellular_aging_analysis.py`**

Evaluates the effects of APOE4, disease status, sex, and their interactions on predicted cellular biological age.

Analyses include:

- discovery and replication datasets
- combined discovery + replication donor-level analyses
- sex-stratified models
- APOE4 × disease interactions
- sex × APOE4 × disease interactions
- disease effects within sex × APOE4 strata
- APOE4 effects within sex × disease strata
- inhibitory-neuron subtype analyses
- optional APOE4-associated aging-gene analyses

---

### 13. APOE4-associated aging-acceleration onset

**`13_APOE4_AASO_analysis.py`**

Extends the APOE4 analysis to AASO.

The script uses fixed RAA reference models and fixed Q75 thresholds and performs:

- discovery analyses
- replication analyses
- combined discovery + replication analyses
- APOE4 × disease interactions
- sex × APOE4 × disease interactions
- sex- and genotype-stratified AD effects on AASO
- inhibitory-neuron subtype analyses
- optional sex-specific genes associated with earlier AASO

---

### 14. Artemisinin intervention and mouse-clock validation

**`14_APOE4_CMap_artemisinin_analysis.py`**

Analyzes the released mouse intervention metadata:

```text
meta_Apoe4_cmap_artemisinin_mouse_20260705.csv
```

The primary analysis compares human scBAC-equivalent cellular age among:

- `APOE3+DMSO`
- `APOE4+DMSO`
- `APOE4+artemisinin`

at both cell and mouse/sample levels.

The script also applies the fixed published mouse brain-age clock to the intervention dataset as an independent species-matched validation.

---

# Pretrained models

The released pretrained scBAC models, benchmarking model, and mouse brain-age clock files are available from Zenodo:

**Zenodo model archive**

https://zenodo.org/records/21882804/files/scBrainAgeClock_models_file.zip?download=1

For example:

```bash
wget -O scBrainAgeClock_models_file.zip \
"https://zenodo.org/records/21882804/files/scBrainAgeClock_models_file.zip?download=1"

unzip scBrainAgeClock_models_file.zip
```

After downloading and extracting the Zenodo archive, keep the released model directory as:

```text
scBrainAgeClock_models_file/
├── benchmarking_model/
│   └── sc_{celltype}.csv
│
├── CLM_Development/
├── CLM_Adult/
├── CLM_Full/
│
├── ElasticNet_Development/
├── ElasticNet_Adult/
├── ElasticNet_Full/
│
├── Transf_Development/
├── Transf_Adult/
├── Transf_Full/
│
└── mouseBrainAgeClock/
    ├── scMouseBrainAgeClock.csv
    └── scMouseBrainAgeClock_humanGeneID.csv
```

The exact model files and corresponding feature files are stored inside the model-family folders shown above.

For fixed-model inference, `model_root` should point directly to:

```text
/path/to/scBrainAgeClock_models_file
```

That directory should contain the `benchmarking_model/`, `CLM_*`, `ElasticNet_*`, and `Transf_*` folders directly.

---

## Fixed pretrained model filenames

The inference code loads the released model files using the original filenames.

### Transformer

```text
{celltype}_geneatt_ensemble_info_age_cut{age_cut}.pkl
{celltype}_geneatt_model_fold{fold}_age_cut{age_cut}.pt
model_features/{celltype}_geneatt_features_age_cut{age_cut}.csv
```

### Elastic Net

```text
{celltype}_elasticnet_5fold_ensemble_age_cut{age_cut}_n_features12779.pkl
model_features/{celltype}_elasticnet_features_age_cut{age_cut}_n_features12779.csv
```

### CLM

```text
{celltype}_clm_5fold_model_fold{fold}_age_cut{age_cut}_n_features12779.pt
model_features/{celltype}_clm_5fold_features_age_cut{age_cut}_n_features12779.csv
```

### Benchmarking model

```text
benchmarking_model/sc_{celltype}.csv
```

The filename convention uses:

- Development: `age_cut = 18`
- Adult: `age_cut = 18`
- Full lifespan: `age_cut = 0`

---

# Installation

The analysis code is written in Python.

Core dependencies include:

```text
numpy
pandas
scipy
scanpy
anndata
scikit-learn
statsmodels
torch
matplotlib
seaborn
joblib
tqdm
```

Install the core dependencies with:

```bash
pip install numpy pandas scipy scanpy anndata scikit-learn \
statsmodels torch matplotlib seaborn joblib tqdm
```

Some upstream annotation or specialized analyses may require additional packages depending on the script and local environment.

For GPU-based Transformer inference/training, install the PyTorch version appropriate for your CUDA environment.

---

# Recommended directory layout

A convenient repository layout is:

```text
scBACs/
├── *.py
├── dataset/
├── scBrainAgeClock_models_file/
├── results/
└── figures/
```

Most revised scripts use `pathlib.Path` and define a project root near the beginning of the script.

Where supported, the project root can be set with:

```bash
export SCBACS_PROJECT_ROOT=/path/to/scBACs
```

Otherwise, update `PROJECT_ROOT` in the corresponding script.

Because the original analyses integrate multiple public single-cell studies, the complete raw datasets are not bundled directly with this code repository. Place the required processed metadata and/or AnnData files under `dataset/` using the filenames expected by the corresponding scripts.

---

# Running the pipeline

The scripts are numbered according to the approximate analysis order:

```text
01
 ↓
02
 ↓
03
 ↓
04a / 04b / 04c
 ↓
04d
 ↓
05
 ↓
06
 ↓
06b
 ↓
07
 ↓
08
 ↓
09
 ↓
10
 ↓
11
 ↓
12
 ↓
13
 ↓
14
```

However, it is **not necessary to rerun the entire pipeline** to reproduce every downstream analysis.

In particular:

- `04a`–`04c` are model-training scripts.
- `04d` performs LOSO refitting and validation.
- `05` is intended for applying the released fixed models to new datasets.
- Many downstream scripts (`06`–`14`) operate on processed metadata or prediction tables produced during the study.

Users interested only in applying scBACs to new single-cell data can generally start from:

```text
05_predict_scBAC_ensemble_fixed_pretrained.py
```

after downloading the released model archive.

---

# Predicting age in a new dataset

For a new single-cell dataset:

1. Prepare an AnnData (`.h5ad`) object.
2. Ensure the cell-type labels use the expected scBAC cell-type names.
3. Download and extract the pretrained model archive as `scBrainAgeClock_models_file/`.
4. Set `model_root` to the extracted `scBrainAgeClock_models_file/` directory.
5. Run:

```bash
python 05_predict_scBAC_ensemble_fixed_pretrained.py
```

The prediction code independently normalizes the new dataset, aligns genes to the exact ordered feature list used by each pretrained model, assigns zero to missing model genes, and applies the fixed released models.

The main prediction outputs include component-model and ensemble ages such as:

```text
transf_Development
elasticnet_Development
clm_Development

transf_Adult
elasticnet_Adult
clm_Adult

transf_Full
elasticnet_Full
clm_Full

Ensemble_Development
Ensemble_Adult
Ensemble_Full
Ensemble_adult_deve
Benchmarking
```

---

# Reproducibility notes

- Model development uses **donor-level data partitioning** to prevent information leakage across cells from the same donor.
- Fixed-model prediction does **not** retrain or recalibrate the released scBAC models.
- Independent datasets are normalized independently before model inference.
- Missing model features are zero-filled only after alignment to the exact released feature order.
- The final scBAC estimate averages the three component model predictions.
- Development and Adult models are routed according to chronological age for the primary stage-specific prediction.
- Donor-level analyses generally summarize cellular age using the donor median.
- AASO analyses use fixed adult-control RAA references and cell-type-specific Q75 thresholds, except where an analysis explicitly defines its own control reference (for example, the hippocampal neurogenesis validation).

---

# Helper modules

### `elasticnet_clock.py`
Elastic Net model training and prediction utilities.

### `clm_clock.py`
Neural cumulative link model architecture and training utilities.

### `transformer_clock.py`
Transformer model architecture and training utilities.

### `ensemble_brain_age_pred.py`
Unified fixed pretrained scBAC model loader and prediction interface.

### `downstream_analysis_utils.py`
Shared functions for downstream regression, stratification, age calculations, FDR correction, and metadata handling.

### `mouse_clock_utils.py`
Utilities for applying the released/published mouse brain-age clock.

---

# Data and model availability

Pretrained model files are available from Zenodo:

https://zenodo.org/records/21882804/files/scBrainAgeClock_models_file.zip?download=1

The repository contains the analysis code but does not redistribute all third-party raw single-cell datasets. Please refer to the manuscript and its supplementary dataset table for the original study accessions and source publications.

---

# Citation

If you use this code or the pretrained scBAC models, please cite the associated scBAC manuscript.

A full citation can be added here once the final publication information is available.

---

# Contact

For questions regarding the analysis code, model files, or reproduction of the study, please open a GitHub issue in this repository.
