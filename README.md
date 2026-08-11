# scBACs - Single Cell Brain Age Clocks

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-active-success)]()
[![DOI](https://img.shields.io/badge/DOI-TBD-orange)]()

> A deep learning tool for predicting brain cell age from single-cell RNA-seq data

## ✨ Key Features

- **🔬 Multi-cell type support:** Predict ages for 13 major brain cell types
- **🧠 Deep learning models:** Integrated MLP and Transformer architectures
- **📊 Advanced analysis:** Relative age aceleration calculation and age at aging acceleration onset detection
- **🎨 Professional visualizations
- **⚡ Efficient computation:** CPU/GPU support and parallel processing
- **🔄 Easy model management:** One-click pre-trained model downloads
- **💻 Command-line interface:** User-friendly CLI for all functions

## 📦 Installation

### Basic Installation
```bash
pip install scbac
```

### Development Version
```bash
git clone https://github.com/sixguns1984/scBACs.git
cd scBACs
pip install scbac-0.1.0-py3-none-any.whl
```
## Download Pre-trained Models
After installation, download the pre-trained models:

```bash
scbac-install-models --cell-type all
```
## 🚀 Quick Start
### Predict Cell Ages from scRNA-seq Data
```bash
scbac predict \
    --input input_data.h5ad \
    --output results.h5ad \
    --cell-type-column celltype \
    --count-layer counts \
    --device cpu
```

### Analyze cellular level of relative age acceleration (RAA) and cell-type-specific individual age at aging acceleration onset
```bash
scbac analyze \
    --input predicted_ages.csv \
    --output-prefix analysis_results \
    --disease-name AD \
    --status-col status \
    --cell-age-pred-col age_pred \
    --chronological-age-col Age_at_death \
    --sex-col Sex \    
    --donor-col PaticipantID_unique \
    --celltype-col celltype
    
```
## Pre-trained model management
```
# Install all models
scbac-install-models --cell-type all

# Force re-download
scbac-install-models --force-download

# List installed models
scbac-install-models --list-models
```

## 📊 Supported Cell Types
scBACs includes pre-trained models for 13 brain cell types:

| Cell Type | Abbreviation |
|-----------|--------------|
| Astrocytes | Ast |
| Endothelial cells | End |
| Excitatory neurons | Exc |
| Inhibitory neurons | Inh |
| Oligodendrocytes | Oli |
| Oligodendrocyte progenitor cells | OPC |
| Pericytes | Per |
| Microglia | Mic |
| CNS associated macrophage | CAM |
| Fibroblasts | Fib |
| Mural cells | Mural |
| Smooth muscle cells | SMC |
| T cells | Tcell |


## 📋 Data Requirements
### For Age Prediction (AnnData Input)
Cell type annotations: in adata.obs column (specified by --cell-type-column)

Raw counts: in a layer (default: 'counts')

Normalized and scaled data: optional, will be processed if needed

### For cell predicted age at aging acceleration onset (CSV Input)
Required columns:

predicted_age: Predicted cell ages

celltype: Cell type annotations

PaticipantID_unique: Donor/participant ID

Age_at_death: Age at death

Sex: Sex (Male/Female or encoded values)

status: Disease status (e.g., 'AD', 'CT', 'ALS', etc.)
## 📈 Output Files
### Prediction Output
predicted_age column added to AnnData.obs

Full observation DataFrame with predictions

Analysis Output
age_gap: Calculated age gap (residuals)

threshold_age: Aging acceleration turning point cell predicted age

positive_ratio: Proportion of cells with positive age gap

Statistical analysis results (ANOVA, pairwise comparisons)

High-quality publication-ready figures

### 🎨 Visualizations
scBACs generates comprehensive visualizations:

Individual Donor Curves: Age gap vs. predicted age for each donor

Cell Type Comparisons: Aging thresholds across cell types

Heatmap Clustering: Donor-level aging patterns

Statistical Significance: Pairwise differences between cell types

Positive Ratio Analysis: Proportion of aged cells per donor

### 🔬 Command Line Options
predict command
```bash
scbac predict
  --input INPUT          Input .h5ad file
  --output OUTPUT        Output .h5ad file
  --cell-type-column CELL_TYPE_COLUMN
                        Column name for cell type annotations
  --count-layer COUNT_LAYER
                        Layer containing raw counts (default: 'counts')
  --device DEVICE        Compute device: 'cpu' or 'cuda' (default: 'cpu')
  --model-dir MODEL_DIR  Pre-trained model directory (optional)
  
```
analyze command
```bash
scbac analyze
  --input INPUT          Input CSV file with predicted ages
  --output-prefix OUTPUT_PREFIX
                        Prefix for output files
  --disease-name DISEASE_NAME
                        Disease group to analyze (e.g., 'AD')
  --status-col STATUS_COL
                        Column with disease/control status
  --cell-age-pred-col CELL_AGE_PREDICTED
                                   Column with predicted cell age
  --chronological-age-col CHRONOLOGICAL_AGE 
                                           Column with chronological age
  --sex-col SEX_COL Column with Sex
                             
  --donor-col DONOR_COL  Column with donor IDs
  --celltype-col CELLTYPE_COL
                        Column with cell type annotations
 ```
 
  
## 🏗️ Project Structure
```bash 
scbac/
├── scbac/
│   ├── __init__.py                 # Package initialization
│   ├── predictor.py                # Core prediction functions
│   ├── age_gap_analysis.py         # Age gap analysis module
│   ├── data_processing.py          # Data preprocessing
│   ├── models/
│   │   ├── mlp_model.py           # MLP model definition
│   │   └── transformer_model.py   # Transformer model definition
│   ├── cli.py                     # Command-line interface
│   └── install_models.py          # Model installation script
├── setup.py                       # Installation configuration
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file

```
## 📚 Tutorial
See tutorial.py for Python code examples including:

Basic age prediction pipeline

Complete aging analysis workflow

Custom model usage

Batch processing examples

Visualization examples


## 📄 Citation
If you use scBACs in your research, please cite:

```
@software{scBACs2025,
  author = {Luo, Jianfeng; Liu, Ganqiang; Tang Yamei},
  title = {scBACs: Single Cell Brain Age Clocks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sixguns1984/scBACs}
}
```

## 🤝 Contributing
We welcome contributions! Here's how to get started:

Fork the repository

Create a feature branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request


📞 Contact
Issues: GitHub Issues

Email: luojf35@mail.sysu.edu.cn
Email: liugq3@mail.sysu.edu.cn
Email: tangym@mail.sysu.edu.cn

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
Thanks to all contributors and testers

Computational resources provided by Sun Yat-sen University

Open-source community for tools and libraries

<div align="center"> <strong>scBACs: Unraveling cellular aging dynamics in the human brain</strong> </div>

