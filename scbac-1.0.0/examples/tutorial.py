"""Minimal scBAC Python API examples."""
import scanpy as sc
from scbac import (
    PretrainedClock,
    CustomClock,
    train_clock,
    fit_raa_reference,
    apply_raa,
    calculate_aaso,
    load_thresholds,
)

# -----------------------------------------------------------------------------
# 1. Released pretrained model prediction
# -----------------------------------------------------------------------------
# adata = sc.read_h5ad("input.h5ad")
# clock = PretrainedClock(device="cpu")
# pred = clock.predict(
#     adata,
#     celltype_col="celltype",
#     age_col="Age_at_death",
#     count_layer="counts",
#     stage="auto",
# )

# -----------------------------------------------------------------------------
# 2. Train and apply a custom scBAC clock
# -----------------------------------------------------------------------------
# manifest = train_clock(
#     adata,
#     "my_clock",
#     celltype_col="annotation",
#     donor_col="participant",
#     age_col="age_years",
#     status_col="diagnosis",
#     control_label="CT",
#     count_layer="counts",
#     device="cuda",
# )
# custom = CustomClock("my_clock", device="cuda")
# custom_pred = custom.predict(
#     adata,
#     celltype_col="annotation",
#     age_col="age_years",
#     count_layer="counts",
#     stage="auto",
# )

# -----------------------------------------------------------------------------
# 3. Fit RAA and calculate AASO
# -----------------------------------------------------------------------------
# control_raa, _, thresholds = fit_raa_reference(
#     control_dataframe,
#     "my_raa_reference",
#     donor_col="participant",
#     celltype_col="celltype",
#     predicted_age_col="scBAC_age",
#     chronological_age_col="age_years",
#     sex_col="sex",
# )
# all_raa, thresholds = apply_raa(
#     full_dataframe,
#     reference="my_raa_reference",
#     donor_col="participant",
#     celltype_col="celltype",
#     predicted_age_col="scBAC_age",
#     chronological_age_col="age_years",
#     sex_col="sex",
# )
# aaso, curves = calculate_aaso(
#     all_raa,
#     thresholds,
#     donor_col="participant",
#     celltype_col="celltype",
#     predicted_age_col="scBAC_age",
#     raa_col="RAA",
# )
