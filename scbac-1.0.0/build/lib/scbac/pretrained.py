"""High-level API for the released pretrained scBAC models."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from .constants import AGE_CUT, PRETRAINED_CELLTYPES
from .install_models import ensure_pretrained_models
from .utils import validate_pretrained_celltypes
from . import _released_loader


class PretrainedClock:
    """Apply the author's released fixed scBAC ensemble to new AnnData.

    Parameters
    ----------
    model_dir:
        Optional path to ``scBrainAgeClock_models_file``. If omitted, models are
        automatically installed to the scBAC user cache when first needed.
    device:
        ``"cpu"`` or a PyTorch CUDA device such as ``"cuda"``.
    auto_download:
        Automatically obtain the Zenodo model archive when missing.
    """
    def __init__(self, model_dir=None, device="cpu", auto_download=True, strict_five_folds=True):
        self.model_root = ensure_pretrained_models(model_dir, auto_download=auto_download)
        self.device = device
        self.predictor = _released_loader.UnifiedAgePredictor(
            device=device,
            model_root=str(self.model_root),
            strict_five_folds=strict_five_folds,
        )

    def predict(
        self,
        adata,
        *,
        celltype_col="celltype",
        age_col="Age_at_death",
        celltypes=None,
        stage="auto",
        age_cut=AGE_CUT,
        include_benchmarking=True,
        normalize=True,
        chunk_size=50000,
        count_layer=None,
    ):
        """Predict cell age for one or more of the 11 released brain cell types.

        ``stage='auto'`` requires chronological age and routes age <=18 to the
        developmental ensemble and age >18 to the adult ensemble. ``development``,
        ``adult`` and ``full`` can be requested explicitly and do not require an age
        column for routing. ``all`` returns all released stage predictions.
        """
        if celltype_col not in adata.obs.columns:
            raise KeyError("Missing cell-type column {!r}.".format(celltype_col))
        work = adata.copy()
        if count_layer is not None:
            if count_layer not in work.layers:
                raise KeyError("AnnData layer {!r} was not found.".format(count_layer))
            work.X = work.layers[count_layer].copy()
        work.obs["celltype"] = work.obs[celltype_col].astype(str).values
        labels = validate_pretrained_celltypes(work.obs["celltype"])
        requested = list(celltypes) if celltypes is not None else labels
        validate_pretrained_celltypes(requested)
        requested = [ct for ct in PRETRAINED_CELLTYPES if ct in requested and ct in labels]
        if not requested:
            raise ValueError("No requested pretrained cell type is present in the input.")

        stage_norm = str(stage).lower()
        allowed = {"auto", "development", "adult", "full", "all"}
        if stage_norm not in allowed:
            raise ValueError("stage must be one of {}".format(sorted(allowed)))
        if stage_norm == "auto":
            if age_col not in work.obs.columns:
                raise KeyError("stage='auto' requires chronological-age column {!r}.".format(age_col))
            work.obs["Age_at_death"] = pd.to_numeric(work.obs[age_col], errors="coerce")
            if work.obs["Age_at_death"].isna().any():
                warnings.warn("Cells with missing chronological age cannot be stage-routed and will have missing scBAC_age.")
        elif age_col in work.obs.columns:
            work.obs["Age_at_death"] = pd.to_numeric(work.obs[age_col], errors="coerce")

        # Load only stages required by the requested mode to reduce memory.
        stage_map = {
            "development": ["Development"],
            "adult": ["Adult"],
            "full": ["Full"],
            "auto": ["Development", "Adult"],
            "all": ["Development", "Adult", "Full"],
        }
        self.predictor.stages = stage_map[stage_norm]
        self.predictor.model_names = [
            "{}_{}".format(mt, st)
            for mt in self.predictor.model_types for st in self.predictor.stages
        ]
        result = _released_loader.predict_all_celltypes(
            work, requested, self.predictor,
            norm=normalize, chunk_size=chunk_size,
            include_benchmarking=include_benchmarking,
        )
        if result.empty:
            return result

        if stage_norm == "auto":
            # Reconstruct if loader did not create the routing field because only two stages were loaded.
            if "Ensemble_adult_deve" in result.columns:
                result["scBAC_age"] = result["Ensemble_adult_deve"]
            else:
                ages = pd.to_numeric(work.obs.loc[result.index, "Age_at_death"], errors="coerce")
                result["scBAC_age"] = np.where(
                    ages.to_numpy(float) <= age_cut,
                    result.get("Ensemble_Development", np.nan),
                    result.get("Ensemble_Adult", np.nan),
                )
            result["scBAC_stage"] = np.where(
                pd.to_numeric(work.obs.loc[result.index, "Age_at_death"], errors="coerce") <= age_cut,
                "development", "adult"
            )
        elif stage_norm in {"development", "adult", "full"}:
            label = stage_norm.capitalize()
            result["scBAC_age"] = result["Ensemble_{}".format(label)]
            result["scBAC_stage"] = stage_norm
        else:
            if "Ensemble_adult_deve" in result.columns:
                result["scBAC_age"] = result["Ensemble_adult_deve"]
        return result


def predict_pretrained(adata, **kwargs):
    """Convenience function equivalent to ``PretrainedClock(...).predict(...)``."""
    init_keys = {"model_dir", "device", "auto_download", "strict_five_folds"}
    init = {k: kwargs.pop(k) for k in list(kwargs) if k in init_keys}
    return PretrainedClock(**init).predict(adata, **kwargs)
