"""Load and apply user-trained scBAC clock bundles."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .constants import AGE_CUT
from .models import elasticnet, clm, transformer


class CustomClock:
    def __init__(self, model_dir, device="cpu", chunk_size=5000):
        self.model_dir = Path(model_dir).expanduser().resolve()
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Custom scBAC manifest not found: {}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != "scbac-custom-clock-bundle-v1":
            raise ValueError("Unsupported custom scBAC bundle format.")
        self.device = device
        self.chunk_size = int(chunk_size)

    @property
    def celltypes(self):
        return list(self.manifest.get("celltypes", []))

    def _algorithm_dir(self, celltype, stage, algorithm):
        return self.model_dir / "celltypes" / str(celltype) / str(stage) / str(algorithm)

    def _available_algorithms(self, celltype, stage):
        return [a for a in self.manifest.get("algorithms", []) if self._algorithm_dir(celltype, stage, a).is_dir()]

    def predict_stage(self, adata, *, celltype, stage, normalize=True):
        stage = str(stage).lower()
        algorithms = self._available_algorithms(celltype, stage)
        if not algorithms:
            raise FileNotFoundError("No trained algorithms found for {} / {}".format(celltype, stage))
        # Normalize exactly once, then each algorithm performs only feature alignment/inference.
        work = adata.copy()
        if normalize:
            work = elasticnet.normalize_adata(work, copy=False)
        result = pd.DataFrame(index=work.obs_names)
        for algorithm in algorithms:
            model_dir = self._algorithm_dir(celltype, stage, algorithm)
            if algorithm == "elasticnet":
                pred = elasticnet.predict_elasticnet_ensemble(work, model_dir, normalize=False, chunk_size=self.chunk_size)
            elif algorithm == "clm":
                pred = clm.predict_clm_ensemble(work, model_dir, normalize=False, chunk_size=self.chunk_size, device=self.device)
            elif algorithm == "transformer":
                pred = transformer.predict_transformer_ensemble(work, model_dir, normalize=False, chunk_size=self.chunk_size, device=self.device)
            else:
                continue
            result[algorithm + "_age"] = pred
        prediction_cols = [c for c in result.columns if c.endswith("_age")]
        result["scBAC_age"] = result[prediction_cols].mean(axis=1)
        result["scBAC_stage"] = stage
        result["celltype"] = str(celltype)
        return result

    def predict(
        self,
        adata,
        *,
        celltype_col="celltype",
        celltypes=None,
        stage="auto",
        age_col="Age_at_death",
        age_cut=None,
        count_layer=None,
    ):
        if celltype_col not in adata.obs.columns:
            raise KeyError("Missing cell-type column {!r}.".format(celltype_col))
        work = adata.copy()
        if count_layer is not None:
            if count_layer not in work.layers:
                raise KeyError("AnnData layer {!r} was not found.".format(count_layer))
            work.X = work.layers[count_layer].copy()
        selected = self.celltypes if celltypes is None else [str(x) for x in celltypes]
        unknown = [x for x in selected if x not in self.celltypes]
        if unknown:
            raise ValueError("Cell types not present in custom bundle: {}".format(unknown))
        stage = str(stage).lower()
        if stage not in {"auto", "development", "adult", "full", "all"}:
            raise ValueError("stage must be auto, development, adult, full, or all")
        cut = float(self.manifest.get("age_cut", AGE_CUT) if age_cut is None else age_cut)
        output = []
        for ct in selected:
            subset = work[work.obs[celltype_col].astype(str) == ct].copy()
            if subset.n_obs == 0:
                continue
            if stage == "all":
                combined = pd.DataFrame(index=subset.obs_names)
                combined["celltype"] = ct
                for one_stage in self.manifest.get("stages", []):
                    if not self._available_algorithms(ct, one_stage):
                        continue
                    one = self.predict_stage(subset, celltype=ct, stage=one_stage, normalize=True)
                    for column in one.columns:
                        if column in {"celltype", "scBAC_stage"}:
                            continue
                        combined["{}_{}".format(column, one_stage)] = one[column]
                output.append(combined)
                continue
            if stage != "auto":
                output.append(self.predict_stage(subset, celltype=ct, stage=stage, normalize=True))
                continue
            if age_col not in subset.obs.columns:
                raise KeyError("stage='auto' requires chronological-age column {!r}.".format(age_col))
            ages = pd.to_numeric(subset.obs[age_col], errors="coerce").to_numpy(float)
            dev = np.isfinite(ages) & (ages <= cut)
            adult = np.isfinite(ages) & (ages > cut)
            if dev.any():
                output.append(self.predict_stage(subset[dev].copy(), celltype=ct, stage="development", normalize=True))
            if adult.any():
                output.append(self.predict_stage(subset[adult].copy(), celltype=ct, stage="adult", normalize=True))
        if not output:
            return pd.DataFrame()
        result = pd.concat(output, axis=0)
        return result.reindex(work.obs_names[work.obs_names.isin(result.index)])
