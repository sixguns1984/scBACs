"""Train custom stage-specific scBAC clock bundles."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import shutil
import tempfile
import warnings
import json

from .constants import AGE_CUT, ALGORITHMS, DEFAULT_N_FOLDS, DEFAULT_RANDOM_SEED, PACKAGE_VERSION, STAGES
from .utils import json_dump
from .models import elasticnet, clm, transformer


def _stage_label(stage):
    return {"development": "Development", "adult": "Adult", "full": "Full"}[stage]


def _work_model_dir(work_root, algorithm, stage, celltype):
    prefix = {"elasticnet": "ElasticNet", "clm": "CLM", "transformer": "Transf"}[algorithm]
    return Path(work_root) / (prefix + "_" + _stage_label(stage)) / "_training_work" / str(celltype)


def _sanitize_bundle_metadata(model_dir, celltype, stage, algorithm):
    path = Path(model_dir) / "metadata.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata = {k: v for k, v in metadata.items() if not str(k).startswith("public_")}
    metadata["bundle_celltype"] = str(celltype)
    metadata["bundle_stage"] = str(stage)
    metadata["bundle_algorithm"] = str(algorithm)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_clock(
    adata,
    output_dir,
    *,
    model_name="custom_scBAC",
    celltype_col="celltype",
    donor_col="donor_id",
    age_col="Age_at_death",
    dataset_col="dataset",
    status_col=None,
    control_label="CT",
    celltypes=None,
    stages=("development", "adult"),
    algorithms=("elasticnet", "clm", "transformer"),
    age_cut=AGE_CUT,
    n_folds=DEFAULT_N_FOLDS,
    random_seed=DEFAULT_RANDOM_SEED,
    device=None,
    count_layer=None,
    continue_on_error=False,
):
    """Train a custom scBAC ensemble using the current model framework.

    Custom cell-type names are allowed. The user must identify cell type, donor and
    chronological-age columns. If ``status_col`` is omitted, every input cell is used
    as training/control data. Training produces a consistent bundle that can be loaded
    by :class:`scbac.CustomClock`.
    """
    required = [celltype_col, donor_col, age_col]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise KeyError("AnnData.obs is missing required columns: {}".format(missing))
    stages = [str(x).lower() for x in stages]
    algorithms = [str(x).lower() for x in algorithms]
    invalid_stages = [x for x in stages if x not in STAGES]
    invalid_algorithms = [x for x in algorithms if x not in ALGORITHMS]
    if invalid_stages or invalid_algorithms:
        raise ValueError("Invalid stages {} or algorithms {}".format(invalid_stages, invalid_algorithms))

    work_adata = adata.copy()
    if count_layer is not None:
        if count_layer not in work_adata.layers:
            raise KeyError("AnnData layer {!r} was not found.".format(count_layer))
        work_adata.X = work_adata.layers[count_layer].copy()

    # Canonical private names make the low-level training modules agnostic to user metadata names.
    work_adata.obs["__scbac_celltype"] = work_adata.obs[celltype_col].astype(str).values
    work_adata.obs["__scbac_donor"] = work_adata.obs[donor_col].astype(str).values
    work_adata.obs["__scbac_age"] = work_adata.obs[age_col].values
    if dataset_col is not None and dataset_col in work_adata.obs.columns:
        work_adata.obs["__scbac_dataset"] = work_adata.obs[dataset_col].astype(str).values
    else:
        work_adata.obs["__scbac_dataset"] = "dataset"
    if status_col is not None:
        if status_col not in work_adata.obs.columns:
            raise KeyError("Status column {!r} was not found.".format(status_col))
        work_adata.obs["__scbac_status"] = work_adata.obs[status_col].astype(str).values
    else:
        work_adata.obs["__scbac_status"] = str(control_label)

    if celltypes is None:
        celltypes = sorted(work_adata.obs["__scbac_celltype"].dropna().astype(str).unique().tolist())
    else:
        celltypes = [str(x) for x in celltypes]
    if not celltypes:
        raise ValueError("No cell types were selected for training.")

    bundle = Path(output_dir).expanduser().resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError("Output bundle already exists and is not empty: {}".format(bundle))
    bundle.mkdir(parents=True, exist_ok=True)
    model_root = bundle / "celltypes"
    model_root.mkdir()
    records = []

    with tempfile.TemporaryDirectory(prefix="scbac_train_") as tmp:
        tmp = Path(tmp)
        for ct in celltypes:
            for stage in stages:
                for algorithm in algorithms:
                    print("\nTraining custom scBAC: {} | {} | {}".format(ct, stage, algorithm))
                    try:
                        common = dict(
                            adata=work_adata,
                            celltype=ct,
                            stage=stage,
                            output_dir=str(tmp),
                            age_cut=age_cut,
                            celltype_col="__scbac_celltype",
                            donor_col="__scbac_donor",
                            dataset_col="__scbac_dataset",
                            age_col="__scbac_age",
                            status_col="__scbac_status",
                            control_label=str(control_label),
                            n_folds=int(n_folds),
                            random_seed=int(random_seed),
                        )
                        if algorithm == "elasticnet":
                            meta = elasticnet.train_elasticnet_clock(**common)
                        elif algorithm == "clm":
                            meta = clm.train_clm_clock(**common, device=device)
                        else:
                            meta = transformer.train_transformer_clock(**common, device=device)
                        src = _work_model_dir(tmp, algorithm, stage, ct)
                        if not src.is_dir():
                            raise RuntimeError("Training completed but auxiliary model directory was not found: {}".format(src))
                        dst = model_root / ct / stage / algorithm
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src, dst)
                        _sanitize_bundle_metadata(dst, ct, stage, algorithm)
                        records.append({"celltype": ct, "stage": stage, "algorithm": algorithm, "status": "ok", "metadata": meta})
                    except Exception as exc:
                        records.append({"celltype": ct, "stage": stage, "algorithm": algorithm, "status": "failed", "error": str(exc)})
                        if continue_on_error:
                            warnings.warn("Training failed for {} / {} / {}: {}".format(ct, stage, algorithm, exc))
                        else:
                            raise

    manifest = {
        "format": "scbac-custom-clock-bundle-v1",
        "package_version": PACKAGE_VERSION,
        "model_name": str(model_name),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "celltypes": celltypes,
        "stages": stages,
        "algorithms": algorithms,
        "age_cut": float(age_cut),
        "n_folds": int(n_folds),
        "random_seed": int(random_seed),
        "normalization": "scanpy.pp.normalize_per_cell + scanpy.pp.log1p",
        "input_columns": {
            "celltype": celltype_col,
            "donor": donor_col,
            "chronological_age": age_col,
            "dataset": dataset_col,
            "status": status_col,
            "control_label": str(control_label),
        },
        "training_records": records,
    }
    json_dump(manifest, bundle / "manifest.json")
    return manifest
