"""Relative age acceleration (RAA) reference fitting and application."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .paths import get_package_raa_root, get_runtime_raa_root
from .utils import encode_sex, json_dump


def _as_frame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    return pd.read_csv(path)


def _canonical_raa_frame(df, donor_col, celltype_col, predicted_age_col, chronological_age_col, sex_col):
    required = [donor_col, celltype_col, predicted_age_col, chronological_age_col, sex_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("RAA input is missing required columns: {}".format(missing))
    out = df.copy()
    out["__donor"] = out[donor_col].astype(str)
    out["__celltype"] = out[celltype_col].astype(str)
    out["__predicted_age"] = pd.to_numeric(out[predicted_age_col], errors="coerce")
    out["__chronological_age"] = pd.to_numeric(out[chronological_age_col], errors="coerce")
    out["__sex_encoded"] = encode_sex(out[sex_col]).to_numpy(float)
    out["__donor_cell_count"] = out.groupby("__donor", observed=True)["__donor"].transform("size").astype(float)
    return out


def fit_raa_reference(
    control_data,
    output_dir,
    *,
    donor_col="donor_id",
    celltype_col="celltype",
    predicted_age_col="scBAC_age",
    chronological_age_col="Age_at_death",
    sex_col="Sex",
    group_col=None,
    control_groups=None,
    quantiles=(5, 25, 50, 75, 95),
    min_cells=10,
):
    """Fit cell-type-specific control RAA models and save thresholds.

    The reference model is ``predicted_age ~ chronological_age + sex + donor_cell_count``.
    The saved ``raa_model.pkl`` and ``thresholds.json`` form one portable RAA bundle.
    """
    raw = _as_frame(control_data)
    if group_col is not None and control_groups is not None:
        if group_col not in raw.columns:
            raise KeyError("Control group column {!r} was not found.".format(group_col))
        raw = raw[raw[group_col].isin(list(control_groups))].copy()
    canonical = _canonical_raa_frame(raw, donor_col, celltype_col, predicted_age_col, chronological_age_col, sex_col)
    models = {}
    raa = pd.Series(np.nan, index=canonical.index, dtype=float)
    feature_names = ["const", "chronological_age", "sex_encoded", "donor_cell_count"]
    for ct, g in canonical.groupby("__celltype", observed=True):
        valid = g.dropna(subset=["__predicted_age", "__chronological_age", "__sex_encoded", "__donor_cell_count"])
        if len(valid) < min_cells:
            mean = float(valid["__predicted_age"].mean()) if len(valid) else np.nan
            models[str(ct)] = {"type": "mean", "mean": mean, "n_obs": int(len(valid))}
            if np.isfinite(mean):
                raa.loc[valid.index] = valid["__predicted_age"].to_numpy(float) - mean
            continue
        X = pd.DataFrame({
            "chronological_age": valid["__chronological_age"].to_numpy(float),
            "sex_encoded": valid["__sex_encoded"].to_numpy(float),
            "donor_cell_count": valid["__donor_cell_count"].to_numpy(float),
        }, index=valid.index)
        X = sm.add_constant(X, has_constant="add")
        constant = [c for c in X.columns if c != "const" and X[c].nunique() <= 1]
        if constant:
            X = X.drop(columns=constant)
        fit = sm.OLS(valid["__predicted_age"].to_numpy(float), X).fit()
        expected = fit.predict(X)
        raa.loc[valid.index] = valid["__predicted_age"].to_numpy(float) - expected
        models[str(ct)] = {
            "type": "ols",
            "coef": {str(k): float(v) for k, v in fit.params.items()},
            "features": [str(x) for x in X.columns],
            "rsquared": float(fit.rsquared),
            "n_obs": int(fit.nobs),
        }

    thresholds = {}
    canonical["__RAA"] = raa
    for ct, g in canonical.groupby("__celltype", observed=True):
        values = pd.to_numeric(g["__RAA"], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        qvals = np.percentile(values, list(quantiles))
        thresholds[str(ct)] = {"q{}".format(q): float(v) for q, v in zip(quantiles, qvals)}
        thresholds[str(ct)].update({"n_cells": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0})

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "scbac-raa-reference-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "linear",
        "formula": "predicted_age ~ chronological_age + sex + donor_cell_count",
        "sex_encoding": "Female=0, Male=1",
        "source_columns": {
            "donor": donor_col, "celltype": celltype_col, "predicted_age": predicted_age_col,
            "chronological_age": chronological_age_col, "sex": sex_col,
        },
        "n_cell_types": len(models),
    }
    with (output_dir / "raa_model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "metadata": metadata}, handle)
    json_dump({"thresholds": thresholds, "metadata": {**metadata, "quantiles": list(quantiles)}}, output_dir / "thresholds.json")
    result = raw.copy()
    result["RAA"] = raa.reindex(result.index).to_numpy(float)
    return result, models, thresholds


def _resolve_raa_files(reference=None, clock_name="Ensemble_Adult", model_dir=None):
    if reference is not None:
        p = Path(reference).expanduser().resolve()
        roots = [p if p.is_dir() else p.parent]
        explicit_model = p if p.is_file() and p.suffix == ".pkl" else None
    else:
        roots = [
            get_package_raa_root() / clock_name,
            get_runtime_raa_root(model_dir) / clock_name,
            get_runtime_raa_root(model_dir),
        ]
        explicit_model = None
    model_candidates = ["raa_model.pkl", "adult_linear_celllevel.pkl"]
    threshold_candidates = ["thresholds.json"]
    for root in roots:
        if not root.exists():
            continue
        model_file = explicit_model if explicit_model is not None else next((root / x for x in model_candidates if (root / x).exists()), None)
        if model_file is None:
            legacy = sorted(root.glob("RAA_model_*.pkl"))
            model_file = legacy[-1] if legacy else None
        threshold_file = next((root / x for x in threshold_candidates if (root / x).exists()), None)
        if threshold_file is None:
            legacy_json = sorted(root.glob("thresholds_*.json"))
            threshold_file = legacy_json[-1] if legacy_json else None
        if model_file is not None:
            return model_file, threshold_file
    raise FileNotFoundError(
        "No RAA reference model was found. Supply reference=..., set SCBAC_RAA_DIR, "
        "or place the author's files under scbac/pretrained_raa/{}/.".format(clock_name)
    )


def load_raa_reference(reference=None, clock_name="Ensemble_Adult", model_dir=None):
    model_file, threshold_file = _resolve_raa_files(reference, clock_name, model_dir)
    with model_file.open("rb") as handle:
        saved = pickle.load(handle)
    models = saved["models"] if isinstance(saved, dict) and "models" in saved else saved
    metadata = saved.get("metadata", {}) if isinstance(saved, dict) else {}
    thresholds = None
    if threshold_file is not None:
        with threshold_file.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        thresholds = loaded.get("thresholds", loaded)
    return models, thresholds, metadata, model_file, threshold_file


def _legacy_expected(valid, model_info, chronological_age_col):
    if model_info.get("type") == "mean":
        return np.full(len(valid), float(model_info.get("mean", np.nan)))
    coef = model_info.get("coef", {})
    features = model_info.get("features", list(coef))
    values = {}
    for feature in features:
        if feature == "const":
            values[feature] = np.ones(len(valid), dtype=float)
        elif feature in {"chronological_age", chronological_age_col, "Age_at_death"}:
            values[feature] = valid["__chronological_age"].to_numpy(float)
        elif feature in {"sex_encoded", "Sex"}:
            values[feature] = valid["__sex_encoded"].to_numpy(float)
        elif feature == "donor_cell_count":
            values[feature] = valid["__donor_cell_count"].to_numpy(float)
        elif feature in valid.columns:
            values[feature] = pd.to_numeric(valid[feature], errors="coerce").to_numpy(float)
        else:
            raise KeyError("RAA model requires unsupported/missing feature {!r}.".format(feature))
    expected = np.zeros(len(valid), dtype=float)
    for feature in features:
        expected += float(coef.get(feature, 0.0)) * values[feature]
    return expected


def apply_raa(
    data,
    *,
    reference=None,
    clock_name="Ensemble_Adult",
    model_dir=None,
    donor_col="donor_id",
    celltype_col="celltype",
    predicted_age_col="scBAC_age",
    chronological_age_col="Age_at_death",
    sex_col="Sex",
    output_col="RAA",
):
    """Apply a custom or author-provided RAA reference to cell-level data."""
    raw = _as_frame(data)
    canonical = _canonical_raa_frame(raw, donor_col, celltype_col, predicted_age_col, chronological_age_col, sex_col)
    models, thresholds, metadata, model_file, threshold_file = load_raa_reference(reference, clock_name, model_dir)
    out = raw.copy()
    out[output_col] = np.nan
    for ct, g in canonical.groupby("__celltype", observed=True):
        if str(ct) not in models:
            continue
        valid = g.dropna(subset=["__predicted_age", "__chronological_age", "__sex_encoded", "__donor_cell_count"])
        if valid.empty:
            continue
        expected = _legacy_expected(valid, models[str(ct)], chronological_age_col)
        out.loc[valid.index, output_col] = valid["__predicted_age"].to_numpy(float) - expected
    return out, thresholds


def load_thresholds(reference=None, clock_name="Ensemble_Adult", model_dir=None):
    _, thresholds, _, _, threshold_file = load_raa_reference(reference, clock_name, model_dir)
    if thresholds is None:
        raise FileNotFoundError("RAA reference was found, but no thresholds JSON is available.")
    return thresholds
