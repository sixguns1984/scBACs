"""Small reusable helpers."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .constants import PRETRAINED_CELLTYPES, SEX_MAP


def encode_sex(values):
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric.isin([0, 1]))
    return series.map(SEX_MAP)


def validate_pretrained_celltypes(values):
    labels = sorted(set(pd.Series(values).dropna().astype(str)))
    unsupported = [x for x in labels if x not in PRETRAINED_CELLTYPES]
    if unsupported:
        raise ValueError(
            "Unsupported cell-type labels for the released pretrained scBAC models: {}. "
            "Use the exact canonical labels: {}".format(
                unsupported, ", ".join(PRETRAINED_CELLTYPES)
            )
        )
    return labels


def json_dump(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def bh_fdr(pvalues):
    p = pd.to_numeric(pd.Series(pvalues), errors="coerce")
    out = np.full(len(p), np.nan, dtype=float)
    valid = p.notna().values
    if not valid.any():
        return out
    pv = p.loc[valid].to_numpy(float)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(len(pv), dtype=float)
    restored[order] = adjusted
    out[valid] = restored
    return out
