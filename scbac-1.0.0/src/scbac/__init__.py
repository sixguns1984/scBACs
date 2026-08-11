"""scBAC: single-cell Brain Age Clocks.

Heavy scientific modules are imported lazily so package metadata/model-management
commands remain lightweight. Normal train/predict workflows still require the full
``requirements.txt`` environment.
"""
from .constants import PACKAGE_VERSION as __version__, PRETRAINED_CELLTYPES

__all__ = [
    "__version__", "PRETRAINED_CELLTYPES",
    "PretrainedClock", "predict_pretrained", "train_clock", "CustomClock",
    "fit_raa_reference", "apply_raa", "load_raa_reference", "load_thresholds",
    "calculate_aaso", "smooth_donor_raa_curve", "first_threshold_crossing", "analyze_aging",
    "install_pretrained_models", "ensure_pretrained_models", "model_status",
]

_LAZY = {
    "PretrainedClock": (".pretrained", "PretrainedClock"),
    "predict_pretrained": (".pretrained", "predict_pretrained"),
    "train_clock": (".training", "train_clock"),
    "CustomClock": (".custom", "CustomClock"),
    "fit_raa_reference": (".raa", "fit_raa_reference"),
    "apply_raa": (".raa", "apply_raa"),
    "load_raa_reference": (".raa", "load_raa_reference"),
    "load_thresholds": (".raa", "load_thresholds"),
    "calculate_aaso": (".aaso", "calculate_aaso"),
    "smooth_donor_raa_curve": (".aaso", "smooth_donor_raa_curve"),
    "first_threshold_crossing": (".aaso", "first_threshold_crossing"),
    "analyze_aging": (".analysis", "analyze_aging"),
    "install_pretrained_models": (".install_models", "install_pretrained_models"),
    "ensure_pretrained_models": (".install_models", "ensure_pretrained_models"),
    "model_status": (".install_models", "model_status"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    import importlib
    module_name, attribute = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
