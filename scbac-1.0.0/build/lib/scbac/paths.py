"""Filesystem locations for downloaded models and package-provided RAA assets."""

from pathlib import Path
import os

from .constants import MODEL_ROOT_NAME


def get_scbac_home():
    """Return the user-writable scBAC home directory."""
    configured = os.environ.get("SCBAC_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".scbac").resolve()


def get_model_root(model_dir=None):
    """Return the root containing CLM_*, ElasticNet_*, Transf_* and benchmarking_model."""
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        # Accept either .../scBrainAgeClock_models_file or its parent.
        nested = path / MODEL_ROOT_NAME
        if nested.is_dir() and not (path / "CLM_Adult").exists():
            return nested
        return path
    env_model = os.environ.get("SCBAC_MODEL_DIR")
    if env_model:
        return get_model_root(env_model)
    return get_scbac_home() / "models" / MODEL_ROOT_NAME


def get_package_raa_root():
    return Path(__file__).resolve().parent / "pretrained_raa"


def get_runtime_raa_root(model_dir=None):
    """Writable/downloaded location for author-provided fixed RAA models and thresholds."""
    env_raa = os.environ.get("SCBAC_RAA_DIR")
    if env_raa:
        return Path(env_raa).expanduser().resolve()
    return get_model_root(model_dir) / "RAA_models"
