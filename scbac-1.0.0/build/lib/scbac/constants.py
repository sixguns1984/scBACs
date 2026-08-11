"""Constants used by the public scBAC package."""

PACKAGE_NAME = "scbac"
PACKAGE_VERSION = "1.0.0"

ZENODO_MODEL_URL = (
    "https://zenodo.org/records/21882804/files/"
    "scBrainAgeClock_models_file.zip?download=1"
)
MODEL_ARCHIVE_NAME = "scBrainAgeClock_models_file.zip"
MODEL_ROOT_NAME = "scBrainAgeClock_models_file"

AGE_CUT = 18.0
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_SEED = 42

# These are the exact labels expected by the released pretrained scBAC models.
PRETRAINED_CELLTYPES = [
    "Exc",
    "Inh",
    "Ast",
    "Oli",
    "OPC",
    "Mic",
    "End",
    "Fib",
    "Per",
    "CAM",
    "T_cell",
]

PRETRAINED_CELLTYPE_NAMES = {
    "Exc": "Excitatory neurons",
    "Inh": "Inhibitory neurons",
    "Ast": "Astrocytes",
    "Oli": "Oligodendrocytes",
    "OPC": "Oligodendrocyte progenitor cells",
    "Mic": "Microglia",
    "End": "Endothelial cells",
    "Fib": "Fibroblasts",
    "Per": "Pericytes",
    "CAM": "CNS-associated macrophages",
    "T_cell": "T cells",
}

STAGES = ["development", "adult", "full"]
ALGORITHMS = ["elasticnet", "clm", "transformer"]

# Released model folders in the Zenodo archive.
PRETRAINED_STAGE_DIRS = {
    ("elasticnet", "development"): "ElasticNet_Development",
    ("elasticnet", "adult"): "ElasticNet_Adult",
    ("elasticnet", "full"): "ElasticNet_Full",
    ("clm", "development"): "CLM_Development",
    ("clm", "adult"): "CLM_Adult",
    ("clm", "full"): "CLM_Full",
    ("transformer", "development"): "Transf_Development",
    ("transformer", "adult"): "Transf_Adult",
    ("transformer", "full"): "Transf_Full",
}

SEX_MAP = {
    "Female": 0.0,
    "female": 0.0,
    "F": 0.0,
    "f": 0.0,
    "Male": 1.0,
    "male": 1.0,
    "M": 1.0,
    "m": 1.0,
    0: 0.0,
    1: 1.0,
    0.0: 0.0,
    1.0: 1.0,
}
