# Author-provided RAA references

This directory is intentionally included in the Python package so the released
RAA reference models and threshold files can be shipped alongside scBAC.

Recommended layout:

```text
pretrained_raa/
└── Ensemble_Adult/
    ├── adult_linear_celllevel.pkl   # or raa_model.pkl
    └── thresholds_YYYYMMDD_HHMMSS.json   # or thresholds.json
```

The loader supports both the manuscript-era filenames
`adult_linear_celllevel.pkl` + `thresholds_*.json` and the package-native names
`raa_model.pkl` + `thresholds.json`.

Additional reference clocks can be added as sibling directories, for example
`Benchmarking/` or `Ensemble_Full/`.
