"""Command-line interface for scBAC."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from .constants import AGE_CUT, PRETRAINED_CELLTYPES
from .data import read_anndata, read_table, attach_predictions, write_table
from .install_models import install_pretrained_models, model_status
from .pretrained import PretrainedClock
from .training import train_clock
from .custom import CustomClock
from .raa import fit_raa_reference, apply_raa, load_thresholds
from .aaso import calculate_aaso
from .analysis import analyze_aging


def _csv_list(value):
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _write_prediction_output(adata, pred, output):
    output = Path(output)
    if output.suffix.lower() == ".h5ad":
        out = attach_predictions(adata, pred)
        output.parent.mkdir(parents=True, exist_ok=True)
        out.write_h5ad(output)
    else:
        frame = adata.obs.copy()
        for c in pred.columns:
            if c == "celltype" and c in frame.columns:
                continue
            frame.loc[pred.index, c] = pred[c]
        write_table(frame.reset_index().rename(columns={"index": "cell_id"}), output, index=False)


def cmd_models(args):
    if args.models_command == "status":
        print(json.dumps(model_status(args.model_dir), indent=2))
    elif args.models_command == "install":
        install_pretrained_models(args.model_dir, force=args.force, quiet=args.quiet)
    return 0


def cmd_predict(args):
    adata = read_anndata(args.input, count_layer=args.count_layer)
    celltypes = _csv_list(args.celltypes)
    if args.source == "pretrained":
        clock = PretrainedClock(model_dir=args.model_dir, device=args.device, auto_download=not args.no_auto_download)
        pred = clock.predict(
            adata, celltype_col=args.celltype_col, age_col=args.age_col,
            celltypes=celltypes, stage=args.stage, include_benchmarking=args.benchmarking,
            chunk_size=args.chunk_size,
        )
    else:
        if not args.custom_model:
            raise ValueError("--custom-model is required when --source custom")
        clock = CustomClock(args.custom_model, device=args.device, chunk_size=args.chunk_size)
        pred = clock.predict(
            adata, celltype_col=args.celltype_col, celltypes=celltypes,
            stage=args.stage, age_col=args.age_col, count_layer=None,
        )
    _write_prediction_output(adata, pred, args.output)
    print("Prediction output:", args.output)
    return 0


def cmd_train(args):
    adata = read_anndata(args.input, count_layer=args.count_layer)
    manifest = train_clock(
        adata, args.output_dir, model_name=args.model_name,
        celltype_col=args.celltype_col, donor_col=args.donor_col, age_col=args.age_col,
        dataset_col=args.dataset_col, status_col=args.status_col, control_label=args.control_label,
        celltypes=_csv_list(args.celltypes), stages=_csv_list(args.stages), algorithms=_csv_list(args.algorithms),
        age_cut=args.age_cut, n_folds=args.n_folds, random_seed=args.seed, device=args.device,
        count_layer=None, continue_on_error=args.continue_on_error,
    )
    print("Custom model bundle:", Path(args.output_dir).resolve())
    print("Trained records:", sum(r.get("status") == "ok" for r in manifest["training_records"]))
    return 0


def cmd_raa_fit(args):
    data = read_table(args.input)
    result, _, _ = fit_raa_reference(
        data, args.output_dir, donor_col=args.donor_col, celltype_col=args.celltype_col,
        predicted_age_col=args.predicted_age_col, chronological_age_col=args.age_col,
        sex_col=args.sex_col, group_col=args.group_col, control_groups=_csv_list(args.control_groups),
    )
    if args.output:
        write_table(result, args.output)
    print("RAA reference bundle:", Path(args.output_dir).resolve())
    return 0


def cmd_raa(args):
    data = read_table(args.input)
    out, _ = apply_raa(
        data, reference=args.reference, clock_name=args.clock_name, model_dir=args.model_dir,
        donor_col=args.donor_col, celltype_col=args.celltype_col,
        predicted_age_col=args.predicted_age_col, chronological_age_col=args.age_col,
        sex_col=args.sex_col, output_col=args.output_col,
    )
    write_table(out, args.output)
    return 0


def cmd_aaso(args):
    data = read_table(args.input)
    thresholds = load_thresholds(reference=args.reference, clock_name=args.clock_name, model_dir=args.model_dir)
    out, _ = calculate_aaso(
        data, thresholds, donor_col=args.donor_col, celltype_col=args.celltype_col,
        predicted_age_col=args.predicted_age_col, raa_col=args.raa_col,
        quantile=args.quantile, min_cells=args.min_cells, n_points=args.n_points,
        metadata_cols=_csv_list(args.metadata_cols), include_censored=args.include_censored,
    )
    write_table(out, args.output)
    return 0


def cmd_analyze(args):
    data = read_table(args.input)
    control_data = read_table(args.control_input) if args.control_input else None
    analyze_aging(
        data, args.output_prefix, donor_col=args.donor_col, celltype_col=args.celltype_col,
        predicted_age_col=args.predicted_age_col, chronological_age_col=args.age_col,
        sex_col=args.sex_col, raa_reference=args.reference, raa_clock_name=args.clock_name,
        control_data=control_data, control_group_col=args.control_group_col,
        control_groups=_csv_list(args.control_groups), quantile=args.quantile,
        metadata_cols=_csv_list(args.metadata_cols), status_col=args.status_col,
        disease_name=args.disease_name, control_name=args.control_name,
    )
    print("Analysis prefix:", Path(args.output_prefix).resolve())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="scbac", description="Single-cell Brain Age Clocks")
    parser.add_argument("--version", action="version", version="scbac 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="Install or inspect released pretrained models")
    sp = p.add_subparsers(dest="models_command", required=True)
    pi = sp.add_parser("install")
    pi.add_argument("--model-dir", default=None); pi.add_argument("--force", "--force-download", action="store_true"); pi.add_argument("--quiet", action="store_true")
    ps = sp.add_parser("status"); ps.add_argument("--model-dir", default=None)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("predict", help="Predict cellular age from .h5ad")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--source", choices=["pretrained", "custom"], default="pretrained")
    p.add_argument("--custom-model"); p.add_argument("--model-dir")
    p.add_argument("--cell-type-column", "--celltype-col", dest="celltype_col", default="celltype")
    p.add_argument("--chronological-age-col", "--age-col", dest="age_col", default="Age_at_death")
    p.add_argument("--celltypes", help="Comma-separated cell types; default: all present/supported")
    p.add_argument("--stage", choices=["auto", "development", "adult", "full", "all"], default="auto")
    p.add_argument("--count-layer", default=None)
    p.add_argument("--device", default="cpu"); p.add_argument("--chunk-size", type=int, default=50000)
    p.add_argument("--benchmarking", action="store_true"); p.add_argument("--no-auto-download", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("train", help="Train a custom ElasticNet+CLM+Transformer clock bundle")
    p.add_argument("--input", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--model-name", default="custom_scBAC")
    p.add_argument("--cell-type-column", "--celltype-col", dest="celltype_col", required=True)
    p.add_argument("--donor-col", required=True); p.add_argument("--age-col", required=True)
    p.add_argument("--dataset-col", default=None); p.add_argument("--status-col", default=None); p.add_argument("--control-label", default="CT")
    p.add_argument("--celltypes", default=None); p.add_argument("--stages", default="development,adult"); p.add_argument("--algorithms", default="elasticnet,clm,transformer")
    p.add_argument("--age-cut", type=float, default=AGE_CUT); p.add_argument("--n-folds", type=int, default=5); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count-layer", default=None); p.add_argument("--device", default="cpu"); p.add_argument("--continue-on-error", action="store_true")
    p.set_defaults(func=cmd_train)

    def add_raa_columns(q):
        q.add_argument("--donor-col", required=True); q.add_argument("--celltype-col", required=True)
        q.add_argument("--predicted-age-col", "--cell-age-pred-col", dest="predicted_age_col", required=True); q.add_argument("--age-col", "--chronological-age-col", dest="age_col", required=True); q.add_argument("--sex-col", required=True)

    p = sub.add_parser("raa-fit", help="Fit a control RAA reference and thresholds")
    p.add_argument("--input", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--output", default=None)
    add_raa_columns(p); p.add_argument("--group-col", default=None); p.add_argument("--control-groups", default=None)
    p.set_defaults(func=cmd_raa_fit)

    p = sub.add_parser("raa", help="Apply a saved/custom/pretrained RAA reference")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); add_raa_columns(p)
    p.add_argument("--reference", default=None); p.add_argument("--clock-name", default="Ensemble_Adult"); p.add_argument("--model-dir", default=None); p.add_argument("--output-col", default="RAA")
    p.set_defaults(func=cmd_raa)

    p = sub.add_parser("aaso", help="Calculate donor-level AASO from cell-level RAA")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--donor-col", required=True); p.add_argument("--celltype-col", required=True); p.add_argument("--predicted-age-col", required=True); p.add_argument("--raa-col", default="RAA")
    p.add_argument("--reference", default=None); p.add_argument("--clock-name", default="Ensemble_Adult"); p.add_argument("--model-dir", default=None)
    p.add_argument("--quantile", default="q75"); p.add_argument("--min-cells", type=int, default=10); p.add_argument("--n-points", type=int, default=100); p.add_argument("--metadata-cols", default=None); p.add_argument("--include-censored", action="store_true")
    p.set_defaults(func=cmd_aaso)

    p = sub.add_parser("analyze", help="End-to-end RAA + AASO workflow")
    p.add_argument("--input", required=True); p.add_argument("--output-prefix", required=True); add_raa_columns(p)
    p.add_argument("--reference", default=None); p.add_argument("--clock-name", default="Ensemble_Adult")
    p.add_argument("--control-input", default=None); p.add_argument("--control-group-col", default=None); p.add_argument("--control-groups", default=None)
    p.add_argument("--quantile", default="q75"); p.add_argument("--metadata-cols", default=None)
    p.add_argument("--status-col", default=None); p.add_argument("--disease-name", default=None); p.add_argument("--control-name", default=None)
    p.set_defaults(func=cmd_analyze)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        parser.exit(2, "scbac error: {}\n".format(exc))


if __name__ == "__main__":
    raise SystemExit(main())
