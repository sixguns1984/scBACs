from pathlib import Path
import numpy as np
import pandas as pd

from scbac.constants import PRETRAINED_CELLTYPES
from scbac.raa import fit_raa_reference, apply_raa, load_thresholds
from scbac.aaso import calculate_aaso


def test_pretrained_celltype_list():
    assert PRETRAINED_CELLTYPES == ["Exc", "Inh", "Ast", "Oli", "OPC", "Mic", "End", "Fib", "Per", "CAM", "T_cell"]


def test_raa_fit_apply_and_aaso(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for donor_i in range(12):
        donor = "D{}".format(donor_i)
        age = 30 + donor_i * 3
        sex = "Male" if donor_i % 2 else "Female"
        for j in range(30):
            pred = age + (j - 15) * 0.3 + rng.normal(0, 0.2)
            rows.append({"donor": donor, "ct": "Exc", "pred": pred, "age": age, "sex": sex})
    df = pd.DataFrame(rows)
    ref = tmp_path / "raa"
    fitted, models, thresholds = fit_raa_reference(
        df, ref, donor_col="donor", celltype_col="ct", predicted_age_col="pred",
        chronological_age_col="age", sex_col="sex"
    )
    assert (ref / "raa_model.pkl").exists()
    assert (ref / "thresholds.json").exists()
    assert "Exc" in thresholds and "q75" in thresholds["Exc"]
    applied, loaded = apply_raa(
        df, reference=ref, donor_col="donor", celltype_col="ct", predicted_age_col="pred",
        chronological_age_col="age", sex_col="sex"
    )
    assert applied["RAA"].notna().all()
    # Force an easily crossed threshold to exercise the AASO implementation.
    loaded = {"Exc": {"q75": -0.25}}
    aaso, curves = calculate_aaso(
        applied, loaded, donor_col="donor", celltype_col="ct", predicted_age_col="pred",
        raa_col="RAA", min_cells=10
    )
    assert isinstance(aaso, pd.DataFrame)
    assert len(curves) > 0
