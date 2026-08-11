from scbac.paths import get_model_root


def test_explicit_model_root(tmp_path):
    p = tmp_path / "scBrainAgeClock_models_file"
    p.mkdir()
    assert get_model_root(p) == p.resolve()
