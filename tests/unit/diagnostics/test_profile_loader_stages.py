import cProfile
import importlib.util
import pathlib
import pstats
import time

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "diagnostics"
    / "profile_loader_stages.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("profile_loader_stages", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_of_names_each_loader_stage():
    mod = _module()
    assert mod.stage_of("m4raw_dataset.py:_group_shifts") == "registration shifts"
    assert mod.stage_of("m4raw_dataset.py:_load_kspace") == "h5 read"
    assert mod.stage_of("rep_mask_patterns.py:_draw_poisson") == "mask transform"
    assert mod.stage_of("builtins:len") is None


def test_stage_totals_attributes_time_to_the_stage_that_spent_it():
    mod = _module()

    def _load_kspace():
        time.sleep(0.02)

    profile = cProfile.Profile()
    profile.enable()
    for _ in range(2):
        _load_kspace()
    profile.disable()
    totals = mod.stage_totals(pstats.Stats(profile), 2)
    assert totals["h5 read"] >= 0.018
