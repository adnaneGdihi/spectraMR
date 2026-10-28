"""Loading arms for the cross-run report."""

import pytest

from spectramr.infrastructure.reporting.comparison.arms import (
    align_cases,
    load_arms,
    resolve_runs,
    short_labels,
)

from ._runs import ACCELS, N_CALLS, make_run


def test_short_labels_strip_the_shared_prefix_and_suffix():
    names = ["n2n_a_r2r_m4raw", "n2n_b_pair_m4raw", "n2n_g_theta20_m4raw"]
    assert short_labels(names) == ["a_r2r", "b_pair", "g_theta20"]


def test_short_labels_keep_names_that_would_collide():
    assert short_labels(["x_a_y", "x_a_y"]) == ["x_a_y", "x_a_y"]


def test_arms_are_read_at_their_selected_pass(tmp_path):
    make_run(tmp_path, "exp_a_run", selected=200)
    make_run(tmp_path, "exp_b_run", selected=None)
    arms = load_arms(resolve_runs([tmp_path]), view="selected")
    assert arms.labels == ["a", "b"]
    assert arms.steps == {"a": 200, "b": 300}


def test_per_case_rows_are_one_pass_per_arm(tmp_path):
    make_run(tmp_path, "exp_a_run", selected=100)
    make_run(tmp_path, "exp_b_run", selected=300)
    arms = load_arms(resolve_runs([tmp_path]))
    per = arms.per_case
    assert dict(per.groupby("method")["step"].unique().map(list)) == {"a": [100], "b": [300]}
    psnr = per[per["base_metric"] == "psnr"]
    assert len(psnr) == 2 * len(ACCELS) * N_CALLS
    # pairing survives the two arms being read at different passes
    keys = psnr.groupby("method")["pair_key"].apply(set)
    assert keys["a"] == keys["b"]


def test_image_cases_align_by_reference_and_acceleration(tmp_path):
    make_run(tmp_path, "exp_a_run")
    make_run(tmp_path, "exp_b_run")
    arms = load_arms(resolve_runs([tmp_path]))
    keys = align_cases(arms.cases, max_rows=10)
    assert len(keys) == 4  # 2 slices x 2 R
    assert {k.split("|")[0] for k in keys} == {"2.0", "8.0"}


def test_arms_on_different_references_do_not_align(tmp_path):
    make_run(tmp_path, "exp_a_run", reference_seed=0)
    make_run(tmp_path, "exp_b_run", reference_seed=5)
    arms = load_arms(resolve_runs([tmp_path]))
    assert align_cases(arms.cases, max_rows=10) == []


def test_one_run_is_not_a_comparison(tmp_path):
    with pytest.raises(ValueError, match="at least two runs"):
        load_arms([make_run(tmp_path, "solo")])


def test_arms_name_the_unit_of_n_from_the_validation_batch(tmp_path):
    import json

    for name in ("exp_a_run", "exp_b_run"):
        run = make_run(tmp_path, name)
        (run / "resolved_config.json").write_text(
            json.dumps({"validation": {"loader": {"batch_size": 2}}})
        )
    arms = load_arms(resolve_runs([tmp_path]), with_images=False)
    assert arms.call_unit == "validation calls (means of 2 slices)"

def _with_note(run, note):
    import json

    (run / "resolved_config.json").write_text(json.dumps({"metadata": {"comparison_note": note}}))
    return run


def test_arms_carry_each_runs_comparison_note(tmp_path):
    _with_note(make_run(tmp_path, "exp_a_run"), "two attention blocks")
    make_run(tmp_path, "exp_b_run")
    arms = load_arms(resolve_runs([tmp_path]), with_images=False)
    assert arms.notes == {"a": "two attention blocks"}


def test_a_comparison_note_that_is_not_a_string_raises(tmp_path):
    _with_note(make_run(tmp_path, "exp_a_run"), ["not", "a", "string"])
    make_run(tmp_path, "exp_b_run")
    with pytest.raises(ValueError, match="comparison_note"):
        load_arms(resolve_runs([tmp_path]), with_images=False)


def test_runs_with_the_same_name_get_parent_qualified_labels(tmp_path):
    a = make_run(tmp_path / "seedA", "run")
    b = make_run(tmp_path / "seedB", "run")
    arms = load_arms([a, b], with_images=False)
    assert len(set(arms.labels)) == 2
    assert set(arms.per_case["method"]) == set(arms.labels)


def test_duplicate_explicit_labels_raise(tmp_path):
    a, b = make_run(tmp_path, "exp_a_run"), make_run(tmp_path, "exp_b_run")
    with pytest.raises(ValueError, match="unique"):
        load_arms([a, b], labels=["x", "x"], with_images=False)


def test_target_key_digest_is_a_non_security_sha1(monkeypatch):
    import hashlib

    from spectramr.infrastructure.reporting.comparison.arms import _target_key

    real = hashlib.sha1
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("usedforsecurity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha1", spy)
    _target_key({"acceleration": 4, "target": [[1.0, 2.0]]})
    assert seen == [False]
