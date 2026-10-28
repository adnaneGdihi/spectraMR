"""The corpus drain rewrites a pre-1.0 checkpoint path to the producer's best file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _drain():
    path = REPO / "scripts" / "migrations" / "drain_checkpoint_paths_v1.py"
    spec = importlib.util.spec_from_file_location("drain_checkpoint_paths_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path, producer_format: str | None) -> Path:
    fmt = f"checkpoint:\n  format: {producer_format}\n" if producer_format else ""
    (tmp_path / "producer.yaml").write_text(
        f"training:\n  output_dir: experiments/results/a\n{fmt}"
    )
    consumer = tmp_path / "consumer.yaml"
    consumer.write_text(
        "model:\n  # warm-start from a's checkpoint_best.pt\n"
        "  checkpoint_path: experiments/results/a/checkpoints/checkpoint_best.pt\n"
    )
    return consumer


def test_a_pre_v1_value_becomes_the_producers_best(tmp_path: Path) -> None:
    d = _drain()
    consumer = _corpus(tmp_path, None)
    changes = d.drain([consumer], d.producer_formats([tmp_path], tracked=False), write=True)
    assert changes == [
        (
            consumer,
            "experiments/results/a/checkpoints/checkpoint_best.pt",
            "experiments/results/a/checkpoints/best.safetensors",
        )
    ]
    assert "checkpoints/best.safetensors" in consumer.read_text()


def test_a_pt_producer_keeps_best_pt(tmp_path: Path) -> None:
    d = _drain()
    consumer = _corpus(tmp_path, "pt")
    d.drain([consumer], d.producer_formats([tmp_path], tracked=False), write=True)
    assert "checkpoints/best.pt\n" in consumer.read_text()
    assert d.check([tmp_path], tracked=False) == []


def test_check_reports_a_stale_value_and_a_wrong_extension(tmp_path: Path) -> None:
    d = _drain()
    _corpus(tmp_path, None)
    findings = d.check([tmp_path], tracked=False)
    assert len(findings) == 1 and "checkpoint_best.pt" in findings[0]
    (tmp_path / "consumer.yaml").write_text(
        "model:\n  checkpoint_path: experiments/results/a/checkpoints/best.pt\n"
    )
    findings = d.check([tmp_path], tracked=False)
    assert len(findings) == 1 and "best.pt" in findings[0]


@pytest.mark.parametrize(
    "name", ["checkpoint_epoch_3.pt", "checkpoint_step_9.pt", "checkpoint_last.pt"]
)
def test_check_reports_every_stale_file_name_shape(tmp_path: Path, name: str) -> None:
    d = _drain()
    (tmp_path / "c.yaml").write_text(f"model:\n  checkpoint_path: r/checkpoints/{name}\n")
    assert len(d.check([tmp_path], tracked=False)) == 1


def test_an_unknown_producer_defaults_to_safetensors(tmp_path: Path) -> None:
    d = _drain()
    assert d.target_for("x/y/checkpoints/best.pt", {}) == "x/y/checkpoints/best.safetensors"
    assert d.target_for("x/y/checkpoints/best.safetensors", {}) is None


def test_the_rewrite_keeps_comments_and_other_values(tmp_path: Path) -> None:
    d = _drain()
    consumer = _corpus(tmp_path, None)
    d.drain([consumer], {}, write=True)
    assert "# warm-start from a's checkpoint_best.pt" in consumer.read_text()


def test_a_path_inside_prose_is_rewritten_with_its_value(tmp_path: Path) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: r/checkpoints/best.pt\nnote: |\n  see r/checkpoints/best.pt here\n")
    d.drain([c], {}, write=True)
    assert (
        c.read_text()
        == "a: r/checkpoints/best.safetensors\nnote: |\n  see r/checkpoints/best.safetensors here\n"
    )


def test_a_rewrite_that_changes_more_than_the_value_is_refused(tmp_path: Path) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    text = "a: r/checkpoints/best.pt\nr/checkpoints/best.pt: 1\n"  # the path is also a mapping key
    c.write_text(text)
    with pytest.raises(ValueError):
        d.drain([c], {}, write=True)
    assert c.read_text() == text


def test_a_longer_extension_is_not_a_prefix_collision(tmp_path: Path) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: r/checkpoints/best.pt\nb: r/checkpoints/best.pth\n")
    d.drain([c], {"r": "pt"}, write=True)
    assert c.read_text() == "a: r/checkpoints/best.pt\nb: r/checkpoints/best.pt\n"
    c.write_text("a: r/checkpoints/best.pt\nb: r/checkpoints/best.pth\n")
    d.drain([c], {}, write=True)
    assert c.read_text() == "a: r/checkpoints/best.safetensors\nb: r/checkpoints/best.safetensors\n"


def test_a_name_that_continues_the_path_is_left_alone(tmp_path: Path) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    text = (
        "a: r/checkpoints/best.pt\nbackup: r/checkpoints/best.pt.bak\nn: r/checkpoints/best.ptx\n"
    )
    c.write_text(text)
    d.drain([c], {}, write=True)
    assert c.read_text() == text.replace(
        "a: r/checkpoints/best.pt\n", "a: r/checkpoints/best.safetensors\n"
    )
    assert d.target_for("r/checkpoints/best.pt.bak", {}) is None


def test_prose_naming_a_path_resolves_the_producer_of_that_path(tmp_path: Path) -> None:
    d = _drain()
    value = "loads r/checkpoints/best.pt as the prior."
    assert d.target_for(value, {"r": "pt"}) is None
    assert (
        d.target_for(value, {"r": "safetensors"})
        == "loads r/checkpoints/best.safetensors as the prior."
    )
    (tmp_path / "p.yaml").write_text("training:\n  output_dir: r\ncheckpoint:\n  format: pt\n")
    (tmp_path / "c.yaml").write_text(f"note: {value}\n")
    assert d.check([tmp_path], tracked=False) == []


def test_a_sentence_final_path_is_still_matched() -> None:
    d = _drain()
    assert d.target_for("see r/checkpoints/best.pt.", {}) == "see r/checkpoints/best.safetensors."


def test_unparseable_and_multi_document_files_are_never_clean(tmp_path: Path) -> None:
    d = _drain()
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\nb: r/checkpoints/checkpoint_best.pt\n")
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        d.check([tmp_path], tracked=False)
    bad.write_text("a: 1\n---\nb: r/checkpoints/checkpoint_best.pt\n")
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        d.check([tmp_path], tracked=False)


def test_check_reports_best_ema_and_a_non_default_producers_wrong_extension(tmp_path: Path) -> None:
    d = _drain()
    (tmp_path / "p.yaml").write_text("training:\n  output_dir: r\ncheckpoint:\n  format: pt\n")
    (tmp_path / "c.yaml").write_text(
        "a: r/checkpoints/best.safetensors\nb: r/checkpoints/checkpoint_best_ema.pt\n"
    )
    findings = d.check([tmp_path], tracked=False)
    assert len(findings) == 2


def test_a_non_string_change_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: r/checkpoints/best.pt\nn: 1\n")
    # A rewrite that also touched the integer would have to change the text it re-parses.
    monkeypatch.setattr(d, "_rewrite_text", lambda text, pairs: text.replace("n: 1", "n: 2"))
    with pytest.raises(ValueError):
        d.drain([c], {}, write=True)
    assert c.read_text() == "a: r/checkpoints/best.pt\nn: 1\n"


def test_producers_that_disagree_on_a_run_dir_raise(tmp_path: Path) -> None:
    d = _drain()
    (tmp_path / "a.yaml").write_text("training:\n  output_dir: r\ncheckpoint:\n  format: pt\n")
    (tmp_path / "b.yaml").write_text("training:\n  output_dir: r\n")
    with pytest.raises(ValueError, match=r"output_dir"):
        d.producer_formats([tmp_path], tracked=False)
    (tmp_path / "b.yaml").write_text("training:\n  output_dir: r\ncheckpoint:\n  format: pt\n")
    assert d.producer_formats([tmp_path], tracked=False) == {"r": "pt"}


def test_an_unresolved_producer_is_logged_with_its_file(tmp_path: Path, capsys) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: nowhere/checkpoints/best.pt\n")
    d.drain([c], {}, write=False)
    assert "c.yaml" in capsys.readouterr().err


def test_the_default_format_is_the_core_owner() -> None:
    from spectramr.core.checkpoint_formats import DEFAULT_CHECKPOINT_FORMAT

    assert _drain().default_format() is DEFAULT_CHECKPOINT_FORMAT


@pytest.mark.parametrize("tail", ["-v2", "+ema", "~", "@step", "*", "\u00e4", "/m.bin", "?dl=1"])
def test_a_name_that_continues_past_the_extension_is_not_a_v1_path(
    tmp_path: Path, tail: str
) -> None:
    d = _drain()
    value = (
        f"https://h/r/checkpoints/best.pt{tail}"
        if tail == "?dl=1"
        else f"r/checkpoints/best.pt{tail}"
    )
    assert d.target_for(value, {}) is None
    (tmp_path / "c.yaml").write_text(f"a: '{value}'\n")
    assert d.check([tmp_path], tracked=False) == []


@pytest.mark.parametrize("after", [")", "]", "}", ",", ";", ":", "'", '"', "`", " x", "."])
def test_a_delimiter_after_the_extension_ends_the_path(after: str) -> None:
    d = _drain()
    assert d.target_for(f"see r/checkpoints/best.pt{after}", {}) == (
        f"see r/checkpoints/best.safetensors{after}"
    )


def test_the_per_string_guard_refuses_a_wrong_extension(tmp_path: Path, monkeypatch) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: [r/checkpoints/checkpoint_best.pt]\n")
    assert d.drain([c], {"r": "pt"}, write=False)[0][2] == "r/checkpoints/best.pt"
    monkeypatch.setattr(
        d,
        "_rewrite_text",
        lambda text, pairs: text.replace("checkpoint_best.pt", "best.safetensors"),
    )
    with pytest.raises(ValueError):
        d.drain([c], {"r": "pt"}, write=True)
    assert c.read_text() == "a: [r/checkpoints/checkpoint_best.pt]\n"


def test_check_reports_a_best_ema_path_by_the_needle_alone(tmp_path: Path) -> None:
    d = _drain()
    assert d.target_for("r/checkpoints/best_ema.pt", {}) is None
    (tmp_path / "c.yaml").write_text("a: r/checkpoints/best_ema.pt\n")
    assert len(d.check([tmp_path], tracked=False)) == 1


def test_a_dot_slash_run_is_the_same_run(tmp_path: Path) -> None:
    d = _drain()
    (tmp_path / "p.yaml").write_text("training:\n  output_dir: ./r/\ncheckpoint:\n  format: pt\n")
    (tmp_path / "c.yaml").write_text(
        "a: r/checkpoints/best.safetensors\nb: ./r/checkpoints/best.safetensors\n"
    )
    assert len(d.check([tmp_path], tracked=False)) == 2
    d.drain([tmp_path / "c.yaml"], d.producer_formats([tmp_path], tracked=False), write=True)
    assert (
        tmp_path / "c.yaml"
    ).read_text() == "a: r/checkpoints/best.pt\nb: ./r/checkpoints/best.pt\n"


def test_a_run_name_with_a_space_is_rewritten_with_its_producers_format(tmp_path: Path) -> None:
    d = _drain()
    (tmp_path / "p.yaml").write_text("training:\n  output_dir: my run\ncheckpoint:\n  format: pt\n")
    c = tmp_path / "c.yaml"
    c.write_text(
        "a: my run/checkpoints/checkpoint_best.pt  # my run/checkpoints/checkpoint_best.pt\n"
    )
    d.drain([c], d.producer_formats([tmp_path], tracked=False), write=True)
    assert c.read_text() == "a: my run/checkpoints/best.pt  # my run/checkpoints/best.pt\n"


def test_a_flow_list_path_is_rewritten_not_refused(tmp_path: Path) -> None:
    d = _drain()
    c = tmp_path / "c.yaml"
    c.write_text("a: [r/checkpoints/checkpoint_best.pt, {k: r/checkpoints/best.pt}]\n")
    d.drain([c], {"r": "pt"}, write=True)
    assert c.read_text() == "a: [r/checkpoints/best.pt, {k: r/checkpoints/best.pt}]\n"
