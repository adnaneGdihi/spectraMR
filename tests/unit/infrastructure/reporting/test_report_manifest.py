import json

from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.report_manifest import write_report_manifest


def test_manifest_lists_artifacts_with_checksums(tmp_path):
    f = tmp_path / "figures"; f.mkdir()
    (f / "fig_a.pdf").write_bytes(b"%PDF-1.4 demo")
    figures = {"fig_a": f / "fig_a.pdf", "fig_b": None}
    meta = RunMetadata(git_commit="abc1234", seed=7, dataset_version="m4raw")
    path = write_report_manifest(tmp_path, figures=figures, tables={}, metadata=meta)
    data = json.loads(path.read_text())
    assert data["provenance"]["git_commit"] == "abc1234"
    entry = next(a for a in data["artifacts"] if a["id"] == "fig_a")
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert any(a["id"] == "fig_b" and a["status"] == "skipped" for a in data["artifacts"])


class TestSkippedFiguresCarryAReason:
    """A bare ``status: skipped`` conflated three different situations.

    "this run has no learning curve to draw", "this plotter crashed" and "this
    figure id is not registered" all wrote the same manifest entry, and they need
    different responses from whoever reads it. This matters most for reports that
    follow a prediction run: the requested figure set is identical to training's
    (it derives from the task preset, not the caller), so the *only* legible
    difference between the two reports is which figures had data -- and that
    difference has to be recorded, not inferred from an absence.
    """

    def _metadata(self):
        return RunMetadata(git_commit="abc1234", seed=7, dataset_version="m4raw")

    def test_a_reason_is_stamped_when_supplied(self, tmp_path):
        path = write_report_manifest(
            tmp_path,
            figures={"fig_1_2_learning_curves": None},
            tables={},
            metadata=self._metadata(),
            figure_reasons={"fig_1_2_learning_curves": "no_data"},
        )
        entry = next(
            a
            for a in json.loads(path.read_text())["artifacts"]
            if a["id"] == "fig_1_2_learning_curves"
        )
        assert entry["status"] == "skipped"
        assert entry["reason"] == "no_data"

    def test_an_absent_reason_omits_the_key_rather_than_inventing_one(self, tmp_path):
        path = write_report_manifest(
            tmp_path,
            figures={"fig_x": None},
            tables={},
            metadata=self._metadata(),
        )
        entry = next(
            a for a in json.loads(path.read_text())["artifacts"] if a["id"] == "fig_x"
        )
        assert entry["status"] == "skipped"
        assert "reason" not in entry, (
            "a fabricated reason is worse than none -- it reads as a diagnosis"
        )

    def test_an_emitted_figure_carries_no_reason(self, tmp_path):
        real = tmp_path / "fig_ok.pdf"
        real.write_bytes(b"%PDF-1.4\n")
        path = write_report_manifest(
            tmp_path,
            figures={"fig_ok": real},
            tables={},
            metadata=self._metadata(),
            figure_reasons={"fig_ok": "no_data"},  # stale/wrong input on purpose
        )
        entry = next(
            a for a in json.loads(path.read_text())["artifacts"] if a["id"] == "fig_ok"
        )
        assert entry["status"] == "ok"
        assert "reason" not in entry


def test_runs_are_recorded_only_when_supplied(tmp_path):
    meta = RunMetadata(git_commit="abc1234", seed=7, dataset_version="m4raw")
    bare = write_report_manifest(tmp_path / "a", figures={}, tables={}, metadata=meta)
    assert "runs" not in json.loads(bare.read_text())
    runs = [{"arm": "crashed", "included": False, "status": "failed"}]
    path = write_report_manifest(tmp_path / "b", figures={}, tables={}, metadata=meta, runs=runs)
    assert json.loads(path.read_text())["runs"] == runs
