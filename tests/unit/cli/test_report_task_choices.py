"""The `report --task` choices must cover every preset that exists.

`TASK_PRESETS["calibration"]` existed while the CLI's `choices=` list omitted it,
so a curated preset was advertised in the code and unreachable from the command
line -- an unwired knob (non-negotiable 8). The list cannot be derived at parser
build time (importing `reporting.pipeline` costs ~3.3s and the parser is built
for every command, `--help` included), so agreement is enforced here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[3] / "src" / "spectramr" / "cli" / "app.py"


def _cli_task_choices() -> list[str]:
    """The literal `choices=` list bound to `report --task`.

    Read from source rather than by building the parser: importing `cli.app`
    pulls a wide surface, and this test's whole point is to avoid paying that
    import to learn a list of strings.
    """
    src = APP.read_text()
    m = re.search(r'"--task",\s*default=None,\s*(?:#[^\n]*\n\s*)*choices=\[(.*?)\]', src, re.S)
    assert m, "could not locate the report --task choices list in cli/app.py"
    return sorted(re.findall(r'"([a-z_]+)"', m.group(1)))


def test_report_task_choices_cover_every_preset() -> None:
    from spectramr.infrastructure.reporting.pipeline import TASK_PRESETS

    cli = _cli_task_choices()
    presets = sorted(TASK_PRESETS)
    assert cli == presets, (
        f"CLI --task choices and TASK_PRESETS disagree.\n"
        f"  unreachable from the CLI: {sorted(set(presets) - set(cli))}\n"
        f"  offered but has no preset (would raise): {sorted(set(cli) - set(presets))}"
    )


def test_calibration_is_selectable() -> None:
    """Named explicitly: it is the one that was missing, and the reason this exists."""
    assert "calibration" in _cli_task_choices()


def test_the_extractor_is_not_vacuous() -> None:
    """Guards the regex, not the code under test.

    If the `choices=` block is ever reformatted past what this pattern matches,
    the search would return an empty list and the comparison above could pass by
    finding nothing on both sides. Assert it really parsed something.
    """
    choices = _cli_task_choices()
    assert len(choices) >= 7, choices
    assert "default" in choices


def test_the_config_enum_agrees_with_both() -> None:
    """The third enumeration -- the one a CLI-only check cannot see.

    `report.task` is validated config-side by the `ReportTask` enum, so the tasks
    are spelled out in *three* places: this enum, `TASK_PRESETS`, and the CLI
    `choices=` literal. A task present in the first two but absent from the enum
    passes argparse and is then rejected by config validation -- reachable on the
    command line, unusable in a YAML. Pin all three together.
    """
    from spectramr.config.schemas.enums import ReportTask
    from spectramr.infrastructure.reporting.pipeline import TASK_PRESETS

    enum_values = sorted(m.value for m in ReportTask)
    assert enum_values == sorted(TASK_PRESETS), (
        f"ReportTask and TASK_PRESETS disagree.\n"
        f"  accepted in YAML but has no preset: "
        f"{sorted(set(enum_values) - set(TASK_PRESETS))}\n"
        f"  has a preset but rejected in YAML: "
        f"{sorted(set(TASK_PRESETS) - set(enum_values))}"
    )
    assert enum_values == _cli_task_choices()


def _cli_style_choices() -> list[str]:
    src = APP.read_text()
    m = re.search(r'"--style",\s*default=None,\s*(?:#[^\n]*\n\s*)*choices=\[(.*?)\]', src, re.S)
    assert m, "could not locate the report --style choices list in cli/app.py"
    return sorted(re.findall(r'"([a-z_]+)"', m.group(1)))


def test_report_style_choices_equal_the_schema_enum() -> None:
    """`--style` is written out for the same import-cost reason as `--task`; a
    style the schema accepts but the CLI cannot select is the same unwired knob."""
    from spectramr.config.schemas.enums import ReportStyle

    assert _cli_style_choices() == sorted(s.value for s in ReportStyle)
