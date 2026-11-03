"""Campaign corpus gate over ``experiments/campaigns/*.yaml``.

Checks, per campaign:
  - it loads through ``CampaignConfigSchema`` (which also enforces ``compare_to`` targets);
  - every declared ``config`` path exists under the repo root;
  - a campaign whose name has no ``smoke``/``hpo``/``example`` token has an enabled baseline;
  - no enabled arm has ``metadata.status`` in the stub vocabulary.

Violations are a ratchet: ``experiments/campaigns/gate_baseline.json`` lists today's
violators. A new violator fails; a baseline entry that no longer violates also fails,
so the list only shrinks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spectramr.config.schemas.campaign import CampaignConfigSchema

REPO_ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN_DIR = REPO_ROOT / "experiments" / "campaigns"
BASELINE_PATH = CAMPAIGN_DIR / "gate_baseline.json"
EXEMPT_TOKENS = frozenset({"smoke", "hpo", "example"})
STUB_STATUSES = frozenset({"needs_implementation", "needs_data", "blocked", "inert"})


def _exempt(stem: str) -> bool:
    return any(token in EXEMPT_TOKENS for token in stem.split("_"))


def _arm_status(path: Path) -> object:
    data = yaml.safe_load(path.read_text()) or {}
    meta = data.get("metadata") or {}
    return meta.get("status")


def collect_violations(campaign_dir: Path, root: Path) -> set[str]:
    """Return one key per violated check, for every campaign under ``campaign_dir``."""
    violations: set[str] = set()
    for path in sorted(campaign_dir.glob("*.yaml")):
        stem = path.stem
        try:
            cfg = CampaignConfigSchema.from_yaml(str(path))
        except (ValueError, OSError, yaml.YAMLError):
            violations.add(f"load:{stem}")
            continue
        for exp in cfg.all_declared_experiments:
            if not (root / exp.config).exists():
                violations.add(f"missing_config:{stem}:{exp.config}")
        if not _exempt(stem) and not cfg.baseline_experiments:
            violations.add(f"no_baseline:{stem}")
        for exp in cfg.enabled_experiments:
            arm = root / exp.config
            if not arm.exists():
                continue
            status = _arm_status(arm)
            if status in STUB_STATUSES:
                violations.add(f"stub_arm:{stem}:{exp.config}:{status}")
    return violations


def _baseline() -> set[str]:
    return set(json.loads(BASELINE_PATH.read_text())["violations"])


def _write_campaign(root: Path, stem: str, body: str) -> Path:
    path = root / f"{stem}.yaml"
    path.write_text(body)
    return path


def _arm(root: Path, rel: str, status: str | None = None) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = f"metadata:\n  status: {status}\n" if status else ""
    target.write_text(meta)


# ── The real corpus ─────────────────────────────────────────────


def test_corpus_introduces_no_new_violations():
    current = collect_violations(CAMPAIGN_DIR, REPO_ROOT)
    new = sorted(current - _baseline())
    assert not new, f"new campaign gate violations (fix them, do not baseline): {new}"


def test_baseline_lists_only_live_violations():
    stale = sorted(_baseline() - collect_violations(CAMPAIGN_DIR, REPO_ROOT))
    assert not stale, f"remove these from gate_baseline.json (now fixed): {stale}"


# ── Planted violations: each check must turn red ────────────────


def test_planted_missing_config_is_flagged(tmp_path: Path):
    _write_campaign(
        tmp_path,
        "p_missing",
        "name: p_missing\nexperiments:\n  - name: B\n    config: nope/absent.yaml\n    role: baseline\n",
    )
    assert "missing_config:p_missing:nope/absent.yaml" in collect_violations(tmp_path, tmp_path)


def test_planted_campaign_without_a_baseline_is_flagged(tmp_path: Path):
    _arm(tmp_path, "arms/v.yaml")
    _write_campaign(
        tmp_path,
        "p_nobase",
        "name: p_nobase\nexperiments:\n  - name: V\n    config: arms/v.yaml\n    role: variant\n",
    )
    assert "no_baseline:p_nobase" in collect_violations(tmp_path, tmp_path)


def test_smoke_campaign_without_a_baseline_is_exempt(tmp_path: Path):
    _arm(tmp_path, "arms/v.yaml")
    _write_campaign(
        tmp_path,
        "debug_smoke_x",
        "name: debug_smoke_x\nexperiments:\n  - name: V\n    config: arms/v.yaml\n    role: variant\n",
    )
    assert not any(v.startswith("no_baseline") for v in collect_violations(tmp_path, tmp_path))


def test_planted_enabled_stub_arm_is_flagged(tmp_path: Path):
    _arm(tmp_path, "arms/b.yaml", status="needs_implementation")
    _write_campaign(
        tmp_path,
        "p_stub",
        "name: p_stub\nexperiments:\n  - name: B\n    config: arms/b.yaml\n    role: baseline\n",
    )
    assert "stub_arm:p_stub:arms/b.yaml:needs_implementation" in collect_violations(
        tmp_path, tmp_path
    )


def test_disabled_stub_arm_is_not_flagged(tmp_path: Path):
    _arm(tmp_path, "arms/b.yaml", status="ready")
    _arm(tmp_path, "arms/s.yaml", status="needs_data")
    _write_campaign(
        tmp_path,
        "p_disabled",
        "name: p_disabled\nexperiments:\n"
        "  - name: B\n    config: arms/b.yaml\n    role: baseline\n"
        "  - name: S\n    config: arms/s.yaml\n    role: variant\n    enabled: false\n",
    )
    assert not any(v.startswith("stub_arm") for v in collect_violations(tmp_path, tmp_path))


def test_planted_dangling_compare_to_is_a_load_violation(tmp_path: Path):
    _arm(tmp_path, "arms/b.yaml")
    _arm(tmp_path, "arms/x.yaml")
    _write_campaign(
        tmp_path,
        "p_dangling",
        "name: p_dangling\nexperiments:\n"
        "  - name: B\n    config: arms/b.yaml\n    role: baseline\n"
        "  - name: X\n    config: arms/x.yaml\n    compare_to: [ghost]\n",
    )
    assert "load:p_dangling" in collect_violations(tmp_path, tmp_path)


def test_ratchet_would_reject_a_violation_outside_the_baseline(tmp_path: Path):
    _write_campaign(
        tmp_path,
        "p_new",
        "name: p_new\nexperiments:\n  - name: B\n    config: gone.yaml\n    role: baseline\n",
    )
    new = collect_violations(tmp_path, tmp_path) - _baseline()
    assert "missing_config:p_new:gone.yaml" in new


@pytest.mark.parametrize("token", ["smoke", "hpo", "example"])
def test_exempt_tokens_match_whole_words(token: str):
    assert _exempt(f"x_{token}_y")
    assert not _exempt(f"x_{token}s_y")
