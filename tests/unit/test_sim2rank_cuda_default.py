"""Regression tests for the sim2rank + meta-evaluate CUDA-canonical default.

The 2026-05-25 change flipped CPU → CUDA as the canonical execution backend
for both the Sim2Rank metric-validation pipeline and the meta-evaluate CLI.

Rationale (supersedes the 2026-05-14 CPU-canonical decision):

- The native-data path now runs ESPIRiT coil estimation, complex k-space
  degradation sweeps, and Inception feature extraction — all GPU-bound. The
  phase / g-factor metrics are scored against the *real* coil maps + complex
  image (not the FourierBridge's synthetic stand-ins), which adds real
  linear-algebra (SVD, SENSE combine) to the hot path.
- ``--device`` defaults to ``auto`` (CUDA when available, else CPU) on every
  CLI, resolved by ``spectramr.cli.app._resolve_device``. The pipeline stays
  FP32 throughout (bf16 remains opt-in).
- The SLURM templates request one GPU (``--gres=gpu:1``) and pass
  ``--device cuda``.

These tests pin the CUDA default on all sim2rank CLI entry points + both
SLURM templates so a future "let's flip back to cpu" refactor either updates
the tests + docs (intended) or fails CI (unintended).
"""

from __future__ import annotations

import re

import pytest

from tests.utils.repo_scripts import require_repo_file

# Entry points that expose ``--device`` and must default to ``"auto"``.
_CLI_ENTRY_POINTS = (
    "scripts/sim2rank/sim2rank.py",
    "scripts/sim2rank/sim2rank_diagnostics.py",
    "scripts/sim2rank/run_meta_eval_all_datasets.py",
    "scripts/sim2rank/run_all_generations.py",
)


def _read(relpath: str) -> str:
    """Read a subject that ships only in the private tree.

    ``scripts/sim2rank/`` is denied by the public allowlist (65 files, 36.8k LOC
    of research tooling), so in the export every one of these reads raised
    ``FileNotFoundError``. The guard is per-test rather than module-level: the
    device-resolver test below has no ``scripts/`` subject and must keep running
    in both trees.
    """
    return require_repo_file(relpath).read_text()


@pytest.mark.parametrize("relpath", _CLI_ENTRY_POINTS)
def test_cli_device_default_is_auto(relpath: str) -> None:
    """Every sim2rank CLI must default ``--device`` to ``"auto"`` (CUDA-first)."""
    text = _read(relpath)
    good = re.compile(
        r"""add_argument\(\s*["']--device["'][^)]*default\s*=\s*["']auto["']""",
        re.DOTALL,
    )
    assert good.search(text), (
        f"{relpath} must declare ``--device`` with default ``'auto'`` — "
        "CUDA is the canonical sim2rank execution backend (2026-05-25)."
    )
    # And — critically — there must be no lingering ``default="cpu"`` for --device.
    bad = re.compile(
        r"""add_argument\(\s*["']--device["'][^)]*default\s*=\s*["']cpu["']""",
        re.DOTALL,
    )
    assert not bad.search(text), (
        f"{relpath} still declares ``--device`` with a cpu default; "
        'flip to ``default="auto"`` or update this test.'
    )


def test_cli_resolves_device_via_shared_helper() -> None:
    """Each CLI must route ``args.device`` through ``_resolve_device`` rather
    than the bare ``torch.device(args.device)`` (which can't honour ``auto``)."""
    for relpath in _CLI_ENTRY_POINTS:
        text = _read(relpath)
        assert "_resolve_device(args.device)" in text, (
            f"{relpath} must resolve --device via _resolve_device(args.device)."
        )
        assert "torch.device(args.device)" not in text, (
            f"{relpath} still uses torch.device(args.device); 'auto' won't resolve."
        )


def test_resolve_device_prefers_cuda_and_refuses_to_guess_cpu() -> None:
    """``_resolve_device('auto')`` picks CUDA, or raises — never CPU silently.

    The second half of the old assertion ("CPU otherwise") is the behaviour
    non-negotiable 9b abolished, and it duly failed on the CPU test array.
    sim2rank's documented CPU exception is reached by ASKING — ``--device cpu``,
    the line at the bottom — not by ``auto`` giving up.
    """
    torch = pytest.importorskip("torch")
    from spectramr.cli.app import _resolve_device
    from spectramr.core.compute_device import AcceleratorRequiredError

    if torch.cuda.is_available():
        assert _resolve_device("auto").type == "cuda"
    else:
        with pytest.raises(AcceleratorRequiredError):
            _resolve_device("auto")

    # The explicit opt-in works on either host — that is what makes it an
    # exception rather than a fallback.
    assert _resolve_device("cpu").type == "cpu"


def test_slurm_submit_requests_gpu() -> None:
    """``submit_sim2rank.sbatch`` must request a GPU (``--gres=gpu``)."""
    text = _read("scripts/sim2rank/submit_sim2rank.sbatch")
    assert any(
        ln.strip().startswith("#SBATCH") and "--gres=gpu" in ln for ln in text.splitlines()
    ), "submit_sim2rank.sbatch must request a GPU (CUDA-canonical as of 2026-05-25)."


def test_slurm_submit_device_default_is_cuda() -> None:
    """The SLURM submission script's ``DEVICE`` env-default must be ``cuda``."""
    text = _read("scripts/sim2rank/submit_sim2rank.sbatch")
    assert 'DEVICE="${DEVICE:-cuda}"' in text, (
        "submit_sim2rank.sbatch must default DEVICE to 'cuda'. "
        "Users can still override with `DEVICE=cpu sbatch ...`."
    )


def test_meta_eval_sbatch_requests_gpu() -> None:
    """``mata_eval.sbatch`` must request a GPU."""
    text = _read("mata_eval.sbatch")
    assert any(
        ln.strip().startswith("#SBATCH") and ("--gres=gpu" in ln or "--gpus=" in ln)
        for ln in text.splitlines()
    ), "mata_eval.sbatch must request a GPU (CUDA-canonical as of 2026-05-25)."


def test_meta_eval_sbatch_invokes_cli_with_device_cuda() -> None:
    """The meta-evaluate sbatch must pass ``--device cuda`` to the CLI."""
    raw = _read("mata_eval.sbatch")
    active_lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")]
    active = "\n".join(active_lines)
    assert re.search(r"--device\s+cuda(\s|\\|$)", active), (
        "mata_eval.sbatch must invoke `python -m spectramr.cli meta-evaluate --device cuda ...`."
    )
    assert not re.search(r"--device\s+cpu", active), (
        "mata_eval.sbatch must not pass `--device cpu` — CUDA is canonical."
    )


def test_readme_documents_cuda_canonical() -> None:
    """The README must call out CUDA as the canonical execution backend."""
    text = _read("scripts/sim2rank/README.md")
    assert "CUDA is the canonical execution backend" in text or ("CUDA is canonical" in text), (
        "README must document that CUDA is the canonical sim2rank execution "
        "backend (native-data path runs ESPIRiT + complex k-space + Inception, "
        "all GPU-bound)."
    )


# ── #379: the synthetic sweep had never run on the default device ─────────────


def _synthetic_sweep():
    """Import ``_generate_synthetic_axis_sweeps`` from the sim2rank script.

    Loaded by path because ``scripts/sim2rank/sim2rank.py`` is a script, not a
    package module.
    """
    import importlib.util
    import sys

    path = require_repo_file("scripts/sim2rank/sim2rank.py")
    spec = importlib.util.spec_from_file_location("_s2r_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s2r_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod._generate_synthetic_axis_sweeps


@pytest.mark.gpu
def test_synthetic_sweep_runs_on_cuda() -> None:
    """The regression for #379.

    ``--synthetic`` seeds a CPU generator (deliberately -- a CUDA generator draws
    from a different stream, so the same ``--seed`` would give different sweeps on
    different hardware). It then added that CPU sample straight to ``img_gt``,
    which lives on ``--device``. Since the accelerated-run contract makes an
    accelerator the default, the synthetic path could only ever run on the one
    device the contract discourages: it worked under ``--device cpu`` and raised
    everywhere else.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    sweep = _synthetic_sweep()
    img = torch.rand(1, 1, 16, 16, device="cuda")
    out = sweep(img, n_timesteps=3, axes=["noise", "motion"], seed=0)
    assert set(out) == {"noise", "motion"}
    for axis, imgs in out.items():
        assert len(imgs) == 3, axis
        for t in imgs:
            assert t.device.type == "cuda", f"{axis} left a sample on {t.device}"
            assert t.shape == img.shape
            assert torch.isfinite(t).all()


def test_synthetic_sweep_is_device_independent() -> None:
    """The CPU generator is the point, so the SAME seed must give the SAME sweep
    regardless of device -- otherwise the fix could have been 'move the generator
    to CUDA', which would silently break reproducibility across hardware."""
    torch = pytest.importorskip("torch")
    sweep = _synthetic_sweep()
    img_cpu = torch.rand(1, 1, 8, 8)
    cpu = sweep(img_cpu, n_timesteps=2, axes=["noise"], seed=7)["noise"]
    if not torch.cuda.is_available():
        pytest.skip("cross-device comparison needs CUDA")
    gpu = sweep(img_cpu.cuda(), n_timesteps=2, axes=["noise"], seed=7)["noise"]
    for a, b in zip(cpu, gpu, strict=True):
        assert torch.allclose(a, b.cpu(), atol=1e-6)


def test_synthetic_sweep_severity_is_monotone_on_cpu() -> None:
    """Device-agnostic sanity: the sweep must actually sweep. A constant-severity
    sweep would satisfy the device assertions above while measuring nothing."""
    torch = pytest.importorskip("torch")
    sweep = _synthetic_sweep()
    img = torch.rand(1, 1, 32, 32)
    imgs = sweep(img, n_timesteps=4, axes=["noise"], seed=0)["noise"]
    errs = [float((t - img).abs().mean()) for t in imgs]
    assert errs == sorted(errs), f"severity is not increasing: {errs}"
    assert errs[-1] > errs[0] * 1.5, f"sweep barely moves: {errs}"


def test_synthetic_branch_leaves_no_variable_unbound() -> None:
    """The bug the device fix uncovered.

    ``clean_refs`` and ``axis_populations`` were initialised only inside the
    real-data arm of ``if args.synthetic: ... else: ...``, but the per-axis
    distributional block that reads them sits OUTSIDE the branch. Under
    ``--synthetic`` the names were therefore unbound, and the run died with
    ``UnboundLocalError`` immediately after the sweep -- with the metric
    evaluation already paid for and nothing written to disk.

    It stayed hidden because #379 crashed the synthetic path earlier, at the
    first axis: one bug was standing in front of the other.

    Asserted structurally rather than by running the pipeline (a full sweep is
    minutes of GPU): every name the post-branch code reads must be assigned at
    the function's top level BEFORE the branch, so both arms bind it.
    """
    import ast

    path = require_repo_file("scripts/sim2rank/sim2rank.py")
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_sim2rank"
    )

    branch = next(
        (
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and any(
                isinstance(d, ast.Attribute) and d.attr == "synthetic" for d in ast.walk(n.test)
            )
        ),
        None,
    )
    assert branch is not None, "the --synthetic branch moved; update this guard"

    bound_before: set[str] = set()
    for stmt in fn.body:
        if stmt is branch:
            break
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound_before.add(node.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound_before.add(node.target.id)

    for name in ("clean_refs", "axis_populations"):
        assert name in bound_before, (
            f"{name!r} is not bound before the `if args.synthetic` branch. If only "
            "the real-data arm assigns it, --synthetic raises UnboundLocalError at "
            "the first read after the branch."
        )
