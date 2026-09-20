"""Every dc_method this generator builds takes the k-space flag, so no retry is needed.

The generator's DC call was wrapped in two nested ``except TypeError`` retries, the
last of which dropped ``is_kspace_domain``. That kwarg defaults to ``False`` on 6
of the 7 layers, so the retry would IFFT a tensor that is already k-space, blend,
and FFT back -- with the same shape and different physics.

The retry could never be entered by a signature mismatch, because all seven accept
both kwargs. Its only reachable entry was a genuine ``TypeError`` raised inside a
layer's forward, which is exactly when a silent retry is most harmful. These tests
pin both halves: the signatures that make the retry dead, and the numerical gap
that made it dangerous.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.data_consistency import (  # noqa: E402
    AdaptiveDataConsistency,
    HardDataConsistency,
    NoiseAdaptiveDataConsistency,
    SimpleDataConsistency,
    SoftDataConsistency,
    TargetAwareFSDC,
)
from spectramr.infrastructure.physics.kan_data_consistency import (  # noqa: E402
    KANAdaptiveDataConsistency,
)


def _layers() -> dict[str, torch.nn.Module]:
    """One instance per branch of the generator's dc_method dispatch."""
    return {
        "soft": SoftDataConsistency(lambda_init=0.5),
        "adaptive": AdaptiveDataConsistency(),
        "noise_adaptive": NoiseAdaptiveDataConsistency(beta=0.5),
        "hard": HardDataConsistency(train_noise_level=0.0, eval_noise_level=0.0),
        "target_aware_fsdc": TargetAwareFSDC(target_channels=2, center_fraction=0.08),
        "simple": SimpleDataConsistency(method="simple", weight=0.5),
        "kan_adaptive": KANAdaptiveDataConsistency(),
    }


def _batch():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    measured = torch.randn(1, 2, 16, 16, dtype=torch.complex64)
    mask = (torch.rand(1, 1, 16, 16) > 0.5).to(torch.complex64)
    return x, measured, mask


@pytest.mark.parametrize("name", sorted(_layers()))
def test_every_reachable_dc_method_accepts_the_kspace_call(name):
    """The call the generator now makes, unguarded, on every branch it can build."""
    layer = _layers()[name].eval()
    x, measured, mask = _batch()
    with torch.no_grad():
        out = layer(x, measured, mask, acs_mask=None, is_kspace_domain=True)
    assert out.shape == x.shape


@pytest.mark.parametrize("name", sorted(_layers()))
def test_every_reachable_dc_method_names_both_kwargs(name):
    """Signature-level: what makes the deleted retry unreachable rather than rare."""
    params = inspect.signature(_layers()[name].forward).parameters
    assert "is_kspace_domain" in params
    assert "acs_mask" in params


@pytest.mark.parametrize(
    "name",
    ["soft", "adaptive", "noise_adaptive", "hard", "simple", "kan_adaptive"],
)
def test_dropping_the_flag_changes_the_physics(name):
    """The retry's cost, measured rather than asserted.

    `target_aware_fsdc` is excluded because its own default is True, so the
    unflagged call is identical there -- which is why a per-layer default cannot
    stand in for the caller passing the flag.
    """
    layer = _layers()[name].eval()
    x, measured, mask = _batch()
    with torch.no_grad():
        flagged = layer(x, measured, mask, acs_mask=None, is_kspace_domain=True)
        unflagged = layer(x, measured, mask)
    rel = (flagged - unflagged).abs().max().item() / (flagged.abs().max().item() + 1e-12)
    assert rel > 1e-3, f"{name}: dropping is_kspace_domain changed nothing (rel={rel:.2e})"


def test_target_aware_fsdc_defaults_to_kspace_and_the_others_do_not():
    """The defaults disagree, so omitting the flag has no single meaning."""
    defaults = {
        name: inspect.signature(layer.forward).parameters["is_kspace_domain"].default
        for name, layer in _layers().items()
    }
    assert defaults["target_aware_fsdc"] is True
    assert all(v is False for k, v in defaults.items() if k != "target_aware_fsdc")


def _dc_layer_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``self.dc_layer(...)`` call in the module."""
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "dc_layer"
    ]


def _statements_beside_the_dc_call(tree: ast.AST, call: ast.Call) -> list[ast.stmt]:
    """The statement list that directly contains the DC call.

    Scoped deliberately: a file-wide search for `if <shapes differ>: raise`
    matches unrelated guards in this ~4k-line module and reports a defect as
    fixed. Observed: the first version of this check stayed GREEN with the silent
    skip planted back.
    """
    candidates: list[tuple[int, list[ast.stmt]]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            if any(call in _dc_layer_calls(stmt) for stmt in block):
                candidates.append((getattr(node, "lineno", 0), block))
    assert candidates, "DC call is not in any statement list"
    # INNERMOST, not first: `ast.walk` reaches Module before the function, and
    # the Module's body "contains" the call transitively. Taking the first match
    # searched the whole file, which is how the guard check stayed green with the
    # silent skip planted back.
    return max(candidates, key=lambda c: c[0])[1]


def _generator_tree() -> ast.AST:
    import spectramr.models.generators.kspace_cold_diffusion_generator as mod

    return ast.parse(pathlib.Path(inspect.getfile(mod)).read_text())


def test_the_generator_makes_exactly_one_dc_call_and_it_names_the_domain():
    calls = _dc_layer_calls(_generator_tree())
    assert len(calls) == 1, f"expected one dc_layer call, found {len(calls)}"
    kwargs = {k.arg for k in calls[0].keywords}
    assert "is_kspace_domain" in kwargs
    assert "acs_mask" in kwargs


def test_no_try_block_wraps_the_dc_call():
    """AST, not a source grep: the prose around the call names the retry it removed.

    A string search for ``except TypeError`` matches that comment and reports the
    defect it documents -- which is how this check first went red.
    """
    tree = _generator_tree()
    guarded = [
        t
        for t in ast.walk(tree)
        if isinstance(t, ast.Try)
        for stmt in t.body
        if _dc_layer_calls(stmt)
    ]
    assert not guarded, (
        "a try/except around the DC call re-enables the retry that drops "
        "is_kspace_domain and applies image semantics to a k-space tensor"
    )


# ── #2147: a mismatch is reported, and the domain is not read off the shapes ──
def test_a_shape_mismatch_raises_instead_of_skipping_dc():
    """The `if shapes match` guard had no `else`, so DC vanished in silence.

    A genuine 5-D volume is the known trigger: `x_out` is restored to
    [B, C, H, W, D] while `kspace_measured` stays flattened to [B*D, C, H, W].
    """
    tree = _generator_tree()
    call = _dc_layer_calls(tree)[0]
    siblings = _statements_beside_the_dc_call(tree, call)
    raising = [
        n
        for n in siblings
        if isinstance(n, ast.If)
        and (n.end_lineno or 0) < call.lineno
        and any(isinstance(b, ast.Raise) for b in ast.walk(n))
        and any(
            isinstance(c, ast.Compare) and any(isinstance(o, ast.NotEq) for o in c.ops)
            for c in ast.walk(n.test)
        )
        and "shape" in ast.unparse(n.test)
    ]
    assert raising, (
        "no `shapes differ -> raise` guard sits beside the DC call; a mismatch "
        "would skip declared data consistency in silence"
    )


def test_the_dc_call_is_not_nested_in_a_shape_equality_branch():
    """The domain must not be inferred from `x_out.shape == measured.shape`.

    `FourierBridgeNetwork` returns k-space on both branches, so the equality
    never carried domain information -- it only ever decided whether DC ran.
    """
    tree = _generator_tree()
    call = _dc_layer_calls(tree)[0]
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(_dc_layer_calls(b) for b in node.body):
            continue
        eq = [
            c
            for c in ast.walk(node.test)
            if isinstance(c, ast.Compare) and any(isinstance(o, ast.Eq) for o in c.ops)
        ]
        assert not eq, (
            "the DC call is gated on a shape EQUALITY again; the mismatch must "
            "raise, not silently select a different path"
        )
    assert call is not None
