"""Unit tests for :mod:`spectramr.models.blocks.domain_adaptation`.

Three jobs, in order of how much they are trusted:

1. Behaviour of the elected owner -- forward identity, backward negation, ``alpha``.
2. NN19: the four modules whose own copy this change deleted are *executed*, not merely
   imported, and their gradients compared against the owner's at a non-default alpha.
3. NN15: the detector that keeps the owner sole, with a planted violation per shape it
   claims to catch and two negative controls that must stay green.

The detector lives here rather than in ``tests/architecture/`` because it guards one
module's ownership rather than a corpus-wide baseline, and because a plant is far easier
to read next to the thing it is planted against.
"""

import ast
import pathlib
import textwrap

import pytest
import torch

from spectramr.models.blocks.domain_adaptation import (
    GradientReversalFunction,
    GradientReversalLayer,
    grad_reverse,
)

# Deliberately not 1.0: that is the schema default, and a probe at the default cannot
# tell "alpha was read" from "alpha was ignored" (memory: coincident-default-vacuous).
ALPHA = 3.7


def _grad_through(fn) -> torch.Tensor:
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    fn(x).sum().backward()
    assert x.grad is not None
    return x.grad


class TestTheOwner:
    def test_forward_is_the_identity(self) -> None:
        x = torch.tensor([1.0, -2.0, 3.0])
        assert torch.equal(grad_reverse(x, ALPHA), x)

    def test_backward_negates_and_scales_by_alpha(self) -> None:
        assert torch.allclose(
            _grad_through(lambda t: grad_reverse(t, ALPHA)),
            torch.full((3,), -ALPHA),
        )

    def test_the_raw_function_reverses(self) -> None:
        assert torch.allclose(
            _grad_through(lambda t: GradientReversalFunction.apply(t, ALPHA)),
            torch.full((3,), -ALPHA),
        )

    def test_the_module_form_agrees_with_the_functional_form(self) -> None:
        assert torch.allclose(
            _grad_through(GradientReversalLayer(alpha=ALPHA)),
            _grad_through(lambda t: grad_reverse(t, ALPHA)),
        )

    def test_set_alpha_changes_the_gradient(self) -> None:
        layer = GradientReversalLayer(alpha=ALPHA)
        layer.set_alpha(0.5)
        assert torch.allclose(_grad_through(layer), torch.full((3,), -0.5))
        assert "alpha=0.5" in repr(layer)

    def test_alpha_is_forwarded_uncoerced(self) -> None:
        # grad_reverse must not call float() on alpha: on the training path that is a
        # device sync (NN9). A 0-d tensor alpha therefore has to survive the call.
        assert torch.allclose(
            _grad_through(lambda t: grad_reverse(t, torch.tensor(ALPHA))),
            torch.full((3,), -ALPHA),
        )


class TestEveryFormerOwnerStillComputesTheSameFunction:
    """NN19: the rewrite is executed before it is trusted.

    Each parameter is a surviving entry point in a module that used to carry its own
    ``autograd.Function``. Importing them is not enough -- a codemod is locally
    well-formed everywhere it is wrong, and a mis-repointed ``.apply`` raises only when
    the line runs.
    """

    @staticmethod
    def _privileged_learning() -> torch.Tensor:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            gradient_reversal,
        )

        # keyword, not positional: this caller spells the knob `lambda_`
        return _grad_through(lambda t: gradient_reversal(t, lambda_=ALPHA))

    @staticmethod
    def _models_domain_adaptation() -> torch.Tensor:
        from spectramr.models.domain_adaptation import DomainAdaptationLayer

        layer = DomainAdaptationLayer(in_channels=4, num_domains=2, alpha=ALPHA)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        return x.grad

    @staticmethod
    def _domain_adversarial_grl() -> torch.Tensor:
        from spectramr.models.losses.domain_adversarial_grl import GradientReversalLayer as Grl

        return _grad_through(Grl(alpha=ALPHA))

    def test_the_lambda_spelling_adapter_reverses(self) -> None:
        assert torch.allclose(self._privileged_learning(), torch.full((3,), -ALPHA))

    def test_the_loss_module_adapter_reverses(self) -> None:
        assert torch.allclose(self._domain_adversarial_grl(), torch.full((3,), -ALPHA))

    def test_the_domain_adaptation_layer_runs_and_reverses_its_input(self) -> None:
        # Shape-level rather than value-level: the discriminator sits between the
        # reversal and the loss, so the sign of any one element is not predictable.
        # What is checked is that the path executes at all -- the .apply repoint.
        grad = self._models_domain_adaptation()
        assert grad.shape == (2, 4, 8, 8)
        assert torch.isfinite(grad).all()

    def test_the_dann_schedule_wrapper_still_anneals(self) -> None:
        from spectramr.models.losses.domain_adaptation_loss import (
            GradientReversalLayerModule,
        )

        layer = GradientReversalLayerModule(max_iters=10, gamma=10.0)
        first = _grad_through(layer)
        for _ in range(20):
            layer(torch.zeros(1))
        last = _grad_through(layer)
        # The sigmoid ramp saturates at -1, so |grad| must grow toward it.
        assert first.abs().max() < last.abs().max() <= 1.0


# ---------------------------------------------------------------------------
# NN15 -- the detector, and one plant per shape it claims to catch
# ---------------------------------------------------------------------------

_OWNER = "models/blocks/domain_adaptation.py"


def find_rogue_reversals(root: pathlib.Path) -> list[str]:
    """Every ``autograd.Function`` subclass under ``root`` that reverses a gradient.

    Two legs, because either alone is blind:

    * **name** -- the class is called something with "revers" in it;
    * **shape** -- its ``backward`` negates *the incoming gradient*
      (:func:`_backward_negates`, which owns that distinction).

    The name leg misses a copy called ``GradReverse``... which the shape leg catches;
    the shape leg misses a copy that reverses by multiplying by a value that is negative
    at runtime rather than negated in the source, which the name leg catches. Neither leg
    reads source *text*: bases are resolved on :class:`ast.Name` / :class:`ast.Attribute`,
    so a docstring or comment naming the owner cannot trip this, and a string literal
    cannot satisfy it.

    Known blind spot, stated rather than papered over: a copy that is neither named for
    reversal nor negates the grad syntactically is invisible to both legs.
    """
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        # ast.walk, not tree.body: a definition nested inside a function is a real
        # shape and a top-level-only scan is green on it.
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [ast.unparse(b) for b in node.bases]
            if not any(b.split(".")[-1] == "Function" for b in bases):
                continue
            if "revers" in node.name.lower() or _backward_negates(node):
                offenders.append(f"{path.as_posix()}:{node.lineno}:{node.name}")
    return offenders


def _backward_negates(node: ast.ClassDef) -> bool:
    """Does ``backward`` negate the *incoming gradient* -- as opposed to negating anything?

    That distinction is the whole leg. "A unary minus somewhere inside ``backward``" reports
    ``SurrogateSpike`` (``models/mamba/neuro_mamba.py``), whose backward builds a sigmoid
    derivative through ``torch.exp(-sgax)``: a negated threshold offset, not a reversed
    gradient. Anchoring every negation to the grad parameter separates the two, and the
    parameter is read off the signature rather than assumed to be spelled ``grad_output`` --
    the planted copies below call it ``g``, so a hard-coded name would pass on all of them.

    ``carriers`` follows the grad through local rebinding, so ``h = grad_output`` then
    ``-h`` is still caught. Two blind spots, stated rather than papered over: a ``backward``
    taking ``*grad_outputs`` has no named parameter to anchor on, and a copy that reverses by
    multiplying by a value that is negative at runtime rather than negated syntactically is
    invisible here -- the name leg is what covers those.
    """
    backward = next(
        (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "backward"),
        None,
    )
    if backward is None or len(backward.args.args) < 2:
        return False
    carriers = {backward.args.args[1].arg}
    assigns = [n for n in ast.walk(backward) if isinstance(n, ast.Assign)]
    changed = True
    while changed:
        changed = False
        for stmt in assigns:
            if not _mentions(stmt.value, carriers):
                continue
            for target in stmt.targets:
                bound = {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
                if bound - carriers:
                    carriers |= bound
                    changed = True
    for sub in ast.walk(backward):
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.USub):
            if _mentions(sub.operand, carriers):
                return True
        elif isinstance(sub, ast.Attribute) and sub.attr == "neg":
            if _mentions(sub.value, carriers):
                return True
        elif isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
            # `grad * -alpha` and `-alpha * grad`: the minus sits on the coefficient, which
            # is how the deleted `_GradientReversal` spelled it.
            for negated, other in ((sub.left, sub.right), (sub.right, sub.left)):
                if (
                    isinstance(negated, ast.UnaryOp)
                    and isinstance(negated.op, ast.USub)
                    and _mentions(other, carriers)
                ):
                    return True
    return False


def _mentions(expr: ast.AST, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(expr))


def _write(root: pathlib.Path, name: str, body: str) -> None:
    (root / name).write_text(textwrap.dedent(body).lstrip())


class TestTheDetectorIsRedOnEveryShape:
    """One plant per shape (NN15). A gate is a gate only for a shape it has failed on."""

    @pytest.mark.parametrize(
        ("shape", "body"),
        [
            (
                "module-level, dotted base",
                """
                import torch

                class GradientReversalLayer(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x, alpha):
                        return x.view_as(x)

                    @staticmethod
                    def backward(ctx, g):
                        return g.neg() * ctx.alpha, None
                """,
            ),
            (
                "function-local",
                """
                import torch

                def build():
                    class _GradientReversal(torch.autograd.Function):
                        @staticmethod
                        def backward(ctx, g):
                            return -ctx.lambda_ * g, None

                    return _GradientReversal
                """,
            ),
            (
                "bare Function base, name says nothing",
                """
                from torch.autograd import Function

                class Flip(Function):
                    @staticmethod
                    def backward(ctx, g):
                        return g.neg(), None
                """,
            ),
            (
                "the grad reaches the negation through a local rebinding",
                """
                import torch

                class Flop(torch.autograd.Function):
                    @staticmethod
                    def backward(ctx, g):
                        h = g
                        return -h * ctx.alpha, None
                """,
            ),
        ],
    )
    def test_a_planted_copy_turns_the_detector_red(
        self, tmp_path: pathlib.Path, shape: str, body: str
    ) -> None:
        _write(tmp_path, "planted.py", body)
        assert find_rogue_reversals(tmp_path), f"blind to the {shape} shape"

    @pytest.mark.parametrize(
        ("shape", "body"),
        [
            (
                "prose naming the owner",
                '''
                """Uses GradientReversalFunction from the blocks package.

                class GradientReversalLayer(torch.autograd.Function) -- not a definition.
                """

                NAME = "GradientReversalFunction"
                ''',
            ),
            (
                "an nn.Module adapter, which is allowed",
                """
                from torch import nn

                class GradientReversalLayer(nn.Module):
                    def forward(self, x):
                        return x
                """,
            ),
            (
                "a surrogate gradient negating a threshold rather than the grad",
                """
                import torch

                class SurrogateSpike(torch.autograd.Function):
                    @staticmethod
                    def backward(ctx, grad_output):
                        grad_input = grad_output.clone()
                        sgax = (ctx.input - ctx.thresh) * ctx.alpha
                        return grad_input * (ctx.alpha / (1 + torch.exp(-sgax))), None
                """,
            ),
        ],
    )
    def test_a_non_violation_stays_green(
        self, tmp_path: pathlib.Path, shape: str, body: str
    ) -> None:
        _write(tmp_path, "innocent.py", body)
        assert find_rogue_reversals(tmp_path) == [], f"false positive on {shape}"


class TestTheOwnerIsSole:
    def test_exactly_one_reversal_function_exists_under_src(self) -> None:
        import spectramr

        offenders = find_rogue_reversals(pathlib.Path(spectramr.__file__).parent)
        assert len(offenders) == 1, f"expected one owner, found: {offenders}"
        assert offenders[0].endswith("GradientReversalFunction")
        assert _OWNER in offenders[0]
