"""Regression tests for the cross-contrast k-space cold-diffusion facade (A5).

``CrossContrastKspaceDiffusionStrategy._compute_losses_impl`` used to
``return super()._compute_losses_impl(...)`` whenever ``batch['kspace_source']``
or ``batch['kspace_target']`` was absent. **No dataset under
``src/spectramr/data/`` emits either key**, so that fallthrough was not a rare
edge case — it was the only path: the arm kept its cross-contrast name,
smoke-passed, and trained the parent's vanilla reconstruction objective
(facade, pitfall #16; silent fallback, pitfall #9).

The fix raises ``ValueError`` naming both required keys and listing what the
batch actually supplied. Pinned here:

1. the raise fires when either key is missing,
2. the parent's ``_compute_losses_impl`` is **never** reached (spied), and
   neither is the generator,
3. the message carries the debugging affordance (both key names + the keys the
   batch really had),
4. a batch that *does* carry the pair still computes the cold-diffusion losses.

The strategy is expensive to construct normally, so — following the idiom in
``test_v6_3_promoted.py`` — it is built with ``__new__`` plus attribute
injection. ``sigma_max`` / ``lambda_destination`` / ``lambda_residual`` are
class attributes, so ``__new__`` alone leaves them wired.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy import (
    CrossContrastKspaceDiffusionStrategy,
)
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


class _RecordingGen:
    """Callable spy so a test can assert the generator was never reached."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._fn(*args, **kwargs)


def _identity_gen() -> _RecordingGen:
    """Destination predictor with closed-form behaviour (halves ``k_t``)."""
    return _RecordingGen(lambda k_t, t, contrast_idx=None: k_t * 0.5)


def _make_strategy(gen: _RecordingGen) -> CrossContrastKspaceDiffusionStrategy:
    s = CrossContrastKspaceDiffusionStrategy.__new__(CrossContrastKspaceDiffusionStrategy)
    s.env = types.SimpleNamespace(generator=gen)
    return s


def _spy_on_parent(monkeypatch) -> list:
    """Replace the parent's loss impl with a recorder.

    ``super()`` in the strategy resolves to ``DiffusionTrainingStrategy``, which
    defines ``_compute_losses_impl`` itself, so patching the attribute there is
    what the old fallthrough would have hit.
    """
    calls: list = []

    def _spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return {"loss_total": torch.zeros(())}

    monkeypatch.setattr(DiffusionTrainingStrategy, "_compute_losses_impl", _spy, raising=True)
    return calls


class TestMissingKspacePairFailsLoud:
    def test_missing_pair_raises_and_never_reaches_the_parent(self, monkeypatch):
        # The load-bearing assertion is ``parent_calls == []``: on the pre-fix
        # code this batch returned the parent's plain reconstruction dict with
        # no error at all, which is exactly the facade being removed.
        parent_calls = _spy_on_parent(monkeypatch)
        gen = _identity_gen()
        s = _make_strategy(gen)
        batch = {
            "image": torch.randn(2, 1, 8, 8),
            "target": torch.randn(2, 1, 8, 8),
        }

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        assert parent_calls == [], "the silent fallback to the parent must be gone"
        assert gen.calls == [], "the generator must not run on an unusable batch"
        assert "kspace_source" in str(exc.value)
        assert "kspace_target" in str(exc.value)

    def test_message_lists_the_keys_the_batch_actually_supplied(self):
        # ``match=`` is useless here: the message renders the key list as
        # ``['image', ...]`` and ``[...]`` is a regex character class, so a
        # ``match`` on it would pass against almost anything. Assert on the
        # captured string instead.
        s = _make_strategy(_identity_gen())
        batch = {
            "image": torch.randn(1, 1, 4, 4),
            "target_image": torch.randn(1, 1, 4, 4),
            "contrast_idx": torch.zeros(1, dtype=torch.long),
        }

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        msg = str(exc.value)
        assert "'image'" in msg
        assert "'target_image'" in msg
        assert "'contrast_idx'" in msg

    @pytest.mark.parametrize("present", ["kspace_source", "kspace_target"])
    def test_half_a_pair_still_raises(self, present, monkeypatch):
        # ``k_src is None or k_dst is None`` — one half is not enough, and the
        # message still enumerates what did arrive.
        parent_calls = _spy_on_parent(monkeypatch)
        gen = _identity_gen()
        s = _make_strategy(gen)
        batch = {present: torch.randn(2, 1, 8, 8), "image": torch.randn(2, 1, 8, 8)}

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        assert parent_calls == []
        assert gen.calls == []
        assert f"'{present}'" in str(exc.value)
        assert "'image'" in str(exc.value)


class TestPairedBatchStillTrains:
    def test_paired_kspace_computes_cold_diffusion_losses(self, monkeypatch):
        # Happy path: the guard does not fire, the generator runs once, and the
        # cold-diffusion objective (not the parent's) produces the losses.
        parent_calls = _spy_on_parent(monkeypatch)
        gen = _identity_gen()
        s = _make_strategy(gen)
        batch = {
            "kspace_source": torch.randn(2, 1, 8, 8),
            "kspace_target": torch.randn(2, 1, 8, 8),
        }

        out = s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        assert parent_calls == []
        assert len(gen.calls) == 1
        assert set(out) == {"loss_destination", "loss_residual", "loss_total"}
        assert all(torch.isfinite(v).all() for v in out.values())
        assert torch.allclose(out["loss_total"], out["loss_destination"] + out["loss_residual"])

    def test_contrast_idx_emits_per_contrast_diagnostics(self):
        # ``iteration`` defaults to 0, so the log-interval gate is open and the
        # per-contrast breakdown is emitted for every distinct contrast index.
        s = _make_strategy(_identity_gen())
        batch = {
            "kspace_source": torch.randn(2, 1, 8, 8),
            "kspace_target": torch.randn(2, 1, 8, 8),
            "contrast_idx": torch.tensor([0, 1]),
        }

        out = s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        assert "loss_contrast_0" in out
        assert "loss_contrast_1" in out
        assert torch.isfinite(out["loss_contrast_0"])


class TestNonDictBatchAndMissingGenerator:
    """Ported from the twin PR (#779), which pinned two cases #725 did not.

    The non-dict case is not academic: #725's own message read the batch with
    ``sorted(k for k in batch if k)`` and NO isinstance guard, so on a tensor
    batch -- precisely what the guard exists to reject -- ``if k`` raised
    "Boolean value of Tensor with more than one value is ambiguous" and the
    diagnostic crashed instead of diagnosing.
    """

    def test_non_dict_batch_raises_rather_than_running_the_parent(self, monkeypatch):
        parent_calls = _spy_on_parent(monkeypatch)
        gen = _identity_gen()
        s = _make_strategy(gen)

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=torch.randn(2, 1, 8, 8), target_batch=None, epoch=0)

        assert parent_calls == []
        assert gen.calls == []
        assert "kspace_source" in str(exc.value)

    def test_the_message_names_the_batch_type(self):
        """A dataclass or tensor batch must be diagnosable, not just 'absent'.

        Passed as ``batch=``, NOT as ``input_batch=``. ``_resolve_legacy_batch``
        returns its first argument only when it is a dict; otherwise it re-looks
        for the value under ``kwargs["batch"]`` / ``kwargs["input_batch"]`` --
        and ``input_batch`` is a named parameter, so it is never in ``**kwargs``.
        A non-dict ``input_batch`` is therefore discarded and the guard sees
        ``None``. Both routes raise (the test above covers the other one), but
        only this one can report the real type.
        """
        s = _make_strategy(_identity_gen())

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(
                input_batch=None,
                target_batch=None,
                epoch=0,
                batch=torch.randn(2, 1, 8, 8),
            )

        assert "'Tensor'" in str(exc.value)

    def test_a_discarded_non_dict_input_batch_still_raises_as_none(self):
        """The other route, pinned so the quirk above is documented, not assumed."""
        s = _make_strategy(_identity_gen())

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=torch.randn(2, 1, 8, 8), target_batch=None, epoch=0)

        assert "'NoneType'" in str(exc.value)

    def test_the_message_names_which_key_is_missing(self):
        """Both keys absent, one absent, or the other -- the list says which."""
        s = _make_strategy(_identity_gen())
        batch = {"kspace_source": torch.randn(2, 1, 8, 8)}

        with pytest.raises(ValueError) as exc:
            s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)

        msg = str(exc.value)
        assert "'kspace_target'" in msg

    def test_missing_generator_raises(self):
        """``env.generator is None`` is a wiring failure, not a data failure."""
        s = _make_strategy(None)
        batch = {
            "kspace_source": torch.randn(2, 1, 8, 8),
            "kspace_target": torch.randn(2, 1, 8, 8),
        }

        with pytest.raises(RuntimeError, match=r"env\.generator is"):
            s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)


class TestNoParentFallbackRemains:
    def test_source_has_no_super_call_in_compute_losses(self):
        """The degradation path is gone, not merely narrowed.

        A source-level assertion is the right shape here: the defect was the
        EXISTENCE of a route from a malformed batch to the parent objective, and
        any re-introduction of that route reads the same way.
        """
        import inspect

        src = inspect.getsource(CrossContrastKspaceDiffusionStrategy._compute_losses_impl)
        assert "super()._compute_losses_impl" not in src


class TestModelInputContract:
    """Non-negotiable 14: the declared tag must name a tensor that exists.

    Overriding ``_compute_losses_impl`` also inherits
    ``DiffusionTrainingStrategy``'s ``snapshot_prepared_is_model_input = False``
    / ``snapshot_model_input_tag = "diffusion_step"`` — but the parent emits that
    tag only inside ``_prepare_diffusion_inputs``, which this override never
    reaches. Every artifact therefore pointed at a snapshot that did not exist.

    It matters more here than elsewhere in the family: A5 was a facade in which
    the arm silently trained a plain denoiser, and the interpolant beside both
    endpoints is the artifact that shows the bridge actually fired.
    """

    def test_declares_the_bridge_interpolant_the_generator_saw(self):
        gen = _identity_gen()
        s = _make_strategy(gen)
        k_src = torch.randn(2, 1, 8, 8)
        k_dst = torch.randn(2, 1, 8, 8)

        s._compute_losses_impl(
            input_batch={"kspace_source": k_src, "kspace_target": k_dst},
            target_batch=None,
            epoch=0,
        )

        assert s._declared_model_input is not None
        tensors, extra, _in_kspace_keys = s._declared_model_input

        # The generator's first positional arg is k_t; the declaration must be
        # that exact tensor, not either endpoint.
        (args, _kwargs) = gen.calls[0]
        assert torch.equal(tensors["kspace_interpolant"], args[0])
        assert not torch.equal(tensors["kspace_interpolant"], k_src)
        assert not torch.equal(tensors["kspace_interpolant"], k_dst)
        assert torch.equal(tensors["kspace_source"], k_src)
        assert torch.equal(tensors["kspace_target"], k_dst)
        assert extra["model_input_key"] == "kspace_interpolant"

    def test_all_three_tensors_are_named_as_kspace(self):
        """Named explicitly rather than left to the config union.

        ``save_debug_snapshot`` falls back to a ``"kspace"`` substring match over
        key names when ``in_kspace_keys`` is ``None``. Two of these three keys
        would survive that by luck of naming — but the strategy *raises* unless
        the batch supplies paired k-space, so it knows the answer for all three
        whatever ``data.dataset_type`` says. Declaring it is what keeps the
        previewer from rendering raw k-space as an image (VIS-1).
        """
        s = _make_strategy(_identity_gen())
        s._compute_losses_impl(
            input_batch={
                "kspace_source": torch.randn(2, 1, 8, 8),
                "kspace_target": torch.randn(2, 1, 8, 8),
            },
            target_batch=None,
            epoch=0,
        )

        _tensors, _extra, in_kspace_keys = s._declared_model_input
        assert in_kspace_keys == {
            "kspace_interpolant",
            "kspace_source",
            "kspace_target",
        }

    def test_nothing_is_declared_when_the_guard_fires(self):
        """A raise must not leave a stale declaration behind.

        The declaration happens after the paired-k-space guard, so a batch that
        trips the guard produces no artifact at all — which is correct: there is
        no model input, because no forward pass happened.
        """
        s = _make_strategy(_identity_gen())
        s._declared_model_input = None

        with pytest.raises(ValueError, match="kspace_source"):
            s._compute_losses_impl(
                input_batch={"kspace_source": torch.randn(2, 1, 8, 8)},
                target_batch=None,
                epoch=0,
            )

        assert s._declared_model_input is None


def _paired_batch(**extra) -> dict:
    batch = {
        "kspace_source": torch.randn(2, 1, 8, 8),
        "kspace_target": torch.randn(2, 1, 8, 8),
    }
    batch.update(extra)
    return batch


class TestGeneratorDispatchIsIntrospectedNotSwallowed:
    """SAQ-001 (#1189): dispatch on the signature, never on ``except TypeError``.

    The retired form was ``try: gen(k_t, t, contrast_idx=...) except TypeError:
    gen(k_t, t)``. It cannot tell a generator that has no ``contrast_idx``
    parameter from one whose forward raises ``TypeError`` several frames down,
    so a genuine bug in the conditioned path was answered by silently training
    the *unconditioned* one -- under an arm whose entire reason to exist is the
    conditioning. That is the same facade shape as the k-space guard above, one
    layer lower.

    Note the generators here are plain functions, not ``_RecordingGen``: that
    spy's ``__call__(*args, **kwargs)`` exposes ``VAR_KEYWORD``, so signature
    introspection answers True for *every* kwarg and could not discriminate.
    """

    def test_a_typeerror_raised_inside_the_conditioned_branch_propagates(self):
        # The generator's *conditioning* is broken; unconditioned it works fine.
        # That asymmetry is the whole defect: the old `except TypeError` caught
        # the conditioned failure, re-ran the working unconditioned path, and
        # returned a finite loss, so the arm trained to completion having never
        # once used the contrast index it exists to study.
        #
        # (A generator that fails on BOTH paths cannot tell the two versions
        # apart -- the retry re-raises the same error either way.)
        def _broken_conditioning(k_t, t, contrast_idx=None):
            if contrast_idx is not None:
                raise TypeError("embedding lookup dtype mismatch, conditioned branch")
            return k_t * 0.5

        s = _make_strategy(_broken_conditioning)

        with pytest.raises(TypeError, match="conditioned branch"):
            s._compute_losses_impl(
                input_batch=_paired_batch(contrast_idx=torch.tensor([0, 1])),
                target_batch=None,
                epoch=0,
            )

    def test_a_generator_without_the_parameter_is_called_unconditioned(self):
        seen: list[tuple] = []

        def _no_kwarg(k_t, t):
            seen.append((k_t.shape, t.shape))
            return k_t * 0.5

        s = _make_strategy(_no_kwarg)
        out = s._compute_losses_impl(
            input_batch=_paired_batch(contrast_idx=torch.tensor([0, 1])),
            target_batch=None,
            epoch=0,
        )

        # Reached, with two positional args and no kwarg -- the introspected
        # branch, not a TypeError that something caught.
        assert len(seen) == 1
        assert torch.isfinite(out["loss_total"])

    def test_a_generator_with_the_parameter_receives_it_even_when_the_batch_omits_it(
        self,
    ):
        # Behaviour preserved from the try/except form, which passed
        # ``contrast_idx=None`` to any generator that would accept it. The fix
        # changes which errors surface, not which arguments are sent.
        seen: list = []

        def _kwarg_gen(k_t, t, contrast_idx=None):
            seen.append(contrast_idx)
            return k_t * 0.5

        s = _make_strategy(_kwarg_gen)
        s._compute_losses_impl(input_batch=_paired_batch(), target_batch=None, epoch=0)

        assert seen == [None]


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: its fidelity term is in k-space against the destination contrast.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert CrossContrastKspaceDiffusionStrategy.__dict__["folds_image_losses"] is False
    assert CrossContrastKspaceDiffusionStrategy.__dict__["inline_losses"] == frozenset()
