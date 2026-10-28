"""Tests for ``optimizer_resolution`` — which knobs actually reach the optimizer.

The defect this module was written for: the leaf builder decided what to forward
from a hardcoded family tuple,

    if canonical_name in ("Adam", "AdamW"):
        opt_kwargs["betas"] = self._betas

while the caller set ``beta1``/``beta2`` unconditionally. So **adamax, nadam and
radam silently trained at torch's default betas** no matter what the YAML said,
and ``weight_decay`` went the other way — forwarded to ``lbfgs``/``sparseadam``,
which accept none, so those would ``TypeError``.

A hand-maintained tuple is the wrong shape for this: it was correct when written
and became wrong the moment three more optimizers were registered. These tests
therefore assert the *policy* (signature-derived forwarding, explicit-vs-default
raise semantics) rather than enumerating families, so registering another
optimizer cannot silently reintroduce the gap.

Separately: ``optimization.optimizer.betas`` was read by NOTHING while 27 experiment YAMLs
set it, and ``beta1`` defaults to 0.5 — not 0.9 — so those arms trained at
(0.5, 0.999). Making it live is a real behaviour change, pinned here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.config.schemas.optimization import (  # noqa: E402
    OptimizationConfigSchema,
)
from spectramr.infrastructure.training.optimizer_resolution import (  # noqa: E402
    OptimizerConfigurationError,
    build_optimizer_from_spec,
    resolve_optimizer_spec,
)

#: Every registered optimizer whose signature accepts ``betas``.
BETA_FAMILY = ("adam", "adamw", "adamax", "nadam", "radam")


class TestBetasReachEveryOptimizerThatAcceptsThem:
    @pytest.mark.parametrize("name", BETA_FAMILY)
    def test_scalar_beta_pair_is_forwarded(self, name: str) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type=name, beta1=0.85, beta2=0.995)
        )
        assert spec.params["betas"] == (0.85, 0.995)

    @pytest.mark.parametrize("name", BETA_FAMILY)
    def test_betas_field_is_forwarded(self, name: str) -> None:
        """``optimization.optimizer.betas`` had a validator, was set by 27 YAMLs, and was
        read by nothing at all."""
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type=name, betas=(0.9, 0.999))
        )
        assert spec.params["betas"] == (0.9, 0.999)

    def test_forwarding_tracks_the_signature_not_a_hardcoded_family_list(self) -> None:
        """The property that keeps this fixed: whether ``betas`` is forwarded is
        derived from the optimizer's own signature, so a newly-registered
        optimizer needs no edit here."""
        import inspect

        from spectramr.infrastructure.training.optimizer_registry import (
            OptimizerRegistry,
        )

        for name in OptimizerRegistry.list_available():
            cls = OptimizerRegistry.get(name)
            assert cls is not None
            accepts_betas = "betas" in inspect.signature(cls.__init__).parameters
            spec = resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer_type=name, beta1=0.8, beta2=0.9)
                if accepts_betas
                else OptimizationConfigSchema(optimizer_type=name)
            )
            assert ("betas" in spec.params) is accepts_betas, name


class TestBetasHasOneHome:
    def test_agreeing_declarations_are_accepted(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(betas=(0.8, 0.99), beta1=0.8, beta2=0.99)
        )
        assert spec.params["betas"] == (0.8, 0.99)

    def test_disagreeing_declarations_raise(self) -> None:
        with pytest.raises(OptimizerConfigurationError, match="disagrees"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(betas=(0.9, 0.999), beta1=0.5, beta2=0.999)
            )

    def test_undeclared_betas_are_a_droppable_default(self) -> None:
        """Nothing declared => the schema default is not an explicit request, so
        an optimizer that cannot take betas must not raise on it."""
        spec = resolve_optimizer_spec(OptimizationConfigSchema(optimizer_type="sgd"))
        assert "betas" not in spec.params


class TestExplicitVersusDefault:
    """Filtering decides what to FORWARD, never what to silently ignore."""

    def test_a_default_the_optimizer_cannot_take_is_dropped(self) -> None:
        spec = resolve_optimizer_spec(OptimizationConfigSchema(optimizer_type="lbfgs"))
        assert "weight_decay" not in spec.params
        assert "weight_decay" in spec.dropped_defaults

    def test_an_explicitly_declared_knob_the_optimizer_cannot_take_raises(self) -> None:
        with pytest.raises(OptimizerConfigurationError, match="weight_decay"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer_type="lbfgs", weight_decay=0.01)
            )

    def test_nesterov_reaches_sgd_and_raises_on_adam(self) -> None:
        assert (
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer={"type": "sgd", "nesterov": True})
            ).params["nesterov"]
            is True
        )

        with pytest.raises(OptimizerConfigurationError, match="nesterov"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer={"type": "adam", "nesterov": True})
            )

    def test_amsgrad_reaches_adamw(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer={"type": "adamw", "amsgrad": True})
        )
        assert spec.params["amsgrad"] is True

    def test_optimizer_kwargs_entries_are_always_explicit(self) -> None:
        """The user typed every key there, so a typo must raise even though the
        object carrying it is a plain dict with no `model_fields_set`."""
        with pytest.raises(OptimizerConfigurationError, match="wumbo"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer_kwargs={"wumbo": 1})
            )


class TestKnobsMatchedByNameMustAlsoMatchByShape:
    """Forwarding is signature-driven **by name**, which is one check short.

    ``torch.optim.Adafactor`` accepts ``eps``, but as a 2-tuple
    ``(float | None, float)`` -- not the scalar every other optimizer takes and
    the scalar ``optimization.optimizer.eps`` holds. The name matched, the value was
    forwarded, and ``optimizer_type: adafactor`` died with
    ``TypeError: 'float' object is not subscriptable`` on a tier-1 name the
    enum and the docs both advertise as always constructible.

    The policy is the module's existing one, extended from "does it accept this
    NAME" to "can it accept this VALUE": a schema default it cannot take is
    dropped, an explicit declaration raises.
    """

    def test_adafactor_builds_from_a_bare_declaration(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type="adafactor")
        )
        assert "eps" not in spec.params
        assert "eps" in spec.dropped_defaults

        model = nn.Linear(4, 2)
        optimizer = build_optimizer_from_spec(spec, model.named_parameters())
        assert isinstance(optimizer, torch.optim.Adafactor)

    def test_a_scalar_eps_declared_for_adafactor_raises(self) -> None:
        """Silently dropping an explicit value would be pitfall #15."""
        with pytest.raises(OptimizerConfigurationError, match="eps"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer_type="adafactor", eps=1e-8)
            )

    def test_adafactor_accepts_its_tuple_eps_through_optimizer_kwargs(self) -> None:
        """The escape hatch must exist, or the knob is simply unreachable."""
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(
                optimizer_type="adafactor",
                optimizer_kwargs={"eps": (None, 1e-3)},
            )
        )
        assert spec.params["eps"] == (None, 1e-3)

        model = nn.Linear(4, 2)
        optimizer = build_optimizer_from_spec(spec, model.named_parameters())
        assert optimizer.param_groups[0]["eps"] == (None, 1e-3)

    def test_a_scalar_knob_still_reaches_optimizers_that_take_a_scalar(self) -> None:
        """Guards the over-correction: the shape check must not start dropping
        the ordinary scalar ``eps`` every other optimizer wants."""
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type="adamw", eps=1e-7)
        )
        assert spec.params["eps"] == pytest.approx(1e-7)


class TestOptimizerKwargsVersusSchemaDefaults:
    """``optimizer_kwargs`` conflicts only with what the YAML actually SAID.

    The duplicate-declaration guard compared against the resolved candidate,
    which includes untouched schema defaults -- so ``optimizer_kwargs`` could
    only ever *agree* with a default, never override one. That is what left
    adafactor with no way to declare its tuple ``eps``.
    """

    def test_optimizer_kwargs_may_override_an_untouched_schema_default(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(
                optimizer_type="adamw", optimizer_kwargs={"eps": 1e-5}
            )
        )
        assert spec.params["eps"] == pytest.approx(1e-5)

    def test_optimizer_kwargs_still_conflicts_with_an_explicit_declaration(
        self,
    ) -> None:
        """Two different values for one knob remains an error -- otherwise the
        arm's provenance disagrees with its own YAML."""
        with pytest.raises(OptimizerConfigurationError, match="Declare it once"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(
                    optimizer_type="adamw",
                    weight_decay=0.01,
                    optimizer_kwargs={"weight_decay": 0.02},
                )
            )

    def test_agreeing_with_an_explicit_declaration_is_still_allowed(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(
                optimizer_type="adamw",
                weight_decay=0.01,
                optimizer_kwargs={"weight_decay": 0.01},
            )
        )
        assert spec.params["weight_decay"] == pytest.approx(0.01)


class TestLearningRate:
    def test_multiplier_applies_to_the_base_lr(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(learning_rate=1e-4), lr_multiplier=2.0
        )
        assert spec.lr == pytest.approx(2e-4)

    def test_override_wins_over_the_multiplier(self) -> None:
        """TTUR: an explicit discriminator_learning_rate is absolute."""
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(learning_rate=1e-4),
            lr_multiplier=10.0,
            lr_override=5e-5,
        )
        assert spec.lr == pytest.approx(5e-5)


class TestUnknownOptimizerRaises:
    def test_unregistered_name_raises_with_the_available_list(self) -> None:
        cfg = OptimizationConfigSchema()
        object.__setattr__(cfg.optimizer, "type", "definitely_not_real")
        with pytest.raises(OptimizerConfigurationError, match="Unknown optimizer"):
            resolve_optimizer_spec(cfg)


class TestParamGroupOverrides:
    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.residual_head = nn.Linear(4, 2)

        def forward(self, x):  # pragma: no cover
            return self.residual_head(self.encoder(x))

    def _cfg(self, **overrides):
        return OptimizationConfigSchema(
            learning_rate=1e-3, optimizer={"param_groups": overrides}
        )

    def test_override_produces_a_distinct_group(self) -> None:
        spec = resolve_optimizer_spec(
            self._cfg(residual_head={"learning_rate": 1e-5}), model=self._Net()
        )
        by_name = {g.name: g for g in spec.param_groups}
        assert by_name["residual_head"].lr == pytest.approx(1e-5)
        assert by_name["__default__"].lr == pytest.approx(1e-3)

    def test_groups_reach_the_built_optimizer(self) -> None:
        model = self._Net()
        spec = resolve_optimizer_spec(
            self._cfg(residual_head={"learning_rate": 1e-5}), model=model
        )
        opt = build_optimizer_from_spec(spec, model)
        assert sorted(g["lr"] for g in opt.param_groups) == [1e-5, 1e-3]

    def test_a_key_matching_nothing_raises(self) -> None:
        """A renamed submodule must not silently degrade the arm to a uniform
        learning rate — which is the entire point of the knob."""
        with pytest.raises(OptimizerConfigurationError, match="matches no parameter"):
            resolve_optimizer_spec(
                self._cfg(typo_head={"learning_rate": 1e-5}), model=self._Net()
            )

    def test_freeze_sets_requires_grad_false_rather_than_lr_zero(self) -> None:
        """A zero-LR group still accrues optimizer state and is still
        decoupled-weight-decayed by AdamW, so freeze cannot be lr=0."""
        model = self._Net()
        spec = resolve_optimizer_spec(self._cfg(encoder={"freeze": True}), model=model)
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert "encoder" not in {g.name for g in spec.param_groups}

    def test_colliding_with_a_model_supplied_group_method_raises(self) -> None:
        class _WithGroups(self._Net.__mro__[0]):  # type: ignore[misc]
            def get_parameter_groups(self):  # pragma: no cover - never called
                return []

        with pytest.raises(OptimizerConfigurationError, match="two sources of truth"):
            resolve_optimizer_spec(
                self._cfg(residual_head={"learning_rate": 1e-5}), model=_WithGroups()
            )


class TestSpecIsProvenanceSerialisable:
    def test_records_what_was_forwarded_not_what_was_declared(self) -> None:
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(
                optimizer_type="adamw", learning_rate=1e-4, beta1=0.9, beta2=0.999
            ),
            role="discriminator",
        )
        record = spec.as_provenance()
        assert record["optimizer"] == "adamw"
        assert record["role"] == "discriminator"
        assert record["lr"] == pytest.approx(1e-4)
        assert record["betas"] == (0.9, 0.999)

    def test_dropped_defaults_are_recorded_not_hidden(self) -> None:
        """ "adafactor ignored your weight_decay default" is invisible otherwise."""
        record = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type="lbfgs")
        ).as_provenance()
        assert "weight_decay" in record["dropped_defaults"]


class TestBuildsARealOptimizer:
    @pytest.mark.parametrize("name", (*BETA_FAMILY, "sgd", "adagrad", "rmsprop"))
    def test_every_torch_tier_name_constructs(self, name: str) -> None:
        model = nn.Linear(4, 2)
        cfg = (
            OptimizationConfigSchema(optimizer_type=name, beta1=0.9, beta2=0.999)
            if name in BETA_FAMILY
            else OptimizationConfigSchema(optimizer_type=name)
        )
        opt = build_optimizer_from_spec(resolve_optimizer_spec(cfg, model=model), model)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_forwarded_betas_land_on_the_optimizer_defaults(self) -> None:
        model = nn.Linear(4, 2)
        cfg = OptimizationConfigSchema(optimizer_type="nadam", beta1=0.85, beta2=0.995)
        opt = build_optimizer_from_spec(resolve_optimizer_spec(cfg, model=model), model)
        assert opt.defaults["betas"] == (0.85, 0.995)


class TestFusedOptimizer:
    """``fused`` collapses the elementwise update into one multi-tensor kernel.

    It follows the module's existing policy rather than adding a branch: unset is
    a droppable default, an explicit declaration an optimizer cannot take RAISES.
    That polarity matters more here than for a numerical knob -- fused is a
    THROUGHPUT claim, and a claim that silently does not hold is the same defect
    class as an eager run reporting compiled numbers.
    """

    @staticmethod
    def _spec(**optimizer):
        return resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer=dict(optimizer))
        )

    def test_fused_reaches_an_optimizer_that_accepts_it(self) -> None:
        assert self._spec(type="adamw", fused=True).params.get("fused") is True

    def test_unset_is_dropped_not_forwarded_as_false(self) -> None:
        """None means 'unset'. Forwarding False would override a future torch
        default, which is what the nesterov/amsgrad tri-state exists to avoid."""
        assert "fused" not in self._spec(type="adamw").params

    def test_explicit_false_is_forwarded(self) -> None:
        assert self._spec(type="adamw", fused=False).params.get("fused") is False

    def test_fused_defaults_on_when_every_parameter_is_real_cuda(self, monkeypatch) -> None:
        from spectramr.infrastructure.training import optimizer_resolution as mod

        monkeypatch.setattr(mod, "_fused_supported", lambda model: True)
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer={"type": "adamw"}), model=object()
        )
        assert spec.params.get("fused") is True
        assert spec.as_provenance()["applied_defaults"] == ["fused"]

    def test_explicit_false_beats_the_cuda_default(self, monkeypatch) -> None:
        from spectramr.infrastructure.training import optimizer_resolution as mod

        monkeypatch.setattr(mod, "_fused_supported", lambda model: True)
        spec = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer={"type": "adamw", "fused": False}), model=object()
        )
        assert spec.params["fused"] is False and not spec.applied_defaults

    def test_a_cpu_or_complex_model_is_not_given_fused(self) -> None:
        import torch

        from spectramr.infrastructure.training.optimizer_resolution import _fused_supported

        assert not _fused_supported(torch.nn.Linear(2, 2))  # CPU
        assert not _fused_supported(None)

    def test_declaring_it_on_an_optimizer_without_it_raises(self) -> None:
        """pitfall #15: a declared knob that cannot reach the optimizer is a
        silent no-op. Adafactor has no ``fused`` argument."""
        with pytest.raises(OptimizerConfigurationError, match="fused"):
            self._spec(type="adafactor", fused=True)

    def test_acceptance_tracks_the_signature_not_a_family_list(self) -> None:
        """Same property the betas tests pin: whether ``fused`` is forwarded is
        derived from each optimizer's own signature, so registering another one
        needs no edit here."""
        import inspect

        from spectramr.infrastructure.training.optimizer_registry import (
            OptimizerRegistry,
        )

        checked = 0
        for name in OptimizerRegistry.list_available():
            cls = OptimizerRegistry.get(name)
            if cls is None:
                continue
            try:
                accepts = "fused" in inspect.signature(cls.__init__).parameters
            except (TypeError, ValueError):  # pragma: no cover - C-level init
                continue
            checked += 1
            if accepts:
                assert self._spec(type=name, fused=True).params.get("fused") is True
            else:
                with pytest.raises(OptimizerConfigurationError):
                    self._spec(type=name, fused=True)
        assert checked, "no optimizer was introspectable; the test proved nothing"


class TestOneAnswerForBeta1:
    """#2308: the schema default said 0.5, the live optimizer ran 0.9, and no
    record said which. Unset beta1 now means the optimizer's own default."""

    def test_an_undeclared_pair_runs_and_records_the_constructor_default(self) -> None:
        cfg = OptimizationConfigSchema(optimizer_type="adamw", learning_rate=1e-4)
        assert cfg.optimizer.beta1 is None
        spec = resolve_optimizer_spec(cfg)
        assert "betas" not in spec.params
        record = spec.as_provenance()
        assert record["betas"] == (0.9, 0.999)
        assert record["betas_source"] == "optimizer constructor default"

    def test_beta1_alone_keeps_the_schema_beta2(self) -> None:
        cfg = OptimizationConfigSchema(optimizer_type="adamw", beta1=0.5)
        assert resolve_optimizer_spec(cfg).params["betas"] == (0.5, 0.999)

    def test_beta2_alone_pairs_with_the_optimizer_beta1(self) -> None:
        cfg = OptimizationConfigSchema(optimizer_type="adamw", beta2=0.99)
        assert resolve_optimizer_spec(cfg).params["betas"] == (0.9, 0.99)

    def test_a_declared_pair_is_recorded_as_declared(self) -> None:
        record = resolve_optimizer_spec(
            OptimizationConfigSchema(optimizer_type="adamw", betas=(0.5, 0.9))
        ).as_provenance()
        assert record["betas"] == (0.5, 0.9)
        assert "betas_source" not in record

    def test_a_pair_beside_beta2_alone_is_checked_against_the_optimizer_beta1(self) -> None:
        """With beta1 unset the scalar side is (0.9, beta2), so a pair that
        agrees with it is accepted and one that contradicts beta2 raises."""
        agree = OptimizationConfigSchema(optimizer_type="adamw", betas=(0.9, 0.99), beta2=0.99)
        assert resolve_optimizer_spec(agree).params["betas"] == (0.9, 0.99)
        with pytest.raises(OptimizerConfigurationError, match="disagrees"):
            resolve_optimizer_spec(
                OptimizationConfigSchema(optimizer_type="adamw", betas=(0.9, 0.99), beta2=0.999)
            )

    def test_a_list_in_kwargs_agrees_with_the_same_declared_pair(self) -> None:
        """YAML gives a list; the resolved pair is a tuple. Equal values raised."""
        cfg = OptimizationConfigSchema(
            optimizer_type="adamw",
            betas=(0.9, 0.99),
            optimizer={"kwargs": {"betas": [0.9, 0.99]}},
        )
        assert tuple(resolve_optimizer_spec(cfg).params["betas"]) == (0.9, 0.99)


class TestParamGroupsAddressTheGenerator:
    """The generator's named-parameter prefixes were applied to the discriminator."""

    def test_the_discriminator_ignores_generator_prefixes(self) -> None:
        cfg = OptimizationConfigSchema(
            learning_rate=1e-3,
            optimizer={"param_groups": {"residual_head": {"learning_rate": 1e-5}}},
        )
        discriminator = nn.Sequential(nn.Linear(4, 1))
        spec = resolve_optimizer_spec(cfg, role="discriminator", model=discriminator)
        assert spec.param_groups == ()
