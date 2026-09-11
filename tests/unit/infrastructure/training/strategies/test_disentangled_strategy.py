"""Unit tests for DisentangledTrainingStrategy."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)

# Shared patch targets for BaseTrainingStrategy.__init__ dependencies
_LOSS_BUILDER_PATCH = "spectramr.infrastructure.training.builders.loss_builder.LossBuilder"
_BLOCH_PATCH = (
    "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
)


def _mock_loss_builder():
    """Create a MagicMock LossBuilder whose chained .build() returns {}."""
    mock_lb = MagicMock()
    builder_instance = MagicMock()
    builder_instance.build.return_value = {}
    mock_lb.return_value = builder_instance
    return mock_lb


@pytest.fixture(autouse=True)
def mock_resolve_service():
    """Mock resolve_service to bypass DI container requirements in unit tests."""
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        yield mock_resolve


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator structure (DisentangledModel)
    gen = MagicMock()
    gen.enc_c = MagicMock(return_value=torch.randn(2, 8, 16, 16))  # content code
    gen.enc_s = MagicMock(return_value=(torch.randn(2, 4), torch.randn(2, 4)))  # style (mu, logvar)
    gen.reparameterize = MagicMock(return_value=torch.randn(2, 4))  # style code
    gen.gen = MagicMock(return_value=torch.randn(2, 1, 64, 64))  # decoded image
    gen.predict_physics = MagicMock(return_value=torch.rand(2, 4))  # physics params
    gen.training = True
    gen.train = MagicMock()
    gen.eval = MagicMock()

    # Add to models dict (new API)
    env.models = {"generator": gen}
    env.generator = gen  # Also support direct access

    # Mock optimizers
    opt_g = MagicMock()
    opt_g.zero_grad = MagicMock()
    opt_g.step = MagicMock()
    env.optimizers = {"main": MagicMock(), "opt_g": opt_g}
    env.opt_g = opt_g  # Direct access

    # Mock loss functions
    env.losses = {
        "l1": MagicMock(return_value=torch.tensor(0.5)),
        "kl": MagicMock(return_value=torch.tensor(0.1)),
        "perceptual": MagicMock(return_value=torch.tensor(0.2)),
        "lpips": MagicMock(return_value=torch.tensor(0.3)),
    }

    # Mock config (SSOT)
    env.config = MagicMock()
    env.config.losses.reconstruction.lambda_recon = 1.0
    env.config.losses.latent.lambda_kl = 0.01
    env.config.losses.latent.kl_capacity_target = 20.0
    env.config.losses.latent.kl_capacity_warmup_steps = 10000
    env.config.losses.latent.use_capacity_scheduling = False
    env.config.losses.reconstruction.lambda_perceptual = 0.0  # Disabled
    env.config.loss_schedules = {}  # Default schedules
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    # resolve_amp_precision (base.py __init__) raises on a non-dtype string;
    # a bare MagicMock fails its membership check, so pin a real dtype.
    env.config.optimization.precision.enabled = False
    env.config.optimization.precision.dtype = "float16"
    env.config.model.model_type = "disentangled_model"
    env.config.model.in_channels = 1
    env.config.model.out_channels = 1
    env.model_type = "disentangled_model"

    # Disable enable flags not backed by mock losses to prevent SSOT violations
    env.config.losses.reconstruction.enable_mind_ssc = False
    env.config.losses.reconstruction.enable_latent_consistency = False
    env.config.losses.reconstruction.enable_hfen = False
    env.config.losses.reconstruction.enable_complex_mse = False
    env.config.losses.reconstruction.enable_perceptual = False
    env.config.losses.reconstruction.enable_lpips = False
    env.config.losses.reconstruction.enable_ssim = False
    env.config.losses.reconstruction.enable_hist = False
    env.config.losses.reconstruction.enable_ffl = False
    env.config.losses.reconstruction.enable_dists = False
    env.config.losses.reconstruction.use_magnitude_scaling = False
    env.config.losses.reconstruction.magnitude_scale_recon = 1.0
    env.config.losses.reconstruction.magnitude_scale_structural = 1.0
    env.config.losses.reconstruction.magnitude_scale_physics = 1.0

    # Set lambda fields to numeric values (loss_weights property compares with > 0)
    env.config.losses.reconstruction.lambda_l1 = 0.0
    env.config.losses.reconstruction.lambda_style = 0.0
    env.config.losses.reconstruction.lambda_content = 0.0
    env.config.losses.reconstruction.lambda_bloch = 0.0
    env.config.losses.reconstruction.lambda_anat = 0.0
    env.config.losses.reconstruction.lambda_mind_ssc = 0.0
    env.config.losses.reconstruction.lambda_latent_consistency = 0.0
    env.config.losses.reconstruction.lambda_hist = 0.0
    env.config.losses.reconstruction.lambda_ffl = 0.0
    env.config.losses.reconstruction.lambda_hfen = 0.0
    env.config.losses.reconstruction.lambda_complex_mse = 0.0
    env.config.losses.reconstruction.lambda_lpips = 0.0
    env.config.losses.reconstruction.lambda_ssim = 0.0
    env.config.losses.latent.lambda_kl = 0.01
    env.config.r1_interval = 1  # Add this for UnifiedGANLossComputer
    if hasattr(env.config.losses.reconstruction, "lambda_kl"):
        del env.config.losses.reconstruction.lambda_kl

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate DisentangledTrainingStrategy (NEW API - env only)."""
    with (
        patch(_BLOCH_PATCH),
        patch(_LOSS_BUILDER_PATCH, new_callable=_mock_loss_builder),
    ):
        strategy = DisentangledTrainingStrategy(env=mock_env)
        return strategy


def test_initialization(strategy):
    """Test strategy initialization and SSOT weight loading."""
    assert strategy is not None
    assert strategy.loss_weights["recon"] == 1.0
    assert strategy.loss_weights["kl"] == 0.01


def test_loss_weights_ssot(strategy, mock_env):
    """Test that loss weights are read dynamically from config."""
    # Modify config (access via env.config now)
    mock_env.config.losses.reconstruction.lambda_recon = 5.0
    assert strategy.loss_weights["recon"] == 5.0


def test_compute_losses_impl_structure(strategy, mock_env):
    """``_compute_losses_impl`` is intentionally a no-op for this strategy.

    DisentangledTrainingStrategy overrides ``train_step`` and computes
    losses inline; ``_compute_losses_impl`` returns ``{}`` to satisfy the
    base-class contract without crashing.  Previously this method raised
    ``NotImplementedError`` — the new behaviour is the empty-dict
    contract documented at
    :func:`spectramr.infrastructure.training.strategies.disentangled_strategy.DisentangledTrainingStrategy._compute_losses_impl`.
    """
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 64, 64)
    out = strategy._compute_losses_impl(input_batch, target_batch, 0)
    assert out == {}


def test_train_step_execution(strategy, mock_env):
    """Test full train_step execution."""
    # Ensure config has no MagicMock values for booleans/numeric used in logic
    mock_env.config.losses.reconstruction.use_curriculum_scheduling = False
    mock_env.config.losses.latent.use_capacity_scheduling = False

    # Ensure compute returns a real tensor to pass isfinite() check
    from spectramr.models.losses.computers.base import LossOutput

    with patch.object(strategy.loss_computer, "compute") as mock_compute:
        mock_compute.return_value = LossOutput(
            total=torch.tensor(0.5, requires_grad=True),
            components={"recon": torch.tensor(0.5)},
            metrics={},
        )

        input_batch = torch.randn(2, 1, 64, 64)
        target_batch = torch.randn(2, 1, 64, 64)
        batch = {"input": input_batch, "target": target_batch}

        steps = strategy.train_step(batch, epoch=0)

        # Execute closures
        for step in steps:
            loss = step["closure"]()
            assert loss is not None

    metrics = strategy.get_last_metrics()

    assert "total_loss" in metrics
    assert "recon" in metrics


def _closure_defs() -> dict[str, ast.FunctionDef]:
    """Parse ``train_step`` source and return its nested closure defs by name."""
    src = textwrap.dedent(inspect.getsource(DisentangledTrainingStrategy.train_step))
    tree = ast.parse(src)
    closures: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "gen_closure",
            "disc_closure",
        ):
            closures[node.name] = node
    return closures


def _is_no_grad_with(node: ast.AST) -> bool:
    """True if ``node`` is a ``with torch.no_grad():`` statement."""
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "no_grad"
        ):
            return True
    return False


def _bare_item_calls_outside_no_grad(func: ast.FunctionDef) -> list[int]:
    """Return line numbers of ``.item()`` calls NOT under a ``no_grad`` block."""
    offending: list[int] = []

    def _visit(node: ast.AST, under_no_grad: bool) -> None:
        if _is_no_grad_with(node):
            for child in ast.iter_child_nodes(node):
                _visit(child, True)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "item"
            and not under_no_grad
        ):
            offending.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            _visit(child, under_no_grad)

    for stmt in func.body:
        _visit(stmt, False)
    return offending


def test_no_item_in_training_closure():
    """SGV-003: no bare ``.item()`` (D2H sync) inside the per-step closures.

    The metric-store ``.item()`` calls were a NN#9 violation — they force a
    GPU->CPU synchronisation on every training step. They were replaced by
    ``float(x.detach())`` wrapped in ``with torch.no_grad():``. This AST guard
    asserts no bare ``.item()`` survives outside a ``no_grad`` block in
    ``gen_closure`` or ``disc_closure``.
    """
    closures = _closure_defs()
    assert "gen_closure" in closures, "gen_closure not found in train_step"
    assert "disc_closure" in closures, "disc_closure not found in train_step"

    for name, func in closures.items():
        offending = _bare_item_calls_outside_no_grad(func)
        assert not offending, (
            f"{name} has bare .item() calls (NN#9 D2H sync) at lines "
            f"{offending}; wrap metric stores in 'with torch.no_grad():' "
            f"using float(x.detach())"
        )


def test_closures_store_floats_via_no_grad(strategy, mock_env):
    """SGV-003 behavioural: metric stores resolve to Python floats.

    Runs ``train_step`` with adversarial disabled and asserts the
    ``_last_step_metrics`` populated inside ``gen_closure`` are Python floats
    (not 0-d tensors), confirming the ``float(detach())`` conversion path.

    2026-07-14 (issue #189): this test said "adversarial disabled" but never
    disabled it — it only switched off the two schedulers. ``config`` is a
    MagicMock, so ``gan_config.enable_adversarial`` was a truthy mock and the
    adversarial branch RAN, making ``adv_loss_total`` a MagicMock.
    ``total_gen_loss = loss_output.total + adv_loss_total`` then resolved through
    ``MagicMock.__radd__``, so ``g_total_loss`` was a MagicMock and the float
    assertion could neither pass nor fail for the reason it was written — a dead
    guard on a live invariant (the no-GPU-sync rule in ``performance.md``).
    Disabling adversarial for real is what makes the assertion meaningful.
    """
    mock_env.config.losses.reconstruction.use_curriculum_scheduling = False
    mock_env.config.losses.latent.use_capacity_scheduling = False
    # Actually disable adversarial, so `total = loss_output.total + 0.0` stays a
    # real tensor and the float(detach()) conversion is what is under test.
    mock_env.config.losses.gan.enable_adversarial = False

    from spectramr.models.losses.computers.base import LossOutput

    with patch.object(strategy.loss_computer, "compute") as mock_compute:
        mock_compute.return_value = LossOutput(
            total=torch.tensor(0.5, requires_grad=True),
            components={"recon": torch.tensor(0.5)},
            metrics={},
        )
        batch = {
            "input": torch.randn(2, 1, 64, 64),
            "target": torch.randn(2, 1, 64, 64),
        }
        steps = strategy.train_step(batch, epoch=0)
        for step in steps:
            step["closure"]()

    metrics = strategy.get_last_metrics()
    # The contract INVERTED with #707: values now stay on-device and the
    # training loop's `log_interval`-gated converter is the only host transfer.
    # Converting here fused was still a sync per STEP, because the loop calls
    # this every iteration. What this test guards is unchanged in substance --
    # that the closure stored a real tensor rather than a MagicMock, so the
    # no-GPU-sync invariant is being exercised on live values.
    assert isinstance(metrics["g_total_loss"], torch.Tensor)
    assert metrics["g_total_loss"].numel() == 1
    assert isinstance(metrics["loss"], torch.Tensor)


def test_validation_visual_target_is_ground_truth_4d(strategy, mock_env):
    """Regression: the cached validation REAL must be the TARGET, not the input.

    The image-capture path in ``pipelines/train.py`` saves
    ``strategy._last_visual_target`` as the validation "real" PNG when both
    ``_last_visual_pred`` and ``_last_visual_target`` are set. The 5D branch
    cached the target correctly, but the 4D branch (the path taken by 2-D data
    such as the single-slice cross-field cohort) cached ``x_a_in`` — the SOURCE
    input — so the diagnostics "real" image showed the input, not the
    ground-truth target, even though loss/metrics graded against the target.
    This drives the 4D branch with a constant-valued input (0.2) and target
    (0.8) and asserts the cached real is the target.
    """
    mock_env.config.metrics.output_dir = None  # skip the disk-save block
    gen = mock_env.generator
    gen.use_vae = False  # use enc_s output directly (no reparameterize)
    # Shape-faithful mocks so the per-sample 4D loop's tensors line up.
    gen.enc_c.side_effect = lambda x: torch.randn(x.shape[0], 8, 16, 16)
    gen.enc_s.side_effect = lambda x: torch.randn(x.shape[0], 4)
    gen.gen.side_effect = lambda c, s: torch.full((c.shape[0], 1, 64, 64), 0.5)

    strategy.logging_service = None
    # validation_metrics_computer is a read-only property delegating here.
    _metrics_computer = MagicMock()
    _metrics_computer.compute.return_value = {}
    strategy._get_validation_metrics_computer = MagicMock(return_value=_metrics_computer)

    input_batch = torch.full((1, 1, 64, 64), 0.2)  # SOURCE
    target_batch = torch.full((1, 1, 64, 64), 0.8)  # GROUND-TRUTH TARGET

    strategy.validation_step(input_batch, target_batch)

    assert strategy._last_visual_target is not None
    real_mean = float(strategy._last_visual_target.float().mean())
    assert real_mean == pytest.approx(0.8, abs=1e-3), (
        f"cached validation REAL mean={real_mean:.3f}; expected the target "
        "(0.8), got the source input (0.2) — real image is not ground truth"
    )


def test_fail_fast_missing_losses(mock_env):
    """Test that strategy fails fast if enabled losses are missing."""
    mock_env.config.losses.reconstruction.enable_perceptual = True
    mock_env.config.losses.reconstruction.lambda_perceptual = 1.0
    mock_env.losses.pop("perceptual")  # Remove from available losses

    with pytest.raises(RuntimeError, match="SSOT VIOLATION"):
        with (
            patch(_BLOCH_PATCH),
            patch(_LOSS_BUILDER_PATCH, new_callable=_mock_loss_builder),
        ):
            DisentangledTrainingStrategy(env=mock_env)


class TestGetLastMetricsStaysOnDevice:
    """The fourth implementation of a four-way contract (#707).

    `base`, `gan` and `mixins.adversarial` all stopped converting: the training
    loop calls `get_last_metrics` on EVERY iteration, outside the `log_interval`
    gate, so converting here is a per-step host sync no matter how well it is
    fused. This strategy kept fusing and so disagreed with its siblings about
    what the method returns.
    """

    def test_tensor_entries_are_not_converted(self):
        import torch

        from spectramr.infrastructure.training.strategies.disentangled_strategy import (
            DisentangledTrainingStrategy,
        )

        strategy = DisentangledTrainingStrategy.__new__(DisentangledTrainingStrategy)
        strategy._last_step_metrics = {"total_loss": torch.tensor(1.5)}

        out = DisentangledTrainingStrategy.get_last_metrics(strategy)

        assert isinstance(out["total_loss"], torch.Tensor), (
            "converting here re-pays the per-step sync #707 removed"
        )

    def test_non_tensor_entries_still_pass_through(self):
        """`loss_output.to_dict()` may carry non-numeric fields."""
        from spectramr.infrastructure.training.strategies.disentangled_strategy import (
            DisentangledTrainingStrategy,
        )

        strategy = DisentangledTrainingStrategy.__new__(DisentangledTrainingStrategy)
        strategy._last_step_metrics = {"note": "warmup", "step": 3}

        out = DisentangledTrainingStrategy.get_last_metrics(strategy)

        assert out == {"note": "warmup", "step": 3}

    def test_the_returned_dict_is_a_copy(self):
        """A caller mutating the result must not corrupt the next step's state."""
        from spectramr.infrastructure.training.strategies.disentangled_strategy import (
            DisentangledTrainingStrategy,
        )

        strategy = DisentangledTrainingStrategy.__new__(DisentangledTrainingStrategy)
        strategy._last_step_metrics = {"a": 1}

        out = DisentangledTrainingStrategy.get_last_metrics(strategy)
        out["a"] = 999

        assert strategy._last_step_metrics["a"] == 1

    def test_all_four_implementations_agree(self):
        """The contract is what makes the loop the single converter."""
        import inspect

        from spectramr.infrastructure.training.strategies import (
            disentangled_strategy,
            gan,
        )
        from spectramr.infrastructure.training.strategies.base import (
            BaseTrainingStrategy,
        )
        from spectramr.infrastructure.training.strategies.mixins import adversarial

        owners = [
            BaseTrainingStrategy,
            gan.GANTrainingStrategy,
            adversarial.AdversarialMixin,
            disentangled_strategy.DisentangledTrainingStrategy,
        ]
        for owner in owners:
            # Parse rather than grep: every one of these docstrings QUOTES the
            # `{k: float(v)}` it replaced, so a substring scan matches the
            # explanation and reports the fix as the defect.
            fn = ast.parse(textwrap.dedent(inspect.getsource(owner.get_last_metrics))).body[0]
            calls = {
                n.func.id
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "float" not in calls, (
                f"{owner.__name__}.get_last_metrics still converts: the loop "
                "calls it every iteration, so any conversion here is a per-step "
                "host sync (#707)"
            )


class TestValidationImageDirIsARealPath:
    """`validation_step` hands `metrics.output_dir` straight to `os.makedirs`.

    A `MagicMock` config leaf is not merely wrong there, it is *silently* wrong:
    `MagicMock` implements `__fspath__`, so it satisfies `os.PathLike` and
    `os.fspath` synthesises ``MagicMock/<mock name>/<id>`` from it. The call
    succeeded and built that tree in the process CWD -- the repo root -- once
    per run, where it reached the cluster's untracked-file list (#917).

    The `hasattr(...)` chain that used to guard this could not fail for a mock,
    because a mock answers every `hasattr` True. The guard is a type check now.
    """

    @staticmethod
    def _mock_self(output_dir):
        from spectramr.config.schemas.metrics import MetricsConfigSchema

        gen = MagicMock()
        gen.use_vae = False
        gen.enc_c.return_value = torch.randn(1, 4, 8, 8)
        gen.enc_s.return_value = torch.randn(1, 4, 1, 1)
        gen.gen.return_value = torch.randn(1, 1, 8, 8)

        mock_self = MagicMock()
        mock_self.device = torch.device("cpu")
        mock_self._in_channels = 1
        mock_self.env.generator = gen
        mock_self._recon_loss_fn = MagicMock(return_value=torch.tensor(0.5))
        for absent in (
            "_hist_loss_fn",
            "_perceptual_loss_fn",
            "_lpips_loss_fn",
            "_dists_loss_fn",
            "_mind_ssc_loss_fn",
        ):
            setattr(mock_self, absent, None)
        mock_self.validation_metrics_computer.compute.return_value = {"psnr": 20.0}
        mock_self._unpack_batch.side_effect = lambda b: (b["input"], b["target"])
        # The real schema, so the stand-in cannot disagree with it about a leaf.
        mock_self.config = SimpleNamespace(
            metrics=MetricsConfigSchema(output_dir=output_dir)
            if isinstance(output_dir, str)
            else SimpleNamespace(output_dir=output_dir)
        )
        return mock_self

    @staticmethod
    def _run(mock_self):
        x = torch.randn(1, 1, 8, 8)
        return DisentangledTrainingStrategy.validation_step(
            mock_self, batch={"input": x, "target": x}, input_batch=x, target_batch=x
        )

    def test_a_mock_output_dir_creates_no_directory(self, tmp_path, monkeypatch):
        """The defect itself: this wrote `MagicMock/` into whatever CWD it ran in."""
        monkeypatch.chdir(tmp_path)
        self._run(self._mock_self(MagicMock()))

        assert not (tmp_path / "MagicMock").exists(), (
            "a mock output_dir was accepted as a path -- os.fspath(MagicMock) "
            "synthesises one, so PathLike is not a sufficient check"
        )
        assert list(tmp_path.iterdir()) == [], f"stray writes: {list(tmp_path.iterdir())}"

    def test_the_guard_is_not_vacuous(self, tmp_path, monkeypatch):
        """A real str output_dir must still reach `makedirs` and write."""
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "run"
        out.mkdir()
        self._run(self._mock_self(str(out)))

        assert (out / "val_images").is_dir(), (
            "the type check rejected a legitimate str path -- the fix would be "
            "silencing the feature rather than guarding it"
        )


class TestLossWeightsMappingTargetsExist:
    """`loss_weights` reads config fields behind `hasattr`, so a stale key is silent.

    The loop is ``if hasattr(recon_settings, config_key): ... if weight > 0``. A
    renamed or deleted field therefore does not raise -- the loss simply never
    enters the weight dict, and the strategy trains without it. #421 renamed
    ``lambda_content`` to ``lambda_content_consistency``; this pins the follow-up
    and every other target with it.

    Read statically (AST) because the mapping is a literal inside the method and
    the assertion is about field names -- constructing the strategy would pull in
    a model and a device for no extra coverage.
    """

    @staticmethod
    def _mapping() -> dict[str, str]:
        import ast
        from pathlib import Path

        module = (
            Path(__file__).resolve().parents[5]
            / "src/spectramr/infrastructure/training/strategies/disentangled_strategy.py"
        )
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "loss_weights":
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Dict)
                        and sub.keys
                        and all(isinstance(k, ast.Constant) for k in sub.keys)
                    ):
                        mapping = {
                            k.value: v.value
                            for k, v in zip(sub.keys, sub.values, strict=True)
                            if isinstance(v, ast.Constant)
                        }
                        if any(str(k).startswith("lambda_") for k in mapping):
                            return mapping
        raise AssertionError("could not locate the config-field -> loss-name mapping")

    def test_every_mapped_config_key_exists_on_the_schema(self):
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        missing = [
            key for key in self._mapping() if key not in ReconstructionLossesConfig.model_fields
        ]
        assert not missing, (
            f"these config keys no longer exist: {missing}. The `hasattr` guard "
            "means the corresponding losses would be dropped from the weight "
            "dict with no diagnostic."
        )

    def test_content_reads_the_renamed_field(self):
        mapping = self._mapping()
        assert mapping["lambda_content_consistency"] == "content"
        assert "lambda_content" not in mapping


class TestPCGradClosureReturnsAConnectedLoss:
    """The PCGrad branch of ``gen_closure`` (#1952).

    This branch used to ``return loss_output.total.detach().requires_grad_(True)``
    -- a **leaf**. It worked only because ``backward()`` on a leaf is a silent
    no-op, which is exactly the mechanism that lets a severed loss train nothing.
    PCGrad has already written every ``p.grad`` by hand at that point, so the
    closure's job is to hand the executor a tensor that (a) carries a graph, so
    the new pre-backward guard admits it, (b) still reports the real loss value,
    because the executor logs what it is given, and (c) contributes exactly zero
    gradient, so the hand-written projections survive ``backward()``.

    Nothing covered this branch before, in this file or any other -- the only
    ``pcgrad`` test in the tree exercises the projection helper itself
    (``tests/unit/infrastructure/optimization/test_pcgrad.py``), not this seam.
    """

    @staticmethod
    def _drive(strategy, mock_env, *, total_value: float = 0.5):
        """Run ``train_step``'s generator closure with the PCGrad branch live.

        Returns ``(loss, params, recon_total)``.
        """
        from spectramr.models.losses.computers.base import LossOutput

        # A real parameter, so ``autograd.grad`` and ``p.grad`` are real.
        param = torch.nn.Parameter(torch.randn(4, requires_grad=True))
        mock_env.generator.parameters = MagicMock(return_value=[param])

        # Both losses must be grad-connected THROUGH ``param``: PCGrad calls
        # ``autograd.grad(loss, params)`` on each, and a total unrelated to
        # ``params`` yields all-None grads, which silently skips the projection
        # loop and would make this test vacuous.
        recon_total = (param * 2.0).sum() * 0.0 + total_value
        adv_total = (param * 3.0).sum() * 0.0 + 0.25

        strategy._enable_pcgrad = True
        mock_env.config.losses.reconstruction.use_curriculum_scheduling = False
        mock_env.config.losses.latent.use_capacity_scheduling = False
        # The shared ``mock_env`` leaves ``config.metrics`` a bare MagicMock, so
        # the training-metrics mixin compares a MagicMock with ``>`` and raises
        # (``metrics_mixin.py:988``). That is pre-existing fixture rot -- it is
        # why ``test_train_step_execution`` is red on ``dev`` today -- and it is
        # not this seam's subject, so switch the block off rather than repair it
        # here (NN17: the metrics cadence has its own owner).
        mock_env.config.metrics.enable_tracking = False

        adv_computer = MagicMock()
        adv_computer.compute_generator_loss.return_value = LossOutput(
            total=adv_total, components={"adv": adv_total.detach()}, metrics={}
        )
        strategy.adv_loss_computer = adv_computer

        with patch.object(strategy.loss_computer, "compute") as mock_compute:
            mock_compute.return_value = LossOutput(
                total=recon_total, components={"recon": recon_total.detach()}, metrics={}
            )
            batch = {
                "input": torch.randn(2, 1, 64, 64),
                "target": torch.randn(2, 1, 64, 64),
            }
            steps = strategy.train_step(batch, epoch=0)
            loss = steps[0]["closure"]()

        return loss, [param], recon_total

    def test_the_branch_is_actually_reached(self, strategy, mock_env):
        """Guard against a vacuous pass: if the setup stops short of the PCGrad
        branch the closure returns ``total_gen_loss`` instead, and every
        assertion below would hold for the wrong reason."""
        _, params, _ = self._drive(strategy, mock_env)
        # PCGrad writes p.grad by hand BEFORE backward(); the non-PCGrad return
        # path never touches it.
        assert params[0].grad is not None, "PCGrad branch was not entered"

    def test_the_returned_loss_carries_a_graph(self, strategy, mock_env):
        loss, _, _ = self._drive(strategy, mock_env)
        assert loss.grad_fn is not None, (
            "closure returned a leaf; the pre-backward guard would refuse it"
        )

    def test_the_reported_value_is_the_real_loss(self, strategy, mock_env):
        loss, _, recon_total = self._drive(strategy, mock_env)
        # The zero-weighted parameter term supplies the graph without moving the
        # value the executor logs.
        assert loss.item() == pytest.approx(recon_total.item())

    def test_backward_leaves_the_projected_gradients_untouched(self, strategy, mock_env):
        loss, params, _ = self._drive(strategy, mock_env)
        before = [p.grad.clone() for p in params]
        loss.backward()
        for p, b in zip(params, before, strict=True):
            assert torch.equal(p.grad, b), (
                "backward() perturbed the gradients PCGrad had already written"
            )
