"""Pytest configuration for spectraMR project
========================================

This file configures pytest for the spectraMR project with proper path setup
for imports and test discovery.
"""

import contextlib
import functools
import gc
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import yaml

# Include critical variables fixtures
pytest_plugins = ["tests.fixtures.critical_variables"]


def pytest_configure(config):
    """Set environment variables before any test module or plugin is imported.

    WandB wraps sys.stdout on import.  PyTorch's atexit handlers then try to
    write to WandB's wrapped stream after pytest has closed it, causing:
        ValueError: I/O operation on closed file.

    Setting WANDB_MODE=disabled before WandB is imported prevents the patch.
    Setting TORCHDYNAMO_VERBOSE=0 prevents Dynamo from registering the noisy
    atexit dump hooks at all.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")

    # One intra-op thread per xdist worker (#945).
    #
    # Torch sizes its thread pool from the HOST core count, not from its share
    # of it, and nothing here constrained it -- so `-n 8` on a 24-core box ran
    # 8 x 16 = 128 compute threads. Two consequences, both measured on an IDLE
    # machine, `tests/unit/.../strategies/` (5 files, 79 tests):
    #
    #   serial : 79 passed in     6.7 s
    #   -n 8   :  5 FAILED in   412.1 s
    #
    # The wall-clock is a 62x blowup from thrash. The failures are six seeded
    # "loss after training < loss before" assertions: oversubscription changes
    # the reduction order inside the matmuls, and a marginal inequality flips.
    # The failing SUBSET varied run to run, which is what made them impossible
    # to baseline -- and a baseline is the only gate available while CI is off.
    #
    # Pinned only under xdist: a serial run should still use the whole box, and
    # `PYTEST_XDIST_WORKER` is set in the worker processes and absent in a
    # serial session. Set before test modules import, so it lands ahead of any
    # module-scope work.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # Give each worker its SHARE of the box rather than all of it, so the
        # total lands near the core count whatever `-n` was passed:
        #   -n 8  on 24 cores -> 3 threads x 8  = 24
        #   -n 24 on 24 cores -> 1 thread x 24  = 24
        # `PYTEST_XDIST_WORKER_COUNT` is set alongside `PYTEST_XDIST_WORKER`.
        workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 1)
        threads = max(1, (os.cpu_count() or 1) // max(1, workers))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        import torch

        torch.set_num_threads(threads)

    # Suppress torch.jit deprecation warnings from third-party packages
    # (monai, etc.) that still use @torch.jit.script / @torch.jit.interface
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\..+` is deprecated",
        category=DeprecationWarning,
    )


# Source-level signals that a test HARD-REQUIRES CUDA (allocates on / moves to
# the device, or calls a torch.cuda.* runtime op). Deliberately excludes the
# device-agnostic `torch.cuda.is_available()` guard and bare "cuda" strings,
# which appear in device-agnostic tests that must still run in the CPU lane.
_CUDA_USAGE_PATTERNS = (
    ".cuda(",
    '.to("cuda',
    ".to('cuda",
    'device="cuda',
    "device='cuda",
    "torch.cuda.synchronize",
    "torch.cuda.memory_",
    "torch.cuda.max_memory",
    "torch.cuda.reset_peak",
    "torch.cuda.Event",
    "torch.cuda.empty_cache",
    'autocast("cuda',
    "autocast('cuda",
)

# Directories whose tests are GPU lanes by design (benchmarks / timing /
# end-to-end convergence). Marked gpu by path so the idle-GPU watchdog can run
# them as one continuous `-m gpu` batch.
_GPU_PATH_PREFIXES = (
    "tests/performance/",
    "tests/convergence/",
    "tests/benchmarks/",
)

# Directories whose tests are pre-launch smoke checks.
_SMOKE_PATH_PREFIXES = ("tests/smoke/",)


def _gpu_source_blob(item) -> str:
    """Best-effort source text for a test: its function body plus, for
    class-based tests, the enclosing class body (setup/helpers live there)."""
    blob = ""
    func = getattr(item, "function", None)
    if func is not None:
        try:
            blob += inspect.getsource(func)
        except (OSError, TypeError, AttributeError):
            pass
    cls = getattr(item, "cls", None)
    if cls is not None:
        try:
            blob += inspect.getsource(cls)
        except (OSError, TypeError, AttributeError):
            pass
    return blob


def pytest_collection_modifyitems(config, items):
    """Auto-mark ``smoke`` and ``gpu`` tests so the suite can be lane-selected.

    The cluster kills any GPU instance idle for an hour, so accurate ``gpu``
    marking lets ``pytest -m gpu`` run all CUDA work as one continuous batch.
    Marking is by (1) path — ``tests/smoke`` → smoke, GPU-lane dirs → gpu;
    (2) source grep for explicit device allocation / ``torch.cuda.*`` runtime
    ops. Each rule is idempotent and skips items already carrying the marker.
    """
    del config
    for item in items:
        nodeid = item.nodeid

        if nodeid.startswith(_SMOKE_PATH_PREFIXES) and "smoke" not in item.keywords:
            item.add_marker(pytest.mark.smoke)

        if "gpu" in item.keywords:
            continue
        if nodeid.startswith(_GPU_PATH_PREFIXES):
            item.add_marker(pytest.mark.gpu)
            continue
        if any(pat in _gpu_source_blob(item) for pat in _CUDA_USAGE_PATTERNS):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session", autouse=True)
def suppress_noisy_loggers():
    """Belt-and-suspenders: silence Dynamo/WandB loggers + patch dump funcs.

    pytest_configure handles the common case.  This fixture covers the rare
    case where torch._dynamo is imported before our env-var takes effect.

    The core problem: wandb monkey-patches sys.stdout/stderr with its own
    ConsoleCapture wrapper.  When pytest tears down, those streams close.
    PyTorch's atexit handlers (dump_compile_times, dump_cache_stats,
    _log_traced_frames) then try to log via the now-closed streams →
    ``ValueError: I/O operation on closed file``.

    We fix this by:
    1. Silencing the relevant loggers so nothing tries to emit.
    2. Replacing the dump functions with no-ops.
    3. Deregistering the atexit callbacks that call them.
    4. Restoring sys.stdout/stderr to real OS file objects at teardown.
    5. Explicitly closing any FileHandler / StreamHandler instances on the
       silenced loggers at teardown so Python's GC does not emit
       ``ResourceWarning: unclosed file`` (which pytest's
       unraisableexception plugin escalates to a teardown ERROR).
    """
    import atexit
    import sys

    _silenced_logger_names = (
        "torch._dynamo",
        "torch._dynamo.eval_frame",
        "torch._dynamo.utils",
        "torch._subclasses.fake_tensor",
        "wandb",
        "wandb.sdk",
    )
    for name in _silenced_logger_names:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Patch dump functions to no-ops if already imported
    _patched_funcs = []
    try:
        import torch._dynamo.utils as _dutils

        _patched_funcs.append(_dutils.dump_compile_times)
        _dutils.dump_compile_times = lambda *a, **kw: None  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    try:
        import torch._dynamo.eval_frame as _eval_frame

        if hasattr(_eval_frame, "_log_traced_frames"):
            _patched_funcs.append(_eval_frame._log_traced_frames)
            _eval_frame._log_traced_frames = lambda *a, **kw: None  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    try:
        import torch._subclasses.fake_tensor as _ft

        if hasattr(_ft, "dump_cache_stats"):
            _patched_funcs.append(_ft.dump_cache_stats)
            _ft.dump_cache_stats = lambda *a, **kw: None  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    # Deregister the atexit callbacks that hold stale references to the
    # original dump functions.  CPython stores these in atexit._ncallbacks
    # but the public API only offers unregister (3.12+) or _run_exitfuncs.
    # We use the private _clear() when available, otherwise just let our
    # no-op patches absorb the calls.
    if hasattr(atexit, "unregister"):
        for fn in _patched_funcs:
            atexit.unregister(fn)

    # Capture real OS-level file objects so we can restore them at teardown
    _real_stdout = os.fdopen(os.dup(1), "w", closefd=True)
    _real_stderr = os.fdopen(os.dup(2), "w", closefd=True)

    yield

    # Restore sys.stdout/stderr to real file objects so any remaining atexit
    # handlers (from wandb, torch, etc.) don't crash on closed streams.
    # Track whether we actually adopted the duplicated FDs so we can close
    # them otherwise — Python's GC would otherwise report them as unclosed
    # ResourceWarnings, which pytest's unraisableexception plugin escalates
    # into a teardown ERROR on whichever test runs last in the session.
    _stdout_adopted = False
    _stderr_adopted = False
    try:
        if sys.stdout.closed:
            sys.stdout = _real_stdout
            _stdout_adopted = True
        if sys.stderr.closed:
            sys.stderr = _real_stderr
            _stderr_adopted = True
    except (ValueError, AttributeError):
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr
        _stdout_adopted = _stderr_adopted = True

    if not _stdout_adopted:
        try:
            _real_stdout.close()
        except Exception:
            pass
    if not _stderr_adopted:
        try:
            _real_stderr.close()
        except Exception:
            pass

    # CC-T1: close any FileHandler / StreamHandler instances that the
    # silenced loggers accumulated during the session.  Without this,
    # Python's GC emits ``ResourceWarning: unclosed file`` for every
    # open file descriptor attached to these loggers, and
    # pytest's unraisableexception plugin escalates that to a teardown
    # ERROR on whichever test runs last.  We only close handlers whose
    # underlying stream is NOT the real sys.stdout / sys.stderr (those
    # are already managed above).
    _safe_std_fds = {1, 2}  # stdout / stderr file descriptors
    for _logger_name in _silenced_logger_names:
        _lgr = logging.getLogger(_logger_name)
        for _handler in list(_lgr.handlers):
            try:
                # Avoid closing handlers that back the real console
                # streams — those are managed by the adopted-fd block.
                _stream = getattr(_handler, "stream", None)
                if _stream is not None:
                    _fd = getattr(_stream, "fileno", None)
                    try:
                        if callable(_fd) and _fd() in _safe_std_fds:
                            continue
                    except Exception:
                        pass
                _handler.close()
            except Exception:
                pass
            finally:
                try:
                    _lgr.removeHandler(_handler)
                except Exception:
                    pass


try:
    import torch
except ImportError as _torch_import_err:
    if os.environ.get("SPECTRAMR_ALLOW_NO_TORCH") != "1":
        raise RuntimeError(
            "torch is required to run the test suite. "
            "Set SPECTRAMR_ALLOW_NO_TORCH=1 only for collection-only smoke checks."
        ) from _torch_import_err

    import sys
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device = MagicMock
    mock_torch.randn = MagicMock(return_value=MagicMock())
    mock_torch.manual_seed = MagicMock()
    mock_torch.seed = MagicMock()

    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda
    sys.modules["torch.backends"] = MagicMock()
    sys.modules["torch.backends.cudnn"] = MagicMock()

    import torch


@pytest.fixture
def slurm_account(monkeypatch):
    """A site allocation account, for tests that render sbatch directives.

    The tree ships no default account (it is site-specific), so rendering a job
    script raises unless ``$SPECTRAMR_SLURM_ACCOUNT`` is set.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")


@pytest.fixture(autouse=True)
def clean_mocks():
    """Ensure mocks are cleaned up before/after each test to prevent pollution."""
    import sys
    from unittest.mock import Mock

    def cleanup():
        patch.stopall()
        # Remove Mock objects from sys.modules that pollute the namespace
        to_remove = []
        for name, module in list(sys.modules.items()):
            if name.startswith("spectramr.") and isinstance(module, Mock):
                to_remove.append(name)
        for name in to_remove:
            del sys.modules[name]

    cleanup()  # Clean before
    yield
    cleanup()  # Clean after


@pytest.fixture(scope="session", autouse=True)
def setup_pythonpath():
    """Add the project root to the Python path."""
    import sys

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root():
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def experiments_dir(project_root):
    """Safely locates the experiments directory."""
    # Check common experiment config locations
    candidates = [
        project_root / "experiments" / "training",
        project_root / "experiments" / "configs",
        project_root / "experiments",
    ]

    for config_dir in candidates:
        if config_dir.exists():
            return config_dir

    # Return the most likely path even if it doesn't exist
    return project_root / "experiments" / "training"


@pytest.fixture
def load_test_config(experiments_dir):
    """Helper to load a YAML config safely."""

    def _load(filename):
        # Handle cases where filename implies a subdirectory
        file_path = experiments_dir / filename

        # Recursive search if not immediately found
        if not file_path.exists():
            matches = list(experiments_dir.rglob(filename))
            if matches:
                file_path = matches[0]
            else:
                raise FileNotFoundError(
                    f"Config {filename} not found in {experiments_dir}"
                )

        return file_path

    return _load


# Minimal test fixtures for performance
@pytest.fixture(scope="session")
def minimal_device():
    """Get the best available device for minimal testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device(minimal_device):
    """Alias for minimal_device to support tests expecting 'device' fixture."""
    return minimal_device


@pytest.fixture
def minimal_batch(minimal_device):
    """Minimal 2D batch for testing (batch_size=1, 1x16x16)."""
    return torch.randn(1, 1, 16, 16, device=minimal_device)


@pytest.fixture
def minimal_3d_batch(minimal_device):
    """Minimal 3D batch for testing (batch_size=1, 1x8x8x8)."""
    return torch.randn(1, 1, 8, 8, 8, device=minimal_device)


@pytest.fixture
def tiny_batch(minimal_device):
    """Tiny batch for ultra-fast tests (batch_size=1, 1x32x32)."""
    return torch.randn(1, 1, 32, 32, device=minimal_device)


@pytest.fixture
def micro_batch(minimal_device):
    """Micro batch for extremely fast tests (batch_size=1, 1x4x4)."""
    return torch.randn(1, 1, 4, 4, device=minimal_device)


@pytest.fixture
def small_batch(minimal_device):
    """Small batch for fast tests (batch_size=1, 1x8x8)."""
    return torch.randn(1, 1, 8, 8, device=minimal_device)


@pytest.fixture
def single_step_config():
    """Configuration for single training step tests."""
    return {
        "batch_size": 1,
        "epochs": 1,
        "steps_per_epoch": 1,
        "learning_rate": 1e-3,
        "model_type": "standard_unet",
        "training_mode": "gan",
    }


@pytest.fixture
def fast_training_config():
    """Configuration optimized for fast training tests."""
    return {
        "batch_size": 1,
        "epochs": 1,
        "steps_per_epoch": 2,
        "learning_rate": 1e-3,
        "model_type": "standard_unet",
        "training_mode": "gan",
        "optimizer": "adam",
        "loss_weights": {"gan": 1.0, "pixel": 0.1},
    }


HIGH_MEMORY_REPORT_BYTES = 100 * 1024 * 1024


@pytest.fixture(autouse=True)
def memory_monitoring():
    """Reclaim and account CUDA memory around each test.

    A CPU session returns immediately. Every accounting statement here is behind
    a CUDA guard, so the two unconditional ``gc.collect()`` calls this used to
    make bought nothing on a CPU node -- and they are not cheap: once the
    package is imported (~9k entries in ``sys.modules``) a full collection costs
    ~320 ms, which is ~0.64 s on every one of the ~45k tests the cluster array
    collects.

    A CUDA session pays the collection only when the test actually retained
    device memory. Reclaiming unconditionally spent the same ~320 ms on the
    overwhelming majority of tests that never allocate on the device at all.

    The companion ``oom_protection`` fixture was removed with the same edit: its
    ``except RuntimeError`` could never fire, because pytest does not throw a
    test's exception into a yield fixture, and its ``finally`` only repeated the
    reclaim below.
    """
    if not torch.cuda.is_available():
        yield
        return

    memory_before = torch.cuda.memory_allocated()

    yield

    memory_retained = torch.cuda.memory_allocated() - memory_before
    if memory_retained <= 0:
        return
    if memory_retained > HIGH_MEMORY_REPORT_BYTES:
        print(f"High memory usage: {memory_retained / 1024 / 1024:.1f} MB")
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def ensure_di_container_initialized():
    """Ensure DI container is initialized for all tests."""
    import logging as std_logging
    import tempfile

    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.di.di_container import init_container
    from tests.utils.logging_service import make_logging_service

    try:
        # Initialize container if not already done
        container = init_container()

        # Register logging service if not already registered
        try:
            if not container.has_service(ILoggingService):
                # Create temp directory for test logs
                test_log_dir = tempfile.gettempdir()
                logging_service = make_logging_service(test_log_dir)
                container.register(ILoggingService, logging_service)
        except ValueError:
            # Service already registered - this is OK
            pass
    except Exception as e:
        # If initialization fails, just continue - some tests may not need it
        print(f"DI Container Initialization Failed: {e}")
        import traceback

        traceback.print_exc()
        pass

    yield

    # Teardown: close any FileHandlers / TensorBoard writer attached to the
    # GANTraining logger so file descriptors are not leaked across tests.
    # pytest's unraisableexception plugin would otherwise flag the dangling
    # TextIOWrapper objects as ResourceWarning at teardown.
    try:
        gan_logger = std_logging.getLogger("GANTraining")
        for handler in list(gan_logger.handlers):
            if isinstance(handler, std_logging.FileHandler):
                try:
                    handler.close()
                except Exception:
                    pass
                gan_logger.removeHandler(handler)
    except Exception:
        pass

    try:
        if container.has_service(ILoggingService):
            svc = container.resolve(ILoggingService)
            writer = getattr(svc, "_writer", None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                svc._writer = None
    except Exception:
        pass


@pytest.fixture
def cpu_fallback_device():
    """Always use CPU device for consistent testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_minimal_batch(cpu_fallback_device):
    """Minimal batch on CPU for consistent testing."""
    return torch.randn(1, 1, 16, 16, device=cpu_fallback_device)


@pytest.fixture
def cpu_tiny_batch(cpu_fallback_device):
    """Tiny batch on CPU for ultra-fast tests."""
    return torch.randn(1, 1, 8, 8, device=cpu_fallback_device)


@pytest.fixture
def temp_experiment_dir(tmp_path):
    """Temporary directory for experiment artifacts."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model_type": "standard_unet",
        "training_mode": "gan",
        "batch_size": 1,
        "learning_rate": 1e-3,
        "epochs": 1,
        "data": {"input_shape": [1, 16, 16], "output_shape": [1, 16, 16]},
    }


@pytest.fixture
def deterministic_seed():
    """Set deterministic seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    yield
    # Reset to random state after test
    torch.seed()


def requires_lib(library_name: str, install_message: str | None = None) -> Callable:
    """Decorator to skip tests if a required library is not available.

    Args:
        library_name: Name of the required library
        install_message: Optional custom message for installation instructions

    Returns:
        Decorator function that skips the test if library is not available

    Example:
        @requires_lib("pytorch_msssim")
        def test_ssim_metrics():
            # Test will be skipped if pytorch_msssim is not installed
            pass

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                __import__(library_name)
                return func(*args, **kwargs)
            except ImportError:
                default_message = (
                    f"Test requires '{library_name}' library. "
                    f"Install with: pip install {library_name}"
                )
                message = install_message or default_message
                import pytest

                pytest.skip(message)

        return wrapper

    return decorator


# Common library requirements for easy reuse
requires_pytorch_msssim = requires_lib(
    "pytorch_msssim",
    "Test requires pytorch_msssim. Install with: pip install pytorch-msssim",
)

requires_torchvision = requires_lib(
    "torchvision",
    "Test requires torchvision. Install with: pip install torchvision",
)

requires_scikit_learn = requires_lib(
    "sklearn",
    "Test requires scikit-learn. Install with: pip install scikit-learn",
)

requires_pandas = requires_lib(
    "pandas",
    "Test requires pandas. Install with: pip install pandas",
)


@pytest.fixture(scope="function")
def setup_di_container(tmp_path):
    """Set up the DI container for tests."""
    from spectramr.config.config import TrainingConfig
    from spectramr.infrastructure.services.metrics_service import MetricsService
    from spectramr.infrastructure.services.model_card_service import ModelCardService

    import spectramr.infrastructure.di.di_container as di_module
    from spectramr.domain.interfaces.checkpoint_service_interface import (
        ICheckpointService,
    )
    from spectramr.domain.interfaces.model_card_interface import IModelCardService
    from spectramr.domain.interfaces.service_interfaces import (
        ILoggingService,
        IMemoryOptimizationService,
        IMetricsService,
    )
    from spectramr.infrastructure.di.di_container import init_container
    from spectramr.infrastructure.checkpointing.facade import CheckpointStore
    from spectramr.infrastructure.services.memory_optimization_service import (
        NoOpMemoryOptimizationService,
    )
    from spectramr.models.factories.model_factory import ModelFactory
    from spectramr.models.interfaces.models import IModelFactory
    from tests.utils.logging_service import make_logging_service

    # Ensure a clean slate by resetting the global container
    di_module._global_container = None
    di_module._global_container_manual_reset = True
    container = init_container()

    # Register all services
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    container.register(
        ILoggingService,
        provider=lambda: make_logging_service(str(log_dir)),
    )
    container.register(IMemoryOptimizationService, NoOpMemoryOptimizationService)
    container.register(ICheckpointService, CheckpointStore)
    container.register(IMetricsService, MetricsService)
    container.register(IModelCardService, ModelCardService)
    container.register(IModelFactory, ModelFactory, singleton=True)

    # Register training strategies by name (not used by factory, but for DI container testing)
    # container.register(ITrainingStrategy, GANTrainingStrategy, name="gan")
    # container.register(ITrainingStrategy, VAETrainingStrategy, name="vae")
    # container.register(ITrainingStrategy,
    #                    DiffusionTrainingStrategy, name="diffusion")
    # container.register(
    #     ITrainingStrategy,
    #     ReconstructionTrainingStrategy,
    #     name="reconstruction",
    # )

    # Register a default TrainingConfig
    config_path = "experiments/configs/hpo_config.yaml"
    if not os.path.exists(config_path):
        # Create a dummy config if it doesn't exist
        dummy_config = {
            "model_type": "standard_gan",
            "training_mode": "gan",
            "dataset": {"name": "dummy", "path": "/tmp/dummy"},
            "training": {"epochs": 1, "batch_size": 1},
        }
        with open(config_path, "w") as f:
            yaml.dump(dummy_config, f)

    cfg = TrainingConfig.from_yaml(config_path)
    container.register(TrainingConfig, provider=lambda: cfg)

    yield container

    # Teardown: clear the global container
    di_module._global_container = None


import pytest

if not hasattr(pytest, "subtests"):

    class _SubtestsProxy:
        def __call__(self, **kwargs: object):
            return contextlib.nullcontext()

        def __getattr__(self, _name: str):
            def _wrapper(**kwargs: object):
                return contextlib.nullcontext()

            return _wrapper

        def test(self, **kwargs: object):
            return contextlib.nullcontext()

    pytest.subtests = _SubtestsProxy()


class MockLoggingService:
    def log_metrics(self, metrics, *, step=None, prefix=None, level="info"):
        pass


# Dataset mocking fixtures using tmp_path
@pytest.fixture
def mock_cluster_datasets(tmp_path):
    """Create a comprehensive fake filesystem with all cluster dataset structures."""

    # Create base datasets directory
    datasets_dir = tmp_path / "cluster" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # 1. brats_sr - PNG slices
    _create_brats_sr_structure(datasets_dir)

    # 2. fastmri - HDF5 and DICOM
    _create_fastmri_structure(datasets_dir)

    # 3. hcp - NIfTI
    _create_hcp_structure(datasets_dir)

    # 4. m4raw - HDF5
    _create_m4raw_structure(datasets_dir)

    # 5. ulf_paired - NIfTI (BIDS)
    _create_ulf_paired_structure(datasets_dir)

    return datasets_dir


def _create_brats_sr_structure(base_dir):
    """Create brats_sr PNG slice structure."""
    brats_dir = base_dir / "brats_sr"
    brats_dir.mkdir(parents=True, exist_ok=True)

    # Create sample PNG files (simplified structure)
    patients = ["BRATS_001", "BRATS_002", "BRATS_003"]
    for patient in patients:
        patient_dir = brats_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Create slice PNG files
        for i in range(1, 11):  # 10 slices per patient
            (patient_dir / f"slice_{i:03d}.png").write_bytes(b"fake_png_data")


def _create_fastmri_structure(base_dir):
    """Create fastmri HDF5 and DICOM structures."""
    fastmri_dir = base_dir / "fastmri"
    fastmri_dir.mkdir(parents=True, exist_ok=True)

    # HDF5 files
    hdf5_files = [
        "brain_multicoil_train/file001.h5",
        "brain_multicoil_val/file002.h5",
        "knee_multicoil_train/file003.h5",
        "knee_multicoil_val/file004.h5",
    ]

    for hdf5_file in hdf5_files:
        start_path = fastmri_dir / hdf5_file
        start_path.parent.mkdir(parents=True, exist_ok=True)
        start_path.write_bytes(b"fake_hdf5_data")

    # DICOM breast structure (simplified)
    dicom_dir = fastmri_dir / "breast_multicoil"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    patients = ["patient001", "patient002"]
    for patient in patients:
        patient_dir = dicom_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Create scan directories
        for scan in ["scan01", "scan02"]:
            scan_dir = patient_dir / scan
            scan_dir.mkdir(parents=True, exist_ok=True)

            # Create sequence directories
            for seq in ["T1", "T2"]:
                seq_dir = scan_dir / seq
                seq_dir.mkdir(parents=True, exist_ok=True)

                # Create DICOM files (simplified - fewer files)
                for i in range(1, 6):  # 5 slices per sequence
                    (seq_dir / f"slice_{i:03d}.dcm").write_bytes(b"fake_dicom_data")


def _create_hcp_structure(base_dir):
    """Create hcp NIfTI structure."""
    hcp_dir = base_dir / "hcp"
    hcp_dir.mkdir(parents=True, exist_ok=True)

    # Create slice_sequences_image structure
    slice_seq_dir = hcp_dir / "slice_sequences_image"
    slice_seq_dir.mkdir(parents=True, exist_ok=True)

    # Create resampled_image structure
    resampled_dir = slice_seq_dir / "resampled_image"
    resampled_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata and resampled subdirs
    (resampled_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (resampled_dir / "resampled").mkdir(parents=True, exist_ok=True)

    # Add sample NIfTI files
    subjects = ["100206", "100307", "100408"]
    for subject in subjects:
        subject_dir = resampled_dir / "resampled" / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Create sample NIfTI files
        modalities = ["T1w", "T2w", "dwi"]
        for mod in modalities:
            (subject_dir / f"{subject}_{mod}.nii.gz").write_bytes(b"fake_nifti_data")
            (subject_dir / f"{subject}_{mod}.json").write_text('{"fake": "metadata"}')


def _create_m4raw_structure(base_dir):
    """Create m4raw HDF5 structure."""
    m4raw_dir = base_dir / "m4raw"
    m4raw_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories with HDF5 files
    subdirs = ["gre_data", "motion", "multicoil_test", "multicoil_train"]

    for subdir in subdirs:
        subdir_path = m4raw_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        # Create multiple HDF5 files per subdirectory
        for i in range(1, 6):  # 5 files per subdir
            (subdir_path / f"data_{i:03d}.h5").write_bytes(b"fake_hdf5_data")


def _create_ulf_paired_structure(base_dir):
    """Create ulf_paired BIDS-compliant NIfTI structure."""
    ulf_dir = base_dir / "ulf_paired"
    ulf_dir.mkdir(parents=True, exist_ok=True)

    # Create main data directory
    data_dir = ulf_dir / "ulf_paired_64mt_3t" / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 3T_data BIDS structure
    _create_3t_data_structure(data_dir)

    # 64mT_data BIDS structure
    _create_64mt_data_structure(data_dir)


def _create_3t_data_structure(data_dir):
    """Create 3T_data BIDS structure."""
    t3_dir = data_dir / "3T_data"
    t3_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset description
    (t3_dir / "dataset_description.json").write_text(
        '{"Name": "ULF Paired 3T", "BIDSVersion": "1.0.0"}'
    )

    # Create subjects (simplified - fewer subjects)
    subjects_3t = ["0011", "0015", "0023", "0025", "0027"]
    for subj in subjects_3t:
        subj_dir = t3_dir / f"sub-{subj}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Create anat directory with files
        anat_dir = subj_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        modalities = ["FLAIR", "T1w", "T2w"]
        for mod in modalities:
            (anat_dir / f"sub-{subj}_acq-highres_{mod}.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (anat_dir / f"sub-{subj}_acq-highres_{mod}.json").write_text(
                '{"fake": "metadata"}'
            )
            (anat_dir / f"sub-{subj}_acq-lowres_{mod}.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (anat_dir / f"sub-{subj}_acq-lowres_{mod}.json").write_text(
                '{"fake": "metadata"}'
            )

        # Create dwi directory with files
        dwi_dir = subj_dir / "dwi"
        dwi_dir.mkdir(parents=True, exist_ok=True)

        # DWI files
        (dwi_dir / f"sub-{subj}_acq-highres_adc.nii.gz").write_bytes(b"fake_nifti_data")
        (dwi_dir / f"sub-{subj}_acq-highres_adc.json").write_text(
            '{"fake": "metadata"}'
        )
        (dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.bval").write_text("0\n1000\n")
        (dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.bvec").write_text(
            "1 0 0\n0 1 0\n"
        )
        (dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.nii.gz").write_bytes(
            b"fake_nifti_data"
        )
        (dwi_dir / f"sub-{subj}_acq-highres_run-1_dwi.json").write_text(
            '{"fake": "metadata"}'
        )


def _create_64mt_data_structure(data_dir):
    """Create 64mT_data BIDS structure."""
    mt64_dir = data_dir / "64mT_data"
    mt64_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset description
    (mt64_dir / "dataset_description.json").write_text(
        '{"Name": "ULF Paired 64mT", "BIDSVersion": "1.0.0"}'
    )

    # Create subjects (simplified - fewer subjects)
    subjects_64mt = ["0001", "0002", "0003", "0004", "0005"]
    for subj in subjects_64mt:
        subj_dir = mt64_dir / f"sub-{subj}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Create sessions
        sessions = ["01", "02"] if subj in ["0001", "0012", "0015"] else ["01"]
        for ses in sessions:
            ses_dir = subj_dir / f"ses-{ses}"
            ses_dir.mkdir(parents=True, exist_ok=True)

            # Create anat directory
            anat_dir = ses_dir / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)

            # Anat files
            modalities = ["FLAIR", "T1w", "T2w"]
            for mod in modalities:
                (anat_dir / f"sub-{subj}_ses-{ses}_run-1_{mod}.nii.gz").write_bytes(
                    b"fake_nifti_data"
                )
                (anat_dir / f"sub-{subj}_ses-{ses}_run-1_{mod}.json").write_text(
                    '{"fake": "metadata"}'
                )
                if mod == "T1w":
                    (
                        anat_dir
                        / f"sub-{subj}_ses-{ses}_run-1_{mod}_acq-localizer.nii.gz"
                    ).write_bytes(b"fake_nifti_data")
                    (
                        anat_dir
                        / f"sub-{subj}_ses-{ses}_run-1_{mod}_acq-localizer.json"
                    ).write_text('{"fake": "metadata"}')

            # Create dwi directory
            dwi_dir = ses_dir / "dwi"
            dwi_dir.mkdir(parents=True, exist_ok=True)

            # DWI files
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_ADC.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_ADC.json").write_text(
                '{"fake": "metadata"}'
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_dwi.bval").write_text("0\n1000\n")
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_dwi.bvec").write_text(
                "1 0 0\n0 1 0\n"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_dwi.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-1_dwi.json").write_text(
                '{"fake": "metadata"}'
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_ADC.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_ADC.json").write_text(
                '{"fake": "metadata"}'
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_dwi.bval").write_text("0\n1000\n")
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_dwi.bvec").write_text(
                "1 0 0\n0 1 0\n"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_dwi.nii.gz").write_bytes(
                b"fake_nifti_data"
            )
            (dwi_dir / f"sub-{subj}_ses-{ses}_run-2_dwi.json").write_text(
                '{"fake": "metadata"}'
            )


@pytest.fixture
def mock_brats_sr_png(mock_cluster_datasets):
    """Fixture providing brats_sr PNG dataset path."""
    return mock_cluster_datasets / "brats_sr"


@pytest.fixture
def mock_fastmri_hdf5(mock_cluster_datasets):
    """Fixture providing fastmri HDF5 dataset path."""
    return mock_cluster_datasets / "fastmri"


@pytest.fixture
def mock_fastmri_dicom(mock_cluster_datasets):
    """Fixture providing fastmri DICOM dataset path."""
    return mock_cluster_datasets / "fastmri" / "breast_multicoil"


@pytest.fixture
def mock_hcp_nifti(mock_cluster_datasets):
    """Fixture providing hcp NIfTI dataset path."""
    return mock_cluster_datasets / "hcp"


@pytest.fixture
def mock_m4raw_h5(mock_cluster_datasets):
    """Fixture providing m4raw HDF5 dataset path."""
    return mock_cluster_datasets / "m4raw"


@pytest.fixture
def mock_ulf_paired_nifti(mock_cluster_datasets):
    """Fixture providing ulf_paired NIfTI dataset path."""
    return mock_cluster_datasets / "ulf_paired"


# ---------------------------------------------------------------------------
# Deterministic synthetic-MRI fixtures (CPU-only, no GPU needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_image(deterministic_seed):
    """Synthetic real-valued image: [2, 1, 16, 16] float32, CPU.

    Seeded deterministically via the ``deterministic_seed`` fixture
    (``torch.manual_seed(42)``).
    """
    return torch.randn(2, 1, 16, 16, dtype=torch.float32)


@pytest.fixture
def synth_complex_image(deterministic_seed):
    """Synthetic complex image: [2, 1, 16, 16] complex64, CPU.

    Real and imaginary components are independent standard-normal draws
    seeded by the ``deterministic_seed`` fixture.
    """
    real = torch.randn(2, 1, 16, 16, dtype=torch.float32)
    imag = torch.randn(2, 1, 16, 16, dtype=torch.float32)
    return torch.complex(real, imag)


@pytest.fixture
def synth_multicoil(deterministic_seed):
    """Synthetic multi-coil k-space: [1, 4, 16, 16] complex64, CPU.

    Simulates a single-slice acquisition with 4 receive coils.
    Seeded by the ``deterministic_seed`` fixture.
    """
    real = torch.randn(1, 4, 16, 16, dtype=torch.float32)
    imag = torch.randn(1, 4, 16, 16, dtype=torch.float32)
    return torch.complex(real, imag)


@pytest.fixture
def cartesian_mask():
    """Binary Cartesian undersampling mask: [1, 1, 16, 16] float32, CPU.

    Every other column is sampled (columns 0, 2, 4, … are 1; odd
    columns are 0), yielding approximately 50% sampling density.
    This is a deterministic, generator-independent fixture.
    """
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
    mask[:, :, :, ::2] = 1.0
    return mask


@pytest.fixture
def tolerances():
    """Return the project-wide numerical tolerance dictionary.

    Delegates to ``tests.utils.tolerances.TOLERANCES`` so that test
    code can write ``tolerances["fp32"]`` without importing the module
    directly.

    Returns:
        dict[str, float]: Mapping from dtype-name string to absolute
        tolerance value.
    """
    from tests.utils.tolerances import TOLERANCES  # noqa: PLC0415

    return TOLERANCES
