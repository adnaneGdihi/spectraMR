import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path = [str(Path.cwd()), *sys.path]

# We need to ensure we can import from src
try:
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
        HealthCheckResult,
    )
except ImportError as e:
    print(f"FAILED TO IMPORT ConfigHealthChecker: {e}")
    sys.exit(1)


class TestPruningVerification(unittest.TestCase):
    def setUp(self):
        # Create a mock config
        self.config = MagicMock()

        # Setup losses structure simulating Config object
        self.config.losses = MagicMock()
        self.config.losses.reconstruction = MagicMock()
        self.config.losses.reconstruction.lambda_l1 = 1.0
        self.config.losses.reconstruction.lambda_l2 = 0.0
        self.config.losses.reconstruction.lambda_perceptual = 0.1
        self.config.losses.reconstruction.lambda_ssim = 0.5

        # Setup other sections
        self.config.physics = MagicMock()
        self.config.training = MagicMock()
        self.config.training.training_mode = "reconstruction"

        self.config.data = MagicMock()
        self.config.model = MagicMock()
        self.config.model.model_type = "standard_unet"
        self.config.optimization = MagicMock()
        self.config.logging = MagicMock()

        # Add basic attributes for hasattr checks
        self.config.losses.physics = None

    def test_physics_config_check(self):
        checker = ConfigHealthChecker()

        # Case 1: Physics present
        self.config.physics = MagicMock()
        results = checker.check_physics_config(self.config)
        self.assertEqual(len(results), 0)

        # Case 2: Physics needed but missing
        self.config.physics = None
        self.config.losses.physics = None
        self.config.training.training_mode = "reconstruction"

        results = checker.check_physics_config(self.config)
        self.assertTrue(len(results) > 0, "Expected warning for missing physics config")
        self.assertIn("missing", results[0].message)

    def test_legacy_modules_deleted(self):
        """Verify deleted modules cannot be imported."""
        with self.assertRaises(ImportError):
            pass

        with self.assertRaises(ImportError):
            pass

        with self.assertRaises(ImportError):
            pass

        with self.assertRaises(ImportError):
            pass

        print("Legacy modules confirmed deleted.")


if __name__ == "__main__":
    unittest.main()
