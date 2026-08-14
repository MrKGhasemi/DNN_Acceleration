from pathlib import Path
import unittest

from food101_accel.config import ExperimentConfig


class ExperimentConfigTests(unittest.TestCase):
    def test_profiles_expose_only_intended_acceleration_switches(self) -> None:
        baseline = ExperimentConfig(profile="baseline")
        accelerated = ExperimentConfig(profile="accelerated")

        self.assertFalse(baseline.use_compile)
        self.assertFalse(baseline.use_amp)
        self.assertFalse(baseline.allow_tf32)
        self.assertEqual(baseline.matmul_precision, "highest")

        self.assertTrue(accelerated.use_compile)
        self.assertTrue(accelerated.use_amp)
        self.assertTrue(accelerated.allow_tf32)
        self.assertEqual(accelerated.matmul_precision, "high")

    def test_paths_are_serialized_for_json(self) -> None:
        config = ExperimentConfig(
            profile="baseline", data_root=Path("dataset"), output_dir=Path("runs")
        )
        payload = config.as_serializable_dict()
        self.assertEqual(payload["data_root"], "dataset")
        self.assertEqual(payload["output_dir"], "runs")

    def test_invalid_food101_shape_is_rejected(self) -> None:
        config = ExperimentConfig(profile="baseline", image_size=224)
        with self.assertRaisesRegex(ValueError, "384"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
