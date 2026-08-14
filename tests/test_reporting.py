import unittest

from food101_accel.reporting import aggregate_profile_runs, compute_speedups


def run(profile: str, seconds: list[float], throughput: list[float]) -> dict:
    return {
        "profile": profile,
        "epochs": [
            {
                "total_seconds": value,
                "train_samples_per_second": throughput[index],
                "eval_top1": 0.8 + index / 100,
                "eval_top5": 0.95 + index / 100,
            }
            for index, value in enumerate(seconds)
        ],
    }


class ReportingTests(unittest.TestCase):
    def test_cold_and_steady_state_are_kept_separate(self) -> None:
        summary = aggregate_profile_runs(
            [
                run("baseline", [20.0, 20.0], [100.0, 100.0]),
                run("accelerated", [18.0, 10.0], [150.0, 200.0]),
            ],
            discard_first_epochs=1,
        )
        speedups = compute_speedups(summary)

        self.assertAlmostEqual(speedups["cold_epoch_speedup"], 20.0 / 14.0)
        self.assertAlmostEqual(speedups["steady_epoch_speedup"], 2.0)
        self.assertAlmostEqual(speedups["training_throughput_speedup"], 1.75)

    def test_both_profiles_are_required_for_speedup(self) -> None:
        summary = aggregate_profile_runs(
            [run("baseline", [20.0], [100.0])], discard_first_epochs=0
        )
        with self.assertRaisesRegex(ValueError, "both baseline"):
            compute_speedups(summary)


if __name__ == "__main__":
    unittest.main()
