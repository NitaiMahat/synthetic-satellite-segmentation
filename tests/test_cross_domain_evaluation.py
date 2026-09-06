import unittest

from evaluation.evaluate_rural_to_urban import summarize_runs


class CrossDomainEvaluationTests(unittest.TestCase):
    def test_summarize_runs_calculates_seed_means(self) -> None:
        runs = {
            "42": {"miou": 0.2, "pixel_accuracy": 0.4, "per_class_iou": [0.1, 0.3]},
            "7": {"miou": 0.4, "pixel_accuracy": 0.6, "per_class_iou": [0.3, 0.5]},
        }

        summary = summarize_runs(runs)

        self.assertEqual(summary["runs"], 2)
        self.assertAlmostEqual(summary["miou_mean"], 0.3)
        self.assertAlmostEqual(summary["pixel_accuracy_mean"], 0.5)
        self.assertEqual(summary["per_class_iou_mean"], [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
