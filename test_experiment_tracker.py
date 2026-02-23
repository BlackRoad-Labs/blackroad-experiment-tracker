"""Tests for BlackRoad Experiment Tracker."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["EXPERIMENT_DB"] = str(Path(tempfile.mkdtemp()) / "test_experiments.db")
sys.path.insert(0, str(Path(__file__).parent))
from main import Experiment, ExperimentTracker, init_db


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        init_db()
        self.tracker = ExperimentTracker()

    def test_create_experiment(self):
        exp = self.tracker.create("Run 1", "resnet50", hyperparams={"lr": 0.001, "epochs": 10})
        self.assertIsNotNone(exp.id)
        self.assertEqual(exp.model, "resnet50")
        self.assertEqual(exp.status, "running")

    def test_log_metric(self):
        exp = self.tracker.create("Metric Test", "bert")
        self.tracker.log_metric(exp.id, "accuracy", 0.85, step=1)
        self.tracker.log_metric(exp.id, "accuracy", 0.90, step=2)
        loaded = self.tracker.get(exp.id)
        self.assertIn("accuracy", loaded.metrics)
        self.assertEqual(len(loaded.metrics["accuracy"]), 2)

    def test_finish_experiment(self):
        exp = self.tracker.create("Finish Test", "gpt2")
        self.tracker.finish(exp.id, status="completed")
        loaded = self.tracker.get(exp.id)
        self.assertEqual(loaded.status, "completed")

    def test_compare_experiments(self):
        e1 = self.tracker.create("Exp A", "model_v1", hyperparams={"lr": 0.001})
        e2 = self.tracker.create("Exp B", "model_v2", hyperparams={"lr": 0.01})
        self.tracker.log_metric(e1.id, "f1", 0.8)
        self.tracker.log_metric(e2.id, "f1", 0.9)
        cmp = self.tracker.compare([e1.id, e2.id])
        self.assertIn("metric_comparison", cmp)
        self.assertIn("f1", cmp["metric_comparison"])

    def test_best_run(self):
        e1 = self.tracker.create("Best A", "m1")
        e2 = self.tracker.create("Best B", "m2")
        self.tracker.log_metric(e1.id, "acc", 0.75)
        self.tracker.log_metric(e2.id, "acc", 0.92)
        self.tracker.finish(e1.id)
        self.tracker.finish(e2.id)
        best = self.tracker.best_run("acc", maximize=True)
        self.assertIsNotNone(best)
        self.assertEqual(best["id"], e2.id)

    def test_export_report(self):
        tmp = Path(tempfile.mkdtemp())
        exp = self.tracker.create("Report Test", "xgboost", hyperparams={"n_estimators": 100})
        self.tracker.log_metric(exp.id, "rmse", 0.15, step=1)
        self.tracker.log_metric(exp.id, "rmse", 0.12, step=2)
        out = self.tracker.export_report(exp.id, output_path=str(tmp / "report.md"))
        self.assertTrue(Path(out).exists())
        content = Path(out).read_text()
        self.assertIn("xgboost", content)

    def test_list_by_status(self):
        e1 = self.tracker.create("Running", "modelA")
        e2 = self.tracker.create("Completed", "modelB")
        self.tracker.finish(e2.id)
        completed = self.tracker.list(status="completed")
        ids = [e["id"] for e in completed]
        self.assertIn(e2.id, ids)


if __name__ == "__main__":
    unittest.main()
