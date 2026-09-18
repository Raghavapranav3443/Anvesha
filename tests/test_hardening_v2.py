"""Tests for the architectural hardening changes.

Covers:
  - Singleton thread safety (double-checked locking)
  - scipy EDT correctness vs chamfer fallback
  - Experiment logging
  - Investigation plan selection
  - Embedding-augmented routing
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Singleton thread safety
# --------------------------------------------------------------------------- #

class TestSingletonThreadSafety:
    def test_get_controller_returns_same_instance(self):
        """Multiple threads calling get_controller() must get the same object."""
        from anvesha.agent import get_controller
        results = [None] * 10

        def worker(i):
            results[i] = get_controller()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        instances = [r for r in results if r is not None]
        assert len(instances) == 10, f"Expected 10 results, got {len(instances)}"
        assert all(r is instances[0] for r in instances), \
            "All threads must receive the same singleton instance"


# --------------------------------------------------------------------------- #
# scipy EDT correctness
# --------------------------------------------------------------------------- #

class TestChamferDistance:
    def test_single_pixel_distance(self):
        """Distance from a single pixel should be Euclidean."""
        from anvesha.impact import chamfer_distance
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        d = chamfer_distance(mask)
        # Distance from center (5,5) to corner (0,0) = sqrt(50) ~ 7.07
        assert abs(d[0, 0] - np.sqrt(50)) < 0.5, \
            f"Expected ~7.07, got {d[0, 0]:.2f}"
        # Distance to self should be 0
        assert d[5, 5] == 0.0

    def test_all_true_mask_gives_zero(self):
        """When all pixels are True, distance should be zero everywhere."""
        from anvesha.impact import chamfer_distance
        mask = np.ones((20, 20), dtype=bool)
        d = chamfer_distance(mask)
        assert np.allclose(d, 0.0), "All-True mask should give zero distance"

    def test_symmetry(self):
        """Distance should be symmetric around the target."""
        from anvesha.impact import chamfer_distance
        mask = np.zeros((15, 15), dtype=bool)
        mask[7, 7] = True
        d = chamfer_distance(mask)
        # Check vertical symmetry
        assert np.allclose(d[7, :], d[7, ::-1], atol=0.5), "Not vertically symmetric"


# --------------------------------------------------------------------------- #
# Experiment logging
# --------------------------------------------------------------------------- #

class TestExperimentLog:
    def test_log_experiment_creates_file(self, tmp_path):
        """log_experiment should create a JSONL file."""
        from anvesha.experiment_log import log_experiment
        import anvesha.config as cfg_mod
        original_runs = cfg_mod.CONFIG.runs_dir
        cfg_mod.CONFIG.runs_dir = tmp_path
        try:
            record = log_experiment(
                script="test_script",
                args={"lr": 0.001},
                metrics={"val_acc": 0.95},
                checkpoint="weights/test.pt",
            )
            assert record["script"] == "test_script"
            assert record["metrics"]["val_acc"] == 0.95
            assert (tmp_path / "experiments.jsonl").exists()
        finally:
            cfg_mod.CONFIG.runs_dir = original_runs

    def test_recent_experiments_reads_back(self, tmp_path):
        """recent_experiments should return logged records."""
        from anvesha.experiment_log import log_experiment, recent_experiments
        import anvesha.config as cfg_mod
        original_runs = cfg_mod.CONFIG.runs_dir
        cfg_mod.CONFIG.runs_dir = tmp_path
        try:
            log_experiment(script="test_a", metrics={"score": 1})
            log_experiment(script="test_b", metrics={"score": 2})
            records = recent_experiments(10)
            assert len(records) == 2
            assert records[0]["script"] == "test_a"
            assert records[1]["script"] == "test_b"
        finally:
            cfg_mod.CONFIG.runs_dir = original_runs


# --------------------------------------------------------------------------- #
# Investigation plan selection
# --------------------------------------------------------------------------- #

class TestInvestigationPlan:
    def test_urban_query_selects_urban_plan(self):
        from anvesha.agent import AgentController
        plan_key = AgentController._select_investigation_plan(
            "Investigate urban expansion around the water body")
        assert plan_key == "urban"

    def test_vegetation_query_selects_vegetation_plan(self):
        from anvesha.agent import AgentController
        plan_key = AgentController._select_investigation_plan(
            "What happened to the forest cover?")
        assert plan_key == "vegetation"

    def test_default_query_returns_default_plan(self):
        from anvesha.agent import AgentController
        plan_key = AgentController._select_investigation_plan(
            "What changed between these dates?")
        assert plan_key == "default"


# --------------------------------------------------------------------------- #
# Embedding-augmented routing
# --------------------------------------------------------------------------- #

class TestEmbeddingRouting:
    def test_bow_vector_normalised(self):
        from anvesha.agent import _query_bow
        vec = _query_bow("is there water in this image")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"BOW vector not normalised: norm={norm:.6f}"

    def test_cosine_sim_identical(self):
        from anvesha.agent import _cosine_sim, _query_bow
        vec = _query_bow("water body")
        assert abs(_cosine_sim(vec, vec) - 1.0) < 1e-5

    def test_task_centroids_computed(self):
        from anvesha.agent import _get_task_centroids
        import anvesha.agent as agent_mod
        agent_mod._task_centroids = None  # reset
        centroids = _get_task_centroids()
        # Data-derived centroids cover RSVQA types; keyword fallback covers all
        assert len(centroids) >= 4, f"Expected at least 4 centroids, got {len(centroids)}"
        # Check that at least one centroid has correct dimensionality
        first_vec = next(iter(centroids.values()))
        assert first_vec.shape == (512,)

    def test_similar_queries_route_to_same_task(self):
        from anvesha.agent import classify_task
        r1 = classify_task("is there water present", "single")
        r2 = classify_task("can you see a water body", "single")
        # Both should route to single_vqa (the closest feasible task for single-image)
        assert r1["task"] == r2["task"], \
            f"Similar queries should route identically: {r1['task']} vs {r2['task']}"
