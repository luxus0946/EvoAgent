"""REST API 测试（FastAPI TestClient）。"""

import pytest
from fastapi.testclient import TestClient

from app.api import TASKS, app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_problems_list(self):
        resp = client.get("/api/problems")
        assert resp.status_code == 200
        probs = resp.json()["problems"]
        assert "semiconductor" in probs
        assert "rastrigin" in probs


class TestEvolveTask:
    def test_submit_and_poll(self):
        resp = client.post(
            "/api/evolve",
            json={
                "problem": "rastrigin",
                "generations": 2,
                "population": 3,
                "islands": 1,
                "budget": 40,
                "seed": 1,
            },
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        task = client.get(f"/api/tasks/{task_id}").json()
        assert task["status"] == "completed"
        result = task["result"]
        assert result["best_fitness"] is not None
        assert len(result["generation_history"]) == 2
        assert result["total_evals"] > 0

    def test_unknown_task_404(self):
        resp = client.get("/api/tasks/nonexistent")
        assert resp.status_code == 404

    def test_unknown_problem_fails(self):
        resp = client.post(
            "/api/evolve",
            json={
                "problem": "nope",
                "generations": 2,
                "population": 3,
                "islands": 1,
                "budget": 40,
            },
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        task = client.get(f"/api/tasks/{task_id}").json()
        assert task["status"] == "failed"

    def test_invalid_params_rejected(self):
        resp = client.post(
            "/api/evolve",
            json={
                "problem": "rastrigin",
                "generations": 0,
                "population": 3,
                "islands": 1,
                "budget": 40,
            },
        )
        assert resp.status_code == 422