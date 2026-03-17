"""Tests for the FastAPI HTTP endpoints."""
import pytest
import os
os.environ.setdefault("TESTING", "1")  # prevent loading from data/ files

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Patch _load_state to do nothing during tests
    import social_arena.ui.app as app_module
    app_module._load_state = lambda: None
    # Reset global state
    app_module.registered_agents.clear()
    app_module.match_history.clear()
    # Also patch saves so tests don't write files
    app_module._save_agents = lambda: None
    app_module._save_history = lambda: None
    app_module._save_leaderboard = lambda: None
    from social_arena.ui.app import app
    return TestClient(app)


class TestConfigEndpoint:
    def test_get_config_returns_tasks(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert "prisoners_dilemma" in data["tasks"]
        assert "salary_negotiation" in data["tasks"]
        assert "policy_debate" in data["tasks"]
        assert "werewolf" in data["tasks"]

    def test_get_config_returns_providers(self, client):
        r = client.get("/api/config")
        data = r.json()
        assert "providers" in data
        assert "anthropic" in data["providers"]
        assert "openai" in data["providers"]
        assert "gemini" in data["providers"]


class TestAgentEndpoints:
    def test_list_agents_empty_initially(self, client):
        import social_arena.ui.app as app_module
        app_module.registered_agents.clear()
        r = client.get("/api/agents")
        assert r.status_code == 200
        assert r.json() == []

    def test_register_mock_agent(self, client):
        r = client.post("/api/agents", json={
            "name": "TestCoop", "agent_type": "mock", "strategy": "cooperative"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "TestCoop"
        assert "agent_id" in data

    def test_registered_agent_appears_in_list(self, client):
        client.post("/api/agents", json={"name": "ListMe", "agent_type": "mock", "strategy": "random"})
        r = client.get("/api/agents")
        names = [a["name"] for a in r.json()]
        assert "ListMe" in names

    def test_register_all_mock_strategies(self, client):
        for strategy in ["cooperative", "competitive", "tit_for_tat", "random"]:
            r = client.post("/api/agents", json={
                "name": f"Agent_{strategy}", "agent_type": "mock", "strategy": strategy
            })
            assert r.status_code == 200, f"Failed for strategy: {strategy}"

    def test_delete_agent(self, client):
        r = client.post("/api/agents", json={"name": "ToDelete", "agent_type": "mock", "strategy": "cooperative"})
        agent_id = r.json()["agent_id"]
        r2 = client.delete(f"/api/agents/{agent_id}")
        assert r2.status_code == 200
        ids = [a["agent_id"] for a in client.get("/api/agents").json()]
        assert agent_id not in ids

    def test_delete_nonexistent_agent_returns_404(self, client):
        r = client.delete("/api/agents/nonexistent_id")
        assert r.status_code == 404

    def test_invalid_agent_type_returns_400(self, client):
        r = client.post("/api/agents", json={"name": "Bad", "agent_type": "invalid"})
        assert r.status_code == 400


class TestMatchEndpoints:
    def _register_two(self, client):
        a1 = client.post("/api/agents", json={"name": "A1", "agent_type": "mock", "strategy": "cooperative"}).json()["agent_id"]
        a2 = client.post("/api/agents", json={"name": "A2", "agent_type": "mock", "strategy": "competitive"}).json()["agent_id"]
        return a1, a2

    def test_start_match_returns_match_id(self, client):
        a1, a2 = self._register_two(client)
        r = client.post("/api/matches", json={"task": "prisoners_dilemma", "agent_ids": [a1, a2], "params": {"rounds": 3}})
        assert r.status_code == 200
        assert "match_id" in r.json()

    def test_unknown_task_returns_400(self, client):
        a1, a2 = self._register_two(client)
        r = client.post("/api/matches", json={"task": "unknown_task", "agent_ids": [a1, a2], "params": {}})
        assert r.status_code == 400

    def test_too_few_agents_returns_400(self, client):
        a1, _ = self._register_two(client)
        r = client.post("/api/matches", json={"task": "prisoners_dilemma", "agent_ids": [a1], "params": {}})
        assert r.status_code == 400

    def test_unknown_agent_id_returns_400(self, client):
        a1, _ = self._register_two(client)
        r = client.post("/api/matches", json={"task": "prisoners_dilemma", "agent_ids": [a1, "fake_id"], "params": {}})
        assert r.status_code == 400

    def test_match_result_retrievable(self, client):
        import time
        a1, a2 = self._register_two(client)
        match_id = client.post("/api/matches", json={
            "task": "prisoners_dilemma", "agent_ids": [a1, a2], "params": {"rounds": 3}
        }).json()["match_id"]
        # Poll until complete
        for _ in range(30):
            r = client.get(f"/api/matches/{match_id}")
            if r.json().get("status") == "complete":
                break
            time.sleep(0.3)
        result = client.get(f"/api/matches/{match_id}").json()
        assert result["status"] == "complete"
        assert "scores" in result
        assert "winner" in result


class TestLeaderboardEndpoint:
    def test_leaderboard_returns_list(self, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_leaderboard_sorted_by_score(self, client):
        r = client.get("/api/leaderboard")
        scores = [e["score"] for e in r.json()]
        assert scores == sorted(scores, reverse=True)


class TestHistoryEndpoint:
    def test_history_returns_list(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
