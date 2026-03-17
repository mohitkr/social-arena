"""Tests for all mock agent strategies."""
import pytest
from tests.conftest import make_config, make_obs
from social_arena.core.agent import (
    CooperativeMockAgent, CompetitiveMockAgent,
    TitForTatMockAgent, RandomMockAgent,
)


# ── CooperativeMockAgent ──────────────────────────────────────────────────────

class TestCooperativeAgent:
    def test_cooperates_in_prisoners_dilemma(self, coop_agent):
        obs = make_obs(valid_actions=["cooperate", "defect"])
        action = coop_agent.act(obs)
        assert action.action_type == "cooperate"

    def test_cooperates_consistently_over_many_rounds(self, coop_agent):
        for _ in range(20):
            obs = make_obs(valid_actions=["cooperate", "defect"])
            assert coop_agent.act(obs).action_type == "cooperate"

    def test_fair_salary_offer_midpoint(self, coop_agent):
        obs = make_obs(
            valid_actions=["make_offer"],
            private_info={"minimum_acceptable_salary": 80000, "maximum_budget": 110000, "market_rate": 95000},
        )
        action = coop_agent.act(obs)
        assert action.action_type == "make_offer"
        amount = int(action.content)
        assert 80000 <= amount <= 110000

    def test_accepts_above_minimum(self, coop_agent):
        obs = make_obs(
            valid_actions=["accept"],
            task_state={"round": 1, "last_offer": 90000},
            private_info={"minimum_acceptable_salary": 80000, "market_rate": 95000},
        )
        action = coop_agent.act(obs)
        assert action.action_type == "accept"

    def test_gives_speech(self, coop_agent):
        obs = make_obs(
            valid_actions=["speech"],
            task_state={"topic": "Climate Change", "round": 1, "total_rounds": 2, "phase": "pro_opens"},
            role="PRO (arguing FOR)",
        )
        action = coop_agent.act(obs)
        assert action.action_type == "speech"
        assert isinstance(action.content, str)
        assert len(action.content) > 10

    def test_votes_for_valid_player(self, coop_agent):
        obs = make_obs(
            valid_actions=["vote"],
            task_state={"phase": "day_vote", "day": 1, "alive_players": ["Alice", "Bob", "Charlie"], "eliminated": [], "player_count": 3},
            role="villager",
        )
        coop_agent.name = "TestAgent"
        action = coop_agent.act(obs)
        assert action.action_type == "vote"
        assert action.content in ["Alice", "Bob", "Charlie"]


# ── CompetitiveMockAgent ──────────────────────────────────────────────────────

class TestCompetitiveAgent:
    def test_defects_in_prisoners_dilemma(self, comp_agent):
        obs = make_obs(valid_actions=["cooperate", "defect"])
        action = comp_agent.act(obs)
        assert action.action_type == "defect"

    def test_defects_consistently(self, comp_agent):
        for _ in range(20):
            obs = make_obs(valid_actions=["cooperate", "defect"])
            assert comp_agent.act(obs).action_type == "defect"

    def test_employee_anchors_high(self, comp_agent):
        obs = make_obs(
            valid_actions=["make_offer"],
            private_info={"minimum_acceptable_salary": 80000, "market_rate": 95000},
        )
        action = comp_agent.act(obs)
        amount = int(action.content)
        # Competitive employee anchors above market rate
        assert amount >= 95000

    def test_employer_anchors_low(self, comp_agent):
        obs = make_obs(
            valid_actions=["make_offer"],
            private_info={"maximum_budget": 110000, "market_rate": 95000},
        )
        action = comp_agent.act(obs)
        amount = int(action.content)
        # Competitive employer anchors below market rate
        assert amount <= 95000

    def test_gives_strong_speech(self, comp_agent):
        obs = make_obs(
            valid_actions=["speech"],
            task_state={"topic": "UBI", "round": 1, "total_rounds": 2, "phase": "con_responds"},
            role="CON (arguing AGAINST)",
        )
        action = comp_agent.act(obs)
        assert action.action_type == "speech"
        assert len(action.content) > 10


# ── TitForTatMockAgent ────────────────────────────────────────────────────────

class TestTitForTatAgent:
    def test_cooperates_on_first_move(self, tft_agent):
        obs = make_obs(valid_actions=["cooperate", "defect"], history=[])
        action = tft_agent.act(obs)
        assert action.action_type == "cooperate"

    def test_mirrors_defect(self, tft_agent):
        obs = make_obs(
            valid_actions=["cooperate", "defect"],
            history=[{"opponent_action": "defect"}],
        )
        action = tft_agent.act(obs)
        assert action.action_type == "defect"

    def test_mirrors_cooperate(self, tft_agent):
        obs = make_obs(
            valid_actions=["cooperate", "defect"],
            history=[{"opponent_action": "cooperate"}],
        )
        action = tft_agent.act(obs)
        assert action.action_type == "cooperate"

    def test_mirrors_most_recent_action(self, tft_agent):
        history = [
            {"opponent_action": "cooperate"},
            {"opponent_action": "cooperate"},
            {"opponent_action": "defect"},  # most recent
        ]
        obs = make_obs(valid_actions=["cooperate", "defect"], history=history)
        action = tft_agent.act(obs)
        assert action.action_type == "defect"

    def test_forgives_after_opponent_cooperates_again(self, tft_agent):
        history = [
            {"opponent_action": "defect"},
            {"opponent_action": "cooperate"},  # opponent returned to cooperating
        ]
        obs = make_obs(valid_actions=["cooperate", "defect"], history=history)
        action = tft_agent.act(obs)
        assert action.action_type == "cooperate"


# ── RandomMockAgent ───────────────────────────────────────────────────────────

class TestRandomAgent:
    def test_always_returns_valid_action(self, rand_agent):
        for _ in range(50):
            valid = ["cooperate", "defect"]
            obs = make_obs(valid_actions=valid)
            action = rand_agent.act(obs)
            assert action.action_type in valid

    def test_produces_both_choices_over_many_rounds(self, rand_agent):
        choices = set()
        for _ in range(100):
            obs = make_obs(valid_actions=["cooperate", "defect"])
            choices.add(rand_agent.act(obs).action_type)
        assert choices == {"cooperate", "defect"}, "Random agent should produce both actions"

    def test_returns_valid_offer_amount(self, rand_agent):
        obs = make_obs(valid_actions=["make_offer"])
        action = rand_agent.act(obs)
        assert action.action_type == "make_offer"
        amount = int(action.content)
        assert 75000 <= amount <= 115000

    def test_vote_targets_valid_player(self, rand_agent):
        players = ["Alice", "Bob", "Charlie"]
        rand_agent.name = "TestAgent"
        obs = make_obs(
            valid_actions=["vote"],
            task_state={"phase": "day_vote", "day": 1, "alive_players": players, "eliminated": [], "player_count": 3},
        )
        action = rand_agent.act(obs)
        assert action.action_type == "vote"
        assert action.content in players
