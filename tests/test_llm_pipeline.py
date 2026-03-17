"""
Tests for the LLM agent pipeline — verifying that:
1. _build_prompt() includes all required information
2. _parse() correctly extracts actions from various LLM response formats
3. Tasks correctly read action_type (not just content) so LLM agents aren't silently forced to cooperate
4. The full observe→prompt→parse→step pipeline works end-to-end with a mocked LLM
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from tests.conftest import make_config, make_obs
from social_arena.core.types import Action, Observation
from social_arena.core.agent import AgentConfig
from social_arena.core.providers import LLMAgent
from social_arena.core.match import MatchOrchestrator
from social_arena.tasks.prisoners_dilemma import PrisonersDilemmaTask
from social_arena.tasks.salary_negotiation import SalaryNegotiationTask
from social_arena.tasks.werewolf import WerewolfTask


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm_agent(name="LLM", provider="anthropic"):
    config = AgentConfig(
        agent_id=f"llm_{name.lower()}",
        name=name,
        system_prompt="You are a strategic agent.",
        model="claude-haiku-4-5-20251001",
    )
    return LLMAgent(config, provider=provider)


def make_mock_llm(response_json: dict):
    """Return an LLMAgent whose API call is mocked to return response_json."""
    agent = make_llm_agent()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(response_json))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    agent._client = mock_client
    return agent


# ── _build_prompt() ───────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_includes_round_number(self):
        agent = make_llm_agent()
        obs = make_obs(round_number=3)
        prompt = agent._build_prompt(obs)
        assert "Round 3" in prompt

    def test_includes_role(self):
        agent = make_llm_agent()
        obs = make_obs(role="prisoner")
        prompt = agent._build_prompt(obs)
        assert "prisoner" in prompt

    def test_includes_game_state(self):
        agent = make_llm_agent()
        obs = make_obs(task_state={"round": 2, "your_score": 6, "opponent_score": 10})
        prompt = agent._build_prompt(obs)
        assert "your_score" in prompt
        assert "opponent_score" in prompt

    def test_includes_private_info(self):
        agent = make_llm_agent()
        obs = make_obs(private_info={"minimum_acceptable_salary": 80000})
        prompt = agent._build_prompt(obs)
        assert "minimum_acceptable_salary" in prompt
        assert "80000" in prompt

    def test_includes_history(self):
        agent = make_llm_agent()
        history = [{"opponent_action": "defect"}, {"opponent_action": "defect"}]
        obs = make_obs(history=history)
        prompt = agent._build_prompt(obs)
        assert "defect" in prompt
        assert "previous rounds" in prompt.lower() or "history" in prompt.lower()

    def test_no_history_shows_first_move_message(self):
        agent = make_llm_agent()
        obs = make_obs(history=[])
        prompt = agent._build_prompt(obs)
        assert "first move" in prompt.lower() or "no previous" in prompt.lower()

    def test_includes_valid_actions(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        prompt = agent._build_prompt(obs)
        assert "cooperate" in prompt
        assert "defect" in prompt

    def test_pd_prompt_has_strategy_hint(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        prompt = agent._build_prompt(obs)
        assert "defect" in prompt.lower()
        # Should mention strategic consideration
        assert "strategically" in prompt.lower() or "opponent" in prompt.lower()

    def test_pd_prompt_shows_json_example_with_defect(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        prompt = agent._build_prompt(obs)
        # Example JSON should demonstrate exact format
        assert '"action_type": "defect"' in prompt or '"action_type":"defect"' in prompt

    def test_salary_prompt_shows_offer_example(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["make_offer", "accept", "counter_offer"])
        prompt = agent._build_prompt(obs)
        assert "make_offer" in prompt
        assert "accept" in prompt


# ── _parse() ─────────────────────────────────────────────────────────────────

class TestParse:
    def test_parse_clean_defect(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        raw = '{"action_type": "defect", "content": "defect", "reasoning": "punishing"}'
        action = agent._parse(raw, obs)
        assert action.action_type == "defect"
        assert action.content == "defect"

    def test_parse_action_type_with_reasoning_content(self):
        """LLM puts reasoning in content — action_type must still be extracted correctly."""
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        raw = '{"action_type": "defect", "content": "I will defect because the opponent always defects", "reasoning": "tit-for-tat"}'
        action = agent._parse(raw, obs)
        assert action.action_type == "defect", (
            "_parse() must return action_type='defect' even when content is a sentence"
        )

    def test_parse_cooperate(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        raw = '{"action_type": "cooperate", "content": "cooperate"}'
        action = agent._parse(raw, obs)
        assert action.action_type == "cooperate"

    def test_parse_offer_with_number_content(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["make_offer"])
        raw = '{"action_type": "make_offer", "content": "95000", "reasoning": "fair market rate"}'
        action = agent._parse(raw, obs)
        assert action.action_type == "make_offer"
        assert action.content == "95000"

    def test_parse_malformed_falls_back_to_first_valid_action(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        action = agent._parse("this is not json at all", obs)
        assert action.action_type in ("cooperate", "defect")

    def test_parse_extracts_json_embedded_in_text(self):
        """LLM sometimes wraps JSON in markdown — must still extract it."""
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        raw = 'Sure! Here is my response:\n```json\n{"action_type": "defect", "content": "defect"}\n```'
        action = agent._parse(raw, obs)
        assert action.action_type == "defect"

    def test_parse_preserves_reasoning(self):
        agent = make_llm_agent()
        obs = make_obs(valid_actions=["cooperate", "defect"])
        raw = '{"action_type": "defect", "content": "defect", "reasoning": "opponent always defects"}'
        action = agent._parse(raw, obs)
        assert action.reasoning_trace == "opponent always defects"


# ── Task action_type handling ─────────────────────────────────────────────────

class TestTaskActionTypeParsing:
    """Verify that tasks use action_type (not just content) so LLM agents work correctly."""

    def test_pd_uses_action_type_to_defect(self):
        """Core regression: action_type='defect' with non-bare content must result in defect."""
        from social_arena.core.agent import CooperativeMockAgent
        from tests.conftest import make_config as mc

        # Create a minimal task and manually step through it
        a1_cfg = mc("a1", "Agent1")
        a2_cfg = mc("a2", "Agent2")
        a1 = CooperativeMockAgent(a1_cfg)
        a2 = CooperativeMockAgent(a2_cfg)

        task = PrisonersDilemmaTask(rounds=1)
        task.reset([a1, a2])

        # Simulate what an LLM produces: action_type=defect but content=sentence
        llm_action = Action(
            action_type="defect",
            content="I choose to defect because the opponent has been exploiting me",
        )
        # a1 acts via LLM-style action
        task.step(a1.agent_id, llm_action)
        # a2 cooperates normally
        task.step(a2.agent_id, Action(action_type="cooperate", content="cooperate"))

        # If action_type was correctly used, a1 should have defected → score 5
        assert task.cumulative_scores[a1.agent_id] == 5, (
            "LLM action with action_type='defect' and sentence content must be read as defect, not cooperate"
        )
        assert task.cumulative_scores[a2.agent_id] == 0

    def test_pd_falls_back_to_content_if_action_type_invalid(self):
        """If action_type is garbage but content is valid, use content."""
        from social_arena.core.agent import CooperativeMockAgent
        from tests.conftest import make_config as mc

        a1 = CooperativeMockAgent(mc("a1", "A1"))
        a2 = CooperativeMockAgent(mc("a2", "A2"))
        task = PrisonersDilemmaTask(rounds=1)
        task.reset([a1, a2])

        task.step(a1.agent_id, Action(action_type="message", content="cooperate"))
        task.step(a2.agent_id, Action(action_type="cooperate", content="cooperate"))
        assert task.cumulative_scores[a1.agent_id] == 3

    def test_pd_defaults_to_cooperate_on_garbage(self):
        """Completely invalid action defaults to cooperate without crashing."""
        from social_arena.core.agent import CooperativeMockAgent
        from tests.conftest import make_config as mc

        a1 = CooperativeMockAgent(mc("a1", "A1"))
        a2 = CooperativeMockAgent(mc("a2", "A2"))
        task = PrisonersDilemmaTask(rounds=1)
        task.reset([a1, a2])

        task.step(a1.agent_id, Action(action_type="???", content="???"))
        task.step(a2.agent_id, Action(action_type="cooperate", content="cooperate"))
        assert task.cumulative_scores[a1.agent_id] == 3  # defaulted to cooperate


# ── Full end-to-end pipeline with mocked LLM ─────────────────────────────────

class TestLLMEndToEnd:
    """Run full matches with mocked LLM responses to verify the pipeline
    without spending API credits."""

    def _make_defecting_llm(self):
        """LLM that always returns defect (with reasoning text in content)."""
        agent = make_llm_agent("Defector")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "action_type": "defect",
            "content": "I defect because the opponent has shown they will exploit cooperation.",
            "reasoning": "competitive strategy",
        }))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client
        return agent

    def _make_cooperating_llm(self):
        agent = make_llm_agent("Cooperator")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "action_type": "cooperate",
            "content": "cooperate",
            "reasoning": "building trust",
        }))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client
        return agent

    def test_defecting_llm_beats_cooperating_llm(self):
        defector = self._make_defecting_llm()
        cooperator = self._make_cooperating_llm()
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [defector, cooperator])
        assert result.scores[defector.agent_id] > result.scores[cooperator.agent_id], (
            "Mocked LLM defector must score higher than mocked LLM cooperator — "
            "if equal, the task is ignoring action_type and defaulting both to cooperate"
        )
        assert result.winner == defector.agent_id

    def test_defecting_llm_gets_correct_payoff(self):
        """5 rounds of defect-vs-cooperate should give defector 5*5=25 points."""
        defector = self._make_defecting_llm()
        cooperator = self._make_cooperating_llm()
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [defector, cooperator])
        assert result.scores[defector.agent_id] == 25
        assert result.scores[cooperator.agent_id] == 0

    def test_llm_receives_history_in_prompt(self):
        """Verify the prompt passed to the LLM actually contains opponent history."""
        agent = make_mock_llm({"action_type": "defect", "content": "defect"})
        from social_arena.core.agent import CompetitiveMockAgent
        from tests.conftest import make_config as mc
        comp = CompetitiveMockAgent(mc("comp", "Comp"))

        task = PrisonersDilemmaTask(rounds=3)
        MatchOrchestrator().run(task, [agent, comp])

        # Check that the LLM was called with a prompt containing history after round 1
        calls = agent._client.messages.create.call_args_list
        assert len(calls) == 3, "LLM should be called once per round"

        # The second call's prompt should reference opponent's defect from round 1
        round2_prompt = calls[1].kwargs.get("messages", calls[1].args[0] if calls[1].args else [{}])
        if isinstance(round2_prompt, list):
            prompt_text = round2_prompt[-1].get("content", "")
        else:
            prompt_text = str(round2_prompt)
        assert "defect" in prompt_text, (
            "Round 2 prompt must include opponent's round 1 action in history"
        )

    def test_llm_cooperation_rate_matches_mocked_response(self):
        """An LLM mocked to always defect must show 0% cooperation rate."""
        defector = self._make_defecting_llm()
        from social_arena.core.agent import CooperativeMockAgent
        from tests.conftest import make_config as mc
        coop = CooperativeMockAgent(mc("coop", "Coop"))

        task = PrisonersDilemmaTask(rounds=6)
        result = MatchOrchestrator().run(task, [defector, coop])
        rates = result.outcome_metrics["cooperation_rates"]
        assert rates[defector.name] == 0.0, (
            "Mocked LLM defector must have 0% cooperation rate — "
            "non-zero means the task is ignoring its action_type"
        )

    def test_salary_llm_offer_is_used(self):
        """Salary negotiation must parse the offer amount from LLM content correctly."""
        emp_llm = make_mock_llm({"action_type": "make_offer", "content": "95000", "reasoning": "market rate"})
        er_llm = make_mock_llm({"action_type": "accept", "content": "accept", "reasoning": "within budget"})

        task = SalaryNegotiationTask(employee_min=80000, employer_max=110000, market_rate=95000, max_rounds=4)
        result = MatchOrchestrator().run(task, [emp_llm, er_llm])
        # Employer accepted → deal reached
        assert result.outcome_metrics.get("deal_reached") is True
        assert result.outcome_metrics.get("final_salary") == 95000
