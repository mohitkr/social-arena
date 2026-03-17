"""Tests for all four task implementations."""
import pytest
from tests.conftest import make_config
from social_arena.core.agent import CooperativeMockAgent, CompetitiveMockAgent, TitForTatMockAgent, RandomMockAgent
from social_arena.core.match import MatchOrchestrator
from social_arena.tasks.prisoners_dilemma import PrisonersDilemmaTask, PAYOFF
from social_arena.tasks.salary_negotiation import SalaryNegotiationTask
from social_arena.tasks.policy_debate import PolicyDebateTask
from social_arena.tasks.werewolf import WerewolfTask


def make_agent(cls, suffix=""):
    config = make_config(f"{cls.__name__}{suffix}", cls.__name__)
    return cls(config)


# ── Prisoner's Dilemma ────────────────────────────────────────────────────────

class TestPrisonersDilemmaPayoffs:
    def test_mutual_cooperation_payoff(self):
        assert PAYOFF[("cooperate", "cooperate")] == (3, 3)

    def test_mutual_defection_payoff(self):
        assert PAYOFF[("defect", "defect")] == (1, 1)

    def test_defect_vs_cooperate_payoff(self):
        assert PAYOFF[("defect", "cooperate")] == (5, 0)

    def test_cooperate_vs_defect_payoff(self):
        assert PAYOFF[("cooperate", "defect")] == (0, 5)

    def test_all_payoffs_non_negative(self):
        for (a, b), (pa, pb) in PAYOFF.items():
            assert pa >= 0 and pb >= 0


class TestPrisonersDilemmaGame:
    def test_coop_vs_coop_both_score_equally(self):
        a1, a2 = make_agent(CooperativeMockAgent, "1"), make_agent(CooperativeMockAgent, "2")
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert result.scores[a1.agent_id] == result.scores[a2.agent_id]
        assert result.scores[a1.agent_id] == 15  # 3 * 5 rounds

    def test_defector_beats_cooperator(self):
        coop = make_agent(CooperativeMockAgent)
        comp = make_agent(CompetitiveMockAgent)
        task = PrisonersDilemmaTask(rounds=10)
        result = MatchOrchestrator().run(task, [coop, comp])
        assert result.scores[comp.agent_id] > result.scores[coop.agent_id]
        assert result.winner == comp.agent_id

    def test_mutual_defection_low_total_welfare(self):
        a1, a2 = make_agent(CompetitiveMockAgent, "1"), make_agent(CompetitiveMockAgent, "2")
        task = PrisonersDilemmaTask(rounds=10)
        result = MatchOrchestrator().run(task, [a1, a2])
        total = sum(result.scores.values())
        # mutual defect = 1+1=2 per round, 10 rounds = 20
        assert total == 20

    def test_correct_number_of_rounds(self):
        a1, a2 = make_agent(CooperativeMockAgent, "1"), make_agent(CooperativeMockAgent, "2")
        task = PrisonersDilemmaTask(rounds=7)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert result.rounds_played == 7

    def test_outcome_metrics_cooperation_rate(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [a1, a2])
        rates = result.outcome_metrics["cooperation_rates"]
        assert rates[a1.name] == 1.0   # always cooperates
        assert rates[a2.name] == 0.0   # always defects


# ── Salary Negotiation ────────────────────────────────────────────────────────

class TestSalaryNegotiation:
    def test_result_has_two_agents(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = SalaryNegotiationTask(employee_min=80000, employer_max=110000, market_rate=95000, max_rounds=4)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert len(result.scores) == 2

    def test_scores_non_negative(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(RandomMockAgent)
        task = SalaryNegotiationTask(max_rounds=4)
        result = MatchOrchestrator().run(task, [a1, a2])
        for score in result.scores.values():
            assert score >= 0

    def test_outcome_metrics_present(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CooperativeMockAgent, "2")
        task = SalaryNegotiationTask(max_rounds=4)
        result = MatchOrchestrator().run(task, [a1, a2])
        m = result.outcome_metrics
        assert "deal_reached" in m
        assert "employee_batna" in m
        assert "employer_budget" in m

    def test_task_terminates(self):
        """Task must always terminate within max_rounds."""
        a1 = make_agent(RandomMockAgent)
        a2 = make_agent(RandomMockAgent, "2")
        task = SalaryNegotiationTask(max_rounds=6)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert result is not None


# ── Policy Debate ─────────────────────────────────────────────────────────────

class TestPolicyDebate:
    def test_two_agents_produce_result(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = PolicyDebateTask(topic="AI should be regulated", rounds=1)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert len(result.scores) == 2

    def test_scores_sum_to_100(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = PolicyDebateTask(topic="AI should be regulated", rounds=1)
        result = MatchOrchestrator().run(task, [a1, a2])
        total = sum(result.scores.values())
        assert abs(total - 100.0) < 1.0, f"Scores should sum to ~100, got {total}"

    def test_has_winner(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = PolicyDebateTask(topic="AI should be regulated", rounds=1)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert result.winner in result.scores

    def test_custom_topic_used(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        topic = "Pineapple belongs on pizza"
        task = PolicyDebateTask(topic=topic, rounds=1)
        result = MatchOrchestrator().run(task, [a1, a2])
        assert result.outcome_metrics["topic"] == topic

    def test_speech_count_matches_rounds(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        rounds = 2
        task = PolicyDebateTask(topic="Test", rounds=rounds)
        result = MatchOrchestrator().run(task, [a1, a2])
        # Each round: pro_opens, con_responds, pro_rebuts, con_rebuts = 4 speeches per round
        assert result.outcome_metrics["speeches_delivered"] == rounds * 4


# ── Werewolf ──────────────────────────────────────────────────────────────────

class TestWerewolf:
    def _make_agents(self, n):
        classes = [CooperativeMockAgent, CompetitiveMockAgent, TitForTatMockAgent, RandomMockAgent]
        return [make_agent(classes[i % len(classes)], str(i)) for i in range(n)]

    def test_game_terminates(self):
        agents = self._make_agents(4)
        task = WerewolfTask(num_werewolves=1)
        result = MatchOrchestrator().run(task, agents)
        assert result is not None

    def test_result_has_all_agents(self):
        agents = self._make_agents(4)
        task = WerewolfTask(num_werewolves=1)
        result = MatchOrchestrator().run(task, agents)
        assert len(result.scores) == 4

    def test_winner_is_team_based(self):
        agents = self._make_agents(4)
        task = WerewolfTask(num_werewolves=1)
        result = MatchOrchestrator().run(task, agents)
        metrics = result.outcome_metrics
        assert "villagers_win" in metrics
        assert isinstance(metrics["villagers_win"], bool)

    def test_scores_non_negative(self):
        agents = self._make_agents(4)
        task = WerewolfTask(num_werewolves=1)
        result = MatchOrchestrator().run(task, agents)
        for score in result.scores.values():
            assert score >= 0

    def test_six_player_game(self):
        agents = self._make_agents(6)
        task = WerewolfTask(num_werewolves=2)
        result = MatchOrchestrator().run(task, agents)
        assert len(result.scores) == 6
