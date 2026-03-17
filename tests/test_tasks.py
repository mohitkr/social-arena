"""Tests for all four task implementations."""
import pytest
from tests.conftest import make_config
from social_arena.core.agent import CooperativeMockAgent, CompetitiveMockAgent, TitForTatMockAgent, RandomMockAgent
from social_arena.core.types import Action
from social_arena.core.match import MatchOrchestrator
from social_arena.tasks.prisoners_dilemma import PrisonersDilemmaTask, PAYOFF
from social_arena.tasks.salary_negotiation import SalaryNegotiationTask
from social_arena.tasks.policy_debate import PolicyDebateTask
from social_arena.tasks.werewolf import WerewolfTask


def make_agent(cls, suffix=""):
    config = make_config(f"{cls.__name__}{suffix}", cls.__name__)
    return cls(config)


# ── Prisoner's Dilemma Payoffs ────────────────────────────────────────────────

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


# ── Prisoner's Dilemma observe() pipeline ────────────────────────────────────

class TestPrisonersDilemmaObserve:
    """Directly test the task's observe() method to verify it returns the
    *opponent's* history — the exact class of bug that was previously undetected."""

    def _setup_task(self, a1, a2):
        task = PrisonersDilemmaTask(rounds=5)
        task.reset([a1, a2])
        return task

    def test_observe_returns_opponent_action_not_own(self):
        """After A cooperates and B defects, A must see opponent_action='defect'."""
        a1 = make_agent(CooperativeMockAgent, "1")  # will cooperate
        a2 = make_agent(CompetitiveMockAgent, "2")  # will defect
        task = self._setup_task(a1, a2)

        # Simulate one round manually
        task.step(a1.agent_id, Action(action_type="cooperate", content="cooperate"))
        task.step(a2.agent_id, Action(action_type="defect", content="defect"))

        # a1 (cooperator) observes — should see opponent (a2) defected
        obs_a1 = task.observe(a1.agent_id, 2)
        assert len(obs_a1.history) == 1
        assert obs_a1.history[0]["opponent_action"] == "defect", (
            "a1's observation must record opponent a2's action ('defect'), not a1's own ('cooperate')"
        )

        # a2 (defector) observes — should see opponent (a1) cooperated
        obs_a2 = task.observe(a2.agent_id, 2)
        assert len(obs_a2.history) == 1
        assert obs_a2.history[0]["opponent_action"] == "cooperate", (
            "a2's observation must record opponent a1's action ('cooperate'), not a2's own ('defect')"
        )

    def test_observe_history_grows_each_round(self):
        a1 = make_agent(CooperativeMockAgent, "1")
        a2 = make_agent(CooperativeMockAgent, "2")
        task = self._setup_task(a1, a2)

        for i in range(3):
            task.step(a1.agent_id, Action(action_type="cooperate", content="cooperate"))
            task.step(a2.agent_id, Action(action_type="cooperate", content="cooperate"))
            obs = task.observe(a1.agent_id, i + 2)
            assert len(obs.history) == i + 1

    def test_observe_history_shows_correct_sequence(self):
        """Verify history reflects opponent's actions in the correct order."""
        a1 = make_agent(TitForTatMockAgent, "1")
        a2 = make_agent(CompetitiveMockAgent, "2")
        task = self._setup_task(a1, a2)

        # Round 1: a1=cooperate, a2=defect
        task.step(a1.agent_id, Action(action_type="cooperate", content="cooperate"))
        task.step(a2.agent_id, Action(action_type="defect", content="defect"))

        # Round 2: a1=defect (mirroring), a2=defect
        task.step(a1.agent_id, Action(action_type="defect", content="defect"))
        task.step(a2.agent_id, Action(action_type="defect", content="defect"))

        obs_a1 = task.observe(a1.agent_id, 3)
        assert obs_a1.history[0]["opponent_action"] == "defect"  # round 1: a2 defected
        assert obs_a1.history[1]["opponent_action"] == "defect"  # round 2: a2 defected

        # From a2's perspective: a1 cooperated round 1, defected round 2
        obs_a2 = task.observe(a2.agent_id, 3)
        assert obs_a2.history[0]["opponent_action"] == "cooperate"
        assert obs_a2.history[1]["opponent_action"] == "defect"

    def test_tit_for_tat_reads_correct_observation(self):
        """TFT's act() must mirror opponent correctly when fed task's own observe()."""
        tft = make_agent(TitForTatMockAgent, "1")
        comp = make_agent(CompetitiveMockAgent, "2")
        task = self._setup_task(tft, comp)

        # Round 1: tft=cooperate, comp=defect
        task.step(tft.agent_id, Action(action_type="cooperate", content="cooperate"))
        task.step(comp.agent_id, Action(action_type="defect", content="defect"))

        # Feed task's observe() output into TFT — it must choose "defect"
        obs = task.observe(tft.agent_id, 2)
        action = tft.act(obs)
        assert action.action_type == "defect", (
            "TFT must defect in round 2 when opponent defected in round 1 "
            "(fails if task.observe() returned own history instead of opponent's)"
        )


# ── Prisoner's Dilemma Game Logic ─────────────────────────────────────────────

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

    def test_tit_for_tat_retaliates_against_defector(self):
        """Regression: TitForTat must read opponent's action, not its own."""
        tft = make_agent(TitForTatMockAgent)
        comp = make_agent(CompetitiveMockAgent)
        task = PrisonersDilemmaTask(rounds=6)
        result = MatchOrchestrator().run(task, [tft, comp])
        rates = result.outcome_metrics["cooperation_rates"]
        # TitForTat cooperates round 1, then defects every round after
        assert rates[tft.name] < 0.5, "TitForTat should defect most rounds against a pure defector"

    def test_outcome_metrics_cooperation_rate(self):
        a1 = make_agent(CooperativeMockAgent)
        a2 = make_agent(CompetitiveMockAgent)
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [a1, a2])
        rates = result.outcome_metrics["cooperation_rates"]
        assert rates[a1.name] == 1.0   # always cooperates
        assert rates[a2.name] == 0.0   # always defects


# ── Strategy Matchups: all 6 pairs ────────────────────────────────────────────

class TestStrategyMatchups:
    """Run every strategy pair through a full 10-round game and verify
    emergent properties.  These tests catch bugs in the task's observe()
    pipeline that unit tests on agent.act() alone cannot detect."""

    ROUNDS = 10

    def _run(self, cls_a, cls_b):
        a = make_agent(cls_a, "A")
        b = make_agent(cls_b, "B")
        task = PrisonersDilemmaTask(rounds=self.ROUNDS)
        result = MatchOrchestrator().run(task, [a, b])
        rates = result.outcome_metrics["cooperation_rates"]
        return result, rates, a, b

    # Cooperative vs Cooperative
    def test_coop_coop_mutual_cooperation(self):
        result, rates, a, b = self._run(CooperativeMockAgent, CooperativeMockAgent)
        assert rates[a.name] == 1.0
        assert rates[b.name] == 1.0
        assert result.scores[a.agent_id] == result.scores[b.agent_id] == 30  # 3*10

    # Competitive vs Competitive
    def test_comp_comp_mutual_defection(self):
        result, rates, a, b = self._run(CompetitiveMockAgent, CompetitiveMockAgent)
        assert rates[a.name] == 0.0
        assert rates[b.name] == 0.0
        assert sum(result.scores.values()) == 20  # 2*10

    # Cooperative vs Competitive
    def test_coop_comp_defector_wins(self):
        result, rates, a, b = self._run(CooperativeMockAgent, CompetitiveMockAgent)
        assert rates[a.name] == 1.0  # coop never deviates
        assert rates[b.name] == 0.0  # comp never deviates
        assert result.scores[b.agent_id] == 50  # 5*10
        assert result.scores[a.agent_id] == 0   # 0*10

    # TitForTat vs Cooperative — both cooperate from round 2 onward
    def test_tft_coop_sustained_cooperation(self):
        result, rates, a, b = self._run(TitForTatMockAgent, CooperativeMockAgent)
        # TFT cooperates round 1 (first move), mirrors coop's cooperation every round after
        assert rates[a.name] == 1.0, "TFT should cooperate every round against a cooperative agent"
        assert rates[b.name] == 1.0
        assert result.scores[a.agent_id] == result.scores[b.agent_id] == 30

    # TitForTat vs Competitive
    def test_tft_comp_retaliates_consistently(self):
        result, rates, a, b = self._run(TitForTatMockAgent, CompetitiveMockAgent)
        # TFT cooperates round 1, then defects for remaining 9 rounds
        assert rates[a.name] == pytest.approx(1 / self.ROUNDS), (
            f"TFT should cooperate exactly once (round 1) against pure defector, "
            f"got cooperation rate {rates[a.name]}"
        )
        assert rates[b.name] == 0.0

    # TitForTat vs TitForTat — mutual cooperation after round 1
    def test_tft_tft_mutual_cooperation(self):
        result, rates, a, b = self._run(TitForTatMockAgent, TitForTatMockAgent)
        # Both TFT start by cooperating; each mirrors the other's cooperation => perpetual cooperation
        assert rates[a.name] == 1.0, "TFT vs TFT should sustain full cooperation"
        assert rates[b.name] == 1.0
        assert result.scores[a.agent_id] == result.scores[b.agent_id] == 30

    # Random vs Cooperative — random cooperates ~50%, game always completes
    def test_rand_coop_game_completes(self):
        result, rates, a, b = self._run(RandomMockAgent, CooperativeMockAgent)
        assert result.rounds_played == self.ROUNDS
        assert len(result.scores) == 2

    # Random vs Competitive
    def test_rand_comp_game_completes(self):
        result, rates, a, b = self._run(RandomMockAgent, CompetitiveMockAgent)
        assert result.rounds_played == self.ROUNDS
        # Competitive always defects
        assert rates[b.name] == 0.0

    # TitForTat exact round-by-round action verification
    def test_tft_vs_comp_exact_actions_in_transcript(self):
        """Verify per-round actions in the transcript match expected TFT behavior."""
        tft = make_agent(TitForTatMockAgent, "T")
        comp = make_agent(CompetitiveMockAgent, "C")
        task = PrisonersDilemmaTask(rounds=5)
        result = MatchOrchestrator().run(task, [tft, comp])

        tft_id = tft.agent_id
        for i, round_log in enumerate(result.transcript):
            tft_action = next(
                a["action_type"] for a in round_log["actions"] if a["agent_id"] == tft_id
            )
            if i == 0:
                assert tft_action == "cooperate", "TFT must cooperate on round 1"
            else:
                assert tft_action == "defect", f"TFT must defect on round {i+1} (opponent always defects)"


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

    def test_observe_gives_correct_role_and_private_info(self):
        """Employee sees employee_min; employer sees employer_max — not swapped."""
        emp = make_agent(CooperativeMockAgent, "emp")
        er = make_agent(CompetitiveMockAgent, "er")
        task = SalaryNegotiationTask(employee_min=80000, employer_max=110000, market_rate=95000, max_rounds=4)
        task.reset([emp, er])

        obs_employee = task.observe(emp.agent_id, 1)
        obs_employer = task.observe(er.agent_id, 1)

        assert obs_employee.role == "employee"
        assert obs_employer.role == "employer"
        assert obs_employee.private_info.get("minimum_acceptable_salary") == 80000
        assert obs_employee.private_info.get("market_rate") == 95000
        assert obs_employer.private_info.get("maximum_budget") == 110000
        assert obs_employer.private_info.get("market_rate") == 95000
        # Cross-check: employee must NOT see employer's budget
        assert "maximum_budget" not in obs_employee.private_info
        assert "minimum_acceptable_salary" not in obs_employer.private_info

    def test_no_deal_both_score_zero(self):
        """When no deal is reached both agents score 0."""
        # Use two Competitive agents that anchor far apart and never accept
        a1 = make_agent(CompetitiveMockAgent)
        a2 = make_agent(CompetitiveMockAgent, "2")
        task = SalaryNegotiationTask(
            employee_min=100000, employer_max=80000,  # impossible ZOPA
            max_rounds=2
        )
        result = MatchOrchestrator().run(task, [a1, a2])
        m = result.outcome_metrics
        if not m["deal_reached"]:
            for score in result.scores.values():
                assert score == 0.0


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

    def test_observe_assigns_correct_sides(self):
        """First agent is PRO, second is CON; verify via observe()."""
        pro = make_agent(CooperativeMockAgent, "pro")
        con = make_agent(CompetitiveMockAgent, "con")
        task = PolicyDebateTask(topic="Test", rounds=1)
        task.reset([pro, con])

        obs_pro = task.observe(pro.agent_id, 1)
        obs_con = task.observe(con.agent_id, 1)
        assert "PRO" in obs_pro.role
        assert "CON" in obs_con.role

    def test_history_accumulates_speeches(self):
        """After each step, subsequent observations include the speech in history."""
        a1 = make_agent(CooperativeMockAgent, "p")
        a2 = make_agent(CompetitiveMockAgent, "c")
        task = PolicyDebateTask(topic="Test", rounds=2)
        task.reset([a1, a2])

        assert len(task.observe(a1.agent_id, 1).history) == 0
        task.step(a1.agent_id, Action(action_type="speech", content="Opening argument."))
        assert len(task.observe(a2.agent_id, 1).history) == 1


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

    def test_winning_team_all_score_higher_than_losing_team(self):
        """All members of the winning team must have equal score, higher than losers."""
        agents = self._make_agents(4)
        task = WerewolfTask(num_werewolves=1)
        result = MatchOrchestrator().run(task, agents)
        metrics = result.outcome_metrics
        scores = result.scores

        villagers_win = metrics["villagers_win"]
        werewolf_ids = [aid for aid in scores if metrics.get("roles", {}).get(aid) == "werewolf"]

        if werewolf_ids:
            werewolf_score = scores[werewolf_ids[0]]
            villager_scores = [s for aid, s in scores.items() if aid not in werewolf_ids]
            if villagers_win:
                assert all(vs >= werewolf_score for vs in villager_scores)
            else:
                assert all(vs <= werewolf_score for vs in villager_scores)
