"""Tests for MatchOrchestrator and Leaderboard."""
import pytest
from tests.conftest import make_config
from social_arena.core.agent import CooperativeMockAgent, CompetitiveMockAgent
from social_arena.core.match import MatchOrchestrator
from social_arena.core.leaderboard import Leaderboard
from social_arena.tasks.prisoners_dilemma import PrisonersDilemmaTask


def make_pair():
    coop = CooperativeMockAgent(make_config("coop", "Cooperative"))
    comp = CompetitiveMockAgent(make_config("comp", "Competitive"))
    return coop, comp


class TestMatchOrchestrator:
    def test_returns_match_result(self):
        from social_arena.core.types import MatchResult
        coop, comp = make_pair()
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=3), [coop, comp])
        assert isinstance(result, MatchResult)

    def test_match_id_is_set(self):
        coop, comp = make_pair()
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=3), [coop, comp])
        assert result.match_id
        assert len(result.match_id) > 0

    def test_transcript_has_correct_rounds(self):
        coop, comp = make_pair()
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
        assert result.rounds_played == 5
        assert len(result.transcript) == 5

    def test_winner_has_highest_score(self):
        coop, comp = make_pair()
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
        winner_score = result.scores[result.winner]
        for score in result.scores.values():
            assert winner_score >= score

    def test_task_name_in_result(self):
        coop, comp = make_pair()
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=3), [coop, comp])
        assert "Prisoner" in result.task_name


class TestLeaderboard:
    def test_register_agent(self):
        lb = Leaderboard()
        lb.register_agent("a1", "Agent1")
        assert "a1" in lb.ratings

    def test_initial_rating_default(self):
        lb = Leaderboard()
        lb.register_agent("a1", "Agent1")
        ar = lb.ratings["a1"]
        assert ar.matches_played == 0
        assert ar.wins == 0

    def test_update_after_match(self):
        coop = CooperativeMockAgent(make_config("coop", "Cooperative"))
        comp = CompetitiveMockAgent(make_config("comp", "Competitive"))
        lb = Leaderboard()
        lb.register_agent("coop", "Cooperative")
        lb.register_agent("comp", "Competitive")
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
        lb.update(result)
        assert lb.ratings["coop"].matches_played == 1
        assert lb.ratings["comp"].matches_played == 1

    def test_winner_gets_win_credit(self):
        coop = CooperativeMockAgent(make_config("coop", "Cooperative"))
        comp = CompetitiveMockAgent(make_config("comp", "Competitive"))
        lb = Leaderboard()
        lb.register_agent("coop", "Cooperative")
        lb.register_agent("comp", "Competitive")
        result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
        lb.update(result)
        # Competitive should win (defects), cooperative should not
        assert lb.ratings["comp"].wins == 1
        assert lb.ratings["coop"].wins == 0

    def test_rankings_sorted_by_score(self):
        coop = CooperativeMockAgent(make_config("coop", "Cooperative"))
        comp = CompetitiveMockAgent(make_config("comp", "Competitive"))
        lb = Leaderboard()
        lb.register_agent("coop", "Cooperative")
        lb.register_agent("comp", "Competitive")
        # Run multiple matches so scores diverge
        for _ in range(3):
            result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
            lb.update(result)
        rankings = lb.get_rankings()
        scores = [r.conservative_score for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_win_rate_calculation(self):
        coop = CooperativeMockAgent(make_config("coop", "Cooperative"))
        comp = CompetitiveMockAgent(make_config("comp", "Competitive"))
        lb = Leaderboard()
        lb.register_agent("coop", "Cooperative")
        lb.register_agent("comp", "Competitive")
        for _ in range(4):
            result = MatchOrchestrator().run(PrisonersDilemmaTask(rounds=5), [coop, comp])
            lb.update(result)
        ar_comp = lb.ratings["comp"]
        assert ar_comp.win_rate == ar_comp.wins / ar_comp.matches_played
