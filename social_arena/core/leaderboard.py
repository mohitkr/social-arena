import trueskill
from dataclasses import dataclass, field
from typing import Optional
from .types import MatchResult


@dataclass
class AgentRating:
    agent_id: str
    name: str
    rating: trueskill.Rating = field(default_factory=trueskill.Rating)
    matches_played: int = 0
    wins: int = 0
    total_score: float = 0.0
    total_rounds: int = 0
    category_ratings: dict = field(default_factory=dict)

    @property
    def conservative_score(self) -> float:
        """µ - 3σ: conservative skill estimate used for ranking."""
        return self.rating.mu - 3 * self.rating.sigma

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def avg_pts_per_round(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.total_score / self.total_rounds


class Leaderboard:
    """TrueSkill-based skill rating leaderboard."""

    def __init__(self):
        self.env = trueskill.TrueSkill(draw_probability=0.05)
        self.ratings: dict[str, AgentRating] = {}

    def register_agent(self, agent_id: str, name: str):
        if agent_id not in self.ratings:
            self.ratings[agent_id] = AgentRating(
                agent_id=agent_id,
                name=name,
                rating=self.env.create_rating(),
            )

    def update(self, result: MatchResult):
        agents = result.agents
        scores = result.scores
        category = result.task_category

        # Ensure all agents are registered
        for agent_id in agents:
            if agent_id not in self.ratings:
                self.ratings[agent_id] = AgentRating(
                    agent_id=agent_id,
                    name=agent_id,
                    rating=self.env.create_rating(),
                )

        # Build rating groups sorted by score (highest score = rank 1)
        sorted_agents = sorted(agents, key=lambda a: scores.get(a, 0), reverse=True)
        rating_groups = [{a: self.ratings[a].rating} for a in sorted_agents]
        ranks = list(range(len(sorted_agents)))

        updated_groups = self.env.rate(rating_groups, ranks=ranks)

        rounds = result.rounds_played or 1
        for group, agent_id in zip(updated_groups, sorted_agents):
            ar = self.ratings[agent_id]
            ar.rating = group[agent_id]
            ar.matches_played += 1
            ar.total_score += scores.get(agent_id, 0)
            ar.total_rounds += rounds
            if agent_id == result.winner:
                ar.wins += 1

            # Per-category rating
            if category not in ar.category_ratings:
                ar.category_ratings[category] = self.env.create_rating()
            # Simplified: update category rating proportionally
            cat_groups = [{agent_id: ar.category_ratings[category]}]
            cat_rank = sorted_agents.index(agent_id)
            # Only meaningful for multi-agent; for 2-agent just use outcome
            ar.category_ratings[category] = group[agent_id]

    def get_rankings(self) -> list[AgentRating]:
        return sorted(
            self.ratings.values(),
            key=lambda r: r.conservative_score,
            reverse=True,
        )

    def display(self):
        rankings = self.get_rankings()
        print(f"\n{'='*60}")
        print(f"{'SOCIAL ARENA LEADERBOARD':^60}")
        print(f"{'='*60}")
        print(f"{'Rank':<6}{'Agent':<20}{'Score':<12}{'W/M':<10}{'Win Rate'}")
        print(f"{'-'*60}")
        for i, ar in enumerate(rankings, 1):
            print(
                f"{i:<6}{ar.name:<20}{ar.conservative_score:<12.2f}"
                f"{ar.wins}/{ar.matches_played:<8}{ar.win_rate:.1%}"
            )
        print(f"{'='*60}\n")
