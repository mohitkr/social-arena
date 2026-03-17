from ..core.types import Observation, Action, TaskCategory
from ..core.agent import BaseAgent
from .base import BaseTask


PAYOFF = {
    ("cooperate", "cooperate"): (3, 3),
    ("cooperate", "defect"):    (0, 5),
    ("defect",    "cooperate"): (5, 0),
    ("defect",    "defect"):    (1, 1),
}


class PrisonersDilemmaTask(BaseTask):
    name = "Iterated Prisoner's Dilemma"
    category = TaskCategory.COOPERATION
    min_agents = 2
    max_agents = 2

    def __init__(self, rounds: int = 10):
        self.rounds = rounds

    def _reset(self):
        self.current_round = 0
        self.cumulative_scores = {aid: 0 for aid in self.agent_ids}
        self.history: list[dict] = []
        self.pending_actions: dict[str, str] = {}

    def observe(self, agent_id: str, round_number: int) -> Observation:
        opponent_id = [a for a in self.agent_ids if a != agent_id][0]
        opponent_history = [
            {"round": h["round"], "opponent_action": h["actions"].get(agent_id, "?")}
            for h in self.history
        ]
        return Observation(
            task_state={
                "round": round_number,
                "total_rounds": self.rounds,
                "your_score": self.cumulative_scores[agent_id],
                "opponent_score": self.cumulative_scores[opponent_id],
            },
            role="prisoner",
            history=opponent_history,
            private_info={},
            round_number=round_number,
            valid_actions=["cooperate", "defect"],
        )

    def step(self, agent_id: str, action: Action):
        choice = action.content if action.content in ("cooperate", "defect") else "cooperate"
        self.pending_actions[agent_id] = choice

        if len(self.pending_actions) == 2:
            a0, a1 = self.agent_ids
            c0, c1 = self.pending_actions[a0], self.pending_actions[a1]
            p0, p1 = PAYOFF[(c0, c1)]
            self.cumulative_scores[a0] += p0
            self.cumulative_scores[a1] += p1
            self.history.append({
                "round": self.current_round + 1,
                "actions": {a0: c0, a1: c1},
                "payoffs": {a0: p0, a1: p1},
            })
            self.current_round += 1
            self.pending_actions = {}
            print(
                f"  Round {self.current_round}: "
                f"{self.agents[a0].name}={c0}, {self.agents[a1].name}={c1} "
                f"| Scores: {self.cumulative_scores}"
            )

    def is_terminal(self) -> bool:
        return self.current_round >= self.rounds

    def compute_scores(self) -> dict[str, float]:
        return dict(self.cumulative_scores)

    def outcome_metrics(self) -> dict:
        cooperation_rate = {}
        for aid in self.agent_ids:
            cooperations = sum(1 for h in self.history if h["actions"].get(aid) == "cooperate")
            cooperation_rate[self.agents[aid].name] = cooperations / max(len(self.history), 1)
        return {
            "cooperation_rates": cooperation_rate,
            "total_social_welfare": sum(self.cumulative_scores.values()),
            "rounds": self.rounds,
        }
