from abc import ABC, abstractmethod
from ..core.types import Observation, Action, TaskCategory
from ..core.agent import BaseAgent


class BaseTask(ABC):
    name: str
    category: str
    min_agents: int = 2
    max_agents: int = 2

    def reset(self, agents: list[BaseAgent]):
        self.agents = {a.agent_id: a for a in agents}
        self.agent_ids = [a.agent_id for a in agents]
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def observe(self, agent_id: str, round_number: int) -> Observation:
        pass

    @abstractmethod
    def step(self, agent_id: str, action: Action):
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def compute_scores(self) -> dict[str, float]:
        pass

    def outcome_metrics(self) -> dict:
        return {}

    def agent_can_act(self, agent_id: str) -> bool:
        return True

    def forfeit_action(self, agent_id: str) -> Action:
        return Action(action_type="forfeit", content=None)
