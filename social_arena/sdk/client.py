import json
import os
import random
from dataclasses import asdict
from datetime import datetime

from ..core.agent import (
    PromptAgent, AgentConfig, BaseAgent,
    CooperativeMockAgent, CompetitiveMockAgent, TitForTatMockAgent, RandomMockAgent,
)
from ..core.match import MatchOrchestrator
from ..core.leaderboard import Leaderboard
from ..tasks.prisoners_dilemma import PrisonersDilemmaTask
from ..tasks.salary_negotiation import SalaryNegotiationTask
from ..tasks.policy_debate import PolicyDebateTask
from ..tasks.werewolf import WerewolfTask


TASK_REGISTRY = {
    "prisoners_dilemma": PrisonersDilemmaTask,
    "salary_negotiation": SalaryNegotiationTask,
    "policy_debate": PolicyDebateTask,
    "werewolf": WerewolfTask,
}


class SocialArenaClient:
    """
    Local SDK client for Social Arena.
    Lets builders test their agents against bots and each other.
    """

    def __init__(self, results_dir: str = "results"):
        self.leaderboard = Leaderboard()
        self.orchestrator = MatchOrchestrator()
        self.agents: dict[str, BaseAgent] = {}
        self.results_dir = results_dir
        self.match_history: list[dict] = []
        os.makedirs(results_dir, exist_ok=True)

    def register_agent(self, agent_id: str, name: str, system_prompt: str, model: str = "claude-sonnet-4-6") -> PromptAgent:
        config = AgentConfig(agent_id=agent_id, name=name, system_prompt=system_prompt, model=model)
        agent = PromptAgent(config)
        self.agents[agent_id] = agent
        self.leaderboard.register_agent(agent_id, name)
        print(f"[SDK] Registered agent: {name} ({agent_id})")
        return agent

    def register_mock_agent(self, agent_id: str, name: str, strategy: str = "cooperative") -> BaseAgent:
        """Register a rule-based mock agent — no API key required."""
        MOCK_STRATEGIES = {
            "cooperative": CooperativeMockAgent,
            "competitive": CompetitiveMockAgent,
            "tit_for_tat": TitForTatMockAgent,
            "random": RandomMockAgent,
        }
        cls = MOCK_STRATEGIES.get(strategy, CooperativeMockAgent)
        config = AgentConfig(agent_id=agent_id, name=name, system_prompt="", model="mock")
        agent = cls(config)
        self.agents[agent_id] = agent
        self.leaderboard.register_agent(agent_id, name)
        print(f"[SDK] Registered mock agent: {name} ({strategy})")
        return agent

    def run_match(self, task_name: str, agent_ids: list[str], **task_kwargs):
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}")
        agents = [self.agents[aid] for aid in agent_ids]
        task = TASK_REGISTRY[task_name](**task_kwargs)
        result = self.orchestrator.run(task, agents)
        self.leaderboard.update(result)
        self._save_result(result)
        return result

    def _save_result(self, result):
        agent_names = {aid: self.agents[aid].name for aid in result.agents if aid in self.agents}
        record = {
            "match_id": result.match_id,
            "timestamp": datetime.now().isoformat(),
            "task_name": result.task_name,
            "task_category": str(result.task_category),
            "agents": result.agents,
            "agent_names": agent_names,
            "scores": result.scores,
            "winner": result.winner,
            "winner_name": agent_names.get(result.winner, result.winner),
            "rounds_played": result.rounds_played,
            "outcome_metrics": result.outcome_metrics,
            "transcript": result.transcript,
        }
        self.match_history.append(record)
        path = os.path.join(self.results_dir, f"match_{result.match_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    def save_session(self):
        """Write a session summary (all matches + leaderboard) to results/session.json."""
        rankings = [
            {
                "rank": i + 1,
                "agent_id": ar.agent_id,
                "name": ar.name,
                "score": round(ar.conservative_score, 2),
                "wins": ar.wins,
                "matches": ar.matches_played,
                "win_rate": round(ar.win_rate, 3),
            }
            for i, ar in enumerate(self.leaderboard.get_rankings())
        ]
        session = {
            "generated_at": datetime.now().isoformat(),
            "leaderboard": rankings,
            "matches": self.match_history,
        }
        path = os.path.join(self.results_dir, "session.json")
        with open(path, "w") as f:
            json.dump(session, f, indent=2, default=str)
        print(f"[SDK] Session saved to {path}")

    def run_random_match(self, agent_ids: list[str]):
        task_name = random.choice(list(TASK_REGISTRY.keys()))
        return self.run_match(task_name, agent_ids)

    def show_leaderboard(self):
        self.leaderboard.display()

    def list_tasks(self) -> list[str]:
        return list(TASK_REGISTRY.keys())

    @staticmethod
    def create_bot(bot_id: str, strategy: str = "cooperative") -> PromptAgent:
        """Create a built-in bot agent for testing."""
        prompts = {
            "cooperative": (
                "You are a cooperative AI agent competing in social tasks. "
                "Prioritize mutual benefit, building trust, and fair outcomes. "
                "Always respond with valid JSON as instructed."
            ),
            "competitive": (
                "You are a competitive AI agent competing in social tasks. "
                "Prioritize your own payoff and winning. Be strategic and calculated. "
                "Always respond with valid JSON as instructed."
            ),
            "tit_for_tat": (
                "You are a tit-for-tat agent. Cooperate on the first move. "
                "Then mirror your opponent's previous action exactly. "
                "Always respond with valid JSON as instructed."
            ),
        }
        prompt = prompts.get(strategy, prompts["cooperative"])
        config = AgentConfig(
            agent_id=f"bot_{bot_id}_{strategy}",
            name=f"Bot-{strategy.title()}",
            system_prompt=prompt,
            model="claude-haiku-4-5-20251001",
        )
        return PromptAgent(config)
