import uuid
from datetime import datetime
from .types import MatchResult
from .agent import BaseAgent
from ..tasks.base import BaseTask


class MatchOrchestrator:
    """Runs a match between agents on a given task."""

    def run(self, task: BaseTask, agents: list[BaseAgent]) -> MatchResult:
        match_id = str(uuid.uuid4())[:8]
        print(f"\n[Match {match_id}] Starting: {task.name} | Players: {[a.name for a in agents]}")

        task.reset(agents)
        transcript = []
        round_num = 0

        while not task.is_terminal():
            round_num += 1
            round_log = {"round": round_num, "actions": []}

            for agent in agents:
                if not task.agent_can_act(agent.agent_id):
                    continue

                obs = task.observe(agent.agent_id, round_num)
                try:
                    action = agent.act(obs)
                except Exception as e:
                    print(f"  [!] Agent {agent.name} error: {e}. Forfeiting turn.")
                    action = task.forfeit_action(agent.agent_id)

                task.step(agent.agent_id, action)
                round_log["actions"].append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "action_type": action.action_type,
                    "content": action.content,
                })

            transcript.append(round_log)

        scores = task.compute_scores()
        winner = max(scores, key=scores.get) if scores else None
        outcome_metrics = task.outcome_metrics()

        print(f"[Match {match_id}] Complete. Scores: {scores} | Winner: {winner}")

        return MatchResult(
            match_id=match_id,
            task_name=task.name,
            task_category=task.category,
            agents=list(scores.keys()),
            scores=scores,
            winner=winner,
            transcript=transcript,
            outcome_metrics=outcome_metrics,
            rounds_played=round_num,
        )
