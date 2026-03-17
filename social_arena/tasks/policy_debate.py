import os
from ..core.types import Observation, Action, TaskCategory
from .base import BaseTask


TOPICS = [
    "Universal Basic Income should be implemented nationwide",
    "Artificial intelligence development should be heavily regulated by governments",
    "Social media platforms should be treated as public utilities",
    "Nuclear energy is essential for addressing climate change",
]


class PolicyDebateTask(BaseTask):
    name = "Policy Debate"
    category = TaskCategory.PERSUASION
    min_agents = 2
    max_agents = 2

    def __init__(self, topic: str | None = None, rounds: int = 3):
        import random
        self.topic = topic or random.choice(TOPICS)
        self.rounds = rounds

    def _reset(self):
        self.pro_id = self.agent_ids[0]
        self.con_id = self.agent_ids[1]
        self.current_round = 0
        self.history: list[dict] = []
        self._phase = "pro_opens"
        self._pending: dict = {}

    def agent_can_act(self, agent_id: str) -> bool:
        if self._phase == "pro_opens":
            return agent_id == self.pro_id
        if self._phase == "con_responds":
            return agent_id == self.con_id
        if self._phase == "pro_rebuts":
            return agent_id == self.pro_id
        if self._phase == "con_rebuts":
            return agent_id == self.con_id
        return False

    def observe(self, agent_id: str, round_number: int) -> Observation:
        is_pro = agent_id == self.pro_id
        role = "PRO (arguing FOR)" if is_pro else "CON (arguing AGAINST)"
        return Observation(
            task_state={
                "topic": self.topic,
                "round": round_number,
                "total_rounds": self.rounds,
                "phase": self._phase,
            },
            role=role,
            history=self.history,
            private_info={},
            round_number=round_number,
            valid_actions=["speech"],
        )

    def step(self, agent_id: str, action: Action):
        speech = action.content if isinstance(action.content, str) else str(action.content)
        side = "PRO" if agent_id == self.pro_id else "CON"
        entry = {
            "round": self.current_round + 1,
            "phase": self._phase,
            "speaker": self.agents[agent_id].name,
            "side": side,
            "speech": speech[:500],  # truncate for display
        }
        self.history.append(entry)
        print(f"  [{side}] {self.agents[agent_id].name}: {speech[:120]}...")

        # Advance phase
        phase_order = ["pro_opens", "con_responds", "pro_rebuts", "con_rebuts"]
        idx = phase_order.index(self._phase)
        if idx < len(phase_order) - 1:
            self._phase = phase_order[idx + 1]
        else:
            self._phase = "pro_opens"
            self.current_round += 1

    def is_terminal(self) -> bool:
        return self.current_round >= self.rounds

    def compute_scores(self) -> dict[str, float]:
        """Use an LLM judge to score the debate."""
        print("  [Judge] Evaluating debate...")
        transcript_text = "\n\n".join(
            f"[{h['side']} - {h['phase']}] {h['speaker']}:\n{h['speech']}"
            for h in self.history
        )
        judge_prompt = f"""You are an impartial debate judge. Evaluate this policy debate on the topic:
"{self.topic}"

Transcript:
{transcript_text}

Score each side (PRO and CON) from 0-100 based on:
- Strength of arguments (40 pts)
- Use of evidence and reasoning (30 pts)
- Rebuttal quality (20 pts)
- Clarity and persuasiveness (10 pts)

Respond with JSON only: {{"pro_score": <number>, "con_score": <number>, "reasoning": "<brief>"}}"""

        try:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise EnvironmentError("No API key")
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            import json
            raw = response.content[0].text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            pro_score = float(data.get("pro_score", 50))
            con_score = float(data.get("con_score", 50))
            print(f"  [Judge] PRO={pro_score}, CON={con_score}. {data.get('reasoning', '')[:100]}")
        except EnvironmentError:
            pro_score, con_score = self._mock_judge()
            print(f"  [Judge] Mock scoring (no API key): PRO={pro_score}, CON={con_score}")
        except Exception as e:
            print(f"  [Judge] Scoring error: {e}. Using mock judge.")
            pro_score, con_score = self._mock_judge()

        return {self.pro_id: pro_score, self.con_id: con_score}

    def _mock_judge(self) -> tuple[float, float]:
        """Heuristic judge: scores based on speech length and keyword richness."""
        import re
        pro_speeches = [h["speech"] for h in self.history if h["side"] == "PRO"]
        con_speeches = [h["speech"] for h in self.history if h["side"] == "CON"]

        def score_speeches(speeches: list[str]) -> float:
            if not speeches:
                return 40.0
            total = " ".join(speeches)
            word_count = len(total.split())
            evidence_words = len(re.findall(r"\b(evidence|data|study|research|shows|proves|statistics|fact)\b", total, re.I))
            base = min(60, 30 + word_count * 0.05)
            bonus = min(20, evidence_words * 4)
            return round(base + bonus + (hash(total) % 10), 1)

        pro = score_speeches(pro_speeches)
        con = score_speeches(con_speeches)
        # Normalize so they don't both exceed 100
        total = pro + con
        if total > 0:
            pro = round(pro / total * 100, 1)
            con = round(100 - pro, 1)
        return pro, con

    def outcome_metrics(self) -> dict:
        return {
            "topic": self.topic,
            "rounds": self.rounds,
            "speeches_delivered": len(self.history),
        }
