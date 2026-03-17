import json
import random
from .types import AgentConfig, Observation, Action


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock agents — no API key required, useful for local testing
# ---------------------------------------------------------------------------

class CooperativeMockAgent(BaseAgent):
    """Always cooperates, makes fair offers, gives constructive speeches."""

    def act(self, obs: Observation) -> Action:
        action_type = obs.valid_actions[0] if obs.valid_actions else "message"

        if action_type == "cooperate":
            return Action(action_type="cooperate", content="cooperate")

        if action_type in ("make_offer", "counter_offer"):
            # Offer near market rate / middle ground
            market = obs.private_info.get("market_rate", 95_000)
            min_val = obs.private_info.get("minimum_acceptable_salary", market)
            max_val = obs.private_info.get("maximum_budget", market)
            offer = int((min_val + max_val) / 2) if max_val > min_val else market
            return Action(action_type=action_type, content=str(offer))

        if action_type == "accept":
            last = obs.task_state.get("last_offer")
            min_val = obs.private_info.get("minimum_acceptable_salary", 0)
            if last and last >= min_val:
                return Action(action_type="accept", content="accept")
            return Action(action_type="counter_offer", content=str(int(min_val * 1.05)))

        if action_type == "speech":
            topic = obs.task_state.get("topic", "this issue")
            side = "in favor of" if "PRO" in obs.role else "against"
            return Action(
                action_type="speech",
                content=(
                    f"I am {side} {topic}. Evidence shows that cooperation leads to "
                    f"better outcomes for all parties. We should consider the long-term "
                    f"benefits of working together rather than short-term individual gains."
                ),
            )

        if action_type == "vote":
            players = obs.task_state.get("alive_players", [])
            candidates = [p for p in players if p != self.name]
            target = random.choice(candidates) if candidates else (players[0] if players else "unknown")
            return Action(action_type="vote", content=target)

        if action_type == "kill":
            players = obs.task_state.get("alive_players", [])
            candidates = [p for p in players if p != self.name]
            target = random.choice(candidates) if candidates else (players[0] if players else "unknown")
            return Action(action_type="kill", content=target)

        return Action(action_type=action_type, content="cooperate")


class CompetitiveMockAgent(BaseAgent):
    """Defects, anchors hard in negotiations, votes aggressively."""

    def act(self, obs: Observation) -> Action:
        action_type = obs.valid_actions[0] if obs.valid_actions else "message"

        if action_type == "defect" or "defect" in obs.valid_actions:
            return Action(action_type="defect", content="defect")

        if action_type in ("make_offer", "counter_offer"):
            if "minimum_acceptable_salary" in obs.private_info:
                # Employee: anchor high
                market = obs.private_info.get("market_rate", 95_000)
                return Action(action_type=action_type, content=str(int(market * 1.25)))
            else:
                # Employer: anchor low
                market = obs.private_info.get("market_rate", 95_000)
                return Action(action_type=action_type, content=str(int(market * 0.8)))

        if action_type == "speech":
            topic = obs.task_state.get("topic", "this issue")
            side = "strongly in favor of" if "PRO" in obs.role else "firmly against"
            return Action(
                action_type="speech",
                content=(
                    f"I am {side} {topic}. The data is unambiguous — our position "
                    f"is the only rational choice. Any alternative approach would lead "
                    f"to demonstrably worse outcomes. We must act decisively."
                ),
            )

        if action_type in ("vote", "kill"):
            players = obs.task_state.get("alive_players", [])
            candidates = [p for p in players if p != self.name]
            target = random.choice(candidates) if candidates else (players[0] if players else "unknown")
            return Action(action_type=action_type, content=target)

        return Action(action_type=action_type, content="defect")


class TitForTatMockAgent(BaseAgent):
    """Cooperates first, then mirrors opponent's last action."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._last_opponent_action: str | None = None

    def act(self, obs: Observation) -> Action:
        action_type = obs.valid_actions[0] if obs.valid_actions else "message"

        if action_type in ("cooperate", "defect"):
            # Check opponent's last action from history
            if obs.history:
                last = obs.history[-1]
                opponent_action = last.get("opponent_action") or last.get("actions", {})
                if isinstance(opponent_action, dict):
                    other = [v for k, v in opponent_action.items() if k != self.agent_id]
                    if other:
                        self._last_opponent_action = other[0]
                elif isinstance(opponent_action, str) and opponent_action in ("cooperate", "defect"):
                    self._last_opponent_action = opponent_action

            choice = self._last_opponent_action if self._last_opponent_action else "cooperate"
            return Action(action_type=choice, content=choice)

        # Fall back to cooperative for other action types
        return CooperativeMockAgent(self.config).act(obs)


class RandomMockAgent(BaseAgent):
    """Picks a random valid action — useful as a baseline."""

    def act(self, obs: Observation) -> Action:
        action_type = random.choice(obs.valid_actions) if obs.valid_actions else "message"

        if action_type in ("cooperate", "defect"):
            return Action(action_type=action_type, content=action_type)

        if action_type in ("make_offer", "counter_offer"):
            amount = random.randint(75_000, 115_000)
            return Action(action_type=action_type, content=str(amount))

        if action_type == "speech":
            return Action(action_type="speech", content=f"I believe my position on this matter is well-founded and deserves careful consideration by all parties.")

        if action_type in ("vote", "kill"):
            players = obs.task_state.get("alive_players", [])
            candidates = [p for p in players if p != self.name]
            target = random.choice(candidates) if candidates else (players[0] if players else "unknown")
            return Action(action_type=action_type, content=target)

        return Action(action_type=action_type, content="ok")


# ---------------------------------------------------------------------------
# Prompt Agent — requires ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

class PromptAgent(BaseAgent):
    """An agent defined by a system prompt, backed by an Anthropic model."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import anthropic
        self.client = anthropic.Anthropic()

    def act(self, observation: Observation) -> Action:
        prompt = self._build_prompt(observation)
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=1024,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return self._parse_response(raw, observation)

    def _build_prompt(self, obs: Observation) -> str:
        lines = [
            f"## Round {obs.round_number}",
            f"**Your role:** {obs.role}",
            "",
            "**Current game state:**",
            json.dumps(obs.task_state, indent=2),
        ]
        if obs.private_info:
            lines += ["", "**Your private information:**", json.dumps(obs.private_info, indent=2)]
        if obs.history:
            lines += ["", "**History of actions so far:**"]
            for entry in obs.history[-10:]:
                lines.append(json.dumps(entry))
        if obs.valid_actions:
            lines += ["", f"**Valid actions:** {', '.join(obs.valid_actions)}"]
        lines += [
            "",
            "Respond with a JSON object containing:",
            '  "action_type": one of the valid actions above',
            '  "content": your action payload (string, number, or object)',
            '  "reasoning": brief explanation of your reasoning (optional, not shared with opponents)',
        ]
        return "\n".join(lines)

    def _parse_response(self, raw: str, obs: Observation) -> Action:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
            else:
                data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message",
                        "content": raw}
        except (json.JSONDecodeError, IndexError):
            data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message",
                    "content": raw}

        return Action(
            action_type=data.get("action_type", "message"),
            content=data.get("content", raw),
            reasoning_trace=data.get("reasoning"),
        )


class PromptAgent(BaseAgent):
    """An agent defined by a system prompt, backed by an Anthropic model."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic()

    def act(self, observation: Observation) -> Action:
        prompt = self._build_prompt(observation)
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=1024,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return self._parse_response(raw, observation)

    def _build_prompt(self, obs: Observation) -> str:
        lines = [
            f"## Round {obs.round_number}",
            f"**Your role:** {obs.role}",
            "",
            "**Current game state:**",
            json.dumps(obs.task_state, indent=2),
        ]
        if obs.private_info:
            lines += ["", "**Your private information:**", json.dumps(obs.private_info, indent=2)]
        if obs.history:
            lines += ["", "**History of actions so far:**"]
            for entry in obs.history[-10:]:  # last 10 for context window
                lines.append(json.dumps(entry))
        if obs.valid_actions:
            lines += ["", f"**Valid actions:** {', '.join(obs.valid_actions)}"]
        lines += [
            "",
            "Respond with a JSON object containing:",
            '  "action_type": one of the valid actions above',
            '  "content": your action payload (string, number, or object)',
            '  "reasoning": brief explanation of your reasoning (optional, not shared with opponents)',
        ]
        return "\n".join(lines)

    def _parse_response(self, raw: str, obs: Observation) -> Action:
        # Extract JSON from the response
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
            else:
                data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message",
                        "content": raw}
        except (json.JSONDecodeError, IndexError):
            data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message",
                    "content": raw}

        return Action(
            action_type=data.get("action_type", "message"),
            content=data.get("content", raw),
            reasoning_trace=data.get("reasoning"),
        )
