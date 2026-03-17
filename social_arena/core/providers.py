import json
from .types import AgentConfig, Observation, Action
from .agent import BaseAgent

PROVIDER_MODELS = {
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
}

class LLMAgent(BaseAgent):
    """Multi-provider LLM agent supporting Anthropic, OpenAI, and Gemini."""

    def __init__(self, config: AgentConfig, provider: str = "anthropic"):
        super().__init__(config)
        self.provider = provider
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI()
        elif self.provider == "gemini":
            import google.generativeai as genai
            import os
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            self._client = genai
        return self._client

    def act(self, obs: Observation) -> Action:
        prompt = self._build_prompt(obs)
        if self.provider == "anthropic":
            return self._anthropic_act(prompt, obs)
        elif self.provider == "openai":
            return self._openai_act(prompt, obs)
        elif self.provider == "gemini":
            return self._gemini_act(prompt, obs)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _anthropic_act(self, prompt: str, obs: Observation) -> Action:
        client = self._get_client()
        response = client.messages.create(
            model=self.config.model,
            max_tokens=1024,
            system=self.config.system_prompt or "You are a social intelligence agent. Always respond with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse(response.content[0].text, obs)

    def _openai_act(self, prompt: str, obs: Observation) -> Action:
        client = self._get_client()
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=1024,
        )
        return self._parse(response.choices[0].message.content or "", obs)

    def _gemini_act(self, prompt: str, obs: Observation) -> Action:
        genai = self._get_client()
        kwargs = {}
        if self.config.system_prompt:
            kwargs["system_instruction"] = self.config.system_prompt
        model = genai.GenerativeModel(self.config.model, **kwargs)
        response = model.generate_content(prompt)
        return self._parse(response.text, obs)

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
            "Respond ONLY with a JSON object containing:",
            '  "action_type": one of the valid actions above',
            '  "content": your action payload (string, number, or object)',
            '  "reasoning": brief explanation (optional, not shared with opponents)',
        ]
        return "\n".join(lines)

    def _parse(self, raw: str, obs: Observation) -> Action:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
            else:
                data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message", "content": raw}
        except json.JSONDecodeError:
            data = {"action_type": obs.valid_actions[0] if obs.valid_actions else "message", "content": raw}
        return Action(
            action_type=data.get("action_type", "message"),
            content=data.get("content", raw),
            reasoning_trace=data.get("reasoning"),
        )
