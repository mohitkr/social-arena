from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class TaskCategory(str, Enum):
    COOPERATION = "cooperation"
    NEGOTIATION = "negotiation"
    PERSUASION = "persuasion"
    COORDINATION = "coordination"
    SOCIAL_DEDUCTION = "social_deduction"
    ETHICS = "ethics"


@dataclass
class Observation:
    task_state: dict[str, Any]
    role: str
    history: list[dict]
    private_info: dict[str, Any]
    round_number: int
    valid_actions: list[str] = field(default_factory=list)


@dataclass
class Action:
    action_type: str
    content: Any
    reasoning_trace: Optional[str] = None


@dataclass
class AgentConfig:
    agent_id: str
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-6"


@dataclass
class MatchResult:
    match_id: str
    task_name: str
    task_category: str
    agents: list[str]
    scores: dict[str, float]
    winner: Optional[str]
    transcript: list[dict]
    outcome_metrics: dict[str, Any]
    rounds_played: int
