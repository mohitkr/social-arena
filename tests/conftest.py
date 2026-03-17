import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from social_arena.core.agent import (
    AgentConfig, CooperativeMockAgent, CompetitiveMockAgent,
    TitForTatMockAgent, RandomMockAgent,
)
from social_arena.core.types import Observation


def make_config(agent_id="test_agent", name="TestAgent"):
    return AgentConfig(agent_id=agent_id, name=name, system_prompt="", model="mock")


def make_obs(valid_actions=None, history=None, task_state=None, private_info=None, role="player", round_number=1):
    return Observation(
        task_state=task_state or {"round": round_number},
        role=role,
        history=history or [],
        private_info=private_info or {},
        round_number=round_number,
        valid_actions=valid_actions or ["cooperate", "defect"],
    )


@pytest.fixture
def coop_agent():
    return CooperativeMockAgent(make_config("coop", "Cooperative"))

@pytest.fixture
def comp_agent():
    return CompetitiveMockAgent(make_config("comp", "Competitive"))

@pytest.fixture
def tft_agent():
    return TitForTatMockAgent(make_config("tft", "TitForTat"))

@pytest.fixture
def rand_agent():
    return RandomMockAgent(make_config("rand", "Random"))

@pytest.fixture
def all_mock_agents(coop_agent, comp_agent, tft_agent, rand_agent):
    return [coop_agent, comp_agent, tft_agent, rand_agent]
