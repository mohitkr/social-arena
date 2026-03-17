import random
import json
from ..core.types import Observation, Action, TaskCategory
from ..core.agent import BaseAgent
from .base import BaseTask


class WerewolfTask(BaseTask):
    name = "Werewolf"
    category = TaskCategory.SOCIAL_DEDUCTION
    min_agents = 4
    max_agents = 10

    def __init__(self, num_werewolves: int = 1):
        self.num_werewolves = num_werewolves

    def _reset(self):
        ids = list(self.agent_ids)
        random.shuffle(ids)
        werewolves = set(ids[:self.num_werewolves])
        villagers = set(ids[self.num_werewolves:])

        self.roles = {aid: ("werewolf" if aid in werewolves else "villager") for aid in ids}
        self.alive = set(ids)
        self.eliminated = []
        self.current_round = 0
        self.history: list[dict] = []
        self.phase = "day_discussion"  # day_discussion -> day_vote -> night
        self.day_speeches: dict[str, str] = {}
        self.day_votes: dict[str, str] = {}
        self.night_kills: dict[str, str] = {}
        self.scores: dict[str, float] = {aid: 0.0 for aid in ids}
        print(f"  Roles assigned: Werewolves={[self.agents[w].name for w in werewolves]}")

    def agent_can_act(self, agent_id: str) -> bool:
        if agent_id not in self.alive:
            return False
        if self.phase == "night" and self.roles[agent_id] != "werewolf":
            return False
        return True

    def observe(self, agent_id: str, round_number: int) -> Observation:
        role = self.roles[agent_id]
        private_info = {"your_role": role}
        if role == "werewolf":
            allies = [self.agents[w].name for w in self.alive if self.roles.get(w) == "werewolf" and w != agent_id]
            private_info["fellow_werewolves"] = allies

        alive_names = [self.agents[a].name for a in self.alive]
        valid = []
        if self.phase == "day_discussion":
            valid = ["speak"]
        elif self.phase == "day_vote":
            valid = ["vote"]
        elif self.phase == "night":
            valid = ["kill"]

        return Observation(
            task_state={
                "phase": self.phase,
                "day": self.current_round + 1,
                "alive_players": alive_names,
                "eliminated": [self.agents[e].name for e in self.eliminated],
                "player_count": len(self.alive),
            },
            role=role,
            history=self.history[-5:],
            private_info=private_info,
            round_number=round_number,
            valid_actions=valid,
        )

    def step(self, agent_id: str, action: Action):
        name = self.agents[agent_id].name
        if self.phase == "day_discussion":
            speech = action.content if isinstance(action.content, str) else str(action.content)
            self.day_speeches[agent_id] = speech
            print(f"  [Day {self.current_round+1}] {name}: {speech[:100]}...")

            if len(self.day_speeches) >= len(self.alive):
                self.phase = "day_vote"

        elif self.phase == "day_vote":
            target_name = action.content if isinstance(action.content, str) else str(action.content)
            # Resolve name to agent_id
            target_id = self._resolve_name(target_name)
            if target_id and target_id in self.alive:
                self.day_votes[agent_id] = target_id
            print(f"  [Vote] {name} votes to eliminate {target_name}")

            if len(self.day_votes) >= len(self.alive):
                self._resolve_day_vote()

        elif self.phase == "night":
            target_name = action.content if isinstance(action.content, str) else str(action.content)
            target_id = self._resolve_name(target_name)
            if target_id and target_id in self.alive:
                self.night_kills[agent_id] = target_id
            print(f"  [Night] Werewolf {name} targets {target_name}")

            werewolves_alive = [a for a in self.alive if self.roles.get(a) == "werewolf"]
            if len(self.night_kills) >= len(werewolves_alive):
                self._resolve_night()

    def _resolve_name(self, name_str: str) -> str | None:
        for aid, agent in self.agents.items():
            if agent.name.lower() in name_str.lower() or name_str.lower() in agent.name.lower():
                return aid
        # Fallback: pick a random alive non-self target
        alive_list = list(self.alive)
        return random.choice(alive_list) if alive_list else None

    def _resolve_day_vote(self):
        from collections import Counter
        if not self.day_votes:
            self.phase = "night"
            return
        vote_counts = Counter(self.day_votes.values())
        eliminated_id = vote_counts.most_common(1)[0][0]
        self.alive.discard(eliminated_id)
        self.eliminated.append(eliminated_id)
        eliminated_role = self.roles[eliminated_id]
        self.history.append({
            "event": "day_elimination",
            "day": self.current_round + 1,
            "player": self.agents[eliminated_id].name,
            "role_revealed": eliminated_role,
        })
        print(f"  [Day result] Village eliminates {self.agents[eliminated_id].name} (was {eliminated_role})")
        self.day_votes = {}
        self.day_speeches = {}
        self.phase = "night"

    def _resolve_night(self):
        from collections import Counter
        if self.night_kills:
            kill_counts = Counter(self.night_kills.values())
            killed_id = kill_counts.most_common(1)[0][0]
            self.alive.discard(killed_id)
            self.eliminated.append(killed_id)
            self.history.append({
                "event": "night_kill",
                "day": self.current_round + 1,
                "player": self.agents[killed_id].name,
            })
            print(f"  [Night result] {self.agents[killed_id].name} was killed in the night")
        self.night_kills = {}
        self.current_round += 1
        self.phase = "day_discussion"

    def is_terminal(self) -> bool:
        werewolves_alive = [a for a in self.alive if self.roles.get(a) == "werewolf"]
        villagers_alive = [a for a in self.alive if self.roles.get(a) == "villager"]
        if not werewolves_alive:
            print("  [End] Villagers WIN — all werewolves eliminated!")
            return True
        if len(werewolves_alive) >= len(villagers_alive):
            print("  [End] Werewolves WIN — outnumber villagers!")
            return True
        if self.current_round >= 8:
            print("  [End] Game over after 8 days — werewolves survive.")
            return True
        return False

    def compute_scores(self) -> dict[str, float]:
        werewolves_alive = [a for a in self.alive if self.roles.get(a) == "werewolf"]
        villagers_alive = [a for a in self.alive if self.roles.get(a) == "villager"]
        scores = {}

        if not werewolves_alive:
            # Villagers win
            for aid in self.agent_ids:
                scores[aid] = 100.0 if self.roles[aid] == "villager" else 0.0
        elif len(werewolves_alive) >= len(villagers_alive):
            # Werewolves win
            for aid in self.agent_ids:
                scores[aid] = 100.0 if self.roles[aid] == "werewolf" else 0.0
        else:
            for aid in self.agent_ids:
                scores[aid] = 50.0

        # Survival bonus
        for aid in self.alive:
            scores[aid] = scores.get(aid, 0) + 10.0

        return scores

    def outcome_metrics(self) -> dict:
        werewolves_alive = [a for a in self.alive if self.roles.get(a) == "werewolf"]
        return {
            "days_played": self.current_round,
            "villagers_win": len(werewolves_alive) == 0,
            "survivors": [self.agents[a].name for a in self.alive],
            "eliminated_order": [self.agents[e].name for e in self.eliminated],
        }
