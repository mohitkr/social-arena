from ..core.types import Observation, Action, TaskCategory
from .base import BaseTask


class SalaryNegotiationTask(BaseTask):
    name = "Salary Negotiation"
    category = TaskCategory.NEGOTIATION
    min_agents = 2
    max_agents = 2

    def __init__(
        self,
        employee_min: int = 80_000,
        employer_max: int = 120_000,
        market_rate: int = 95_000,
        max_rounds: int = 8,
    ):
        self.employee_min = employee_min
        self.employer_max = employer_max
        self.market_rate = market_rate
        self.max_rounds = max_rounds

    def _reset(self):
        self.employee_id = self.agent_ids[0]
        self.employer_id = self.agent_ids[1]
        self.current_round = 0
        self.history: list[dict] = []
        self.last_offer: int | None = None
        self.deal_reached: bool = False
        self.final_salary: int | None = None
        self.negotiation_failed: bool = False
        self._pending: dict[str, Action] = {}
        self._phase = "employee_opens"  # who acts next

    def agent_can_act(self, agent_id: str) -> bool:
        if self._phase == "employee_opens":
            return agent_id == self.employee_id
        if self._phase == "employer_responds":
            return agent_id == self.employer_id
        if self._phase == "employee_responds":
            return agent_id == self.employee_id
        return False

    def observe(self, agent_id: str, round_number: int) -> Observation:
        is_employee = agent_id == self.employee_id
        role = "employee" if is_employee else "employer"
        private_info = (
            {"minimum_acceptable_salary": self.employee_min, "market_rate": self.market_rate}
            if is_employee
            else {"maximum_budget": self.employer_max, "market_rate": self.market_rate}
        )
        return Observation(
            task_state={
                "round": round_number,
                "max_rounds": self.max_rounds,
                "last_offer": self.last_offer,
                "status": "negotiating",
            },
            role=role,
            history=self.history[-6:],
            private_info=private_info,
            round_number=round_number,
            valid_actions=["make_offer", "accept", "reject", "counter_offer"],
        )

    def step(self, agent_id: str, action: Action):
        entry = {
            "agent": self.agents[agent_id].name,
            "role": "employee" if agent_id == self.employee_id else "employer",
            "action": action.action_type,
            "content": action.content,
            "round": self.current_round + 1,
        }
        self.history.append(entry)

        if action.action_type == "accept" and self.last_offer is not None:
            self.deal_reached = True
            self.final_salary = self.last_offer
            self.current_round = self.max_rounds  # end the game
            print(f"  Deal reached! Salary: ${self.final_salary:,}")
            return

        if action.action_type == "reject" and self.current_round >= self.max_rounds - 1:
            self.negotiation_failed = True
            self.current_round = self.max_rounds
            print("  Negotiation failed — no deal reached.")
            return

        # Extract offer amount
        amount = self._extract_amount(action.content)
        if amount is not None:
            self.last_offer = amount

        # Advance phase
        if self._phase == "employee_opens":
            self._phase = "employer_responds"
        elif self._phase == "employer_responds":
            self._phase = "employee_responds"
            self.current_round += 1
        elif self._phase == "employee_responds":
            self._phase = "employer_responds"
            self.current_round += 1

        print(f"  {self.agents[agent_id].name} [{action.action_type}]: {action.content}")

    def _extract_amount(self, content) -> int | None:
        import re
        if isinstance(content, (int, float)):
            return int(content)
        if isinstance(content, dict):
            for key in ("salary", "offer", "amount", "value"):
                if key in content:
                    return int(content[key])
        if isinstance(content, str):
            nums = re.findall(r"\d[\d,]*", content)
            if nums:
                return int(nums[0].replace(",", ""))
        return None

    def is_terminal(self) -> bool:
        return self.deal_reached or self.negotiation_failed or self.current_round >= self.max_rounds

    def compute_scores(self) -> dict[str, float]:
        if not self.deal_reached or self.final_salary is None:
            # No deal: both get 0
            return {self.employee_id: 0.0, self.employer_id: 0.0}

        salary = self.final_salary
        # Employee: how much above minimum they got (normalized 0-1)
        total_range = self.employer_max - self.employee_min
        employee_surplus = max(0, salary - self.employee_min)
        employer_surplus = max(0, self.employer_max - salary)

        employee_score = employee_surplus / total_range if total_range > 0 else 0.5
        employer_score = employer_surplus / total_range if total_range > 0 else 0.5

        return {
            self.employee_id: round(employee_score * 100, 2),
            self.employer_id: round(employer_score * 100, 2),
        }

    def outcome_metrics(self) -> dict:
        return {
            "deal_reached": self.deal_reached,
            "final_salary": self.final_salary,
            "employee_batna": self.employee_min,
            "employer_budget": self.employer_max,
            "market_rate": self.market_rate,
            "rounds_to_deal": len(self.history),
        }
