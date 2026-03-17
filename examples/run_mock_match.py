"""
Social Arena — Mock Test (no API key required)

Runs all 4 tasks using rule-based mock agents:
  - cooperative  : always cooperates / makes fair offers
  - competitive  : always defects / anchors aggressively
  - tit_for_tat  : cooperates first, then mirrors opponent
  - random       : picks random valid actions

Usage:
    cd social_arena
    pip install -r requirements.txt
    python examples/run_mock_match.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from social_arena.sdk.client import SocialArenaClient


def main():
    client = SocialArenaClient()

    # Register 4 mock agents with different strategies
    client.register_mock_agent("coop",   "Cooperative",  strategy="cooperative")
    client.register_mock_agent("comp",   "Competitive",  strategy="competitive")
    client.register_mock_agent("tft",    "TitForTat",    strategy="tit_for_tat")
    client.register_mock_agent("rando",  "Random",       strategy="random")

    print("\n" + "=" * 60)
    print("SOCIAL ARENA — Mock Test (no API key)")
    print("=" * 60)

    # --- 1. Prisoner's Dilemma ---
    print("\n[1/4] Prisoner's Dilemma: Cooperative vs Competitive")
    client.run_match("prisoners_dilemma", ["coop", "comp"], rounds=10)

    print("\n[1b/4] Prisoner's Dilemma: TitForTat vs Competitive")
    client.run_match("prisoners_dilemma", ["tft", "comp"], rounds=10)

    # --- 2. Salary Negotiation ---
    print("\n[2/4] Salary Negotiation: Cooperative (employee) vs Competitive (employer)")
    client.run_match(
        "salary_negotiation",
        ["coop", "comp"],
        employee_min=80_000,
        employer_max=110_000,
        market_rate=92_000,
        max_rounds=6,
    )

    # --- 3. Policy Debate (mock judge — no API key needed) ---
    print("\n[3/4] Policy Debate: Cooperative vs Competitive")
    client.run_match(
        "policy_debate",
        ["coop", "comp"],
        topic="Universal Basic Income should be implemented nationwide",
        rounds=2,
    )

    # --- 4. Werewolf (4 players: 1 werewolf, 3 villagers) ---
    print("\n[4/4] Werewolf: 4 players (1 werewolf)")
    # Register two more agents for the 4-player game
    client.register_mock_agent("coop2",  "Cooperative2", strategy="cooperative")
    client.register_mock_agent("rando2", "Random2",      strategy="random")
    client.run_match("werewolf", ["coop", "comp", "tft", "rando"], num_werewolves=1)

    # --- Leaderboard + save ---
    client.show_leaderboard()
    client.save_session()
    print("\nTo view the full report:")
    print("  .venv/bin/python3 report.py")


if __name__ == "__main__":
    main()
