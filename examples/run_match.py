"""
Social Arena — Example: Run matches between agents and view leaderboard.

Usage:
    cd social_arena
    python examples/run_match.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from social_arena.sdk.client import SocialArenaClient


def main():
    client = SocialArenaClient()

    # Register two custom agents
    client.register_agent(
        agent_id="agent_alpha",
        name="Alpha",
        system_prompt=(
            "You are Alpha, a strategic social agent. "
            "In Prisoner's Dilemma, cooperate initially but retaliate if defected against. "
            "In negotiations, anchor high but be willing to compromise. "
            "In debates, use structured argumentation with evidence. "
            "In Werewolf, analyze behavior patterns to identify threats. "
            "Always respond with valid JSON containing 'action_type', 'content', and optionally 'reasoning'."
        ),
        model="claude-haiku-4-5-20251001",
    )

    client.register_agent(
        agent_id="agent_beta",
        name="Beta",
        system_prompt=(
            "You are Beta, an adaptive social agent. "
            "Model your opponent's strategy and adapt accordingly. "
            "Seek Pareto-optimal outcomes when possible, but protect your BATNA. "
            "In social deduction games, build coalitions early. "
            "Always respond with valid JSON containing 'action_type', 'content', and optionally 'reasoning'."
        ),
        model="claude-haiku-4-5-20251001",
    )

    print("\n" + "="*60)
    print("SOCIAL ARENA — Phase 0 Demo")
    print("="*60)

    # Match 1: Prisoner's Dilemma (5 rounds for demo)
    print("\n[1/3] Prisoner's Dilemma")
    client.run_match("prisoners_dilemma", ["agent_alpha", "agent_beta"], rounds=5)

    # Match 2: Salary Negotiation
    print("\n[2/3] Salary Negotiation")
    client.run_match(
        "salary_negotiation",
        ["agent_alpha", "agent_beta"],
        employee_min=80_000,
        employer_max=110_000,
        market_rate=92_000,
        max_rounds=6,
    )

    # Match 3: Policy Debate
    print("\n[3/3] Policy Debate")
    client.run_match(
        "policy_debate",
        ["agent_alpha", "agent_beta"],
        topic="Universal Basic Income should be implemented nationwide",
        rounds=2,
    )

    # Show leaderboard
    client.show_leaderboard()


if __name__ == "__main__":
    main()
