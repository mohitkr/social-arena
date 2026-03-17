"""
Social Arena — Report Generator

Reads results/session.json and produces:
  1. A terminal summary
  2. results/report.html — open in any browser

Usage:
    .venv/bin/python3 report.py
    open results/report.html
"""
import json
import os
import sys
import webbrowser

RESULTS_DIR = "results"
SESSION_FILE = os.path.join(RESULTS_DIR, "session.json")


def load_session():
    if not os.path.exists(SESSION_FILE):
        print("No session.json found. Run a match first:")
        print("  .venv/bin/python3 examples/run_mock_match.py")
        sys.exit(1)
    with open(SESSION_FILE) as f:
        return json.load(f)


def print_terminal_summary(session):
    lb = session["leaderboard"]
    matches = session["matches"]

    print(f"\n{'='*62}")
    print(f"{'SOCIAL ARENA — SESSION REPORT':^62}")
    print(f"{'Generated: ' + session['generated_at'][:19]:^62}")
    print(f"{'='*62}")

    print(f"\n{'LEADERBOARD':^62}")
    print(f"{'─'*62}")
    print(f"{'Rank':<6}{'Agent':<18}{'TrueSkill':>10}{'W/M':>8}{'Win Rate':>10}")
    print(f"{'─'*62}")
    for entry in lb:
        bar = "█" * int(entry["win_rate"] * 20)
        print(
            f"{entry['rank']:<6}{entry['name']:<18}{entry['score']:>10.2f}"
            f"{str(entry['wins'])+'/'+str(entry['matches']):>8}{entry['win_rate']:>9.1%}"
        )
    print(f"{'─'*62}")

    print(f"\n{'MATCH HISTORY':^62}")
    print(f"{'─'*62}")
    for m in matches:
        names = list(m["agent_names"].values())
        score_str = " | ".join(
            f"{m['agent_names'].get(aid, aid)}: {s:.1f}"
            for aid, s in m["scores"].items()
        )
        winner_name = m.get("winner_name", m.get("winner", "?"))
        print(f"  [{m['match_id']}] {m['task_name']}")
        print(f"    Players : {', '.join(names)}")
        print(f"    Scores  : {score_str}")
        print(f"    Winner  : {winner_name}  ({m['rounds_played']} rounds)")
        if m.get("outcome_metrics"):
            for k, v in m["outcome_metrics"].items():
                print(f"    {k}: {v}")
        print()


def generate_html(session) -> str:
    lb = session["leaderboard"]
    matches = session["matches"]

    lb_rows = ""
    for e in lb:
        win_pct = e["win_rate"] * 100
        bar_width = int(win_pct * 1.5)
        lb_rows += f"""
        <tr>
          <td class="rank">#{e['rank']}</td>
          <td class="name">{e['name']}</td>
          <td class="score">{e['score']:.2f}</td>
          <td>{e['wins']}/{e['matches']}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar" style="width:{bar_width}px">{win_pct:.0f}%</div>
            </div>
          </td>
        </tr>"""

    match_cards = ""
    for m in matches:
        score_items = "".join(
            f'<div class="score-item"><span class="agent-name">{m["agent_names"].get(aid, aid)}</span>'
            f'<span class="agent-score">{s:.1f}</span></div>'
            for aid, s in m["scores"].items()
        )
        metrics = "".join(
            f'<div class="metric"><span>{k}</span><span>{v}</span></div>'
            for k, v in (m.get("outcome_metrics") or {}).items()
            if not isinstance(v, list)
        )
        winner_name = m.get("winner_name", m.get("winner", "?"))
        category_class = str(m.get("task_category", "")).replace("TaskCategory.", "").lower()
        match_cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="task-name">{m['task_name']}</span>
            <span class="tag {category_class}">{category_class}</span>
            <span class="match-id">#{m['match_id']}</span>
          </div>
          <div class="scores">{score_items}</div>
          <div class="winner">Winner: <strong>{winner_name}</strong> &nbsp;·&nbsp; {m['rounds_played']} rounds</div>
          {f'<div class="metrics">{metrics}</div>' if metrics else ''}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Social Arena — Session Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f1117; color: #e0e0e0; padding: 24px; }}
    h1 {{ color: #4a9eff; font-size: 2rem; margin-bottom: 4px; }}
    .subtitle {{ color: #888; margin-bottom: 32px; font-size: 0.9rem; }}
    h2 {{ color: #4a9eff; font-size: 1.2rem; margin: 32px 0 12px; border-bottom: 1px solid #2a2a3a; padding-bottom: 6px; }}

    /* Leaderboard */
    table {{ width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 8px; overflow: hidden; }}
    th {{ background: #1e3a5f; color: #4a9eff; text-align: left; padding: 10px 14px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #2a2a3a; font-size: 0.95rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #22263a; }}
    .rank {{ color: #888; font-weight: bold; width: 50px; }}
    .name {{ font-weight: 600; color: #fff; }}
    .score {{ font-family: monospace; color: #4a9eff; font-size: 1.1rem; }}
    .bar-wrap {{ background: #2a2a3a; border-radius: 4px; height: 20px; width: 160px; }}
    .bar {{ background: linear-gradient(90deg, #4a9eff, #7b4fff); border-radius: 4px; height: 100%;
             display: flex; align-items: center; padding-left: 6px; font-size: 0.75rem; color: #fff;
             min-width: 30px; }}

    /* Match cards */
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: #1a1d27; border-radius: 10px; padding: 16px; border: 1px solid #2a2a3a; }}
    .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
    .task-name {{ font-weight: 700; color: #fff; flex: 1; }}
    .match-id {{ color: #555; font-size: 0.75rem; font-family: monospace; }}
    .tag {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 600; text-transform: uppercase; }}
    .cooperation {{ background: #1a3a2a; color: #4caf88; }}
    .negotiation {{ background: #3a2a1a; color: #e6a030; }}
    .persuasion {{ background: #2a1a3a; color: #a060ff; }}
    .social_deduction {{ background: #3a1a1a; color: #ff6060; }}
    .scores {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
    .score-item {{ background: #22263a; border-radius: 6px; padding: 6px 12px;
                   display: flex; flex-direction: column; align-items: center; min-width: 80px; }}
    .agent-name {{ font-size: 0.75rem; color: #888; }}
    .agent-score {{ font-size: 1.2rem; font-weight: 700; color: #4a9eff; font-family: monospace; }}
    .winner {{ font-size: 0.85rem; color: #aaa; margin-bottom: 8px; }}
    .metrics {{ border-top: 1px solid #2a2a3a; margin-top: 10px; padding-top: 10px; }}
    .metric {{ display: flex; justify-content: space-between; font-size: 0.8rem; color: #888; padding: 2px 0; }}
    .metric span:last-child {{ color: #ccc; }}
  </style>
</head>
<body>
  <h1>Social Arena</h1>
  <div class="subtitle">Session Report &nbsp;·&nbsp; {session['generated_at'][:19]}</div>

  <h2>Leaderboard</h2>
  <table>
    <thead>
      <tr><th>Rank</th><th>Agent</th><th>TrueSkill (µ−3σ)</th><th>W/M</th><th>Win Rate</th></tr>
    </thead>
    <tbody>{lb_rows}</tbody>
  </table>

  <h2>Match History</h2>
  <div class="cards">{match_cards}</div>
</body>
</html>"""


def main():
    session = load_session()
    print_terminal_summary(session)

    html = generate_html(session)
    html_path = os.path.join(RESULTS_DIR, "report.html")
    with open(html_path, "w") as f:
        f.write(html)

    abs_path = os.path.abspath(html_path)
    print(f"HTML report saved → {abs_path}")
    webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
