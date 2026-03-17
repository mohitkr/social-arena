import asyncio
import json
import os
import queue
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.agent import (
    AgentConfig, BaseAgent,
    CooperativeMockAgent, CompetitiveMockAgent, TitForTatMockAgent, RandomMockAgent,
)
from ..core.providers import LLMAgent, PROVIDER_MODELS
from ..core.match import MatchOrchestrator
from ..core.leaderboard import Leaderboard
from ..tasks.prisoners_dilemma import PrisonersDilemmaTask
from ..tasks.salary_negotiation import SalaryNegotiationTask
from ..tasks.policy_debate import PolicyDebateTask
from ..tasks.werewolf import WerewolfTask

app = FastAPI(title="Social Arena")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
AGENTS_FILE    = DATA_DIR / "agents.json"
HISTORY_FILE   = DATA_DIR / "history.json"
LEADERBOARD_FILE = DATA_DIR / "leaderboard.json"

# ── Global State ──────────────────────────────────────────────────────────────
leaderboard = Leaderboard()
orchestrator = MatchOrchestrator()

registered_agents: dict[str, dict] = {}   # agent_id -> metadata + agent object
match_queues: dict[str, queue.Queue] = {} # match_id -> log queue
match_results: dict[str, dict] = {}       # match_id -> result dict
match_history: list[dict] = []

# ── Persistence helpers ───────────────────────────────────────────────────────
def _save_agents():
    rows = []
    for aid, meta in registered_agents.items():
        row = {k: v for k, v in meta.items() if k != "agent"}  # skip live object
        row["agent_id"] = aid
        rows.append(row)
    AGENTS_FILE.write_text(json.dumps(rows, indent=2))

def _save_history():
    HISTORY_FILE.write_text(json.dumps(match_history[:200], indent=2, default=str))

def _save_leaderboard():
    data = {}
    for aid, ar in leaderboard.ratings.items():
        data[aid] = {"mu": ar.rating.mu, "sigma": ar.rating.sigma,
                     "wins": ar.wins, "matches": ar.matches_played,
                     "total_score": ar.total_score, "total_rounds": ar.total_rounds}
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2))

def _rebuild_agent(row: dict) -> BaseAgent:
    """Reconstruct a live agent object from a saved metadata row."""
    config = AgentConfig(
        agent_id=row["agent_id"],
        name=row["name"],
        system_prompt=row.get("system_prompt", ""),
        model=row.get("model", ""),
    )
    if row["agent_type"] == "mock":
        cls = MOCK_STRATEGIES.get(row.get("strategy", "cooperative"), CooperativeMockAgent)
        return cls(config)
    # llm
    provider = row.get("provider", "anthropic")
    api_key = row.get("api_key")
    if api_key:
        env_vars = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
        ev = env_vars.get(provider)
        if ev:
            os.environ[ev] = api_key
    return LLMAgent(config, provider=provider)

def _load_state():
    import trueskill as ts

    # Agents
    if AGENTS_FILE.exists():
        rows = json.loads(AGENTS_FILE.read_text())
        for row in rows:
            aid = row["agent_id"]
            try:
                agent = _rebuild_agent(row)
                meta = {k: v for k, v in row.items() if k != "agent_id"}
                meta["agent"] = agent
                registered_agents[aid] = meta
                leaderboard.register_agent(aid, row["name"])
            except Exception as e:
                print(f"[state] Could not restore agent {row.get('name')}: {e}")

    # Leaderboard ratings (restore after agents so ratings dict exists)
    if LEADERBOARD_FILE.exists():
        data = json.loads(LEADERBOARD_FILE.read_text())
        for aid, r in data.items():
            if aid in leaderboard.ratings:
                ar = leaderboard.ratings[aid]
                ar.rating = leaderboard.env.create_rating(mu=r["mu"], sigma=r["sigma"])
                ar.wins = r["wins"]
                ar.matches_played = r["matches"]
                ar.total_score = r.get("total_score", 0.0)
                ar.total_rounds = r.get("total_rounds", 0)

    # Match history
    if HISTORY_FILE.exists():
        match_history.extend(json.loads(HISTORY_FILE.read_text()))

    print(f"[state] Loaded {len(registered_agents)} agents, {len(match_history)} matches.")

@app.on_event("startup")
async def startup():
    _load_state()

# Thread-local stdout routing
_match_loggers: dict[int, callable] = {}
_logger_lock = threading.Lock()

class _RoutingStdout:
    def __init__(self, real):
        self.real = real
    def write(self, msg):
        tid = threading.get_ident()
        with _logger_lock:
            fn = _match_loggers.get(tid)
        if fn and msg.strip():
            fn(msg.rstrip())
        self.real.write(msg)
    def flush(self):
        self.real.flush()

_real_stdout = sys.stdout
sys.stdout = _RoutingStdout(_real_stdout)

# ── Task Registry ─────────────────────────────────────────────────────────────
TASK_REGISTRY = {
    "prisoners_dilemma": PrisonersDilemmaTask,
    "salary_negotiation": SalaryNegotiationTask,
    "policy_debate": PolicyDebateTask,
    "werewolf": WerewolfTask,
}

TASK_META = {
    "prisoners_dilemma": {
        "label": "Prisoner's Dilemma",
        "category": "cooperation",
        "description": "Iterated cooperation/defection dilemma across N rounds.",
        "min_agents": 2, "max_agents": 2,
        "params": [
            {"name": "rounds", "type": "number", "default": 10, "label": "Rounds", "min": 3, "max": 100},
        ],
    },
    "salary_negotiation": {
        "label": "Salary Negotiation",
        "category": "negotiation",
        "description": "Employee vs employer negotiate salary under private budget constraints.",
        "min_agents": 2, "max_agents": 2,
        "params": [
            {"name": "employee_min", "type": "number", "default": 80000, "label": "Employee Min ($)"},
            {"name": "employer_max", "type": "number", "default": 110000, "label": "Employer Budget ($)"},
            {"name": "market_rate", "type": "number", "default": 92000, "label": "Market Rate ($)"},
            {"name": "max_rounds", "type": "number", "default": 6, "label": "Max Rounds", "min": 2, "max": 20},
        ],
    },
    "policy_debate": {
        "label": "Policy Debate",
        "category": "persuasion",
        "description": "Two agents argue opposing sides of a policy topic before an LLM judge.",
        "min_agents": 2, "max_agents": 2,
        "params": [
            {"name": "topic", "type": "text", "default": "Universal Basic Income should be implemented nationwide", "label": "Debate Topic"},
            {"name": "rounds", "type": "number", "default": 2, "label": "Rounds", "min": 1, "max": 5},
        ],
    },
    "werewolf": {
        "label": "Werewolf",
        "category": "social_deduction",
        "description": "Social deduction: hidden werewolves vs villagers, day/night cycle.",
        "min_agents": 4, "max_agents": 10,
        "params": [
            {"name": "num_werewolves", "type": "number", "default": 1, "label": "Werewolves", "min": 1, "max": 3},
        ],
    },
}

MOCK_STRATEGIES = {
    "cooperative": CooperativeMockAgent,
    "competitive": CompetitiveMockAgent,
    "tit_for_tat": TitForTatMockAgent,
    "random": RandomMockAgent,
}

# ── Pydantic Models ───────────────────────────────────────────────────────────
class RegisterAgentRequest(BaseModel):
    name: str
    agent_type: str  # "mock" or "llm"
    strategy: Optional[str] = "cooperative"   # for mock
    provider: Optional[str] = "anthropic"     # for llm
    model: Optional[str] = None               # for llm
    system_prompt: Optional[str] = None       # for llm
    api_key: Optional[str] = None             # for llm — stored in memory, used to set env var

class UpdateAgentRequest(BaseModel):
    system_prompt: str

class StartMatchRequest(BaseModel):
    task: str
    agent_ids: list[str]
    params: dict = {}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text()

@app.get("/api/config")
async def get_config():
    return {
        "tasks": TASK_META,
        "providers": PROVIDER_MODELS,
        "mock_strategies": list(MOCK_STRATEGIES.keys()),
    }

@app.get("/api/agents")
async def list_agents():
    result = []
    for aid, meta in registered_agents.items():
        rating = leaderboard.ratings.get(aid)
        result.append({
            "agent_id": aid,
            "name": meta["name"],
            "agent_type": meta["agent_type"],
            "provider": meta.get("provider", "mock"),
            "model": meta.get("model", meta.get("strategy", "-")),
            "score": round(rating.conservative_score, 2) if rating else 0.0,
            "wins": rating.wins if rating else 0,
            "matches": rating.matches_played if rating else 0,
            "avg_pts_per_round": round(rating.avg_pts_per_round, 2) if rating else 0.0,
            "system_prompt": meta.get("system_prompt", "") if meta.get("agent_type") == "llm" else None,
            "community": meta.get("community", False),
        })
    return result

@app.post("/api/agents")
async def register_agent(req: RegisterAgentRequest):
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    config = AgentConfig(agent_id=agent_id, name=req.name, system_prompt=req.system_prompt or "", model=req.model or "")

    if req.agent_type == "mock":
        strategy = req.strategy or "cooperative"
        cls = MOCK_STRATEGIES.get(strategy, CooperativeMockAgent)
        agent = cls(config)
        meta = {"name": req.name, "agent_type": "mock", "strategy": strategy, "agent": agent}
    elif req.agent_type == "llm":
        provider = req.provider or "anthropic"
        model = req.model or PROVIDER_MODELS.get(provider, [""])[0]
        config.model = model
        # Store the API key in the environment for this provider
        if req.api_key:
            env_vars = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
            env_var = env_vars.get(provider)
            if env_var:
                os.environ[env_var] = req.api_key
        agent = LLMAgent(config, provider=provider)
        meta = {
            "name": req.name, "agent_type": "llm", "provider": provider, "model": model,
            "system_prompt": req.system_prompt or "",
            "community": not bool(req.api_key),  # True = uses server API key
            "agent": agent,
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type: {req.agent_type}")

    registered_agents[agent_id] = meta
    leaderboard.register_agent(agent_id, req.name)
    _save_agents()
    _save_leaderboard()
    return {"agent_id": agent_id, "name": req.name}

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    del registered_agents[agent_id]
    if agent_id in leaderboard.ratings:
        del leaderboard.ratings[agent_id]
    _save_agents()
    _save_leaderboard()
    return {"ok": True}

@app.patch("/api/agents/{agent_id}")
async def update_agent_prompt(agent_id: str, req: UpdateAgentRequest):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    meta = registered_agents[agent_id]
    if meta.get("agent_type") != "llm":
        raise HTTPException(status_code=400, detail="Only LLM agents have a system prompt")
    meta["system_prompt"] = req.system_prompt
    meta["agent"].config.system_prompt = req.system_prompt
    _save_agents()
    return {"ok": True}

@app.get("/api/leaderboard")
async def get_leaderboard():
    rankings = leaderboard.get_rankings()
    return [
        {
            "rank": i + 1,
            "agent_id": ar.agent_id,
            "name": ar.name,
            "score": round(ar.conservative_score, 2),
            "mu": round(ar.rating.mu, 2),
            "sigma": round(ar.rating.sigma, 2),
            "wins": ar.wins,
            "matches": ar.matches_played,
            "win_rate": round(ar.win_rate, 3),
            "avg_pts_per_round": round(ar.avg_pts_per_round, 2),
            "agent_type": registered_agents.get(ar.agent_id, {}).get("agent_type", "?"),
            "provider": registered_agents.get(ar.agent_id, {}).get("provider", registered_agents.get(ar.agent_id, {}).get("strategy", "-")),
        }
        for i, ar in enumerate(rankings)
    ]

@app.post("/api/matches")
async def start_match(req: StartMatchRequest):
    # Validate
    if req.task not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
    meta = TASK_META[req.task]
    if len(req.agent_ids) < meta["min_agents"]:
        raise HTTPException(status_code=400, detail=f"Need at least {meta['min_agents']} agents")
    for aid in req.agent_ids:
        if aid not in registered_agents:
            raise HTTPException(status_code=400, detail=f"Agent not found: {aid}")

    match_id = uuid.uuid4().hex[:8]
    match_queues[match_id] = queue.Queue()
    match_results[match_id] = {"status": "running", "match_id": match_id}

    # Run match in background thread
    def run():
        tid = threading.get_ident()
        q = match_queues[match_id]
        with _logger_lock:
            _match_loggers[tid] = lambda msg: q.put({"type": "log", "message": msg})
        try:
            task_cls = TASK_REGISTRY[req.task]
            # Convert param types
            params = {}
            for p in meta["params"]:
                val = req.params.get(p["name"], p["default"])
                if p["type"] == "number":
                    val = int(val)
                params[p["name"]] = val
            task = task_cls(**params)
            agents: list[BaseAgent] = [registered_agents[aid]["agent"] for aid in req.agent_ids]
            result = orchestrator.run(task, agents)
            leaderboard.update(result)

            agent_names = {aid: registered_agents[aid]["name"] for aid in result.agents if aid in registered_agents}
            record = {
                "match_id": result.match_id,
                "timestamp": datetime.now().isoformat(),
                "task_name": result.task_name,
                "task_category": str(result.task_category),
                "agents": result.agents,
                "agent_names": agent_names,
                "scores": result.scores,
                "winner": result.winner,
                "winner_name": agent_names.get(result.winner, result.winner or ""),
                "rounds_played": result.rounds_played,
                "outcome_metrics": result.outcome_metrics,
                "transcript": result.transcript,
            }
            match_history.insert(0, record)
            match_results[match_id] = {"status": "complete", **record}
            _save_history()
            _save_leaderboard()
            q.put({"type": "result", "data": record})
        except Exception as e:
            match_results[match_id] = {"status": "error", "error": str(e)}
            q.put({"type": "error", "message": str(e)})
        finally:
            with _logger_lock:
                del _match_loggers[tid]
            q.put(None)  # sentinel

    threading.Thread(target=run, daemon=True).start()
    return {"match_id": match_id}

@app.get("/api/matches/{match_id}/events")
async def stream_match_events(match_id: str):
    if match_id not in match_queues:
        raise HTTPException(status_code=404, detail="Match not found")

    async def generate():
        q = match_queues[match_id]
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if msg is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/matches/{match_id}")
async def get_match(match_id: str):
    result = match_results.get(match_id)
    if not result:
        raise HTTPException(status_code=404, detail="Match not found")
    return result

@app.get("/api/history")
async def get_history():
    return match_history[:50]
