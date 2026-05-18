"""
API Optimizer - FastAPI proxy server
Drop-in replacement for OpenAI and Anthropic APIs.

Routing:
  /openai/...   → proxied to OpenAI
  /anthropic/...→ proxied to Anthropic
  /api/...      → management API
  /             → dashboard (served from dashboard/index.html)
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from database import (
    init_db, get_analytics, cache_list, cache_delete, cache_clear,
    mock_list, mock_add, mock_delete,
    budget_list, budget_add, budget_delete,
    cache_get_exact, hash_prompt
)
from proxy_engine import ProxyEngine, generate_variants
from semantic_cache import semantic_clear
from costs import PRICING

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Global settings (in-memory; persist to file for production) ─────────────

SETTINGS: dict = {
    "cache_enabled": True,
    "semantic_enabled": True,
    "mocks_enabled": True,
    "semantic_threshold": 0.88,
    "cache_ttl": 86400,# i think to add an option to add -1 for infinite time
    "budget_enforce": False,
}

SETTINGS_FILE = Path("data/settings.json")


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            SETTINGS.update(json.loads(SETTINGS_FILE.read_text()))
        except Exception:
            pass


def save_settings():
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=2))


load_settings()
init_db()

engine = ProxyEngine(SETTINGS)

app = FastAPI(
    title="API Optimizer",
    description="Intelligent AI API proxy with semantic caching and cost optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Proxy routes ─────────────────────────────────────────────────────────────

@app.api_route("/openai/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openai(request: Request, path: str):
    return await _proxy(request, "openai", f"/{path}")


@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_anthropic(request: Request, path: str):
    return await _proxy(request, "anthropic", f"/{path}")

@app.api_route("/openrouter/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openrouter(request: Request, path :str):
    return await _proxy(request , "openrouter", f"/{path}")


async def _proxy(request: Request, provider: str, path: str):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    headers = dict(request.headers)

    # Variant generation shortcut
    num_variants = body.pop("_optimizer_variants", None)
    if num_variants and isinstance(num_variants, int) and num_variants > 1:
        variants = await generate_variants(provider, path, headers, body, num_variants)
        return JSONResponse({"variants": variants})

    try:
        resp_body, meta = await engine.handle(provider, path, headers, body)
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(502, str(e))

    response = JSONResponse(resp_body)
    response.headers["X-Cache-Hit"] = str(meta["cache_hit"])
    response.headers["X-Hit-Type"] = meta.get("hit_type") or "miss"
    response.headers["X-Cost-USD"] = f"{meta['cost_usd']:.6f}"
    response.headers["X-Saved-USD"] = f"{meta['saved_usd']:.6f}"
    response.headers["X-Latency-Ms"] = str(meta["latency_ms"])
    if meta.get("similarity"):
        response.headers["X-Similarity"] = f"{meta['similarity']:.4f}"
    return response


# ─── Management API ───────────────────────────────────────────────────────────

# Settings
@app.get("/api/settings")
def get_settings():
    return SETTINGS


@app.put("/api/settings")
def update_settings(updates: dict = Body(...)):
    SETTINGS.update(updates)
    engine.settings.update(updates)
    save_settings()
    return SETTINGS


# Analytics
@app.get("/api/analytics")
def analytics(days: int = 7):
    return get_analytics(days)


# Cache
@app.get("/api/cache")
def list_cache(limit: int = 50):
    entries = cache_list(limit)
    for e in entries:
        e["preview"] = e["prompt"][:120] + ("…" if len(e["prompt"]) > 120 else "")
        e["response_preview"] = e["response"][:80] + ("…" if len(e["response"]) > 80 else "")
    return {"entries": entries, "count": len(entries)}


@app.delete("/api/cache/{entry_id}")
def delete_cache_entry(entry_id: str):
    cache_delete(entry_id)
    from src.semantic_cache import semantic_delete
    semantic_delete(entry_id)
    return {"ok": True}


@app.delete("/api/cache")
def clear_cache():
    cache_clear()
    semantic_clear()
    return {"ok": True}


# Mock templates
@app.get("/api/mocks")
def list_mocks():
    return {"mocks": mock_list()}


@app.post("/api/mocks")
def create_mock(data: dict = Body(...)):
    name = data.get("name", "Unnamed")
    pattern = data.get("pattern")
    response = data.get("response")
    model = data.get("model")
    if not pattern or not response:
        raise HTTPException(400, "pattern and response required")
    mid = mock_add(name, pattern, response, model)
    return {"id": mid, "ok": True}


@app.delete("/api/mocks/{mock_id}")
def delete_mock(mock_id: int):
    mock_delete(mock_id)
    return {"ok": True}


# Budgets
@app.get("/api/budgets")
def list_budgets():
    return {"budgets": budget_list()}


@app.post("/api/budgets")
def create_budget(data: dict = Body(...)):
    name = data.get("name")
    limit_usd = data.get("limit_usd")
    period = data.get("period", "monthly")
    if not name or limit_usd is None:
        raise HTTPException(400, "name and limit_usd required")
    bid = budget_add(name, float(limit_usd), period)
    return {"id": bid, "ok": True}


@app.delete("/api/budgets/{budget_id}")
def delete_budget(budget_id: int):
    budget_delete(budget_id)
    return {"ok": True}


# Pricing reference
@app.get("/api/pricing")
def get_pricing():
    return PRICING


# Health
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ─── Dashboard ────────────────────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

@app.get("/", response_class=HTMLResponse)
def root():
    p = DASHBOARD_DIR / "index.html"
    # p= Path.cwd() / "index.html"
    try:
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
    except:
        return HTMLResponse("<h1>API Optimizer running</h1><p>Visit <code>/docs</code> for the API.</p>")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║          API Optimizer v1.0 - Starting Up            ║
╠══════════════════════════════════════════════════════╣
║  Proxy endpoints:                                    ║
║    OpenAI   → http://localhost:8000/openai/...       ║
║    Anthropic→ http://localhost:8000/anthropic/...    ║
║  Dashboard  → http://localhost:8000/                 ║
║  API docs   → http://localhost:8000/docs             ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src"])
