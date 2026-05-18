"""
Core proxy engine:
- Checks mock templates first
- Exact hash match in SQLite
- Semantic similarity via ChromaDB
- Falls through to live API if no cache hit
- Records everything for analytics
"""
import hashlib
import json
import re
import time
import uuid
import logging
from typing import Optional, AsyncIterator

import httpx

from database import (
    hash_prompt, cache_get_exact, cache_set, log_request,
    mock_list, budget_list, budget_spend, init_db
)
from semantic_cache import semantic_add, semantic_search, semantic_delete
from costs import model_cost, estimate_tokens

logger = logging.getLogger(__name__)

# ─── Provider upstream URLs ───────────────────────────────────────────────────

PROVIDER_URLS = {
    "openai":    "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai"
}
DEFAULT_MODELS = {
    "openrouter": "tencent/hy3-preview:free",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
}

# ─── Mock matching ────────────────────────────────────────────────────────────

def _check_mocks(prompt: str, model: Optional[str]) -> Optional[str]:
    for tmpl in mock_list():
        if not tmpl["enabled"]:
            continue
        if tmpl["model"] and tmpl["model"] != model:
            continue
        try:
            if re.search(tmpl["pattern"], prompt, re.IGNORECASE):
                return tmpl["response"]
        except re.error:
            if tmpl["pattern"].lower() in prompt.lower():
                return tmpl["response"]
    return None


# ─── Budget guard ─────────────────────────────────────────────────────────────

def _check_budgets(cost: float) -> Optional[str]:
    for b in budget_list():
        if b["spent_usd"] + cost > b["limit_usd"]:
            return f"Budget '{b['name']}' exceeded: ${b['spent_usd']:.4f} / ${b['limit_usd']:.2f}"
    return None


# ─── Main proxy logic ─────────────────────────────────────────────────────────

class ProxyEngine:
    def __init__(self, settings: dict):
        self.settings = settings
        init_db()

    # --- Extract a flat prompt string from request body --------------------

    @staticmethod
    def _extract_prompt(body: dict, provider: str) -> tuple[str, str]:
        """Returns (prompt_text, system_text)."""
        system = ""
        if provider == "anthropic":
            system = body.get("system", "")
            messages = body.get("messages", [])
        else:
            messages = body.get("messages", [])
            for m in messages:
                if m.get("role") == "system":
                    system = m.get("content", "")

        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if role != "system":
                parts.append(f"{role}: {content}")
        return "\n".join(parts), system

    # --- Build a synthetic response ----------------------------------------

    @staticmethod
    def _make_cached_response(body: dict, provider: str, text: str, from_cache: str) -> dict:
        model = body.get("model", "unknown")
        if provider == "anthropic":
            return {
                "id": f"msg_cache_{uuid.uuid4().hex[:16]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "_cache_hit": from_cache,
            }
        else:
            return {
                "id": f"chatcmpl-cache-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "_cache_hit": from_cache,
            }

    # --- Extract response text from upstream reply -------------------------

    @staticmethod
    def _extract_response_text(resp_body: dict, provider: str) -> str:
        if provider == "anthropic":
            content = resp_body.get("content", [])
            return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        else:
            choices = resp_body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return json.dumps(resp_body)

    # --- Main handle ---------------------------------------------------------

    async def handle(
        self,
        provider: str,
        path: str,
        headers: dict,
        body: dict,
    ) -> tuple[dict, dict]:  # (response_body, meta)
        t0 = time.time()
        model = body.get("model", "unknown")
        prompt, system = self._extract_prompt(body, provider)
        phash = hash_prompt(prompt, model, system)
        settings = self.settings

        meta = {
            "cache_hit": False,
            "hit_type": None,
            "similarity": None,
            "cost_usd": 0.0,
            "saved_usd": 0.0,
            "latency_ms": 0,
        }

        # 1) Mock check
        if settings.get("mocks_enabled", True):
            mock_text = _check_mocks(prompt, model)
            if mock_text:
                resp = self._make_cached_response(body, provider, mock_text, "mock")
                meta.update({"cache_hit": True, "hit_type": "mock", "latency_ms": int((time.time()-t0)*1000)})
                log_request(provider, model, True, "mock", None, 0, 0, 0, meta["latency_ms"],
                            model_cost(model, estimate_tokens(prompt), estimate_tokens(mock_text)))
                return resp, meta

        # 2) Exact cache hit
        if settings.get("cache_enabled", True):
            hit = cache_get_exact(phash)
            if hit:
                resp = self._make_cached_response(body, provider, hit["response"], "exact")
                saved = model_cost(model, hit["tokens_in"], hit["tokens_out"])
                meta.update({"cache_hit": True, "hit_type": "exact", "saved_usd": saved,
                              "latency_ms": int((time.time()-t0)*1000)})
                log_request(provider, model, True, "exact", None,
                            hit["tokens_in"], hit["tokens_out"], 0, meta["latency_ms"], saved)
                return resp, meta

        # 3) Semantic cache hit
        sem_threshold = settings.get("semantic_threshold", 0.88)
        if settings.get("semantic_enabled", True):
            sem = semantic_search(prompt, model, threshold=sem_threshold)
            if sem:
                hit = cache_get_exact(  # use the entry_id as hash lookup shortcut
                    _lookup_hash_by_id(sem["entry_id"])
                )
                if hit:
                    resp = self._make_cached_response(body, provider, hit["response"], "semantic")
                    saved = model_cost(model, hit["tokens_in"], hit["tokens_out"])
                    meta.update({
                        "cache_hit": True, "hit_type": "semantic",
                        "similarity": sem["similarity"], "saved_usd": saved,
                        "latency_ms": int((time.time()-t0)*1000)
                    })
                    log_request(provider, model, True, "semantic", sem["similarity"],
                                hit["tokens_in"], hit["tokens_out"], 0, meta["latency_ms"], saved)
                    return resp, meta

        # 4) Live API call
        base_url = PROVIDER_URLS.get(provider, "https://api.openai.com")
        upstream_url = f"{base_url}{path}"

        # Strip proxy-specific headers
        fwd_headers = {k: v for k, v in headers.items()
                       if k.lower() not in ("host", "content-length")}
        fwd_headers["content-type"] = "application/json"
        if not body.get(model):
            body["model"] = DEFAULT_MODELS.get(provider)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(upstream_url, headers=fwd_headers, json=body  )
            r.raise_for_status()
            resp_body = r.json()
        except httpx.HTTPStatusError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Upstream error: {e}") from e

        # Extract tokens & cost
        if provider == "anthropic":
            usage = resp_body.get("usage", {})
            tokens_in = usage.get("input_tokens", estimate_tokens(prompt))
            tokens_out = usage.get("output_tokens", 50)
        # elif provider =="openrouter":
        else:
            usage = resp_body.get("usage", {})
            tokens_in = usage.get("prompt_tokens", estimate_tokens(prompt))
            tokens_out = usage.get("completion_tokens", 50)

        cost = model_cost(model, tokens_in, tokens_out)

        # Budget check (warn but don't block in this version)
        budget_warn = _check_budgets(cost)
        if budget_warn:
            resp_body["_budget_warning"] = budget_warn

        budget_spend("default", cost)  # no-op if budget doesn't exist

        # Store in cache
        #if want to store full response
        # response = resp_body
        response_text = self._extract_response_text(resp_body, provider)
        entry_id = str(uuid.uuid4())
        ttl = settings.get("cache_ttl", 86400)
        cache_set(entry_id, phash, prompt, response_text, model, provider,
                  tokens_in, tokens_out, ttl)
        semantic_add(entry_id, prompt, model)

        latency_ms = int((time.time()-t0)*1000)
        meta.update({"cost_usd": cost, "latency_ms": latency_ms})
        log_request(provider, model, False, None, None,
                    tokens_in, tokens_out, cost, latency_ms, 0)

        return resp_body, meta


# ─── Variant generation ───────────────────────────────────────────────────────

async def generate_variants(
    provider: str,
    path: str,
    headers: dict,
    body: dict,
    num_variants: int = 5,
) -> list[dict]:
    """
    Generate N response variants for the same prompt in one batch.
    Each variant is stored in the variants table.
    """
    import src.database as db

    model = body.get("model", "unknown")
    prompt, _ = ProxyEngine._extract_prompt(body, provider)
    group_id = str(uuid.uuid4())
    results = []

    base_url = PROVIDER_URLS.get(provider, "https://api.openai.com")
    upstream_url = f"{base_url}{path}"
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length")}
    fwd_headers["content-type"] = "application/json"

    # Run requests concurrently
    import asyncio
    tasks = []
    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(num_variants):
            b = {**body, "temperature": body.get("temperature", 0.9)}
            tasks.append(client.post(upstream_url, headers=fwd_headers, json=b))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    conn = db.get_db()
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            results.append({"variant": i, "error": str(r)})
            continue
        try:
            rb = r.json()
            text = ProxyEngine._extract_response_text(rb, provider)
            conn.execute(
                "INSERT INTO variants (group_id, prompt, variant_idx, response, model, created_at) VALUES (?,?,?,?,?,?)",
                (group_id, prompt, i, text, model, time.time())
            )
            results.append({"variant": i, "text": text, "group_id": group_id})
        except Exception as e:
            results.append({"variant": i, "error": str(e)})
    conn.commit()
    conn.close()
    return results


# ─── Utility ─────────────────────────────────────────────────────────────────

def _lookup_hash_by_id(entry_id: str) -> Optional[str]:
    """Get prompt_hash for a given cache entry id."""
    conn = __import__("database").get_db()
    row = conn.execute("SELECT prompt_hash FROM cache_entries WHERE id=?", (entry_id,)).fetchone()
    conn.close()
    return row["prompt_hash"] if row else None
