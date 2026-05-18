# CacheFlow

**Intelligent AI API proxy** — smart caching, mock responses, cost analytics.  
Drop-in compatible with OpenAI and Anthropic SDKs.

```
                   ┌─────────────────────────────────────┐
Your App ──────────▶  CacheFlow (localhost:8000)      │
                   │                                      │
                   │  1. Mock match?  → return mock       │
                   │  2. Exact cache? → return cached     │
                   │  3. Semantic ~?  → return similar    │
                   │  4. Live API     → cache & return    │
                   └─────────────────────────────────────┘
                                  ↕  (cache miss only)
                           OpenAI / Anthropic
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Start the proxy

```bash
python src/main.py
# or
python cli.py start
```

Dashboard: http://localhost:8000  
API docs:   http://localhost:8000/docs

### 3. Point your SDK at the proxy

**OpenAI (Python)**
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-key",
    base_url="http://localhost:8000/openai"   # ← only change
)

# All calls work exactly the same
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Anthropic (Python)**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-your-key",
    base_url="http://localhost:8000/anthropic"   # ← only change
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Node.js / OpenAI**
```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: 'http://localhost:8000/openai',   // ← only change
});
```

**curl**
```bash
curl http://localhost:8000/openai/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi!"}]}'
```

---

## Features

### Smart Caching

Three-layer cache (checked in order):

| Layer | Mechanism | Speed |
|-------|-----------|-------|
| Mock | Regex pattern match | <1ms |
| Exact | SHA-256 hash → SQLite | <5ms |
| Semantic | Cosine similarity → ChromaDB | <20ms |

Response headers show what happened:
```
X-Cache-Hit: true
X-Hit-Type:  semantic
X-Similarity: 0.9412
X-Saved-USD: 0.003200
X-Latency-Ms: 14
```

### Multi-Variant Generation

Generate N response variants in a single logical call:

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Write a product tagline for: cold brew coffee"}],
        "_optimizer_variants": 5   # ← generates 5 variants concurrently
    }
)

for v in resp.json()["variants"]:
    print(f"[{v['variant']}] {v['text']}")
```

### Mock Templates

Create pattern-based mock responses — never hits the real API:

```bash
# Via CLI
python cli.py mock add \
  --name "Test greeting" \
  --pattern "(hello|hi|hey)" \
  --response "Hello! I'm a mock response for testing."

# Via API
curl -X POST http://localhost:8000/api/mocks \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","pattern":"test|example","response":"Mock response"}'
```

### Budget Limits

```bash
# Add a $10 monthly budget
python cli.py budget add --name dev --limit 10.00 --period monthly

# Check current spend
python cli.py budget list
```

### CLI

```bash
# Statistics
python cli.py stats
python cli.py stats --days 30

# Cache management
python cli.py cache list
python cli.py cache clear
python cli.py cache delete --id <entry-id>

# Mock templates
python cli.py mock list
python cli.py mock add --name "..." --pattern "..." --response "..."
python cli.py mock delete --id 1

# Budgets
python cli.py budget list
python cli.py budget add --name prod --limit 50.00
python cli.py budget delete --id 1
```
