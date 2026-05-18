#!/usr/bin/env python3
"""
api-optimizer CLI — manage the proxy from the terminal.
Usage:
  python cli.py start
  python cli.py stats
  python cli.py cache list
  python cli.py cache clear
  python cli.py mock add --name "Hello" --pattern "hello|hi" --response "Hi there!"
  python cli.py budget add --name dev --limit 10.00
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path

try:
    import httpx
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

BASE = "http://localhost:8000"
console = Console() if HAS_RICH else None


def api(method, path, **kwargs):
    import urllib.request, urllib.error
    url = BASE + path
    data = json.dumps(kwargs.get("json", {})).encode() if "json" in kwargs else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        sys.exit(1)
    except Exception as e:
        print(f"Connection error: {e}\nIs the optimizer running? Try: python src/main.py")
        sys.exit(1)


def cmd_start(args):
    print("Starting API Optimizer proxy on http://localhost:8000 ...")
    src = Path(__file__).parent / "src" / "main.py"
    subprocess.run([sys.executable, str(src)])


def cmd_stats(args):
    d = api("GET", f"/api/analytics?days={args.days}")
    t = d["totals"]
    print(f"\n{'='*50}")
    print(f"  API Optimizer — Last {args.days} days")
    print(f"{'='*50}")
    print(f"  Total requests : {t.get('total_requests', 0)}")
    print(f"  Cache hit rate : {t.get('hit_rate', 0):.1f}%")
    print(f"  Actual spend   : ${t.get('total_cost', 0):.6f}")
    print(f"  Total saved    : ${t.get('total_saved', 0):.6f}")
    print(f"  Avg latency    : {t.get('avg_latency', 0):.0f} ms")
    print()

    if d.get("by_model"):
        print("  By model:")
        for m in d["by_model"]:
            hit_pct = (m["hits"] / m["reqs"] * 100) if m["reqs"] else 0
            print(f"    {m['model']:35} {m['reqs']:5} reqs  {hit_pct:.0f}% hit  ${m['cost']:.6f}")
    print()


def cmd_cache(args):
    if args.action == "list":
        d = api("GET", "/api/cache?limit=20")
        print(f"\nCached entries ({d['count']} total):")
        print(f"{'ID'[:8]:<10} {'Model':<25} {'Hits':>4}  Prompt")
        print("-" * 70)
        for e in d["entries"]:
            print(f"  {e['id'][:8]}  {e['model']:<25} {e['hit_count']:>4}  {e['preview'][:40]}")
        print()

    elif args.action == "clear":
        confirm = input("Clear all cache? [y/N] ")
        if confirm.lower() == "y":
            api("DELETE", "/api/cache")
            print("Cache cleared.")

    elif args.action == "delete":
        if not args.id:
            print("Usage: cache delete --id <entry-id>")
            sys.exit(1)
        api("DELETE", f"/api/cache/{args.id}")
        print(f"Deleted {args.id}")


def cmd_mock(args):
    if args.action == "list":
        d = api("GET", "/api/mocks")
        print(f"\nMock templates ({len(d['mocks'])}):")
        for m in d["mocks"]:
            model = m["model"] or "all"
            print(f"  [{m['id']}] {m['name']:<20} pattern={m['pattern']!r} model={model}")
        print()

    elif args.action == "add":
        if not args.pattern or not args.response:
            print("--pattern and --response required")
            sys.exit(1)
        d = api("POST", "/api/mocks", json={
            "name": args.name or "Unnamed",
            "pattern": args.pattern,
            "response": args.response,
            "model": args.model,
        })
        print(f"Created mock #{d['id']}")

    elif args.action == "delete":
        api("DELETE", f"/api/mocks/{args.id}")
        print(f"Deleted mock #{args.id}")


def cmd_budget(args):
    if args.action == "list":
        d = api("GET", "/api/budgets")
        for b in d["budgets"]:
            pct = (b["spent_usd"] / b["limit_usd"] * 100) if b["limit_usd"] else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {b['name']:<20} [{bar}] ${b['spent_usd']:.4f} / ${b['limit_usd']:.2f} ({pct:.1f}%)")

    elif args.action == "add":
        if not args.name or args.limit is None:
            print("--name and --limit required")
            sys.exit(1)
        d = api("POST", "/api/budgets", json={
            "name": args.name, "limit_usd": args.limit, "period": args.period or "monthly"
        })
        print(f"Created budget #{d['id']}")

    elif args.action == "delete":
        api("DELETE", f"/api/budgets/{args.id}")
        print(f"Deleted budget #{args.id}")


def main():
    p = argparse.ArgumentParser(description="API Optimizer CLI")
    sub = p.add_subparsers(dest="cmd")

    # start
    sub.add_parser("start", help="Start the proxy server")

    # stats
    s = sub.add_parser("stats", help="Show usage statistics")
    s.add_argument("--days", type=int, default=7)

    # cache
    c = sub.add_parser("cache", help="Manage cache")
    c.add_argument("action", choices=["list", "clear", "delete"])
    c.add_argument("--id")

    # mock
    m = sub.add_parser("mock", help="Manage mock templates")
    m.add_argument("action", choices=["list", "add", "delete"])
    m.add_argument("--name")
    m.add_argument("--pattern")
    m.add_argument("--response")
    m.add_argument("--model")
    m.add_argument("--id", type=int)

    # budget
    b = sub.add_parser("budget", help="Manage budgets")
    b.add_argument("action", choices=["list", "add", "delete"])
    b.add_argument("--name")
    b.add_argument("--limit", type=float)
    b.add_argument("--period", default="monthly")
    b.add_argument("--id", type=int)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return

    {"start": cmd_start, "stats": cmd_stats, "cache": cmd_cache,
     "mock": cmd_mock, "budget": cmd_budget}[args.cmd](args)


if __name__ == "__main__":
    main()
