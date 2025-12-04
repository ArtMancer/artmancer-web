#!/usr/bin/env python3
"""RunPod healthcheck utility.

This script attempts to check the health of a RunPod endpoint in a few ways:
- Try the official `runpod` SDK if it's installed
- Fallback to an HTTP GET to /ping using `requests`
- DNS and TCP connect checks for more detailed diagnostics

Usage:
  RUNPOD_API_KEY=<key> RUNPOD_URL=https://pov3ewvy1mejeo.api.runpod.ai python runpod_healthcheck.py

The script performs exponential backoff retries and prints clear diagnostics to stderr/stdout.
"""

from __future__ import annotations

import os
import sys
import time
import socket
from typing import Optional
from pathlib import Path

DEFAULT_URL = "https://pov3ewvy1mejeo.api.runpod.ai"


def derive_endpoint_id(url: str) -> Optional[str]:
    # Given a host like pov3ewvy1mejeo.api.runpod.ai, return pov3ewvy1mejeo
    try:
        host = url.split("//")[-1].split("/")[0]
        if host.endswith(".api.runpod.ai"):
            return host.replace(".api.runpod.ai", "")
    except Exception:
        return None
    return None


def dns_and_tcp_check(host: str, port: int = 443, timeout: float = 5.0) -> tuple[bool, str]:
    """Return (ok, message) for basic resolution and TCP connect test."""
    try:
        addrs = socket.getaddrinfo(host, port)
        if not addrs:
            return False, "DNS lookup returned no addresses"
    except Exception as e:
        return False, f"DNS lookup failed: {e}"

    try:
        # Try TCP connect to first addr
        addr = addrs[0][4]
        with socket.create_connection(addr, timeout=timeout):
            return True, f"TCP connect to {host}:{port} OK"
    except Exception as e:
        return False, f"TCP connect failed: {e}"


def requests_health_check(url: str, api_key: str, timeout: float = 15.0, retries: int = 4):
    import requests

    headers = {"Authorization": f"Key {api_key}"}
    ping_url = f"{url.rstrip('/')}/ping"
    attempt = 0
    backoff = 5
    while attempt < retries:
        attempt += 1
        try:
            print(f"🔁 request attempt {attempt}/{retries} -> {ping_url}")
            r = requests.get(ping_url, headers=headers, timeout=timeout)
            print(f"🟢 HTTP {r.status_code}: {r.text}")
            return True, r.status_code, r.text
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout on attempt {attempt}; waiting {backoff}s...")
        except Exception as e:
            print(f"❌ Request error on attempt {attempt}: {e}")
        time.sleep(backoff)
        backoff *= 2

    return False, None, "All attempts failed"


def runpod_sdk_healthcheck(endpoint_id: str, api_key: str, timeout: int = 10):
    try:
        import runpod
    except Exception:
        return False, "runpod SDK not available"

    try:
        runpod.api_key = api_key
        endpoint = runpod.Endpoint(endpoint_id)
        if hasattr(endpoint, "health"):
            health = endpoint.health(timeout=timeout)
            return True, health
        return False, "SDK Endpoint has no health() method"
    except Exception as e:
        return False, str(e)


def main():
    # If env vars are not exported, try loading .env from common locations
    def load_dotenv_file(p: Path):
        if not p.exists():
            return
        print(f"🔎 loading env file from {p}")
        try:
            with p.open("r", encoding="utf8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # don't overwrite existing environment vars
                    if os.environ.get(k) is None:
                        os.environ[k] = v
        except Exception as e:
            print(f"⚠️ Failed to load dotenv file {p}: {e}")

    # try several common locations: current cwd, script dir, ../server/.env
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates = [cwd / ".env", script_dir / ".env", script_dir.parent / ".env"]
    for candidate in candidates:
        load_dotenv_file(candidate)

    url = os.getenv("RUNPOD_URL", DEFAULT_URL)
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("Please set RUNPOD_API_KEY in the environment or .env file", file=sys.stderr)
        sys.exit(2)

    host = url.split("//")[-1].split("/")[0]
    print(f"🏷️ Testing RunPod endpoint URL: {url}")
    print(f"🔐 Using RUNPOD_API_KEY starting with: {api_key[:8]}... (keep this secret)")

    try:
        import runpod
    except Exception:
        print("❌ RunPod SDK not installed. Install it with: pip install runpod")
        return 1

    endpoint_id = derive_endpoint_id(url)
    if not endpoint_id:
        print("❌ Could not derive endpoint id from URL; expected format <id>.api.runpod.ai")
        return 1

    print(f"💡 Derived endpoint id for SDK: {endpoint_id}")
    ok, sdk_msg = runpod_sdk_healthcheck(endpoint_id, api_key)
    if ok:
        print("🟢 RunPod SDK health result:", sdk_msg)
        return 0
    else:
        print("❌ RunPod SDK health failed:", sdk_msg)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
