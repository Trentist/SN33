"""Reserve live task bundles from the ReadyAI Conversation Server API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from conversationgenome.api.ApiLib import ApiLib
from conversationgenome.task_bundle.TaskBundle import TaskBundle

_ROOT = Path(__file__).resolve().parents[1]


def load_readyai_api_key() -> str:
    """Load key from env or JSON files (testnet file preferred for local_dev)."""
    env_key = (os.environ.get("READYAI_API_KEY") or "").strip()
    if env_key:
        return env_key

    candidates = []
    explicit = (os.environ.get("READYAI_API_KEY_FILE") or "").strip()
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            _ROOT / "testnet_readyai_api_data.json",
            _ROOT / "readyai_api_data.json",
            Path.cwd() / "testnet_readyai_api_data.json",
            Path.cwd() / "readyai_api_data.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text())
            key = (data.get("api_key") or "").strip()
            if key:
                return key
    raise FileNotFoundError(
        "No ReadyAI API key found. Set READYAI_API_KEY or place "
        "testnet_readyai_api_data.json / readyai_api_data.json in the repo root."
    )


DEFAULT_CACHE_DIR = _ROOT / "local_dev" / "cache" / "testnet_problems"


async def reserve_live_bundle(
    *,
    netuid: Optional[int] = None,
    hotkey: Optional[str] = None,
    api_key: Optional[str] = None,
) -> TaskBundle:
    """
    Pull one live task bundle from ReadyAI (same endpoint validators use).

    Default netuid=138 (testnet). Mainnet (33) requires a mainnet validator key.
    """
    netuid = int(netuid if netuid is not None else os.environ.get("READYAI_NETUID", "138"))
    hotkey = hotkey or os.environ.get("READYAI_HOTKEY", "local-dev-hotkey")
    api_key = api_key or load_readyai_api_key()

    api = ApiLib()
    bundle = await api.reserve_task_bundle(hotkey=hotkey, netuid=netuid, api_key=api_key)
    if not bundle:
        raise RuntimeError(f"ReadyAI API returned no task bundle (netuid={netuid})")
    return bundle


async def reserve_raw_json(
    *,
    netuid: Optional[int] = None,
    hotkey: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Reserve and return the raw JSON payload (for caching)."""
    import requests
    from conversationgenome import __version__ as CGP_VERSION
    from conversationgenome.ConfigLib import c
    from conversationgenome.utils.Utils import Utils

    netuid = int(netuid if netuid is not None else os.environ.get("READYAI_NETUID", "138"))
    api_key = api_key or load_readyai_api_key()
    read_host_url = c.get("env", "CGP_API_READ_HOST", "https://api.conversations.xyz")
    read_host_port = c.get("env", "CGP_API_READ_PORT", "443")
    http_timeout = Utils._float(c.get("env", "HTTP_TIMEOUT", 60))
    url = f"{read_host_url}:{read_host_port}/api/v1/conversation/reserve?cgp_version={CGP_VERSION}&netuid={netuid}"
    headers = {
        "Accept": "application/json",
        "Accept-Language": "en_US",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, json={}, timeout=http_timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Reserve failed HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


async def fetch_and_cache_problems(
    *,
    count: int = 20,
    netuid: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    task_type: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> Path:
    """
    Reserve `count` live testnet problems and save them under cache_dir.
    Returns the cache directory path.
    """
    from conversationgenome.task_bundle.task_bundle_factory import try_parse_task_bundle

    cache_dir = Path(cache_dir or os.environ.get("READYAI_CACHE_DIR") or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    index = {"netuid": int(netuid if netuid is not None else os.environ.get("READYAI_NETUID", "138")), "problems": []}
    if index_path.is_file():
        previous = json.loads(index_path.read_text())
        index["problems"] = list(previous.get("problems") or [])

    seen = {str(row.get("guid")) for row in index["problems"]}
    saved = 0
    attempts = 0
    attempt_cap = max_attempts if max_attempts is not None else (count if not task_type else count * 15)
    while saved < count and attempts < attempt_cap:
        attempts += 1
        raw = await reserve_raw_json(netuid=netuid)
        guid = str(raw.get("guid"))
        if guid in seen:
            guid = f"{guid}_{attempts}"
        seen.add(str(raw.get("guid")))
        bundle = try_parse_task_bundle(raw)
        if not bundle:
            raise RuntimeError(f"Could not parse reserved bundle guid={raw.get('guid')}")
        if task_type and bundle.type != task_type:
            print(
                f"[attempt {attempts}/{attempt_cap}] skip {bundle.type} guid={raw.get('guid')} "
                f"(wanted {task_type}, saved {saved}/{count})"
            )
            continue
        i = len(index["problems"])
        fname = f"{i:02d}_{bundle.type}_{guid}.json"
        out = cache_dir / fname
        out.write_text(json.dumps(raw, indent=2))
        index["problems"].append(
            {
                "file": fname,
                "guid": raw.get("guid"),
                "type": bundle.type,
                "mode": raw.get("mode"),
            }
        )
        print(f"[{saved+1}/{count}] cached {bundle.type} guid={raw.get('guid')} -> {fname}")
        saved += 1

    if saved < count:
        print(f"Stopped after {attempts} reserves with {saved}/{count} matching problems")

    index_path.write_text(json.dumps(index, indent=2))
    print(f"Wrote index: {index_path} ({len(index['problems'])} problems)")
    return cache_dir


def list_cached_problems(cache_dir: Optional[Path] = None) -> list[dict]:
    cache_dir = Path(cache_dir or os.environ.get("READYAI_CACHE_DIR") or DEFAULT_CACHE_DIR)
    index_path = cache_dir / "index.json"
    if not index_path.is_file():
        return []
    return json.loads(index_path.read_text()).get("problems") or []


def load_cached_bundle(index: int = 0, cache_dir: Optional[Path] = None) -> TaskBundle:
    from conversationgenome.task_bundle.task_bundle_factory import try_parse_task_bundle

    cache_dir = Path(cache_dir or os.environ.get("READYAI_CACHE_DIR") or DEFAULT_CACHE_DIR)
    problems = list_cached_problems(cache_dir)
    if not problems:
        raise FileNotFoundError(f"No cached problems in {cache_dir}. Run: python -m local_dev.fetch_problems --count 20")
    if index < 0 or index >= len(problems):
        raise IndexError(f"cache index {index} out of range 0..{len(problems)-1}")
    path = cache_dir / problems[index]["file"]
    raw = json.loads(path.read_text())
    bundle = try_parse_task_bundle(raw)
    if not bundle:
        raise RuntimeError(f"Failed to parse cached bundle {path}")
    return bundle
