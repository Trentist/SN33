"""Persist miner-facing problems received from validators (mainnet/testnet).

Saves the raw CgSynapse.cgp_input task payload plus light metadata so we can
replay / analyze live problems offline without needing ReadyAI validator keys.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = _ROOT / "local_dev" / "cache" / "mainnet_problems"


def problems_dir() -> Path:
    raw = os.environ.get("MINER_PROBLEMS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # Prefer explicit network; default to mainnet cache (finney / netuid 33)
    network = (os.environ.get("NETWORK") or os.environ.get("BT_SUBTENSOR_NETWORK") or "").strip().lower()
    netuid = str(os.environ.get("NETUID") or os.environ.get("BT_NETUID") or "").strip()
    if network in ("test", "testnet") or netuid == "138":
        return _ROOT / "local_dev" / "cache" / "testnet_live_problems"
    if network in ("finney", "mainnet", "main") or netuid == "33":
        return DEFAULT_DIR
    # If both mainnet + testnet caches exist, prefer mainnet; else default mainnet
    return DEFAULT_DIR


def saving_enabled() -> bool:
    return os.environ.get("SAVE_MINER_PROBLEMS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _safe_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return str(obj)


def _task_type(cgp_input: Any) -> str:
    try:
        task = cgp_input[0]["task"]
        if isinstance(task, dict):
            return str(task.get("type") or "unknown")
        return str(getattr(task, "type", "unknown"))
    except Exception:
        return "unknown"


def save_miner_request(
    *,
    cgp_input: Any,
    validator_hotkey: str | None = None,
    netuid: int | None = None,
    axon_hotkey: str | None = None,
    mine_result: Any = None,
    elapsed_s: float | None = None,
    error: str | None = None,
) -> Path | None:
    """Write one received problem (+ optional mine result) to disk. Returns path or None."""
    if not saving_enabled():
        return None

    out_dir = problems_dir()
    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    ms = int(time.time() * 1000) % 1000
    ttype = _task_type(cgp_input)
    # Short hotkey suffix for uniqueness
    hk = (validator_hotkey or "unknown")[-8:]
    fname = f"{stamp}_{ms:03d}_{ttype}_{hk}.json"
    path = tasks_dir / fname

    payload = {
        "saved_at": ts.isoformat(),
        "netuid": netuid,
        "network": os.environ.get("NETWORK") or os.environ.get("BT_SUBTENSOR_NETWORK"),
        "validator_hotkey": validator_hotkey,
        "miner_hotkey": axon_hotkey,
        "task_type": ttype,
        "elapsed_s": elapsed_s,
        "error": error,
        # Exact miner-facing synapse body
        "cgp_input": _safe_json(cgp_input),
        "mine_result": _safe_json(mine_result) if mine_result is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Update index
    index_path = out_dir / "index.json"
    index: dict[str, Any]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            index = {"problems": []}
    else:
        index = {"problems": []}

    problems = index.get("problems") or []
    problems.append(
        {
            "file": fname,
            "saved_at": payload["saved_at"],
            "task_type": ttype,
            "netuid": netuid,
            "validator_hotkey": validator_hotkey,
            "elapsed_s": elapsed_s,
            "error": error,
            "has_result": mine_result is not None and error is None,
        }
    )
    # Keep last 5000 entries in index
    index["problems"] = problems[-5000:]
    index["updated_at"] = payload["saved_at"]
    index["count"] = len(index["problems"])
    index["dir"] = str(tasks_dir)
    index_path.write_text(json.dumps(index, indent=2))

    return path


def list_saved_problems(limit: int = 50) -> list[dict[str, Any]]:
    index_path = problems_dir() / "index.json"
    if not index_path.is_file():
        return []
    data = json.loads(index_path.read_text())
    problems = data.get("problems") or []
    return list(reversed(problems[-limit:]))
