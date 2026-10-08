"""Monitor configuration resolved from env / pm2 miner defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MonitorConfig:
    uid: int
    hotkey: str
    netuid: int
    entity: str
    project: str
    out_dir: Path
    pm2_name: str
    pm2_out_log: Path
    pm2_err_log: Path
    interval_s: float
    wandb_runs: int


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env", override=False)
    except Exception:
        pass


def default_config() -> MonitorConfig:
    _load_dotenv()
    uid = int(os.environ.get("MONITOR_UID", os.environ.get("MINER_UID", "244")))
    hotkey = os.environ.get(
        "MONITOR_HOTKEY",
        os.environ.get("MINER_HOTKEY", "5D7SBJHbqXYTQHnWKDgeDw8eVHpEa82Mcjb9sMGmpJgfmiUT"),
    )
    pm2_name = os.environ.get("MONITOR_PM2_NAME", "sn33-uid244-ready-1")
    out = Path(
        os.environ.get(
            "MONITOR_OUT_DIR",
            str(_ROOT / "local_dev" / "cache" / "monitor" / f"uid{uid}"),
        )
    )
    home = Path.home()
    return MonitorConfig(
        uid=uid,
        hotkey=hotkey,
        netuid=int(os.environ.get("MONITOR_NETUID", "33")),
        entity=os.environ.get("MONITOR_WANDB_ENTITY", "afterparty"),
        project=os.environ.get("MONITOR_WANDB_PROJECT", "conversationgenome"),
        out_dir=out,
        pm2_name=pm2_name,
        pm2_out_log=Path(
            os.environ.get(
                "MONITOR_PM2_OUT_LOG",
                str(home / ".pm2" / "logs" / f"{pm2_name}-out.log"),
            )
        ),
        pm2_err_log=Path(
            os.environ.get(
                "MONITOR_PM2_ERR_LOG",
                str(home / ".pm2" / "logs" / f"{pm2_name}-error.log"),
            )
        ),
        interval_s=float(os.environ.get("MONITOR_INTERVAL_S", "120")),
        wandb_runs=int(os.environ.get("MONITOR_WANDB_RUNS", "40")),
    )
