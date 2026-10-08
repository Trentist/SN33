"""Fetch SN33 WandB summary scores for a miner UID/hotkey (public GraphQL)."""

from __future__ import annotations

import json
import statistics
import urllib.request
from typing import Any


def _gql(query: str, variables: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 sn33-monitor"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_uid_scores(
    *,
    uid: int,
    hotkey: str,
    entity: str = "afterparty",
    project: str = "conversationgenome",
    max_runs: int = 40,
) -> dict[str, Any]:
    runs_q = f'''
    query {{
      project(name: "{project}", entityName: "{entity}") {{
        runs(first: {max_runs}, order: "-created_at") {{
          edges {{ node {{ name displayName state createdAt }} }}
        }}
      }}
    }}
    '''
    data = _gql(runs_q)
    runs = [e["node"] for e in data["data"]["project"]["runs"]["edges"]]

    rows: list[dict[str, Any]] = []
    for r in runs:
        q = """
        query($name: String!) {
          project(name: "%s", entityName: "%s") {
            run(name: $name) { summaryMetrics }
          }
        }
        """ % (project, entity)
        try:
            d = _gql(q, {"name": r["name"]})
            sm = d["data"]["project"]["run"]["summaryMetrics"]
            if not sm:
                continue
            smj = json.loads(sm) if isinstance(sm, str) else sm
        except Exception:
            continue

        hk = smj.get(f"hotkey.{uid}")
        final = smj.get(f"final_miner_score.{uid}")
        adj = smj.get(f"adjusted_score.{uid}")
        if final is None and hk is None:
            continue
        if hk and hk != hotkey:
            continue

        finals = [
            (int(k.split(".")[-1]), float(v))
            for k, v in smj.items()
            if k.startswith("final_miner_score.") and isinstance(v, (int, float))
        ]
        rank = None
        n_scored = len(finals)
        top = median = pct = None
        if finals and isinstance(final, (int, float)):
            finals_sorted = sorted(finals, key=lambda x: -x[1])
            rank = next((i + 1 for i, (u, _) in enumerate(finals_sorted) if u == uid), None)
            top = finals_sorted[0][1]
            median = statistics.median([v for _, v in finals_sorted])
            if rank and n_scored:
                pct = 100.0 * (n_scored - rank + 1) / n_scored

        rows.append(
            {
                "created_at": r["createdAt"],
                "run_id": r["name"],
                "run_name": r["displayName"],
                "state": r["state"],
                "final_miner_score": final,
                "adjusted_score": adj,
                "hotkey": hk,
                "rank": rank,
                "n_scored": n_scored,
                "percentile": pct,
                "median": median,
                "top": top,
                "task_id": smj.get(f"task_id.{uid}"),
                "_step": smj.get("_step"),
            }
        )

    finals = [r["final_miner_score"] for r in rows if isinstance(r["final_miner_score"], (int, float))]
    aggregate = None
    if finals:
        aggregate = {
            "n": len(finals),
            "mean": sum(finals) / len(finals),
            "min": min(finals),
            "max": max(finals),
        }

    return {
        "uid": uid,
        "hotkey": hotkey,
        "entity": entity,
        "project": project,
        "dashboard": f"https://wandb.ai/{entity}/{project}",
        "runs_scanned": len(runs),
        "matches": rows,
        "aggregate": aggregate,
    }
