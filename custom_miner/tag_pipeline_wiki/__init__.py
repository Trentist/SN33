"""WikiGround conversation tagging — external KB, not ReadyAI GT memory.

segment → mentions → Wikidata link → KG expand → evidence gate → select
"""

from __future__ import annotations

from custom_miner.tag_pipeline_wiki.pipeline import select_wiki_ground

__all__ = ["select_wiki_ground"]
