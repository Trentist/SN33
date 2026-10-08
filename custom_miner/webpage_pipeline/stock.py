"""Render the validator's own webpage prompts.

The competitive miner overlays ``custom_miner/prompts`` onto every LlmLib call.
Those overlays are conversation- and microservices-biased. Webpage ground truth
is built from the stock templates, so this module loads that directory directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader

from conversationgenome.utils.Utils import Utils

from custom_miner.webpage_pipeline.gate import clean_tags

_STOCK_DIR = Path(__file__).resolve().parents[2] / "conversationgenome" / "llm" / "prompts"
_ENV = Environment(loader=FileSystemLoader(str(_STOCK_DIR)))
_NEIGHBORHOOD_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).resolve().parent / "prompts"))
)


def _render(env: Environment, name: str, **kwargs) -> str:
    return env.get_template(name).render(**kwargs)


def website_prompt(page: str, *, coding: bool) -> str:
    template = "website_to_metadata_coding.j2" if coding else "website_to_metadata.j2"
    return _render(_ENV, template, website_content=page)


def enrichment_prompt(snippet: str, *, coding: bool) -> str:
    template = "enrichment_to_metadata_coding.j2" if coding else "enrichment_to_metadata.j2"
    return _render(_ENV, template, enrichment_content=snippet)


def combine_prompt(groups: list[list[str]]) -> str:
    """Match ``PromptManager.combine_named_entities_prompt`` formatting."""
    chunks: list[str] = []
    for i, group in enumerate(groups):
        chunks.append(f"Set {i} :\n{','.join(group)}\n\n")
    return _render(_ENV, "combine_named_entities_prompt.j2", entities_str="".join(chunks))


def neighborhood_prompt(page: str, snippets: list[str], core_tags: list[str]) -> str:
    snippet_text = "\n---\n".join(snippets)[:1500]
    return _render(
        _NEIGHBORHOOD_ENV,
        "neighborhood.j2",
        page=(page or "")[:1000],
        snippets=snippet_text,
        core_tags=", ".join(core_tags[:12]),
    )


def paraphrase_prompt(tags: list[str]) -> str:
    """Ask for synonyms of the combine tags. Those strings feed the unique-score term."""
    return _render(
        _NEIGHBORHOOD_ENV,
        "paraphrase.j2",
        tags=", ".join(tags[:8]),
    )


def parse_tag_response(raw: str | None) -> list[str]:
    """Pull a comma-delimited tag list out of an LLM reply."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = text.replace("```", "").strip()
    lines = [re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        richest = max(lines, key=lambda line: line.count(","))
        text = richest if richest.count(",") >= 2 else ", ".join(lines)
    text = re.sub(r"(?:^|,)\s*\d+[\.\)]\s*", ", ", text)
    parts = Utils.clean_tags(re.split(r"[,;\n|]", text))
    return clean_tags(parts)
