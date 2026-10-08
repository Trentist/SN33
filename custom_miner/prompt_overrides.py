"""Install overlay PromptManager: custom prompts win, stock prompts fall back."""

from __future__ import annotations

from pathlib import Path

from jinja2 import ChoiceLoader
from jinja2 import Environment
from jinja2 import FileSystemLoader

from conversationgenome.llm.prompt_manager import PromptManager

_CUSTOM_DIR = Path(__file__).parent / "prompts"
_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "conversationgenome" / "llm" / "prompts"

_installed = False


class OverlayPromptManager(PromptManager):
    def __init__(self, custom_dir: Path = _CUSTOM_DIR, default_dir: Path = _DEFAULT_DIR):
        if not default_dir.is_dir():
            raise FileNotFoundError(f"Default prompt directory not found: {default_dir}")
        loaders = []
        if custom_dir.is_dir():
            loaders.append(FileSystemLoader(str(custom_dir)))
        loaders.append(FileSystemLoader(str(default_dir)))
        self.env = Environment(loader=ChoiceLoader(loaders))


def install_custom_prompts() -> OverlayPromptManager:
    """Monkeypatch the module-level prompt_manager used by LlmLib."""
    global _installed
    overlay = OverlayPromptManager()
    import conversationgenome.llm.LlmLib as llmlib_mod
    import conversationgenome.llm.prompt_manager as pm_mod

    pm_mod.prompt_manager = overlay
    llmlib_mod.prompt_manager = overlay
    _installed = True
    return overlay


def is_installed() -> bool:
    return _installed
