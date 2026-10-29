"""ComfyUI のカスタムノード入口（buqi-code/buqi-minimax-h3-multigpu の 2 台版、2026-10-28）。"""

from .minimax_sp import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
