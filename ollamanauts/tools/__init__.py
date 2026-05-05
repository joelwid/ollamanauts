from collections.abc import Callable

from .subagents import make_deploy_subagent_tool

DEFAULT_TOOLS: tuple[Callable[..., object], ...] = ()
