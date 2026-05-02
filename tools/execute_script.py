from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "agent_scripts"


def execute_script(script_name: str, params: dict[str, object] | None = None) -> dict[str, object]:
    """Execute an agent-created script by calling its run(params) function.

    Args:
        script_name: Name of the script to execute from agent_scripts.
        params: Parameters passed into the script's run(params) function.

    Returns:
        Script execution metadata, captured stdout, and the run(params) return value.
    """
    filename = script_name if script_name.endswith(".py") else f"{script_name}.py"
    target = (SCRIPTS_DIR / filename).resolve()
    try:
        target.relative_to(SCRIPTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("script_name must stay inside the agent_scripts directory") from exc

    if not target.exists():
        raise FileNotFoundError(f"Script does not exist: {filename}")

    spec = importlib.util.spec_from_file_location(f"agent_script_{target.stem}", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script: {filename}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run = getattr(module, "run", None)
    if not callable(run):
        raise AttributeError(f"Script does not define callable run(params): {filename}")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run(params or {})

    return {
        "path": str(target.relative_to(PROJECT_ROOT)),
        "stdout": buffer.getvalue(),
        "result": result,
    }
