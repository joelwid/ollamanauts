from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "agent_scripts"


def read_script(script_name: str) -> dict[str, str]:
    """Read an agent-created script from the agent_scripts directory.

    Args:
        script_name: Name of the script to read from agent_scripts.

    Returns:
        Script metadata and full file contents.
    """
    filename = script_name if script_name.endswith(".py") else f"{script_name}.py"
    target = (SCRIPTS_DIR / filename).resolve()
    try:
        target.relative_to(SCRIPTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("script_name must stay inside the agent_scripts directory") from exc

    if not target.exists():
        raise FileNotFoundError(f"Script does not exist: {filename}")

    return {
        "path": str(target.relative_to(PROJECT_ROOT)),
        "content": target.read_text(encoding="utf-8"),
    }
