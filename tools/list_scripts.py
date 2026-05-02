from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "agent_scripts"


def list_scripts() -> list[dict[str, str]]:
    """List agent-created scripts and their module docstrings.

    Returns:
        A list of Python scripts in agent_scripts with their docstrings.
    """
    SCRIPTS_DIR.mkdir(exist_ok=True)
    results: list[dict[str, str]] = []

    for script_path in sorted(SCRIPTS_DIR.glob("*.py")):
        source = script_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        docstring = ast.get_docstring(module) or ""
        results.append(
            {
                "path": str(script_path.relative_to(PROJECT_ROOT)),
                "docstring": docstring,
            }
        )

    return results
