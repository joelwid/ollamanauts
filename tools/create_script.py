from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "agent_scripts"


def create_script(filename: str, content: str, overwrite: bool = False) -> dict[str, str]:
    """Create a Python script in the agent_scripts directory.

    Args:
        filename: Name of the Python file to create, such as hello.py.
        content: Full Python source code to write into the file.
        overwrite: Whether to replace an existing file with the same name.

    Returns:
        Metadata about the created script path.
    """
    if not filename.endswith(".py"):
        raise ValueError("filename must end with .py")

    target = (SCRIPTS_DIR / filename).resolve()
    try:
        target.relative_to(SCRIPTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("filename must stay inside the agent_scripts directory") from exc

    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(f"Script already exists: {filename}")

    SCRIPTS_DIR.mkdir(exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": str(target.relative_to(PROJECT_ROOT)),
        "status": "overwritten" if existed else "created",
    }
