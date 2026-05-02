from __future__ import annotations

import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "agent_scripts"


def _indent_block(text: str, spaces: int = 4) -> str:
    stripped = text.strip()
    if not stripped:
        return " " * spaces + "pass"
    return textwrap.indent(stripped, " " * spaces)


def create_script(
    filename: str,
    imports: str,
    constants: str,
    run_body: str,
    parse_args_body: str,
    params_expression: str,
    module_docstring: str = "",
    overwrite: bool = False,
) -> dict[str, str]:
    """Create a Python script from the standard agent template.

    Args:
        filename: Name of the Python file to create, such as hello.py.
        imports: Import statements to place at the top of the script.
        constants: Module-level constant definitions placed after imports.
        run_body: Body of the run(params) function.
        parse_args_body: Statements in the __main__ block before params assignment.
        params_expression: Python expression assigned to params in the __main__ block.
        module_docstring: Optional module docstring placed after imports.
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

    imports_block = imports.strip()
    constants_block = constants.strip()
    docstring_block = ""
    if module_docstring.strip():
        docstring_block = f'"""{module_docstring.strip()}"""'

    sections = []
    if imports_block:
        sections.append(imports_block)
    if constants_block:
        sections.append(constants_block)
    if docstring_block:
        sections.append(docstring_block)

    header = "\n\n".join(sections)
    if header:
        header = f"{header}\n\n"

    content = (
        f"{header}"
        "def run(params):\n"
        f"{_indent_block(run_body)}\n\n\n"
        'if __name__ == "__main__":\n'
        f"{_indent_block(parse_args_body)}\n"
        f"    params = {params_expression.strip()}\n"
        "    run(params)\n"
    )

    SCRIPTS_DIR.mkdir(exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": str(target.relative_to(PROJECT_ROOT)),
        "status": "overwritten" if existed else "created",
    }
