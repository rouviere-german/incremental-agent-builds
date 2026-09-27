"""The AST outline program run in the isolated child interpreter (stdlib only).

The parent starts ``python -I -c <this file's source>``, so this module must not import anything
from the package. ``outline`` is also imported in-process by the workload generator to snapshot the
initial API with exactly the same normalization.
"""

import ast
import json
import sys
from typing import Any


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0].strip()


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(node.args)
    returns = " -> " + ast.unparse(node.returns) if node.returns is not None else ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return prefix + node.name + "(" + args + ")" + returns


def _class_symbol(node: ast.ClassDef) -> dict[str, Any]:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    methods = [
        {
            "name": item.name,
            "signature": _signature(item),
            "doc": _first_line(ast.get_docstring(item)),
        }
        for item in node.body
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        and not item.name.startswith("_")
    ]
    return {
        "kind": "class",
        "name": node.name,
        "signature": "class " + node.name + ("(" + bases + ")" if bases else ""),
        "doc": _first_line(ast.get_docstring(node)),
        "methods": methods,
    }


def outline(source: str) -> dict[str, Any]:
    """Top-level imports and public symbols of a module; comments and layout are ignored."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return {"error": type(exc).__name__}
    imports: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend({"module": a.name, "level": 0, "names": []} for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = sorted(alias.name for alias in node.names)
            imports.append({"module": node.module or "", "level": node.level, "names": names})
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                symbols.append(
                    {
                        "kind": "function",
                        "name": node.name,
                        "signature": _signature(node),
                        "doc": _first_line(ast.get_docstring(node)),
                    }
                )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            symbols.append(_class_symbol(node))
    return {"doc": ast.get_docstring(tree) or "", "imports": imports, "symbols": symbols}


def main() -> None:
    sys.setrecursionlimit(2000)
    for line in sys.stdin.buffer:
        request = json.loads(line)
        try:
            result = outline(request["source"])
        except (RecursionError, MemoryError) as exc:
            result = {"error": type(exc).__name__}
        response = {"id": request["id"], "result": result}
        sys.stdout.buffer.write(json.dumps(response, sort_keys=True).encode("ascii") + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
