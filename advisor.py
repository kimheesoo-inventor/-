#!/usr/bin/env python3
"""Code review advisor: scans a file and prints suggestions."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_LINE_LEN = 100
MAX_FUNC_LINES = 50
MAX_NESTING = 4
MAX_PARAMS = 5


@dataclass
class Finding:
    line: int
    severity: str
    message: str

    def format(self, path: str) -> str:
        return f"{path}:{self.line}: [{self.severity}] {self.message}"


def scan_line_length(lines: list[str]) -> list[Finding]:
    out = []
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")
        if len(stripped) > MAX_LINE_LEN:
            out.append(Finding(i, "style", f"line exceeds {MAX_LINE_LEN} chars ({len(stripped)})"))
    return out


def scan_markers(lines: list[str]) -> list[Finding]:
    pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    out = []
    for i, line in enumerate(lines, 1):
        m = pattern.search(line)
        if m:
            out.append(Finding(i, "note", f"{m.group(1)} marker — track or resolve"))
    return out


def scan_trailing_whitespace(lines: list[str]) -> list[Finding]:
    out = []
    for i, line in enumerate(lines, 1):
        body = line.rstrip("\n")
        if body != body.rstrip():
            out.append(Finding(i, "style", "trailing whitespace"))
    return out


def scan_tabs_mixed(lines: list[str]) -> list[Finding]:
    has_tab = any("\t" in ln for ln in lines)
    has_space_indent = any(ln.startswith("    ") for ln in lines)
    if has_tab and has_space_indent:
        return [Finding(1, "style", "mixed tabs and spaces for indentation")]
    return []


def _func_depth(node: ast.AST, depth: int = 0) -> int:
    nesters = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)
    best = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesters):
            best = max(best, _func_depth(child, depth + 1))
        else:
            best = max(best, _func_depth(child, depth))
    return best


def scan_python_ast(source: str) -> list[Finding]:
    out: list[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Finding(e.lineno or 1, "error", f"syntax error: {e.msg}")]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            length = end - start + 1
            if length > MAX_FUNC_LINES:
                out.append(Finding(start, "complexity",
                                   f"function '{node.name}' is {length} lines (>{MAX_FUNC_LINES})"))
            param_count = len(node.args.args) + len(node.args.kwonlyargs)
            if param_count > MAX_PARAMS:
                out.append(Finding(start, "complexity",
                                   f"function '{node.name}' has {param_count} params (>{MAX_PARAMS})"))
            depth = _func_depth(node)
            if depth > MAX_NESTING:
                out.append(Finding(start, "complexity",
                                   f"function '{node.name}' nested {depth} deep (>{MAX_NESTING})"))
            if ast.get_docstring(node) is None and not node.name.startswith("_"):
                out.append(Finding(start, "doc",
                                   f"public function '{node.name}' has no docstring"))
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            out.append(Finding(node.lineno, "bug",
                               "bare except catches everything, including KeyboardInterrupt"))
        elif isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) and comp.value is None:
                    out.append(Finding(node.lineno, "style",
                                       "use 'is None' / 'is not None' instead of '== None'"))
    return out


def review(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    findings: list[Finding] = []
    findings += scan_line_length(lines)
    findings += scan_markers(lines)
    findings += scan_trailing_whitespace(lines)
    findings += scan_tabs_mixed(lines)
    if path.suffix == ".py":
        findings += scan_python_ast(text)
    findings.sort(key=lambda f: (f.line, f.severity))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code review advisor")
    ap.add_argument("paths", nargs="+", help="file(s) to review")
    ap.add_argument("--quiet", action="store_true", help="only print summary")
    args = ap.parse_args(argv)

    total = 0
    for p in args.paths:
        path = Path(p)
        if not path.is_file():
            print(f"{p}: not a file", file=sys.stderr)
            continue
        findings = review(path)
        total += len(findings)
        if not args.quiet:
            for f in findings:
                print(f.format(str(path)))
        print(f"{path}: {len(findings)} finding(s)")

    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
