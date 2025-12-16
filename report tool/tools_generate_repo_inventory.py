#!/usr/bin/env python3

"""
Generate a file-by-file repository inventory in Markdown.

Usage (from repo root):
    python tools_generate_repo_inventory.py

Output:
    appendix_repo_inventory.md

Design goals:
- Zero external dependencies (stdlib only)
- Robust to encoding errors
- Skips large/generated folders by default
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "results", "data", "datasets", "dist", "build", ".venv", "venv",
    ".idea", ".vscode", ".DS_Store",
}

EXCLUDE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".bin", ".pt", ".pth", ".onnx", ".npy", ".npz",
}

MAX_SUMMARY_CHARS = 160
MAX_TOPLEVEL_NAMES = 12


@dataclass(frozen=True)
class FileInfo:
    relpath: str
    ftype: str
    summary: str


def _is_excluded(rel: Path) -> bool:
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return True
    if rel.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def _summarize_python(path: Path) -> str:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        mod = ast.parse(src)
        doc = ast.get_docstring(mod) or ""
        doc_first = ""
        for line in doc.splitlines():
            line = line.strip()
            if line:
                doc_first = line
                break

        names: List[str] = []
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)

        bits: List[str] = []
        if doc_first:
            bits.append(doc_first)

        if names:
            shown = names[:MAX_TOPLEVEL_NAMES]
            suffix = " ..." if len(names) > MAX_TOPLEVEL_NAMES else ""
            bits.append("Top-level: " + ", ".join(shown) + suffix)

        out = " | ".join(bits).strip()
        return out[:MAX_SUMMARY_CHARS]
    except Exception as e:
        return f"(python parse error: {type(e).__name__})"


def _summarize_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Skip pure comment lines (best-effort)
            if line.startswith("#") or line.startswith("//"):
                continue
            return line[:MAX_SUMMARY_CHARS]
    except Exception:
        pass
    return ""


def summarize_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".py":
        return _summarize_python(path)
    # For other text-ish formats, try first meaningful line
    if suf in {".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".sh", ".bat"}:
        return _summarize_text(path)
    return ""


def iter_repo_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if _is_excluded(rel):
            continue
        yield rel, p


def build_inventory(root: Path) -> List[FileInfo]:
    out: List[FileInfo] = []
    for rel, abs_path in iter_repo_files(root):
        ftype = abs_path.suffix.lower().lstrip(".") or "file"
        summary = summarize_file(abs_path).replace("\n", " ").strip()
        out.append(FileInfo(relpath=rel.as_posix(), ftype=ftype, summary=summary))
    return out


def write_markdown(inventory: List[FileInfo], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Appendix - Repository inventory (auto-generated)\n\n")
    lines.append("This appendix is auto-generated to ensure the report can accurately cover **every file**.\n\n")
    lines.append("## How to regenerate\n\n")
    lines.append("From repo root:\n\n")
    lines.append("```bash\npython tools_generate_repo_inventory.py\n```\n\n")
    lines.append("---\n\n")
    lines.append("| File | Type | Summary |\n")
    lines.append("|---|---|---|\n")
    for fi in inventory:
        summary = fi.summary.replace("|", "\\|")
        lines.append(f"| `{fi.relpath}` | {fi.ftype} | {summary} |\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(".").resolve()
    inventory = build_inventory(root)
    out_path = root / "appendix_repo_inventory.md"
    write_markdown(inventory, out_path)
    print(f"Wrote {out_path} ({len(inventory)} files)")


if __name__ == "__main__":
    main()
