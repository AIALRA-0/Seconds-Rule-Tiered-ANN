#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os

TEXT_EXTS = {
    ".txt", ".py", ".sh", ".json", ".yaml", ".yml", ".ini", ".cfg"
}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules",
    "venv", ".venv", "env", ".env",
    "dist", "build", "target", ".idea", ".vscode"
}

def looks_binary(data: bytes) -> bool:
    return b"\x00" in data

def decode_text(data: bytes) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")

def main():
    root = Path(".").resolve()
    out_path = (root / "all_text_dump.txt").resolve()
    script_path = Path(__file__).resolve()

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for name in filenames:
                p = (Path(dirpath) / name).resolve()

                if p == out_path:
                    continue
                if p == script_path:   # Do not dump this script itself
                    continue
                if p.is_symlink():
                    continue
                if p.suffix.lower() not in TEXT_EXTS:
                    continue

                try:
                    data = p.read_bytes()
                except Exception:
                    continue

                if looks_binary(data):
                    continue

                text = decode_text(data)
                rel = p.relative_to(root)

                # Add the file name (relative path) at the beginning of each section
                out.write(f"{rel}\n")
                out.write("-" * 80 + "\n")
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
                out.write("\n")  # Leave a blank line between sections

    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()
