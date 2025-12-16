#!/usr/bin/env bash
set -euo pipefail

if ! command -v tree >/dev/null 2>&1; then
  echo "tree command not found. Install it or update tree.txt manually."
  exit 1
fi

tree -a -I ".venv|__pycache__|data|results" > tree.txt
( cd src && tree -a -I "__pycache__" > tree.txt )
( cd scripts && tree -a -I "__pycache__" > tree.txt )
( cd configs && tree -a -I "__pycache__" > tree.txt )
( cd tests && tree -a -I "__pycache__" > tree.txt )

echo "Updated tree.txt files."
