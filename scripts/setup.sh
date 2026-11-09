#!/usr/bin/env bash
# Install the harness in .venv and prepare its default Ollama models.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [--python PATH] [--skip-models]

Creates .venv, installs the harness and test tools, then pulls its three models.
Install Python 3.11+, Git, ripgrep and Ollama 0.35+ before running this script.
Ollama must be running unless --skip-models is selected.

  --python PATH   Python interpreter to create .venv (default: python3)
  --skip-models   Install Python requirements without contacting Ollama
EOF
}

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
setup_python=python3
skip_models=false
while (($#)); do
    case "$1" in
        --python)
            if (($# < 2)); then
                printf '%s\n' 'Missing interpreter after --python.' >&2
                exit 2
            fi
            setup_python=$2
            shift 2
            ;;
        --skip-models) skip_models=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for command_name in git rg "$setup_python"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Missing requirement: %s. Install it and run setup again.\n' "$command_name" >&2
        exit 1
    fi
done
"$setup_python" -c 'import sys; sys.exit("Python 3.11 or newer is required.") if sys.version_info < (3, 11) else None'

if ! "$skip_models"; then
    if ! command -v ollama >/dev/null 2>&1; then
        printf '%s\n' 'Missing Ollama. Install it from https://ollama.com/download, then run setup again.' >&2
        exit 1
    fi
    if ! ollama list >/dev/null; then
        printf '%s\n' 'Cannot reach Ollama. Start it (ollama serve), then run setup again.' >&2
        exit 1
    fi
    "$setup_python" - <<'PY'
import json
import os
import re
import sys
from urllib.request import urlopen

host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if "://" not in host:
    host = "http://" + host
try:
    with urlopen(host + "/api/version", timeout=10) as response:
        version = json.load(response).get("version", "")
except (OSError, ValueError, AttributeError):
    sys.exit("Cannot read the Ollama server version. Check Ollama and OLLAMA_HOST.")
match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", version) if isinstance(version, str) else None
if match is None or tuple(int(part or 0) for part in match.groups()) < (0, 35, 0):
    sys.exit("Tev1 requires Ollama 0.35 or newer. Upgrade Ollama, restart it, then retry setup.")
print("Ollama server " + version + " supports Tev1 decisions.")
PY
fi

venv_dir="$project_dir/.venv"
if [[ ! -d "$venv_dir" ]]; then
    if ! "$setup_python" -m venv "$venv_dir"; then
        printf '%s\n' 'Could not create .venv. Install the Python venv component for your system and retry.' >&2
        exit 1
    fi
fi
venv_python="$venv_dir/bin/python"
if [[ ! -x "$venv_python" ]]; then
    printf '%s\n' 'The existing .venv has no usable Python interpreter. Repair or remove it, then retry.' >&2
    exit 1
fi
"$venv_python" -c 'import sys; sys.exit("The existing .venv requires Python 3.11 or newer.") if sys.version_info < (3, 11) else None'

package_spec="$project_dir[dev]"
"$venv_python" -m pip install --editable "$package_spec"

if ! "$skip_models"; then
    for model_name in qwen3:8b tev1:4b qwen3:14b; do
        ollama pull "$model_name"
    done
fi

printf '\n%s\n' 'Setup complete.' 'Activate with: source .venv/bin/activate'
printf '%s\n' 'Then run: patchrook-agent "Describe the coding task" --workspace /path/to/project'
