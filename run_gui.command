#!/bin/bash
# Desktop app launcher for macOS / Linux
cd "$(dirname "$0")"

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m spam_detector.gui
