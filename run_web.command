#!/bin/bash
# Web app launcher for macOS / Linux
cd "$(dirname "$0")"

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting AI Spam & Phishing Detector..."
echo "Open http://127.0.0.1:5000 in your browser."
exec "$PYTHON" -m spam_detector.app
