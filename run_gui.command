#!/bin/bash
# Desktop app launcher for macOS / Linux
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m spam_detector.gui
