#!/bin/bash
# Web app launcher for macOS / Linux
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting AI Spam & Phishing Detector..."
echo "Open http://127.0.0.1:5000 in your browser."
python3 -m spam_detector.app
