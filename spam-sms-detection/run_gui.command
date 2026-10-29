#!/bin/bash
# macOS / Linux startup script for Spam Detector Flask GUI
cd "$(dirname "$0")"
echo "Starting AI Spam & Phishing Detector..."
echo "Open your browser at: http://127.0.0.1:5000"
python3 flask_app.py
