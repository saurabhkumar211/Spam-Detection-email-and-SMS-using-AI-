"""Project paths, shared by the apps and the training script.

The package lives in src/spam_detector, while data and models stay at the
project root. SPAM_ROOT lets a container or a test point somewhere else.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("SPAM_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_FILE = DATA_DIR / "spam.csv"
MODEL_FILE = MODEL_DIR / "model.pkl"
VECTORIZER_FILE = MODEL_DIR / "vectorizer.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"
