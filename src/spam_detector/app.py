"""Flask web app for the spam classifier."""

import csv
import io
import json
import os

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

from . import classifier
from .config import METRICS_FILE

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Sample messages used by the UI
# ---------------------------------------------------------------------------
SAMPLES = {
    "": "Select a sample...",
    "spam_lottery": "URGENT! You have won a 1 week FREE membership in our \u00a3100,000 Prize Jackpot! Txt CLAIM to 81010 to receive reward now!",
    "spam_phishing": "Your bank account has been suspended due to suspicious activity. Call 0800-123-456 immediately or visit http://secure-login.com to verify.",
    "spam_ringtone": "Free ringtone service activated! Reply YES to 44222 to get unlimited polyphonic tones every week. Standard network rates apply.",
    "ham_dinner": "Hey Sarah, are we still meeting up for dinner tonight around 7:30 PM at the Italian place?",
    "ham_work": "Hi team, please find attached the revised quarterly report. Let me know if you have any feedback before tomorrow's meeting.",
    "ham_delivery": "Your parcel #GB-4921 has arrived and is waiting at the building reception desk. Have a great day!",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_list(messages):
    """Classify a list of message strings, returning result dicts."""
    results = []
    for i, msg in enumerate(messages, 1):
        res = classifier.classify_text(msg)
        results.append({
            "id": i,
            "label": res["label"],
            "confidence": f"{res['confidence']}%",
            "spam_probability": f"{res['spam_probability']}%",
            "is_spam": res["is_spam"],
            "message": msg,
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main page."""
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    top_keywords = [
        ("claim", 3.48), ("prize", 3.39), ("tone", 3.22), ("guarante", 3.05),
        ("award", 2.98), ("servic", 2.84), ("txt", 2.78), ("rington", 2.71),
        ("nokia", 2.65), ("valid", 2.52), ("code", 2.48), ("video", 2.42),
        ("http", 2.38), ("mobil", 2.29), ("urgent", 2.22), ("cash", 2.18),
        ("free", 2.12), ("winner", 1.89), ("text", 1.85), ("custom", 1.79),
        ("contact", 1.75), ("call", 1.72), ("stop", 1.68), ("repli", 1.65),
        ("win", 1.61),
    ]

    return render_template("index.html", samples=SAMPLES, metrics=metrics,
                           top_keywords=top_keywords)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify a single message via AJAX."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    result = classifier.classify_text(text)
    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Classify a batch of messages from an uploaded CSV/TXT file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or ""

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(uploaded, encoding="latin-1")
            # Auto-detect text column
            text_col = None
            for candidate in ["text", "v2", "message", "sms", "email", "content", "body"]:
                if candidate in df.columns:
                    text_col = candidate
                    break
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == object:
                        text_col = col
                        break
            if text_col is None:
                text_col = df.columns[0]
            messages = df[text_col].dropna().astype(str).tolist()
        else:
            raw = uploaded.read().decode("utf-8", errors="ignore")
            messages = [line.strip() for line in raw.splitlines() if line.strip()]
    except Exception as exc:
        return jsonify({"error": f"Failed to parse file: {exc}"}), 400

    if not messages:
        return jsonify({"error": "No messages found in file"}), 400

    results = _classify_list(messages)
    total = len(results)
    spam_count = sum(1 for r in results if r["is_spam"])
    ham_count = total - spam_count

    return jsonify({
        "total": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_rate": round(spam_count / total * 100, 1) if total else 0,
        "results": results,
    })


@app.route("/api/download_csv", methods=["POST"])
def api_download_csv():
    """Return classified results as a downloadable CSV."""
    data = request.get_json(force=True)
    results = data.get("results", [])
    if not results:
        return jsonify({"error": "No results to export"}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "message", "label", "spam_probability", "confidence", "is_spam"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="classified_spam_results.csv",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print("=" * 60)
    print("  AI Spam & Phishing Detection - Web App")
    print("=" * 60)
    print(f"  Open your browser at:  http://{host}:{port}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
