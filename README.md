# Spam and phishing detection for email and SMS

A machine learning app that reads a message and decides whether it is spam or legitimate. One classifier sits behind two interfaces: a Flask web app and a CustomTkinter desktop app. Both show a label, a spam probability, and the words that pushed the decision.

The model is trained on the SMS Spam Collection dataset and evaluated on a held-out 20% split. The full pipeline runs on a laptop CPU in about a minute.

## Results

Held-out test set: 1,034 messages, 903 legitimate and 131 spam.

| Metric | Score |
| --- | --- |
| Accuracy | 98.26% |
| Precision | 98.29% |
| Recall | 87.79% |
| F1 score | 92.74% |
| ROC-AUC | 99.53% |

|  | Predicted ham | Predicted spam |
| --- | --- | --- |
| Actual ham | 901 | 2 |
| Actual spam | 16 | 115 |

Two legitimate messages were flagged as spam, and sixteen spam messages were missed. That trade-off is deliberate. In a messaging product, hiding a real message is worse than letting one spam text through, so the decision threshold sits at 0.5. You can move it in `classifier.py` if you care more about recall.

## How it works

1. Normalize the text: lowercase, tokenize with NLTK, remove stopwords and punctuation, then apply Porter stemming.
2. Vectorize with TF-IDF: unigrams and bigrams, 4,000 features, sublinear term frequency.
3. Classify with a soft-voting ensemble of Multinomial Naive Bayes, calibrated LinearSVC, ExtraTrees and LogisticRegression, weighted 1.5, 2.0, 1.5 and 1.0.
4. Explain the result with the tokens that pushed the message toward spam. These come from the Naive Bayes feature log probabilities, not from a hand-written keyword list.

The training script evaluates on the 80/20 split, then refits the final model on every cleaned row before saving it.

## Project structure

```
.
├── src/spam_detector/
│   ├── __init__.py
│   ├── config.py           # Project paths
│   ├── classifier.py       # Preprocessing, prediction and explanations
│   ├── app.py              # Flask web app
│   ├── gui.py              # Desktop app
│   ├── train.py            # Training and evaluation
│   └── templates/
│       └── index.html      # Web interface
├── tests/
│   └── test_classifier.py
├── data/
│   └── spam.csv            # SMS Spam Collection
├── models/
│   ├── model.pkl           # Ensemble plus attribution model
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── metrics.json        # Benchmark scores
├── notebook/
│   └── spam_detection.ipynb
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── run_web.command         # macOS / Linux launcher for the web app
└── run_gui.command         # macOS / Linux launcher for the desktop app
```

## Install

Use Python 3.11 or newer. The model files here were built with Python 3.14 and scikit-learn 1.9.1.

```bash
git clone https://github.com/saurabhkumar211/Spam-Detection-email-and-SMS-using-AI-.git
cd Spam-Detection-email-and-SMS-using-AI-
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The editable install adds the three commands used below and pulls the packages listed in `requirements.txt`. On Windows, activate the environment with `venv\Scripts\activate`.

NLTK downloads the `punkt`, `punkt_tab` and `stopwords` corpora the first time `classifier.py` is imported. The downloads need a network connection. After that, the apps work offline.

If you use Homebrew Python on macOS, Tk is packaged separately. Install it with `brew install python-tk@3.14` before running the desktop app. Without it you will see `ModuleNotFoundError: No module named '_tkinter'`.

## Usage

### Web app

```bash
spam-detector-web
```

Open http://127.0.0.1:5000. The page has a single-message checker with sample messages, batch upload with CSV export, and the model metrics shown at the bottom.

To expose it on a different host or port, set `HOST` and `PORT`:

```bash
HOST=0.0.0.0 PORT=8080 spam-detector-web
```

### Desktop app

```bash
spam-detector-gui
```

The desktop app has dark and light themes, live confidence bars, keyword breakdowns, and batch processing. On macOS you can also double-click `run_web.command` or `run_gui.command`.

### Batch files

Upload a `.csv` or `.txt` file. For CSV, the app looks for a column named `text`, `v2`, `message`, `sms`, `email`, `content` or `body`, then falls back to the first text column it finds. Classified rows can be downloaded as CSV.

### API

```bash
curl -X POST http://127.0.0.1:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You have won a free prize, claim now"}'
```

```json
{
  "is_spam": true,
  "label": "Spam",
  "spam_probability": 97.36,
  "ham_probability": 2.64,
  "confidence": 97.36,
  "flagged_keywords": [["claim", 5.28], ["http", 4.02]]
}
```

There are two more endpoints:

- `POST /api/batch` accepts a multipart `file` field and returns per-row results with spam and ham counts.
- `POST /api/download_csv` takes the JSON results and returns a CSV file.

### Docker

```bash
docker build -t spam-detector .
docker run -p 5000:5000 spam-detector
```

The image sets `HOST=0.0.0.0`. Local runs bind to `127.0.0.1` unless you override it.

## Retrain

```bash
spam-detector-train
```

Training removes nulls and duplicates from `data/spam.csv`, builds the features, runs the stratified split, prints the benchmark, refits on all rows, and overwrites the three files in `models/`. Expect about a minute on a laptop CPU.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover text cleaning and a handful of known spam and ham messages.

## Dataset

The SMS Spam Collection starts with 5,572 rows. After removing duplicates and nulls, 5,169 remain: 4,516 legitimate messages and 653 spam messages. Spam is 12.6% of the data, and the stratified split keeps that ratio in both train and test.

## Design notes

The choices that shaped the result:

- TF-IDF with bigrams and 4,000 features. With roughly five thousand short messages, sparse linear models train in seconds and generalize well. A transformer would be overkill and slower to demo.
- A soft-voting ensemble instead of a single model. Naive Bayes handles word frequencies well, LinearSVC gives a clean boundary, ExtraTrees picks up interactions between tokens, and logistic regression adds a stable linear view. The SVM carries the highest weight at 2.0, with Naive Bayes and ExtraTrees at 1.5 and logistic regression at 1.0.
- Precision over recall. The model blocks very few legitimate messages (2 out of 903) and accepts that it misses some spam (16 out of 131).
- Explanations come from model weights. `classifier.py` reads the Naive Bayes log probability difference for each token, so the highlighted keywords reflect what the model learned.
- One preprocessing path. Every interface calls the same `classifier.py`, which keeps the web app, desktop app and training data on identical cleaning steps.

## Limitations

- The training data is SMS, not email. Long bodies and HTML email are out of distribution, so treat email results as untested.
- There is no URL reputation, domain age or sender check. A short, well-written phishing message can still pass.
- Recall is 87.8%, which means roughly one in eight spam messages gets through.
- The saved artifacts depend on the scikit-learn version that created them. If loading fails after an upgrade, run `spam-detector-train` again.

## Possible next steps

- Add URL and sender features. They matter more for email than word frequencies do.
- Train separate models for SMS and email rather than sharing one.
- Calibrate probabilities per channel so a confidence score means the same thing everywhere.
- Store corrected labels from user feedback and retrain on them.

## Notebook

`notebook/spam_detection.ipynb` holds the exploratory work: class balance, message length distributions, word clouds, confusion matrices, and a comparison of Naive Bayes, SVM, ExtraTrees, Random Forest, Logistic Regression, k-nearest neighbours, Decision Tree and Gaussian Naive Bayes. Its baseline artifacts are written to `models/nb_baseline*.pkl` and are separate from the production ensemble.
