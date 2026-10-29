# Spam and phishing detection for email and SMS

A machine learning app that reads a message and decides whether it is spam or legitimate. The Flask web app shows the label, the spam probability, and the words that pushed the decision.

The model trains on the SMS Spam Collection and is evaluated on a held-out 20% split. Training takes about a minute on a laptop CPU.

## Results

The test set has 1,034 messages: 903 legitimate and 131 spam.

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

Two legitimate messages were flagged and sixteen spam messages were missed. That is deliberate. In a messaging product, hiding a real message is worse than letting one spam text through, so the threshold sits at 0.5. You can change it in `classifier.py`.

## How it works

1. Clean the text: lowercase, tokenize with NLTK, remove stopwords and punctuation, then apply Porter stemming.
2. Vectorize with TF-IDF: unigrams and bigrams, 4,000 features, sublinear term frequency.
3. Classify with a soft-voting ensemble of Multinomial Naive Bayes, calibrated LinearSVC, ExtraTrees and LogisticRegression, weighted 1.5, 2.0, 1.5 and 1.0.
4. Show the tokens that pushed the message toward spam, taken from the Naive Bayes feature log probabilities.

Training scores the 80/20 split, then refits on every cleaned row before saving the model.

## Project structure

```
.
├── src/spam_detector/
│   ├── __init__.py
│   ├── config.py           # Project paths
│   ├── classifier.py       # Preprocessing, prediction and explanations
│   ├── app.py              # Flask web app
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
└── run_web.command         # macOS / Linux launcher
```

## Install

Python 3.11 or newer. The model files were built with Python 3.14 and scikit-learn 1.9.1.

```bash
git clone https://github.com/saurabhkumar211/Spam-Detection-email-and-SMS-using-AI-.git
cd Spam-Detection-email-and-SMS-using-AI-
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

This installs the dependencies and two commands: `spam-detector-web` and `spam-detector-train`. On Windows, activate the environment with `venv\Scripts\activate`.

The first run downloads the NLTK `punkt`, `punkt_tab` and `stopwords` corpora, so it needs a network connection once.

## Usage

```bash
spam-detector-web
```

Open <http://127.0.0.1:5000>. The page checks a single message and loads sample messages. It also accepts a CSV or TXT file for batch classification and CSV export.

To change the host or port, set `HOST` and `PORT`:

```bash
HOST=0.0.0.0 PORT=8080 spam-detector-web
```

You can also run the file directly:

```bash
cd src/spam_detector
python app.py
```

On macOS you can double-click `run_web.command`. It uses `venv/bin/python` when that folder exists.

### Batch files

Upload a `.csv` or `.txt` file. For CSV, the app looks for a column named `text`, `v2`, `message`, `sms`, `email`, `content` or `body`, then falls back to the first text column it finds.

### API

`POST /api/classify` takes a JSON body and returns the label and probabilities:

```bash
curl -X POST http://127.0.0.1:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You have won a free prize, claim now"}'
```

```json
{
  "is_spam": true,
  "label": "Spam",
  "spam_probability": 98.23,
  "ham_probability": 1.77,
  "confidence": 98.23,
  "flagged_keywords": [["claim", 5.28], ["prize", 5.14], ["free", 1.76]]
}
```

`POST /api/batch` accepts a `.csv` or `.txt` file and returns per-row results. `POST /api/download_csv` returns the results as a CSV file.

## Retrain

```bash
spam-detector-train
```

Training removes nulls and duplicates from `data/spam.csv`, prints the benchmark, refits on all rows, and overwrites the three files in `models/`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Seven tests cover text cleaning and a few known spam and ham messages.

## Dataset and limits

The SMS Spam Collection has 5,572 rows. After removing duplicates and nulls, 5,169 remain: 4,516 legitimate and 653 spam, so spam is 12.6% of the data.

- The training data is SMS, not email. Long email bodies are untested.
- There is no URL or sender check, so a careful phishing message can pass.
- Recall is 87.8%, which means about one in eight spam messages is missed.

## Design notes

- TF-IDF with bigrams and 4,000 features. Five thousand short messages do not need a bigger model, and sparse linear models train in seconds.
- Soft voting across four models. Naive Bayes handles word frequencies, LinearSVC draws a clean boundary, ExtraTrees picks up token interactions, and logistic regression adds a stable linear view. The SVM carries the highest weight at 2.0.

## Notebook

`notebook/spam_detection.ipynb` has the exploratory work: class balance, message lengths, word clouds, confusion matrices, and a comparison of eight classifiers.
