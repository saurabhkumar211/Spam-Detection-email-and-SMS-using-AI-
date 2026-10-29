# Spam and phishing detection for email and SMS

A machine learning app that reads a message and decides whether it is spam or legitimate. The Flask web app shows the label, the spam probability, and the words that pushed the decision.

The model trains on the SMS Spam Collection and is evaluated on a held-out 20% split. The stack is Python, Flask, scikit-learn, NLTK, and pandas. Training takes about a minute on a laptop CPU.

## Results

The test set has 1,034 messages: 903 legitimate and 131 spam. The scores below come from `models/metrics.json`.

| Metric | Score |
| --- | --- |
| Accuracy | 98.26% |
| Precision | 98.29% |
| Recall | 87.79% |
| F1 score | 92.74% |
| ROC-AUC | 99.53% |

The decision threshold is 0.5. You can change it in `classifier.py`.

## How it works

1. Clean the text: lowercase, tokenize with NLTK, remove stopwords and punctuation, then apply Porter stemming.
2. Vectorize with TF-IDF: unigrams and bigrams, 4,000 features, sublinear term frequency.
3. Classify with a soft-voting ensemble of Multinomial Naive Bayes, calibrated LinearSVC, ExtraTrees and LogisticRegression, weighted 1.5, 2.0, 1.5 and 1.0.
4. Show the tokens that pushed the message toward spam, taken from the Naive Bayes feature log probabilities.

Training scores the 80/20 split, then refits on every cleaned row before saving the model.

## ML notebook

The main ML record is [`notebook/spam_detection.ipynb`](notebook/spam_detection.ipynb). It starts from `data/spam.csv` and covers data cleaning, exploratory data analysis, text preprocessing, word clouds, and top spam and ham words. It then tests CountVectorizer and TF-IDF features and compares eleven classifiers on the TF-IDF set. The production version of the pipeline is `src/spam_detector/train.py`.

### Model comparison

The table shows the notebook's results on its 80/20 split.

| Model | Accuracy | Precision |
| --- | --- | --- |
| ExtraTrees | 97.40% | 100.00% |
| SVC | 96.59% | 97.62% |
| XGBoost | 96.50% | 97.60% |
| Random Forest | 96.41% | 98.36% |
| Bagging | 95.87% | 92.42% |
| Multinomial NB | 95.78% | 100.00% |
| AdaBoost | 95.34% | 93.44% |
| Logistic Regression | 94.44% | 94.44% |
| Gradient Boosting | 94.26% | 96.08% |
| Decision Tree | 92.38% | 85.44% |
| K-Neighbors | 89.87% | 100.00% |

ExtraTrees was the best single model at 97.40% accuracy. A soft-voting ensemble of SVC, Multinomial NB, and ExtraTrees reached 97.13% accuracy. Stacking the same three with a Random Forest final estimator reached 97.85%. The production pipeline uses four models and is scored separately in [Results](#results).

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
│   └── spam_detection.ipynb  # ML walkthrough and experiments
├── pyproject.toml
└── requirements.txt
```

## Install

Python 3.11 or newer. The model files were built with Python 3.14 and scikit-learn 1.9.1.

```bash
git clone https://github.com/saurabhkumar211/Spam-Detection-email-and-SMS-using-AI-.git
cd Spam-Detection-email-and-SMS-using-AI-
python -m venv venv
```

Activate the virtual environment, then install the package:

```bash
python -m pip install -e .
```

This installs the dependencies and two commands: `spam-detector-web` and `spam-detector-train`.

The first run downloads the NLTK `punkt`, `punkt_tab` and `stopwords` corpora, so it needs a network connection once.

## Usage

```bash
spam-detector-web
```

Open <http://127.0.0.1:5000>. The page checks a single message and loads sample messages. It also accepts a CSV or TXT file for batch classification and CSV export.

The app binds to `127.0.0.1:5000` by default. Set `HOST` and `PORT` to change the address.

You can also run the file directly:

```bash
cd src/spam_detector
python app.py
```

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

## Future work

The next version should add forensic metadata and URL analysis. Sender details, message headers, link reputation, and domain age would let the model judge the message text and its links together.
