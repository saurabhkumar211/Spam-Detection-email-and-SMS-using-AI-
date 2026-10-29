# 🛡️ AI Spam & Phishing Detection (Email and SMS)

A machine learning application that detects spam, phishing, and scam messages across email and SMS communications in real time using NLP and Multinomial Naive Bayes.

---

## 🚀 Key Features

* **Dual GUI Options**:
  * **Desktop GUI (`CustomTkinter`)**: Native desktop window with Dark/Light theme switching, live confidence meters, keyword diagnostic breakdowns, and batch CSV processing.
  * **Web Dashboard (`Streamlit`)**: Modern multi-tab browser application with probability charts, sample loaders, batch CSV classifier with export, and model analytics.
* **Explainable AI Diagnostics**: Displays tokenized & stemmed representations along with high-risk spam keywords that triggered the prediction.
* **Batch Classification**: Upload any `.csv` or `.txt` file, analyze thousands of messages simultaneously, preview stats, and download classified results.
* **High Accuracy & Precision**:
  * **Accuracy**: 95.94%
  * **Precision**: 100.0% (Zero false positives on test data)

---

## 🖥️ How to Run

### Option 1: Desktop GUI (CustomTkinter)
Run via Python or double-click `run_desktop_gui.bat`:
```powershell
python launch_gui.py
```
*(Or inside `spam-sms-detection/`: `python gui_app.py`)*

### Option 2: Web Dashboard (Streamlit)
Run via Python or double-click `run_web_gui.bat`:
```powershell
python -m streamlit run spam-sms-detection/app.py
```

---

## 📂 Project Structure

```
├── launch_gui.py             # Root launcher for Desktop GUI
├── run_desktop_gui.bat       # Windows 1-click batch launcher for Desktop GUI
├── run_web_gui.bat           # Windows 1-click batch launcher for Web GUI
├── requirements.txt          # Python dependencies
└── spam-sms-detection/
    ├── app.py                # Enhanced Streamlit Web GUI
    ├── gui_app.py            # CustomTkinter Desktop GUI
    ├── classifier.py         # Shared NLP pipeline & classification engine
    ├── model.pkl             # Trained MultinomialNB model
    ├── vectorizer.pkl        # Fitted TF-IDF Vectorizer
    ├── spam.csv              # Training dataset (5,169 records)
    └── spam-detection.ipynb  # Exploratory Data Analysis & Model training
```

---

## 🧠 Model Pipeline

1. **Text Normalization**: Lowercase conversion & word tokenization (`nltk.word_tokenize`).
2. **Noise Reduction**: Alphanumeric filtering, English stopwords removal (`nltk.corpus.stopwords`), and punctuation stripping.
3. **Stemming**: Root word extraction using Porter Stemmer (`nltk.stem.PorterStemmer`).
4. **Vectorization**: TF-IDF weighting (`sklearn.feature_extraction.text.TfidfVectorizer`).
5. **Classification**: Multinomial Naive Bayes (`sklearn.naive_bayes.MultinomialNB`).
