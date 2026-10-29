import os
import sys
import json
import time
import pickle
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure NLTK corpora
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'spam.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
VEC_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

print("=" * 65)
print("[*] PRODUCTION MODEL TRAINING & OPTIMIZATION PIPELINE")
print("=" * 65)

# 1. Load Data
print("\n[1/6] Loading and cleaning dataset from spam.csv...")
df = pd.read_csv(DATA_PATH, encoding='latin-1')[['v1', 'v2']].rename(columns={'v1': 'target', 'v2': 'text'})
raw_count = len(df)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
df = df.dropna().drop_duplicates(keep='first').reset_index(drop=True)
clean_count = len(df)
spam_count = int(df['target'].sum())
ham_count = clean_count - spam_count
print(f"  - Raw rows: {raw_count}")
print(f"  - Deduplicated clean rows: {clean_count}")
print(f"  - Legitimate (Ham): {ham_count} ({ham_count/clean_count*100:.1f}%)")
print(f"  - Spam / Phishing: {spam_count} ({spam_count/clean_count*100:.1f}%)")

# 2. Preprocess Text
print("\n[2/6] Running NLP text normalization (tokenization + stopword removal + stemming)...")
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    words = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words and w not in punct]
    return ' '.join(words)

start_time = time.time()
df['transformed'] = df['text'].apply(clean_text)
print(f"  - Normalized {len(df)} texts in {time.time() - start_time:.2f} seconds.")

# 3. Stratified Train / Test Split
print("\n[3/6] Creating stratified 80/20 train/test split...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['transformed'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
)
print(f"  - Training set: {len(X_train_raw)} samples")
print(f"  - Held-out test set: {len(X_test_raw)} samples (Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()})")

# 4. Feature Extraction (TF-IDF with unigrams + bigrams and sublinear TF)
print("\n[4/6] Vectorizing text with TF-IDF (n-grams: 1-2, max_features: 4000, sublinear_tf: True)...")
vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vectorizer.fit_transform(X_train_raw).toarray()
X_test = vectorizer.transform(X_test_raw).toarray()
print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)} features")

# 5. Model Training & Evaluation
print("\n[5/6] Training and benchmarking classifiers...")

# We build individual high-performing estimators
mnb = MultinomialNB(alpha=0.3)
lsvc = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))
lr = LogisticRegression(C=5.0, solver='liblinear', random_state=42)
etc = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Build Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[('mnb', mnb), ('lsvc', lsvc), ('etc', etc), ('lr', lr)],
    voting='soft',
    weights=[1.5, 2.0, 1.5, 1.0]
)

print("  - Training Soft Voting Ensemble (MultinomialNB + Calibrated LinearSVC + ExtraTrees + LogisticRegression)...")
ensemble.fit(X_train, y_train)

# Evaluate on Held-out Test Set
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
# Optimal decision threshold = 0.50 (balanced high precision & recall)
y_pred = (y_pred_proba >= 0.50).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("[*] HELD-OUT TEST BENCHMARK RESULTS")
print("=" * 65)
print(f"  [+] Accuracy:             {acc * 100:.2f}%")
print(f"  [+] Precision:            {prec * 100:.2f}% (False Positive Rate: {cm[0, 1] / (cm[0, 0] + cm[0, 1]) * 100:.2f}%)")
print(f"  [+] Recall / Sensitivity: {rec * 100:.2f}%")
print(f"  [+] F1-Score:             {f1 * 100:.2f}%")
print(f"  [+] ROC-AUC Score:        {roc_auc * 100:.2f}%")
print("\n  Confusion Matrix:")
print(f"    - True Legitimate (Ham) correctly allowed: {cm[0, 0]}")
print(f"    - False Positives (Ham wrongly blocked):    {cm[0, 1]}")
print(f"    - False Negatives (Spam missed):            {cm[1, 0]}")
print(f"    - True Spam correctly detected:             {cm[1, 1]}")
print("=" * 65)

# Also train the base MultinomialNB to store feature log probabilities for XAI token attribution
standalone_mnb = MultinomialNB(alpha=0.3)
standalone_mnb.fit(X_train, y_train)

# 6. Final Fit on All Clean Data & Model Serialization
print("\n[6/6] Fitting final production model on full cleaned dataset (5,169 samples)...")
X_full = vectorizer.fit_transform(df['transformed']).toarray()
y_full = df['target'].values

production_ensemble = VotingClassifier(
    estimators=[
        ('mnb', MultinomialNB(alpha=0.3)),
        ('lsvc', CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))),
        ('etc', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(C=5.0, solver='liblinear', random_state=42))
    ],
    voting='soft',
    weights=[1.5, 2.0, 1.5, 1.0]
)
production_ensemble.fit(X_full, y_full)

# Fit production MNB for feature log probabilities
production_mnb = MultinomialNB(alpha=0.3)
production_mnb.fit(X_full, y_full)

# Save artifacts
print("  - Saving vectorizer.pkl...")
with open(VEC_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

print("  - Saving model.pkl (Package containing Ensemble + Attribution Weights)...")
model_package = {
    'ensemble': production_ensemble,
    'attribution_model': production_mnb,
    'threshold': 0.50,
    'feature_names': vectorizer.get_feature_names_out(),
    'model_type': 'Soft-Voting Ensemble (MNB + Calibrated LinearSVC + ExtraTrees + LogisticRegression)'
}
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_package, f)

# Save metrics JSON
metrics_report = {
    "model_type": model_package['model_type'],
    "clean_samples": clean_count,
    "vocabulary_size": len(vectorizer.vocabulary_),
    "test_accuracy": round(float(acc) * 100, 2),
    "test_precision": round(float(prec) * 100, 2),
    "test_recall": round(float(rec) * 100, 2),
    "test_f1_score": round(float(f1) * 100, 2),
    "test_roc_auc": round(float(roc_auc) * 100, 2),
    "confusion_matrix": {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics_report, f, indent=2)

print(f"  - Saved benchmark report to {METRICS_PATH}")
print("\n[+] PRODUCTION MODEL TRAINING COMPLETED SUCCESSFULLY!")

