import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

# Load model and vectorizer
with open(VECTORIZER_PATH, 'rb') as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    raw_model_data = pickle.load(f)

if isinstance(raw_model_data, dict):
    model = raw_model_data['ensemble']
    attribution_model = raw_model_data.get('attribution_model', model)
    threshold = raw_model_data.get('threshold', 0.50)
    model_type = raw_model_data.get('model_type', 'Ensemble')
else:
    model = raw_model_data
    attribution_model = raw_model_data
    threshold = 0.50
    model_type = 'Single Classifier'

feature_names = tfidf.get_feature_names_out()
if hasattr(attribution_model, 'feature_log_prob_'):
    word_spam_bias = {
        feature_names[i]: (attribution_model.feature_log_prob_[1][i] - attribution_model.feature_log_prob_[0][i])
        for i in range(len(feature_names))
    }
else:
    word_spam_bias = {}

def transform_text(text: str) -> str:
    """Preprocess text by lowercasing, tokenizing, removing stopwords/punctuation, and stemming."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = [i for i in tokens if i.isalnum() and i not in stop_words and i not in punct]
    stemmed = [ps.stem(i) for i in clean_tokens]
    return " ".join(stemmed)

def classify_text(text: str) -> dict:
    """
    Classify input text into Spam or Ham using calibrated probabilities and thresholding.
    Returns:
        dict containing:
            - is_spam (bool)
            - label (str: 'Spam' | 'Ham')
            - spam_probability (float)
            - ham_probability (float)
            - confidence (float)
            - transformed_text (str)
            - flagged_keywords (list of tuples: (token, score))
    """
    cleaned = text.strip()
    if not cleaned:
        return {
            "is_spam": False,
            "label": "Not Spam",
            "spam_probability": 0.0,
            "ham_probability": 100.0,
            "confidence": 100.0,
            "transformed_text": "",
            "flagged_keywords": []
        }

    transformed = transform_text(cleaned)
    vec = tfidf.transform([transformed]).toarray()
    
    probabilities = model.predict_proba(vec)[0]  # [p_ham, p_spam]
    ham_prob = float(probabilities[0])
    spam_prob = float(probabilities[1])
    is_spam = bool(spam_prob >= threshold)
    
    # Extract keywords present in input that strongly indicate spam
    tokens = transformed.split()
    flagged = []
    for t in set(tokens):
        if t in word_spam_bias and word_spam_bias[t] > 1.2:
            flagged.append((t, round(float(word_spam_bias[t]), 2)))
    flagged.sort(key=lambda x: x[1], reverse=True)

    confidence = spam_prob if is_spam else ham_prob

    return {
        "is_spam": is_spam,
        "label": "Spam" if is_spam else "Not Spam",
        "spam_probability": round(spam_prob * 100, 2),
        "ham_probability": round(ham_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "transformed_text": transformed,
        "flagged_keywords": flagged
    }
