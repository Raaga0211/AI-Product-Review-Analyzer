import pandas as pd
import numpy as np
import re
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv(r"C:\Users\RAAGA\mini-project\women_reviews_with_emojis.csv")

# ============================================
# CREATE SENTIMENT LABELS
# ============================================

def get_sentiment(rating):

    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Rating"].apply(get_sentiment)

# ============================================
# CLEAN TEXT
# ============================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ============================================
# COMBINE TEXT
# ============================================

df["Combined_Text"] = (
    df["Title"].fillna("").astype(str)
    + " "
    + df["Emoji_Review"].fillna("").astype(str)
)

df["Combined_Text"] = df["Combined_Text"].apply(clean_text)

# ============================================
# EMOJI SCORES
# ============================================

emoji_dict = {
    "😊": 2,
    "❤️": 2,
    "😍": 2,
    "😁": 2,
    "😃": 2,
    "✨": 1,
    "🔥": 1,
    "🤔": 0,
    "🙂": 0,
    "😌": 0,
    "😞": -2,
    "😢": -2,
    "😭": -2,
    "😠": -2,
    "😡": -2
}

def get_emoji_score(text):

    if pd.isna(text):
        return 0

    score = 0

    for token in str(text).split():
        score += emoji_dict.get(token, 0)

    return score

df["emoji_score"] = df["Emoji"].apply(get_emoji_score)

# ============================================
# FEATURES
# ============================================

X_text = df["Combined_Text"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_text = vectorizer.fit_transform(X_text)

emoji_feature = df["emoji_score"].values.reshape(-1,1)

X = hstack([X_text, emoji_feature])

# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================
# TRAIN MODEL
# ============================================

model = LogisticRegression(
    C=10,
    max_iter=5000
)

model.fit(X_train, y_train)

# ============================================
# EVALUATION
# ============================================

predictions = model.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nFiles Saved:")
print("sentiment_model.pkl")
print("tfidf_vectorizer.pkl")