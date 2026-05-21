import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) or '.'))

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def get_weighted_w2v(tokens, model, tfidf, size=100):
    size = getattr(model, 'vector_size', size)
    vec = np.zeros(size)
    weight_sum = 0
    for word in tokens:
        if word in model.wv and word in tfidf.vocabulary_:
            weight = tfidf.idf_[tfidf.vocabulary_[word]]
            vec += model.wv[word] * weight
            weight_sum += weight
    if weight_sum == 0:
        return np.zeros(size)
    return vec / weight_sum

DATA_DIR = os.path.dirname(os.path.abspath(__file__) or '.')
MODEL_DIR = os.path.join(DATA_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

data_file = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
try:
    df = pd.read_csv(data_file, encoding='latin-1', header=None)
except FileNotFoundError:
    print(f"\n⚠️ Dataset not found at {data_file}")
    print("Please place the dataset inside the 'data' folder next to train.py.")
    sys.exit(1)

df = df[[0, 5]]
df.columns = ['label', 'text']

df = df.sample(n=10000, random_state=42)

df['label'] = df['label'].apply(lambda x: 1 if x == 4 else 0)

df['tokens'] = df['text'].apply(clean_text)
df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=1000)
tfidf.fit(train_df['clean_text'])

w2v = Word2Vec(train_df['tokens'], vector_size=100, window=5, min_count=1)
w2v.save(os.path.join(MODEL_DIR, "word2vec.model"))

X_train = np.array([get_weighted_w2v(t, w2v, tfidf) for t in train_df['tokens']])
X_test = np.array([get_weighted_w2v(t, w2v, tfidf) for t in test_df['tokens']])

y_train = train_df['label']
y_test = test_df['label']

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))

print("\n✅ Training Complete!")
