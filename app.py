import gradio as gr
import joblib
import numpy as np
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

MODEL_DIR = "models"

clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
w2v = Word2Vec.load(os.path.join(MODEL_DIR, "word2vec.model"))

def clean_text(text):
    text = text.lower()
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

def analyze(text):
    if not text.strip():
        return "⚠️ Please enter text", "", 0

    tokens = clean_text(text)
    vec = get_weighted_w2v(tokens, w2v, tfidf).reshape(1, -1)

    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0].max()

    sentiment = "Positive" if pred == 1 else "Negative"
    emoji = "😊" if sentiment == "Positive" else "😔"

    result = f"## {sentiment} {emoji}"
    confidence_text = f"Confidence: {prob:.2%}"

    return result, confidence_text, float(prob)

with gr.Blocks(theme=gr.themes.Monochrome()) as app:

    gr.Markdown("# 🌙 Sentiment Analyzer")
    gr.Markdown("### Analyze emotions with AI ⚡")

    text_input = gr.Textbox(
        placeholder="Type something like: I love this product!",
        lines=4
    )

    btn = gr.Button("Analyze Sentiment")

    output = gr.Markdown()
    confidence = gr.Text()
    conf_bar = gr.Slider(0, 1, step=0.01, label="Confidence")

    btn.click(
        fn=analyze,
        inputs=text_input,
        outputs=[output, confidence, conf_bar]
    )

    gr.Markdown("---")
    gr.Markdown("Built by Sathvik 🚀")

app.launch()
