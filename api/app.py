import os
import sys
import json
import joblib
import torch
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax

# Add src folder to sys.path to avoid ModuleNotFoundError
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.LSTM import LSTMSentiment, SimpleTokenizer  # LSTM model and tokenizer

# ---------------- App ----------------
app = FastAPI(title="Sentiment Analysis API")

MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------- Load TF-IDF + Logistic Regression ----------------
tfidf_path = os.path.join(MODEL_DIR, "TF_IDF_Logistic_Reg")
tfidf_vectorizer = joblib.load(os.path.join(tfidf_path, "tfidf_vectorizer.joblib"))
logreg_model = joblib.load(os.path.join(tfidf_path, "logreg_tfidf.joblib"))

# ---------------- Load DistilBERT ----------------
bert_path = os.path.join(MODEL_DIR, "distilbert")
bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()

# ---------------- Load LSTM ----------------
lstm_path = os.path.join(MODEL_DIR, "lstm")

# Load tokenizer word2idx (must match training)
tokenizer_word2idx_path = os.path.join(lstm_path, "tokenizer_word2idx.joblib")
tokenizer = SimpleTokenizer()
tokenizer.word2idx = joblib.load(tokenizer_word2idx_path)

# Recreate LSTM model
vocab_size = len(tokenizer.word2idx)
lstm_model = LSTMSentiment(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, output_dim=2, n_layers=2)

# Load state_dict
state_dict = torch.load(os.path.join(lstm_path, "lstm_model.pth"), map_location="cpu")
lstm_model.load_state_dict(state_dict)
lstm_model.eval()

# ---------------- Schemas ----------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., description="logistic | bert | lstm")

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model_used: str

class MetricsResponse(BaseModel):
    logistic: Optional[dict] = None
    bert: Optional[dict] = None
    lstm: Optional[dict] = None

# ---------------- Utils ----------------
def sentiment_label(idx: int):
    return "positive" if idx == 1 else "negative"

# ---------------- Predict ----------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    model_type = req.model.lower()

    if model_type == "logistic":
        X = tfidf_vectorizer.transform([text])
        probs = logreg_model.predict_proba(X)[0]
        pred = int(np.argmax(probs))

        return PredictResponse(
            sentiment=sentiment_label(pred),
            confidence=float(np.max(probs)),
            model_used="tfidf_logistic"
        )

    elif model_type == "bert":
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs = softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs))

        return PredictResponse(
            sentiment=sentiment_label(pred),
            confidence=float(torch.max(probs)),
            model_used="distilbert"
        )

    elif model_type == "lstm":
        # Tokenize input text
        tokens = tokenizer(text)
        max_len = 100  # same as training
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))  # pad/truncate
        inputs = torch.tensor([tokens], dtype=torch.long)  # batch of 1

        with torch.no_grad():
            logits = lstm_model(inputs)
            probs = softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs))

        return PredictResponse(
            sentiment=sentiment_label(pred),
            confidence=float(torch.max(probs)),
            model_used="lstm"
        )

    else:
        raise HTTPException(status_code=400, detail="Model must be logistic | bert | lstm")

# ---------------- Metrics ----------------
@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    metrics = {}

    try:
        with open(os.path.join(tfidf_path, "metrics_logreg.json")) as f:
            metrics["logistic"] = json.load(f)

        with open(os.path.join(bert_path, "metrics.json")) as f:
            metrics["bert"] = json.load(f)

        with open(os.path.join(lstm_path, "metrics.json")) as f:
            metrics["lstm"] = json.load(f)

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"status": "running"}
