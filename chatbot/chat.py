import os
import joblib
import pandas as pd
import torch
import numpy as np
from typing import TypedDict, Optional, Dict, Any

from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from LSTM import LSTMSentiment, SimpleTokenizer


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = D:/sentiment_analysis/src

TFIDF_DIR = os.path.join(BASE_DIR, "models", "TF_IDF_Logistic_Reg")
BERT_DIR = os.path.join(BASE_DIR, "models", "distilbert")
LSTM_DIR = os.path.join(BASE_DIR, "models", "lstm")

LLM_PATH = r"D:\sentiment_analysis\src\models\gpt4all-falcon-newbpe-q4_0.gguf"
CSV_PATH = os.path.join(BASE_DIR, "csv", "imdb_reviews.csv")


#  Load Models
# TF-IDF + Logistic Regression
tfidf_vectorizer = joblib.load(os.path.join(TFIDF_DIR, "tfidf_vectorizer.joblib"))
logreg_model = joblib.load(os.path.join(TFIDF_DIR, "logreg_tfidf.joblib"))

# DistilBERT
bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_DIR)
bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
bert_model.eval()

# LSTM
tokenizer = SimpleTokenizer()
tokenizer.word2idx = joblib.load(os.path.join(LSTM_DIR, "tokenizer_word2idx.joblib"))

vocab_size = len(tokenizer.word2idx)
lstm_model = LSTMSentiment(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=128,
    output_dim=2,
    n_layers=2
)
state_dict = torch.load(os.path.join(LSTM_DIR, "lstm_model.pth"), map_location="cpu")
lstm_model.load_state_dict(state_dict)
lstm_model.eval()


#  LLM
llm = GPT4All(
    model=LLM_PATH,
    max_tokens=256,
    verbose=False
)

#  Prompts
explain_prompt = PromptTemplate(
    input_variables=["text", "sentiment", "confidence", "model"],
    template=(
        "You are a sentiment analysis expert.\n\n"
        "Review: {text}\n"
        "Predicted Sentiment: {sentiment}\n"
        "Confidence: {confidence}\n"
        "Model Used: {model}\n\n"
        "Explain the decision in 2–3 sentences.\n"
        "List exactly 3 keywords influencing the prediction.\n\n"
        "Answer:"
    )
)


dataset_prompt = PromptTemplate(
    input_variables=["top_words"],
    template=(
        "These are common words in positive reviews:\n{top_words}\n\n"
        "Explain why these words appear more frequently."
    )
)


# Prediction Functions
def predict_logistic(text):
    X = tfidf_vectorizer.transform([text])
    probs = logreg_model.predict_proba(X)[0]
    pred = int(np.argmax(probs))
    return "positive" if pred == 1 else "negative", float(np.max(probs))


def predict_bert(text):
    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    return "positive" if pred == 1 else "negative", float(torch.max(probs))


def predict_lstm(text, max_len=100):
    tokens = tokenizer(text)[:max_len]
    tokens += [0] * (max_len - len(tokens))
    inputs = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = lstm_model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    return "positive" if pred == 1 else "negative", float(torch.max(probs))


#  State Definition
class ChatState(TypedDict):
    user_input: str
    model_choice: str
    prediction: Optional[Dict[str, Any]]
    explanation: Optional[str]
    dataset_insight: Optional[str]


# LangGraph Nodes
def predict_node(state: ChatState):
    text = state["user_input"]
    model_choice = state.get("model_choice", "logistic").lower()

    if model_choice == "logistic":
        sentiment, confidence = predict_logistic(text)
    elif model_choice == "bert":
        sentiment, confidence = predict_bert(text)
    elif model_choice == "lstm":
        sentiment, confidence = predict_lstm(text)
    else:
        raise ValueError("Model must be one of: logistic | bert | lstm")

    return {
        "prediction": {
            "sentiment": sentiment,
            "confidence": confidence,
            "model": model_choice,
        }
    }


def explain_node(state: ChatState):
    pred = state["prediction"]

    prompt = explain_prompt.format(
        text=state["user_input"],
        sentiment=pred["sentiment"],
        confidence=pred["confidence"],
        model=pred["model"],
    )

    explanation = llm(prompt)

    return {"explanation": explanation}


def dataset_node(state: ChatState, df: pd.DataFrame):
    pos_words = " ".join(
        df[df["label"] == 1]["text"].astype(str)
    ).split()[:50]

    prompt = dataset_prompt.format(top_words=pos_words)

    # ✅ CORRECT GPT4All CALL
    insight = llm(prompt)

    return {"dataset_insight": insight}


# Build Workflow
def build_graph(df=None):
    graph = StateGraph(ChatState)

    graph.add_node("PREDICT", predict_node)
    graph.add_node("EXPLAIN", explain_node)

    graph.add_edge("PREDICT", "EXPLAIN")

    if df is not None:
        graph.add_node("DATASET", lambda state: dataset_node(state, df))
        graph.add_edge("EXPLAIN", "DATASET")
        graph.add_edge("DATASET", END)
    else:
        graph.add_edge("EXPLAIN", END)

    graph.set_entry_point("PREDICT")
    return graph.compile()


# ---------------- CLI Runner ----------------
def run_chat(df_path=None):
    df = pd.read_csv(df_path) if df_path else None
    workflow = build_graph(df)

    print("\nLangChain + LangGraph Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("You: ")
        if text.lower() == "exit":
            break

        model_choice = input("Choose model (logistic | bert | lstm): ").lower()

        state = {
            "user_input": text,
            "model_choice": model_choice,
            "prediction": None,
            "explanation": None,
            "dataset_insight": None,
        }

        final_state = workflow.invoke(state)

        print("\n🔹 Sentiment:", final_state["prediction"]["sentiment"])
        print("🔹 Confidence:", final_state["prediction"]["confidence"])
        print("\n✨ Explanation:\n", final_state["explanation"])

        if df is not None:
            print("\n📊 Dataset Insight:\n", final_state["dataset_insight"])

        print("\n" + "-" * 40 + "\n")


# Main
if __name__ == "__main__":
    run_chat(CSV_PATH)
