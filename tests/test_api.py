import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

# Predict Endpoint
@pytest.mark.parametrize(
    "text, model, expected_model",
    [
        ("I love this product!", "logistic", "tfidf_logistic"),
        ("This is terrible.", "bert", "distilbert"),
        ("It was okay.", "lstm", "lstm"),
    ]
)
def test_predict_valid(text, model, expected_model):
    response = client.post("/predict", json={"text": text, "model": model})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == expected_model
    assert data["sentiment"] in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_invalid_model():
    response = client.post("/predict", json={"text": "Test", "model": "invalid"})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_empty_text():
    response = client.post("/predict", json={"text": "", "model": "logistic"})
    assert response.status_code == 422  # Pydantic validation error

# Metrics Endpoint
def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "logistic" in data or "bert" in data or "lstm" in data

# Health Endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}
