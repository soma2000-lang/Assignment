import pytest
from fastapi.testclient import TestClient
from typing import List

from models.rating_predict import  predict_with_model 
from .api_service import api
from models.RAG import RAG_pipeline

client = TestClient(api)

class MockModel:
    def predict(self, X):
        return [5.0]  # Mocked prediction

def mock_predict_with_model(model, X):
    return [5.0]  # Mocked prediction

@pytest.fixture
def override_predict_with_model(monkeypatch):
    monkeypatch.setattr("models.rating_predict.predict_with_model", mock_predict_with_model)

@pytest.fixture
def override_RAG_pipeline(monkeypatch):
    def mock_RAG_pipeline(query, texts, bm25_1, bm25_2):
        return ["Mocked answer"]
    monkeypatch.setattr("models.RAG.RAG_pipeline", mock_RAG_pipeline)

def test_predict_rating(override_predict_with_model):
    response = client.post("/predict_rating", json={"review": "This is a test review"})
    assert response.status_code == 200
    assert response.json() == {"request_id": 1, "rating": 5.0}

def test_get_answer(override_RAG_pipeline):
    response = client.post("/get_answer", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
    assert response.json() == {"request_id": 2, "results": ["Mocked answer"]}
